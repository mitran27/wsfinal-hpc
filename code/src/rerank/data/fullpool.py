"""Training pools over the full MS MARCO passage collection (8.8M passages).

The course corpus (1M passages) holds the judged positive of only 60,201 of the
502,939 judged training queries. Training on all of them needs pools retrieved
from the full collection (`data.collection_train`); validation, calibration and
every reported evaluation stay on the course corpus.

Same BM25 as the course pools: tokenizer, k1, b and depth from the config. Each
posting's BM25 impact is precomputed once, with the same float32 operations as
`BM25.search`, so a query costs one scatter-add per term.

    rerank fullpool      ->  work/index_full/, work/splits/train500.qids, work/candidates/train500.tsv
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from rerank.config import Config
from rerank.data import io
from rerank.data.bm25 import tokenize
from rerank.data.splits import HELD_OUT, normalize, split_spec


def build_full_split(cfg: Config) -> dict:
    """All judged training queries, minus held-out overlaps (id or normalized text) and the val/calib queries."""
    d = cfg.data
    qrels = io.load_qrels(d.qrels_train)
    queries = io.load_queries(d.queries_train, set(qrels))
    held_ids, held_text = set(), set()
    for name in HELD_OUT:
        held = split_spec(cfg, name).queries()
        held_ids |= set(held)
        held_text |= {normalize(t) for t in held.values()}
    reserved = set(io.read_qids(cfg.splits_dir / "val.qids")) | set(io.read_qids(cfg.splits_dir / "calib.qids"))
    stats = {"train_queries_with_qrels": len(qrels), "missing_text": 0, "overlap_id": 0,
             "overlap_text": 0, "val_or_calib": 0}
    keep = []
    for qid in sorted(qrels, key=io.qid_sort_key):
        if qid not in queries:
            stats["missing_text"] += 1
        elif qid in held_ids:
            stats["overlap_id"] += 1
        elif normalize(queries[qid]) in held_text:
            stats["overlap_text"] += 1
        elif qid in reserved:
            stats["val_or_calib"] += 1
        else:
            keep.append(qid)
    stats["train500"] = len(keep)
    io.write_qids(keep, cfg.splits_dir / "train500.qids")
    (cfg.splits_dir / "train500.stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def build_full_index(collection: Path, out_dir: Path, chunk: int = 500_000) -> dict:
    """Document-major chunks (one CSR block per `chunk` passages), joined and turned column-major once."""
    start = time.time()
    vocab: dict[str, int] = {}
    pids: list[str] = []
    lengths: list[int] = []
    indptrs, indices, data = [], [], []
    ind, dat, ptr = [], [], [0]

    def flush() -> None:
        indptrs.append(np.asarray(ptr, dtype=np.int64))
        indices.append(np.asarray(ind, dtype=np.int32))
        data.append(np.asarray(dat, dtype=np.float32))
        ind.clear(), dat.clear(), ptr.clear(), ptr.append(0)

    with open(collection, encoding="utf-8") as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            pids.append(line[:tab])
            tokens = tokenize(line[tab + 1:])
            lengths.append(len(tokens))
            counts: dict[int, int] = {}
            for t in tokens:
                tid = vocab.setdefault(t, len(vocab))
                counts[tid] = counts.get(tid, 0) + 1
            ind.extend(counts.keys())
            dat.extend(counts.values())
            ptr.append(len(ind))
            if len(ptr) > chunk:
                flush()
    if len(ptr) > 1:
        flush()

    offsets = np.cumsum([0] + [len(i) for i in indices[:-1]])
    indptr = np.concatenate([indptrs[0]] + [p[1:] + o for p, o in zip(indptrs[1:], offsets[1:])])
    indices, data = np.concatenate(indices), np.concatenate(data)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(len(pids), len(vocab)))
    del indices, data, indptr
    matrix = matrix.tocsc()
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "bm25_matrix.npz", matrix, compressed=False)
    np.savez(out_dir / "bm25_meta.npz", pids=np.asarray(pids), lengths=np.asarray(lengths, np.int32))
    (out_dir / "bm25_vocab.json").write_text(json.dumps(vocab))
    stats = {"passages": len(pids), "terms": len(vocab), "postings": int(matrix.nnz),
             "build_seconds": round(time.time() - start, 1)}
    (out_dir / "bm25_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class ImpactIndex:
    """BM25 with each posting's impact precomputed: idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avglen))."""

    def __init__(self, index_dir: Path, k1: float, b: float) -> None:
        m = sp.load_npz(index_dir / "bm25_matrix.npz").tocsc()
        meta = np.load(index_dir / "bm25_meta.npz")
        self.pids = meta["pids"]
        lengths = meta["lengths"].astype(np.float32)
        self.vocab: dict[str, int] = json.loads((index_dir / "bm25_vocab.json").read_text())
        k1 = np.float32(k1)
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        df = np.diff(m.indptr).astype(np.float32)
        n = len(self.pids)
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        tf = m.data
        self.impact = np.repeat(idf, np.diff(m.indptr)) * tf * (k1 + 1) / (tf + norm[m.indices])
        self.indptr, self.indices = m.indptr, m.indices
        self.max_impact = np.maximum.reduceat(self.impact, self.indptr[:-1])   # every term has ≥ 1 posting
        self.fallbacks = 0

    def search(self, query: str, k: int, rare_df: int = 200_000) -> list[tuple[str, float]]:
        """Exact top k. Fast path: score only passages holding a rare query term (df ≤ rare_df). Any other passage
        scores at most U = the sum of the common terms' largest impacts; if U is below the k-th best candidate
        score, no other passage can enter the top k. Otherwise (or with too few candidates) score every passage."""
        tids = [self.vocab[t] for t in dict.fromkeys(tokenize(query)) if t in self.vocab]
        if not tids:
            return []
        rare = [t for t in tids if self.indptr[t + 1] - self.indptr[t] <= rare_df]
        if rare:
            cand = np.unique(np.concatenate([self.indices[self.indptr[t]:self.indptr[t + 1]] for t in rare]))
            if len(cand) >= k:
                s = np.zeros(len(cand), dtype=np.float32)
                for t in tids:                         # query order, as in the full scan: identical float32 sums
                    lo, hi = self.indptr[t], self.indptr[t + 1]
                    docs = self.indices[lo:hi]
                    pos = np.minimum(np.searchsorted(docs, cand), hi - lo - 1)
                    hit = docs[pos] == cand
                    s[hit] += self.impact[lo + pos[hit]]
                top = np.argpartition(-s, k - 1)[:k]
                bound = float(sum(self.max_impact[t] for t in tids if t not in rare))
                if bound * (1 + 1e-5) < s[top].min():          # margin covers float32 rounding of the sums
                    top = top[np.argsort(-s[top], kind="stable")]
                    return [(str(self.pids[cand[i]]), float(s[i])) for i in top]
        self.fallbacks += 1
        return self.full_search(query, k)

    def full_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.pids), dtype=np.float32)
        hit = False
        for term in dict.fromkeys(tokenize(query)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            hit = True
            lo, hi = self.indptr[tid], self.indptr[tid + 1]
            scores[self.indices[lo:hi]] += self.impact[lo:hi]
        if not hit:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(str(self.pids[i]), float(scores[i])) for i in top if scores[i] > 0]


_INDEX: ImpactIndex | None = None


def _search_batch(args):
    items, depth = args
    return [(qid, _INDEX.search(text, depth)) for qid, text in items]


def retrieve_full(cfg: Config, workers: int, limit: int | None = None) -> dict:
    """Top-`depth` BM25 candidates over the full collection for every train500 query, in parallel (fork)."""
    global _INDEX
    _INDEX = ImpactIndex(cfg.work / "index_full", cfg.retrieval.k1, cfg.retrieval.b)
    qids = io.read_qids(cfg.splits_dir / "train500.qids")[:limit]
    text = io.load_queries(cfg.data.queries_train, set(qids))
    qrels = io.load_qrels(cfg.data.qrels_train)
    items = [(q, text[q]) for q in qids]
    batches = [(items[i:i + 500], cfg.retrieval.depth) for i in range(0, len(items), 500)]
    out = cfg.candidates_dir / ("train500.tsv" if limit is None else f"train500.first{limit}.tsv")
    tmp = out.with_suffix(".tmp")
    start, done, with_pos = time.time(), 0, 0
    with mp.get_context("fork").Pool(workers) as pool, open(tmp, "w", encoding="utf-8") as f:
        f.write(io.CANDIDATE_HEADER + "\n")
        for result in pool.imap(_search_batch, batches):
            for qid, hits in result:
                judged = qrels.get(qid, {})
                with_pos += any(judged.get(pid, 0) >= 1 for pid, _ in hits)
                for rank, (pid, score) in enumerate(hits, start=1):
                    f.write(f"{qid}\t{pid}\t{rank}\t{score:.6f}\t{judged.get(pid, io.UNJUDGED)}\n")
            done += len(result)
            if done % 20_000 < 500:
                el = time.time() - start
                print(f"{done}/{len(items)} queries, {el / 60:.1f} min, eta {el / done * (len(items) - done) / 60:.0f} min",
                      flush=True)
    tmp.rename(out)
    stats = {"queries": len(items), "with_positive_in_top": with_pos,
             "ms_per_query_per_worker": round(1000 * (time.time() - start) * workers / max(len(items), 1), 1)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    return stats
