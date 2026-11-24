"""BM25 candidate generation over the course corpus.

Same scoring as assignments 2 and 3 (k1 = 1.2, b = 0.75, positive IDF, ASCII
alphanumeric tokenizer), stored as a column-major sparse term-document matrix
so that one query term's postings are one contiguous column.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from rerank.config import Config
from rerank.data import io
from rerank.data.splits import SplitSpec


def tokenize(text: str) -> list[str]:
    """Lowercase ASCII letters and digits; everything else separates tokens."""
    tokens, current = [], []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            current.append(ch.lower())
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def build_index(collection: Path, out_dir: Path) -> dict:
    """Tokenize the collection once and save the sparse index."""
    start = time.time()
    vocab: dict[str, int] = {}
    pids: list[str] = []
    lengths: list[int] = []
    rows, cols, tfs = [], [], []
    buf_r, buf_c, buf_t = [], [], []

    def flush() -> None:
        rows.append(np.asarray(buf_r, dtype=np.int32))
        cols.append(np.asarray(buf_c, dtype=np.int32))
        tfs.append(np.asarray(buf_t, dtype=np.float32))
        buf_r.clear(), buf_c.clear(), buf_t.clear()

    with open(collection, encoding="utf-8") as f:
        for doc, line in enumerate(f):
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
            row = len(pids) - 1
            buf_r.extend([row] * len(counts))
            buf_c.extend(counts.keys())
            buf_t.extend(counts.values())
            if len(buf_r) > 5_000_000:
                flush()
    flush()

    matrix = sp.csc_matrix(
        (np.concatenate(tfs), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(pids), len(vocab)), dtype=np.float32,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "bm25_matrix.npz", matrix)
    np.savez(out_dir / "bm25_meta.npz", pids=np.asarray(pids), lengths=np.asarray(lengths, np.int32))
    (out_dir / "bm25_vocab.json").write_text(json.dumps(vocab))
    stats = {"passages": len(pids), "terms": len(vocab), "postings": int(matrix.nnz),
             "build_seconds": round(time.time() - start, 1)}
    (out_dir / "bm25_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class BM25:
    """The saved index held in memory, ready to score queries (OR matching)."""

    def __init__(self, index_dir: Path, k1: float = 1.2, b: float = 0.75) -> None:
        self.matrix = sp.load_npz(index_dir / "bm25_matrix.npz").tocsc()
        meta = np.load(index_dir / "bm25_meta.npz")
        self.pids = meta["pids"]
        lengths = meta["lengths"].astype(np.float32)
        self.vocab: dict[str, int] = json.loads((index_dir / "bm25_vocab.json").read_text())
        self.k1 = np.float32(k1)
        self.norm = self.k1 * (1 - b + b * lengths / lengths.mean())
        df = np.diff(self.matrix.indptr).astype(np.float32)
        n = len(self.pids)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.pids), dtype=np.float32)
        hit = False
        for term in dict.fromkeys(tokenize(query)):   # dedupe, keep order
            tid = self.vocab.get(term)
            if tid is None:
                continue
            hit = True
            lo, hi = self.matrix.indptr[tid], self.matrix.indptr[tid + 1]
            docs = self.matrix.indices[lo:hi]
            tf = self.matrix.data[lo:hi]
            scores[docs] += self.idf[tid] * tf * (self.k1 + 1) / (tf + self.norm[docs])
        if not hit:
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(str(self.pids[i]), float(scores[i])) for i in top if scores[i] > 0]


def retrieve_split(cfg: Config, spec: SplitSpec, index: BM25 | None = None) -> dict:
    """Write the top-`depth` BM25 candidates for one split, with judgment status."""
    index = index or BM25(cfg.index_dir, cfg.retrieval.k1, cfg.retrieval.b)
    queries = spec.queries()
    qrels = spec.qrels()
    start = time.time()
    candidates: dict[str, list[io.Candidate]] = {}
    for qid, text in queries.items():
        judged = qrels.get(qid, {})
        hits = index.search(text, cfg.retrieval.depth)
        candidates[qid] = [
            io.Candidate(pid, rank, score, judged.get(pid, io.UNJUDGED))
            for rank, (pid, score) in enumerate(hits, start=1)
        ]
    elapsed = time.time() - start
    io.write_candidates(candidates, cfg.candidates_dir / f"{spec.name}.tsv")

    with_relevant = [q for q in queries if any(g >= spec.threshold for g in qrels.get(q, {}).values())]
    retrieved_rel = sum(
        1 for q in with_relevant if any(c.grade >= spec.threshold for c in candidates[q])
    )
    stats = {
        "split": spec.name,
        "queries": len(queries),
        "queries_with_relevant": len(with_relevant),
        "queries_with_relevant_retrieved": retrieved_rel,
        "ms_per_query": round(1000 * elapsed / max(len(queries), 1), 2),
    }
    out = cfg.candidates_dir / f"{spec.name}.stats.json"
    out.write_text(json.dumps(stats, indent=2))
    return stats
