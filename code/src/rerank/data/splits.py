"""Query splits.

Model fitting, model selection and calibration use disjoint subsets of MS MARCO
*training* queries. The course dev set and TREC DL 2019/2020 are held out for
reporting only. Training-pool queries are restricted to those whose judged
positive exists in the 1M-passage corpus, and any query whose id or normalized
text appears in a held-out set is removed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from rerank.config import Config
from rerank.data import io

TRAIN_POOL = ("train", "val", "calib")
HELD_OUT = ("dev", "dl19", "dl20")
ALL_SPLITS = TRAIN_POOL + HELD_OUT


@dataclass(frozen=True)
class SplitSpec:
    """Where one split's queries and judgments live, and its relevance threshold."""

    name: str
    queries_path: Path
    qrels_path: Path
    threshold: int                 # grade at or above which a passage counts as relevant
    qids_path: Path | None = None  # restricts the query file; None means "the qrels queries"

    def qids(self) -> list[str]:
        if self.qids_path is not None:
            return io.read_qids(self.qids_path)
        return sorted(_qrels(self.qrels_path), key=io.qid_sort_key)

    def queries(self) -> dict[str, str]:
        wanted = self.qids()
        text = io.load_queries(self.queries_path, set(wanted))
        return {q: text[q] for q in wanted if q in text}

    def qrels(self) -> dict[str, dict[str, int]]:
        qrels = _qrels(self.qrels_path)
        if self.qids_path is None:
            return qrels
        return {q: qrels[q] for q in self.qids() if q in qrels}


@cache
def _qrels(path: Path) -> dict[str, dict[str, int]]:
    return io.load_qrels(path)


def split_spec(cfg: Config, name: str) -> SplitSpec:
    d = cfg.data
    if name in TRAIN_POOL:
        return SplitSpec(name, d.queries_train, d.qrels_train, 1, cfg.splits_dir / f"{name}.qids")
    if name == "dev":
        return SplitSpec(name, d.queries_dev, d.qrels_dev, 1)
    if name == "dl19":
        return SplitSpec(name, d.queries_trec, d.qrels_dl19, 2)   # TREC DL: grades 2-3 relevant
    if name == "dl20":
        return SplitSpec(name, d.queries_trec, d.qrels_dl20, 2)
    raise ValueError(f"unknown split {name!r}; expected one of {ALL_SPLITS}")


def normalize(text: str) -> str:
    from rerank.data.bm25 import tokenize
    return " ".join(tokenize(text))


def build_splits(cfg: Config) -> dict:
    corpus = io.collection_ids(cfg.data.collection)
    qrels = io.load_qrels(cfg.data.qrels_train)
    queries = io.load_queries(cfg.data.queries_train, set(qrels))

    held_ids: set[str] = set()
    held_text: set[str] = set()
    for name in HELD_OUT:
        spec = split_spec(cfg, name)
        held = spec.queries()
        held_ids |= set(held)
        held_text |= {normalize(t) for t in held.values()}

    stats = {"train_queries_with_qrels": len(qrels), "missing_text": 0,
             "no_positive_in_corpus": 0, "overlap_id": 0, "overlap_text": 0}
    eligible = []
    for qid in sorted(qrels, key=io.qid_sort_key):
        if qid not in queries:
            stats["missing_text"] += 1
        elif not any(g >= 1 and pid in corpus for pid, g in qrels[qid].items()):
            stats["no_positive_in_corpus"] += 1
        elif qid in held_ids:
            stats["overlap_id"] += 1
        elif normalize(queries[qid]) in held_text:
            stats["overlap_text"] += 1
        else:
            eligible.append(qid)
    stats["eligible"] = len(eligible)

    s = cfg.splits
    need = s.train_queries + s.val_queries + s.calib_queries
    if len(eligible) < s.val_queries + s.calib_queries + 1:
        raise ValueError(f"only {len(eligible)} eligible training queries")
    random.Random(s.seed).shuffle(eligible)
    val = eligible[: s.val_queries]
    calib = eligible[s.val_queries: s.val_queries + s.calib_queries]
    train = eligible[s.val_queries + s.calib_queries: need]
    for name, qids in (("train", train), ("val", val), ("calib", calib)):
        io.write_qids(sorted(qids, key=io.qid_sort_key), cfg.splits_dir / f"{name}.qids")
        stats[name] = len(qids)
    (cfg.splits_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
