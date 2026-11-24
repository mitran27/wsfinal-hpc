"""Cache of frozen-backbone [CLS] vectors, one row per query-passage pair.

Layout of `work/features/<backbone tag>/<split>/`:

    pairs.tsv     the candidate rows that were encoded (candidate-file format);
                  row i of pairs.tsv is row i of vectors.npy
    vectors.npy   float16 [num_pairs, hidden]
    meta.json     backbone tag, hidden size, pair count

The training split encodes the sampled groups (`train.groups.tsv`); every other
split encodes its full top-`depth` BM25 pool.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rerank.config import Config
from rerank.data import io
from rerank.data.splits import split_spec


def source_candidates(cfg: Config, split: str) -> Path:
    name = "train.groups.tsv" if split == "train" else f"{split}.tsv"
    return cfg.candidates_dir / name


def encode_split(cfg: Config, split: str, encoder=None, limit: int | None = None) -> Path:
    """Encode every pair of one split with the frozen backbone and write the cache."""
    from rerank.models.backbone import PairEncoder, load_backbone, pick_device, pick_dtype

    candidates = io.read_candidates(source_candidates(cfg, split))
    if limit is not None:
        candidates = dict(list(candidates.items())[:limit])
    queries = split_spec(cfg, split).queries()
    flat = [(qid, c) for qid, cands in candidates.items() for c in cands]
    passages = io.load_collection(cfg.data.collection, {c.pid for _, c in flat})

    if encoder is None:
        bb = cfg.backbone
        device = pick_device()
        encoder = PairEncoder(load_backbone(bb), bb.max_length, bb.max_query_tokens,
                              device, pick_dtype(bb.dtype, device))

    out = cfg.feature_dir / split
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "vectors.partial.npy"
    vectors = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(len(flat), encoder.hidden_size))
    # Encode in length order to minimise padding; write back in file order.
    order = sorted(range(len(flat)), key=lambda i: len(passages[flat[i][1].pid]))
    bs = cfg.backbone.batch_size
    start = time.time()
    for b in range(0, len(order), bs):
        idx = order[b: b + bs]
        qs = [queries[flat[i][0]] for i in idx]
        ps = [passages[flat[i][1].pid] for i in idx]
        vectors[idx] = encoder.encode(qs, ps).numpy().astype(np.float16)
        if (b // bs) % 200 == 0:
            print(f"[encode {split}] {b + len(idx):,}/{len(flat):,} pairs "
                  f"({time.time() - start:.0f}s)", flush=True)
    vectors.flush()
    del vectors
    tmp.replace(out / "vectors.npy")
    io.write_candidates(candidates, out / "pairs.tsv")
    meta = {"backbone": cfg.backbone.tag, "hidden": encoder.hidden_size, "pairs": len(flat),
            "queries": len(candidates), "seconds": round(time.time() - start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


@dataclass(frozen=True)
class Group:
    qid: str
    pids: tuple[str, ...]
    grades: np.ndarray   # int, -1 = unjudged
    start: int
    end: int


class FeatureStore:
    """Read-only view of one cached split, grouped by query."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        self.dim = int(self.vectors.shape[1])
        self.groups: list[Group] = []
        row = 0
        for qid, cands in io.read_candidates(directory / "pairs.tsv").items():
            n = len(cands)
            self.groups.append(Group(qid, tuple(c.pid for c in cands),
                                     np.array([c.grade for c in cands]), row, row + n))
            row += n
        if row != len(self.vectors):
            raise ValueError(f"{directory}: pairs.tsv has {row} rows, vectors.npy {len(self.vectors)}")

    def __len__(self) -> int:
        return len(self.groups)

    def positive_rate(self, threshold: int) -> float:
        grades = np.concatenate([g.grades for g in self.groups])
        return float((grades >= threshold).mean())

    def batch(self, indices, device: torch.device, depth: int | None = None,
              permute: np.random.Generator | None = None) -> dict:
        """Pad a list of groups into tensors.

        depth keeps only each group's first `depth` candidates (BM25 order);
        permute shuffles candidates inside each group (stability tests).
        """
        groups = [self.groups[i] for i in indices]
        orders = []
        for g in groups:
            o = np.arange(g.end - g.start)[:depth]
            orders.append(permute.permutation(o) if permute is not None else o)
        width = max(len(o) for o in orders)
        feats = torch.zeros(len(groups), width, self.dim)
        grades = torch.full((len(groups), width), -1, dtype=torch.long)
        valid = torch.zeros(len(groups), width, dtype=torch.bool)
        pids = []
        for i, (g, o) in enumerate(zip(groups, orders)):
            feats[i, : len(o)] = torch.from_numpy(np.asarray(self.vectors[g.start + o], dtype=np.float32))
            grades[i, : len(o)] = torch.from_numpy(g.grades[o])
            valid[i, : len(o)] = True
            pids.append([g.pids[j] for j in o])
        return {"feats": feats.to(device), "grades": grades.to(device), "valid": valid.to(device),
                "qids": [g.qid for g in groups], "pids": pids}
