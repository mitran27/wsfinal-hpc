"""Stability checks on cached features.

- shuffle: score each pool in BM25 order and in a random order; compare scores
  by passage id. The candidate model has no position information, so the
  difference should be at floating-point noise level.
- depth: rerank only the first k BM25 candidates for several k and report the
  ranking metrics, to see how sensitive candidate attention is to pool size
  (training groups hold 32 candidates; evaluation pools hold 100).
"""

from __future__ import annotations

import numpy as np
import torch

from rerank.eval import metrics
from rerank.features import FeatureStore
from rerank.train.trainer import score_store


def shuffle_test(model, store: FeatureStore, device: torch.device, seed: int = 0) -> dict:
    base = score_store(model, store, device)
    shuffled = score_store(model, store, device, permute=np.random.default_rng(seed))
    diffs, flips = [], 0
    for qid, items in base.items():
        other = dict(shuffled[qid])
        d = np.array([abs(s - other[p]) for p, s in items])
        diffs.append(d.max())
        top_a = max(items, key=lambda t: t[1])[0]
        top_b = max(other.items(), key=lambda t: t[1])[0]
        flips += top_a != top_b
    return {"max_abs_diff": float(np.max(diffs)), "mean_max_abs_diff": float(np.mean(diffs)),
            "top1_changes": int(flips), "queries": len(base)}


def depth_test(model, store: FeatureStore, qrels, threshold: int, device: torch.device,
               depths=(10, 20, 32, 50, 100)) -> dict[int, dict]:
    return {k: metrics.evaluate_run(score_store(model, store, device, depth=k), qrels, threshold)
            for k in depths}
