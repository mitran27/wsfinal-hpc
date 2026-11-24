"""Ranking and probability metrics.

Ranking measures follow trec_eval, as verified in assignment 3:
- nDCG uses linear gain (the grade) and log2(rank + 1) discount; the ideal
  ordering is over every judged passage of the query.
- MRR@10, Recall and MAP binarize at the split's threshold (1 for MS MARCO,
  2 for TREC DL).
- Queries with no relevant judgment are skipped.
- Runs are re-sorted by score with ties broken by descending passage id.

Probability measures (Brier score, reliability bins) use judged pairs only.
"""

from __future__ import annotations

import math

import numpy as np

MEASURES = ("mrr@10", "recall@100", "ndcg@10", "ndcg@100", "map")


def ordered(entries: list[tuple[str, float]]) -> list[str]:
    out = sorted(entries, key=lambda t: t[0], reverse=True)
    out.sort(key=lambda t: -t[1])
    return [pid for pid, _ in out]


def _dcg(grades: list[int]) -> float:
    return sum(g / math.log2(r + 1) for r, g in enumerate(grades, start=1) if g > 0)


def ndcg(ranked: list[str], judged: dict[str, int], k: int) -> float:
    best = _dcg(sorted(judged.values(), reverse=True)[:k])
    return _dcg([judged.get(p, 0) for p in ranked[:k]]) / best if best > 0 else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for r, p in enumerate(ranked[:k], start=1):
        if p in relevant:
            return 1.0 / r
    return 0.0


def recall(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for r, p in enumerate(ranked, start=1):
        if p in relevant:
            hits += 1
            total += hits / r
    return total / len(relevant) if relevant else 0.0


def per_query(run: dict[str, list[tuple[str, float]]], qrels: dict[str, dict[str, int]],
              threshold: int) -> dict[str, dict[str, float]]:
    out = {}
    for qid, judged in qrels.items():
        relevant = {p for p, g in judged.items() if g >= threshold}
        if not relevant:
            continue
        ranked = ordered(run.get(qid, []))
        out[qid] = {
            "mrr@10": reciprocal_rank(ranked, relevant, 10),
            "recall@100": recall(ranked, relevant, 100),
            "ndcg@10": ndcg(ranked, judged, 10),
            "ndcg@100": ndcg(ranked, judged, 100),
            "map": average_precision(ranked, relevant),
        }
    return out


def aggregate(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {m: math.nan for m in MEASURES}
    return {m: float(np.mean([s[m] for s in scores.values()])) for m in MEASURES} | {"queries": len(scores)}


def evaluate_run(run, qrels, threshold: int) -> dict[str, float]:
    return aggregate(per_query(run, qrels, threshold))


def ceiling_run(candidates: dict[str, list], qrels: dict[str, dict[str, int]]) -> dict[str, list[tuple[str, float]]]:
    """Best possible reordering of each retrieved pool: sort by judged grade."""
    return {q: [(c.pid, float(qrels.get(q, {}).get(c.pid, 0)) - 1e-6 * c.rank) for c in cands]
            for q, cands in candidates.items()}


# ---------------------------------------------------------------- probability

def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2)) if len(probs) else math.nan


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(sel.sum()),
                     "mean_prob": float(probs[sel].mean()) if sel.any() else math.nan,
                     "frac_relevant": float(labels[sel].mean()) if sel.any() else math.nan})
    return rows


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    rows = reliability(probs, labels, bins)
    n = max(len(probs), 1)
    return float(sum(r["count"] / n * abs(r["mean_prob"] - r["frac_relevant"]) for r in rows if r["count"]))


# ------------------------------------------------------------------ inference

def paired_bootstrap(a: dict[str, float], b: dict[str, float], samples: int = 2000,
                     seed: int = 0) -> dict[str, float]:
    """Mean of (b - a) over shared queries with a 95% percentile bootstrap interval."""
    qids = sorted(set(a) & set(b))
    diff = np.array([b[q] - a[q] for q in qids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(samples, len(diff)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_zero": float((means <= 0).mean()), "queries": len(qids)}
