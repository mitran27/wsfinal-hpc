"""Decision layer: how many results to return per query, including none (abstention).

Given calibrated relevance probabilities p₁ ≥ p₂ ≥ … for one query's ranked candidates, returning the
top k has expected F1

    E[F1(k)] = Σₜ Σᵣ P(TP = t) · P(R_out = r) · 2t / (k + t + r)        for k ≥ 1

where TP is the number of relevant passages among the top k and R_out the number below it. Both are
Poisson-binomial sums of independent Bernoullis, computed exactly by dynamic programming. Returning
nothing is correct only when nothing is relevant, so E[F1(0)] = P(no relevant passage), taken from the
set head when there is one, else Π(1 − pᵢ). The cutoff is the k with the highest expected F1, so
abstention (k = 0) comes from the same objective instead of a separate rule.
For independent labels the F1-optimal set is always a top-k of the sorted probabilities, so searching
over k is exact.
"""

from __future__ import annotations

import numpy as np


def _prefix_pmfs(p: np.ndarray) -> list[np.ndarray]:
    """pmf of Σ_{i<k} Xᵢ for every k = 0..n."""
    out = [np.array([1.0])]
    for pi in p:
        prev = out[-1]
        nxt = np.zeros(len(prev) + 1)
        nxt[:-1] += prev * (1 - pi)
        nxt[1:] += prev * pi
        out.append(nxt)
    return out


def expected_f1_curve(probs: np.ndarray, p_none: float | None = None, max_k: int | None = None) -> np.ndarray:
    """E[F1(k)] for k = 0..max_k, with `probs` sorted in decreasing order."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    n = len(p)
    max_k = n if max_k is None else min(max_k, n)
    prefix = _prefix_pmfs(p)
    suffix = _prefix_pmfs(p[::-1])            # suffix[m] = pmf of the last m items
    curve = np.zeros(max_k + 1)
    curve[0] = float(np.prod(1 - p)) if p_none is None else float(p_none)
    for k in range(1, max_k + 1):
        tp, rest = prefix[k], suffix[n - k]
        t = np.arange(len(tp))[:, None]
        r = np.arange(len(rest))[None, :]
        curve[k] = float((tp[:, None] * rest[None, :] * (2 * t / (k + t + r))).sum())
    return curve


def decide(probs: np.ndarray, p_any: float | None = None, max_k: int = 10) -> dict:
    """Return count, abstention flag and expected F1 for one query's sorted probabilities."""
    curve = expected_f1_curve(probs, None if p_any is None else 1.0 - p_any, max_k)
    k = int(np.argmax(curve))
    return {"k": k, "abstain": k == 0, "expected_f1": float(curve[k]), "curve": curve.tolist()}


def global_threshold_count(probs: np.ndarray, t: float, max_k: int = 10) -> int:
    """Baseline policy: return every passage with p ≥ t (at most max_k)."""
    return int(min((np.asarray(probs) >= t).sum(), max_k))


# ---------------------------------------------------------------- evaluation

def set_scores(returned: list[str], relevant: set[str]) -> dict:
    """Precision, recall and F1 of a returned set; an empty answer is right only when nothing is relevant."""
    if not returned:
        ok = float(not relevant)
        return {"precision": ok, "recall": ok, "f1": ok}
    tp = len(set(returned) & relevant)
    precision = tp / len(returned)
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * tp / (len(returned) + len(relevant)) if (returned or relevant) else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def risk_coverage(confidence: list[float], correct: list[float]) -> list[tuple[float, float]]:
    """(coverage, mean quality of answered queries) when answering the most confident first."""
    order = np.argsort(-np.asarray(confidence))
    q = np.asarray(correct, dtype=float)[order]
    cum = np.cumsum(q)
    n = len(q)
    return [((i + 1) / n, float(cum[i] / (i + 1))) for i in range(n)]


def count_error(probs: np.ndarray, n_relevant: int) -> float:
    """|expected number of relevant passages − actual|, as in Dinzinger et al.'s count MAE."""
    return abs(float(np.sum(probs)) - n_relevant)
