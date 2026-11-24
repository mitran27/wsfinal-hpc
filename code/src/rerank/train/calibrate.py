"""Monotonic logistic (Platt) calibration: p = sigmoid(a * logit + b) with a > 0.

The slope is parameterised as softplus(alpha), so it stays positive and the
calibrated probabilities keep the model's ranking. The intercept absorbs the
shift caused by training on sampled negatives. Fit only on *judged* pairs from
a split disjoint from the one being reported, and only when both classes are
present; otherwise report raw probabilities with their judgment coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    n: int = 0
    positives: int = 0

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray, min_per_class: int = 10) -> "PlattCalibrator":
        labels = np.asarray(labels, dtype=np.float64)
        pos = int(labels.sum())
        neg = len(labels) - pos
        if pos < min_per_class or neg < min_per_class:
            raise ValueError(f"calibration needs both classes: {pos} positive, {neg} negative judged pairs")
        z = torch.tensor(logits, dtype=torch.float64)
        y = torch.tensor(labels)
        alpha = torch.tensor(0.5413, dtype=torch.float64, requires_grad=True)   # softplus(.5413) = 1
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([alpha, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(F.softplus(alpha) * z + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        return cls(float(F.softplus(alpha.detach())), float(b.detach()), len(labels), pos)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(logits) + self.b)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vars(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlattCalibrator":
        return cls(**json.loads(path.read_text()))


def read_scores(path: Path, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(logits, labels) of judged rows from a score table written by `predict`."""
    z, y = [], []
    with open(path) as f:
        next(f)
        for line in f:
            _, _, logit, _, _, grade = line.rstrip("\n").split("\t")
            if int(grade) >= 0:
                z.append(float(logit))
                y.append(float(int(grade) >= threshold))
    return np.asarray(z), np.asarray(y)
