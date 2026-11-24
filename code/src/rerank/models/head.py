"""Relevance heads over the N pooled candidate vectors of one query.

    pooled [B, N, d_backbone]
      -> Linear projection to width w
      -> (variant block, residual, pre-norm)
      -> LayerNorm
      -> Linear w -> 1                      one logit per passage
      -> sigmoid                            P(relevant), same order as the logit

Variants
- "pointwise":   no block; every passage is scored independently.
- "candidate":   one or more pre-norm transformer blocks with multi-head
                 self-attention across the N candidates of the same query.
- "ffn_control": the same blocks with attention replaced by a feed-forward layer
                 of (almost) equal parameter count. It isolates whether a gain
                 comes from candidate interaction or just from extra capacity.

No candidate position embeddings and no BM25 rank features are used, so the
candidate model is permutation-equivariant: shuffling the candidates shuffles
the logits and changes nothing else. Padding is masked out of attention.
"""

from __future__ import annotations

import torch
from torch import nn

from rerank.config import ModelConfig

VARIANTS = ("pointwise", "candidate", "ffn_control")


class FeedForward(nn.Sequential):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))


class CandidateBlock(nn.Module):
    """Pre-norm transformer block whose sequence axis is the candidate set."""

    def __init__(self, width: int, heads: int, ff_width: int, dropout: float, attention: bool = True) -> None:
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(width)
        if attention:
            self.mix = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        else:
            # Parameter-matched stand-in: attention holds 4w^2 + 4w parameters,
            # a w->h->w FFN holds 2wh + h + w.
            hidden = max(1, round((4 * width * width + 4 * width - width) / (2 * width + 1)))
            self.mix = FeedForward(width, hidden, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ff = FeedForward(width, ff_width, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.attention:
            h, _ = self.mix(h, h, h, key_padding_mask=~valid, need_weights=False)
        else:
            h = self.mix(h)
        x = x + self.drop(h)
        return x + self.drop(self.ff(self.norm2(x)))


class OrdinalThresholds(nn.Module):
    """Ordered cut points θ₁ < θ₂ < θ₃ on one score: P(g ≥ k) = σ(s − θₖ) for grades k = 1..levels.

    Every grade probability moves with the same score s, so ranking by s, by P(relevant) or by the
    expected grade gives the same order. θ for `rel_grade` starts at 0, so σ(s − θ_rel) = σ(s) at init.
    """

    def __init__(self, levels: int = 3, rel_grade: int = 2) -> None:
        super().__init__()
        self.levels, self.rel = levels, rel_grade - 1
        self.shift = nn.Parameter(torch.zeros(()))
        self.gaps = nn.Parameter(torch.full((levels - 1,), 0.5413))   # softplus(0.5413) = 1

    def forward(self) -> torch.Tensor:
        steps = torch.cat([self.gaps.new_zeros(1), torch.cumsum(nn.functional.softplus(self.gaps), 0)])
        return steps - steps[self.rel] + self.shift


class Reranker(nn.Module):
    def __init__(self, in_dim: int, variant: str, cfg: ModelConfig) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.proj = nn.Linear(in_dim, cfg.width)
        n_blocks = 0 if variant == "pointwise" else cfg.blocks
        self.blocks = nn.ModuleList(
            CandidateBlock(cfg.width, cfg.heads, cfg.ff_width, cfg.dropout, attention=variant == "candidate")
            for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, 1)
        # [SET]: a learned extra "candidate" that attends with the pool and answers "is anything relevant?".
        # Only the candidate-attention variant can use it; the others fall back to 1 − Π(1 − pᵢ).
        self.use_set = bool(getattr(cfg, "set_token", False)) and variant == "candidate"
        if self.use_set:
            self.set_vec = nn.Parameter(torch.randn(cfg.width) * 0.02)
            self.set_head = nn.Linear(cfg.width, 1)
        self.thresholds = (OrdinalThresholds(rel_grade=getattr(cfg, "rel_grade", 2))
                           if getattr(cfg, "ordinal", False) else None)

    def forward_full(self, feats: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """feats [B, N, in_dim], valid [B, N] -> (scores [B, N], set logit [B] or None)."""
        if not bool(valid.any(1).all()):
            raise ValueError("every group needs at least one valid candidate")
        x = self.proj(feats)
        n = x.shape[1]
        if self.use_set:
            x = torch.cat([x, self.set_vec.expand(x.shape[0], 1, -1)], dim=1)
            valid = torch.cat([valid, valid.new_ones(valid.shape[0], 1)], dim=1)
        for block in self.blocks:
            x = block(x, valid)
        out = self.norm(x)
        scores = self.head(out[:, :n]).squeeze(-1)
        set_logit = self.set_head(out[:, n]).squeeze(-1) if self.use_set else None
        return scores, set_logit

    def forward(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats [B, N, in_dim], valid [B, N] bool -> ranking scores [B, N] (padding is arbitrary)."""
        return self.forward_full(feats, valid)[0]

    def relevance_logits(self, scores: torch.Tensor) -> torch.Tensor:
        """Logit of P(relevant) for binary labels: s − θ_rel with the graded head, else s itself."""
        if self.thresholds is None:
            return scores
        return scores - self.thresholds()[self.thresholds.rel]

    def grade_probs(self, scores: torch.Tensor) -> torch.Tensor:
        """P(g ≥ k) for k = 1..levels, shape [..., levels]. Requires the graded head."""
        return torch.sigmoid(scores.unsqueeze(-1) - self.thresholds())

    def init_prior(self, positive_rate: float) -> None:
        """Start the head at the training positive rate instead of 0.5."""
        p = min(max(positive_rate, 1e-4), 1 - 1e-4)
        with torch.no_grad():
            self.head.bias.fill_(float(torch.logit(torch.tensor(p))))


def mask_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Push padding to the bottom of any ranking."""
    return logits.masked_fill(~valid, float("-inf"))
