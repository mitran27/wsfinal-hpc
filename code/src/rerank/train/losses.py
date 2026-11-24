"""Training objectives over padded candidate groups.

All functions take logits [B, N] and a `valid` mask [B, N]; padding never
contributes to a loss or a reward. Losses are averaged within each group and
then across groups, so long and short groups weigh the same.

- group_bce:  per-passage probability objective (always on).
- ranknet:    supervised pairwise ranking loss, softplus(-(s_i - s_j)) for
              every valid pair with label_i > label_j.
- reinforce:  Plackett-Luce policy over the logits; sample a top-k ranking,
              reward it with binary nDCG@k, subtract the greedy ranking's
              reward as a baseline, and push up the sampled ranking's log-prob.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NEG = -1e9   # finite "minus infinity": keeps logsumexp gradients free of NaNs


def _group_mean(per_group: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if not bool(keep.any()):
        return per_group.sum() * 0.0
    return per_group[keep].mean()


def group_bce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    loss = loss.masked_fill(~valid, 0.0)
    count = valid.sum(1)
    return _group_mean(loss.sum(1) / count.clamp(min=1), count > 0)


def listwise_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over each group: -log(sum_pos e^s / sum_valid e^s).

    With one positive and hard negatives this is the localized contrastive
    estimation loss commonly used to fine-tune cross-encoder rerankers.
    """
    pos = labels.bool() & valid
    all_lse = torch.logsumexp(logits.masked_fill(~valid, NEG), dim=1)
    pos_lse = torch.logsumexp(logits.masked_fill(~pos, NEG), dim=1)
    return _group_mean(all_lse - pos_lse, pos.any(1))


def ranknet(logits: torch.Tensor, gains: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    diff = logits[:, :, None] - logits[:, None, :]
    pairs = (gains[:, :, None] > gains[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    loss = F.softplus(-diff).masked_fill(~pairs, 0.0)
    count = pairs.sum((1, 2))
    return _group_mean(loss.sum((1, 2)) / count.clamp(min=1), count > 0)


def sample_rankings(logits: torch.Tensor, valid: torch.Tensor, k: int,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Exact Plackett-Luce samples of the top k via the Gumbel-top-k trick."""
    u = torch.rand(logits.shape, generator=generator, device="cpu").to(logits.device)
    gumbel = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-10)))
    perturbed = (logits.detach() + gumbel).masked_fill(~valid, float("-inf"))
    return perturbed.topk(k, dim=1).indices


def plackett_luce_log_prob(logits: torch.Tensor, ranking: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """log P(ranking[:, :k]) under PL(softmax over the still-unranked valid items).

    Steps that pick a padding slot (groups with fewer than k valid items) add 0.
    """
    remaining = valid.clone()
    total = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    for t in range(ranking.shape[1]):
        idx = ranking[:, t: t + 1]
        ok = remaining.gather(1, idx).squeeze(1)
        lse = torch.logsumexp(logits.masked_fill(~remaining, NEG), dim=1)
        chosen = logits.gather(1, idx).squeeze(1)
        total = total + torch.where(ok, chosen - lse, torch.zeros_like(lse))
        remaining = remaining.scatter(1, idx, False)
    return total


def ndcg_at_k(ranking: torch.Tensor, gains: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """nDCG@k of each ranking, with the ideal taken over the group's own candidates."""
    k = min(k, ranking.shape[1])
    g = gains.float().masked_fill(~valid, 0.0)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=gains.device, dtype=torch.float))
    dcg = (g.gather(1, ranking[:, :k]) * disc).sum(1)
    ideal = g.sort(dim=1, descending=True).values[:, :k]
    idcg = (ideal * disc[: ideal.shape[1]]).sum(1)
    return torch.where(idcg > 0, dcg / idcg.clamp(min=1e-12), torch.zeros_like(dcg))


def reinforce(logits: torch.Tensor, gains: torch.Tensor, valid: torch.Tensor, k: int = 10,
              samples: int = 4, generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """REINFORCE with a greedy-ranking baseline. Groups with no positive are skipped."""
    k = min(k, logits.shape[1])
    with torch.no_grad():
        greedy = logits.masked_fill(~valid, float("-inf")).topk(k, dim=1).indices
        baseline = ndcg_at_k(greedy, gains, valid, k)
    keep = ((gains > 0) & valid).any(1)
    terms, rewards = [], []
    for _ in range(samples):
        ranking = sample_rankings(logits, valid, k, generator)
        reward = ndcg_at_k(ranking, gains, valid, k)
        terms.append(-(reward - baseline) * plackett_luce_log_prob(logits, ranking, valid))
        rewards.append(reward)
    loss = _group_mean(torch.stack(terms).mean(0), keep)
    stats = {"rl_reward": float(torch.stack(rewards).mean(0)[keep].mean()) if keep.any() else math.nan,
             "rl_greedy": float(baseline[keep].mean()) if keep.any() else math.nan}
    return loss, stats
