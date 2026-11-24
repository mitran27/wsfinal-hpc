"""Train a relevance head on cached backbone vectors, then score splits with it.

Objectives (always with BCE, so logits stay tied to passage-level labels):
    bce           BCE only
    ranknet       BCE + ranknet_weight * RankNet pairwise loss
    rl            BCE + rl_weight * Plackett-Luce REINFORCE on nDCG@rl_cutoff

Matched comparisons: give every run the same `--seed`, the same cached groups
and the same `steps`; continue the three objectives from one supervised
checkpoint with `--init` for the "does RL help" experiment. The checkpoint with
the best validation MRR@10 is kept.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from rerank.config import Config, ModelConfig
from rerank.data import io
from rerank.data.splits import split_spec
from rerank.eval import metrics
from rerank.features import FeatureStore
from rerank.models.backbone import pick_device
from rerank.models.head import Reranker, mask_logits
from rerank.train import losses

OBJECTIVES = ("bce", "ranknet", "rl")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def score_store(model: Reranker, store: FeatureStore, device: torch.device, depth: int | None = None,
                batch_groups: int = 64, permute: np.random.Generator | None = None) -> dict[str, list[tuple[str, float]]]:
    """Logit for every cached pair, as {qid: [(pid, logit), ...]}."""
    model.eval()
    out: dict[str, list[tuple[str, float]]] = {}
    for b in range(0, len(store), batch_groups):
        batch = store.batch(range(b, min(b + batch_groups, len(store))), device, depth, permute)
        # relevance logits: same order as the ranking scores, and the right scale for probabilities
        logits = model.relevance_logits(model(batch["feats"], batch["valid"])).float().cpu()
        for i, (qid, pids) in enumerate(zip(batch["qids"], batch["pids"])):
            out[qid] = list(zip(pids, logits[i, : len(pids)].tolist()))
    return out


def judged_pairs(scored: dict[str, list[tuple[str, float]]], qrels: dict[str, dict[str, int]],
                 threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(logits, binary labels) for retrieved pairs that carry a judgment."""
    z, y = [], []
    for qid, items in scored.items():
        judged = qrels.get(qid, {})
        for pid, logit in items:
            if pid in judged:
                z.append(logit)
                y.append(float(judged[pid] >= threshold))
    return np.asarray(z), np.asarray(y)


def validate(model, store, qrels, threshold, device) -> dict:
    scored = score_store(model, store, device)
    result = metrics.evaluate_run(scored, qrels, threshold)
    z, y = judged_pairs(scored, qrels, threshold)
    result["brier_judged"] = metrics.brier(1 / (1 + np.exp(-z)), y)
    result["judged_pairs"] = int(len(y))
    result["judged_negatives"] = int((y == 0).sum())
    return result


def build_model(in_dim: int, variant: str, mcfg: ModelConfig) -> Reranker:
    return Reranker(in_dim, variant, mcfg)


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[Reranker, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["in_dim"], ckpt["variant"], ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or pick_device()).eval(), ckpt


def train(cfg: Config, variant: str, objective: str = "bce", seed: int | None = None,
          init: Path | None = None, out_dir: Path | None = None, steps: int | None = None) -> Path:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    t = cfg.train
    seed = t.seed if seed is None else seed
    steps = t.steps if steps is None else steps
    seed_everything(seed)
    device = pick_device()

    train_store = FeatureStore(cfg.feature_dir / "train")
    val_store = FeatureStore(cfg.feature_dir / "val")
    val_spec = split_spec(cfg, "val")
    val_qrels = val_spec.qrels()
    threshold = split_spec(cfg, "train").threshold

    model = build_model(train_store.dim, variant, cfg.model)
    if init is not None:
        init_model, init_ckpt = load_checkpoint(init, torch.device("cpu"))
        if init_ckpt["variant"] != variant:
            raise ValueError(f"--init is a {init_ckpt['variant']} checkpoint, not {variant}")
        model.load_state_dict(init_model.state_dict())
    else:
        model.init_prior(train_store.positive_rate(threshold))
    if t.freeze_projection:
        model.proj.requires_grad_(False)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(t.warmup_steps, 1)))
    gen = torch.Generator().manual_seed(seed)

    tag = f"{variant}-{objective}-s{seed}" + ("-ft" if init else "")
    out_dir = out_dir or cfg.models_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "log.jsonl", "w") as log:
        best = {"mrr@10": -math.inf}
        best_state = None
        order: list[int] = []
        start = time.time()
        for step in range(1, steps + 1):
            model.train()
            if len(order) < t.batch_groups:
                order += torch.randperm(len(train_store), generator=gen).tolist()
            idx, order = order[: t.batch_groups], order[t.batch_groups:]
            batch = train_store.batch(idx, device)
            valid, grades = batch["valid"], batch["grades"]
            labels = (grades >= threshold) & valid
            if model.use_set and t.empty_rate > 0:
                # Constructed empty pools: hide the judged positives of some groups so the set head also
                # sees pools with nothing relevant. (Unjudged false negatives make these labels noisy until
                # teacher scores can filter them; see findings N1.)
                empty = (torch.rand(len(idx), generator=gen) < t.empty_rate).to(device)
                valid = valid & ~(labels & empty[:, None])
                labels = labels & valid
            logits, set_logit = model.forward_full(batch["feats"], valid)

            loss = losses.group_bce(model.relevance_logits(logits), labels, valid)
            record = {"step": step, "bce": loss.item()}
            if set_logit is not None:
                set_loss = torch.nn.functional.binary_cross_entropy_with_logits(set_logit, labels.any(1).float())
                loss = loss + t.set_weight * set_loss
                record["set"] = set_loss.item()
            if objective == "ranknet":
                rank_loss = losses.ranknet(logits, labels.float(), valid)
                loss = loss + t.ranknet_weight * rank_loss
                record["ranknet"] = rank_loss.item()
            elif objective == "rl":
                rl_loss, rl_stats = losses.reinforce(logits, labels.float(), valid, t.rl_cutoff, t.rl_samples, gen)
                loss = loss + t.rl_weight * rl_loss
                record |= {"rl": rl_loss.item(), **rl_stats}

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step % t.eval_every == 0 or step == steps:
                val = validate(model, val_store, val_qrels, val_spec.threshold, device)
                record |= {"val": val, "elapsed": round(time.time() - start, 1)}
                print(f"[{tag}] step {step} loss {loss.item():.4f} val MRR@10 {val['mrr@10']:.4f}", flush=True)
                if val["mrr@10"] > best["mrr@10"]:
                    best = val | {"step": step}
                    best_state = copy.deepcopy(model.state_dict())
            log.write(json.dumps(record) + "\n")

    path = out_dir / "best.pt"
    torch.save({
        "state_dict": best_state, "variant": variant, "objective": objective, "seed": seed,
        "in_dim": train_store.dim, "model_cfg": vars(cfg.model), "backbone": cfg.backbone.tag,
        "train_cfg": vars(t) | {"steps": steps}, "init": str(init) if init else None, "val": best,
    }, path)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    return path


def predict(cfg: Config, ckpt_path: Path, split: str, calibrator=None, depth: int | None = None) -> tuple[Path, Path]:
    """Write a TREC run (ranked by logit) and a per-pair score table for one split."""
    model, ckpt = load_checkpoint(ckpt_path)
    if ckpt.get("adapter"):
        raise ValueError("this checkpoint adapts the backbone; cached features do not apply. Use `rerank live`.")
    if ckpt["backbone"] != cfg.backbone.tag:
        raise ValueError(f"checkpoint was trained on {ckpt['backbone']}, config uses {cfg.backbone.tag}")
    store = FeatureStore(cfg.feature_dir / split)
    scored = score_store(model, store, next(model.parameters()).device, depth)
    tag = ckpt_path.parent.name
    run = {q: sorted(items, key=lambda t: -t[1]) for q, items in scored.items()}
    run_path = cfg.runs_dir / f"{tag}.{split}.run"
    io.write_run(run, run_path, tag)

    grades = {g.qid: dict(zip(g.pids, g.grades.tolist())) for g in store.groups}
    score_path = cfg.scores_dir / f"{tag}.{split}.tsv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        f.write("qid\tpid\tlogit\tprob\tprob_calibrated\tgrade\n")
        for qid, items in run.items():
            logits = np.array([s for _, s in items])
            probs = 1 / (1 + np.exp(-logits))
            cal = calibrator.apply(logits) if calibrator is not None else np.full_like(probs, np.nan)
            for (pid, z), p, c in zip(items, probs, cal):
                f.write(f"{qid}\t{pid}\t{z:.6f}\t{p:.6f}\t{c:.6f}\t{grades[qid][pid]}\n")
    return run_path, score_path
