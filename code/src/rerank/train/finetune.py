"""End-to-end fine-tuning: LoRA on the backbone, trained jointly with the relevance head.

Each step takes `queries_per_step` training queries, samples a group of
`group_size` BM25 candidates for each (retrieved positives plus hard negatives
from ranks 1..negative_depth), encodes every query-passage pair through the
adapted backbone, and scores the group with the chosen head. With the
candidate head, attention runs across the group in the same forward pass, so
the backbone learns representations that are useful in context.

    loss = listwise softmax CE over the group + bce_weight * per-passage BCE

The listwise term is what fine-tuned cross-encoder rerankers are usually
trained with; BCE keeps the sigmoid output a per-passage probability.

Output directory (`work/models/ft-<variant>-s<seed>/`):
    adapter.pt   LoRA tensors and their rank/alpha
    best.pt      head checkpoint in the same format as the frozen trainer,
                 plus "adapter" pointing at adapter.pt
`rerank live <best.pt> <splits>` then scores any split end to end.
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

from rerank.config import Config
from rerank.data import io
from rerank.data.groups import sample_group
from rerank.data.splits import split_spec
from rerank.eval import metrics
from rerank.models.backbone import PairEncoder, load_backbone, pick_device, pick_dtype
from rerank.models.head import Reranker
from rerank.models.lora import add_lora, lora_state_dict
from rerank.train import losses
from rerank.train.trainer import seed_everything


def score_queries(encoder: PairEncoder, head: Reranker, queries: dict[str, str],
                  candidates: dict[str, list[io.Candidate]], passages: dict[str, str],
                  depth: int, batch_size: int) -> dict[str, list[tuple[str, float]]]:
    """Score each query's pool end to end: encode all pairs, then run the head over the whole pool."""
    head.eval()
    encoder.model.eval()
    run = {}
    with torch.inference_mode():
        for qid, cands in candidates.items():
            cands = cands[:depth]
            if not cands:
                run[qid] = []
                continue
            texts = [passages[c.pid] for c in cands]
            vecs = torch.cat([encoder.pooled([queries[qid]] * len(texts[i:i + batch_size]), texts[i:i + batch_size])
                              for i in range(0, len(texts), batch_size)])
            valid = torch.ones(1, len(cands), dtype=torch.bool, device=vecs.device)
            logits = head.relevance_logits(head(vecs[None], valid))[0].float().cpu().tolist()
            run[qid] = sorted(zip((c.pid for c in cands), logits), key=lambda t: -t[1])
    return run


def finetune(cfg: Config, variant: str = "candidate", seed: int | None = None,
             steps: int | None = None, out_dir: Path | None = None, encoder: PairEncoder | None = None,
             max_train_queries: int | None = None) -> Path:
    f = cfg.finetune
    seed = f.seed if seed is None else seed
    steps = f.steps if steps is None else steps
    seed_everything(seed)
    rng = random.Random(seed)

    if encoder is None:
        bb = cfg.backbone
        device = pick_device()
        dtype = pick_dtype(bb.dtype, device)
        if dtype == torch.float16:
            dtype = torch.bfloat16      # fp16 training without loss scaling is fragile
        encoder = PairEncoder(load_backbone(bb), bb.max_length, bb.max_query_tokens, device, dtype)
    device = encoder.device
    add_lora(encoder.model, f.lora_rank, f.lora_alpha, f.lora_dropout)
    encoder.model.to(device)
    if f.grad_checkpointing:
        kwargs = {"use_reentrant": False}
        if hasattr(encoder.model, "activation_checkpointing_enable"):          # transformers >= 5
            encoder.model.activation_checkpointing_enable(activation_checkpointing_kwargs=kwargs)
        else:
            encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    train_spec, val_spec = split_spec(cfg, "train"), split_spec(cfg, "val")
    train_cands = io.read_candidates(cfg.candidates_dir / "train.tsv")
    train_qids = [q for q, cs in train_cands.items() if any(c.grade >= train_spec.threshold for c in cs)]
    if max_train_queries is not None:
        train_qids = train_qids[:max_train_queries]
    val_cands = dict(list(io.read_candidates(cfg.candidates_dir / "val.tsv").items())[: f.eval_queries])
    queries = train_spec.queries() | val_spec.queries()
    val_qrels = {q: j for q, j in val_spec.qrels().items() if q in val_cands}
    passages = io.load_collection(cfg.data.collection,
                                  {c.pid for q in train_qids for c in train_cands[q]}
                                  | {c.pid for cs in val_cands.values() for c in cs[: f.eval_depth]})

    head = Reranker(encoder.hidden_size, variant, cfg.model).to(device)
    head.init_prior(1.0 / f.group_size)
    lora_params = [p for p in encoder.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": f.lr_lora},
        {"params": head.parameters(), "lr": f.lr_head},
    ], weight_decay=f.weight_decay)
    total = steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(f.warmup_steps, 1), max(0.0, (total - s) / max(total - f.warmup_steps, 1))))

    out_dir = out_dir or cfg.models_dir / f"ft-{variant}-s{seed}"
    tag = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] {len(train_qids):,} training queries with a retrieved positive; "
          f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M LoRA + "
          f"{sum(p.numel() for p in head.parameters()) / 1e6:.2f}M head parameters", flush=True)

    best = {"mrr@10": -math.inf}
    best_head = best_lora = None
    order: list[str] = []
    start = time.time()
    with open(out_dir / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            encoder.model.train()
            head.train()
            if len(order) < f.queries_per_step:
                fresh = train_qids[:]
                rng.shuffle(fresh)
                order += fresh
            qids, order = order[: f.queries_per_step], order[f.queries_per_step:]
            groups = [sample_group(train_cands[q], f.group_size, f.negative_depth, rng, train_spec.threshold)
                      for q in qids]
            if head.use_set and f.empty_rate > 0:
                # Constructed empty pools for the set head: drop the judged positives before encoding.
                groups = [[c for c in g if c.grade < train_spec.threshold] or g
                          if rng.random() < f.empty_rate else g for g in groups]
            width = max(len(g) for g in groups)
            qs = [queries[q] for q, g in zip(qids, groups) for _ in g]
            ps = [passages[c.pid] for g in groups for c in g]
            pooled = encoder.pooled(qs, ps)

            feats = torch.zeros(len(groups), width, encoder.hidden_size, device=device)
            valid = torch.zeros(len(groups), width, dtype=torch.bool, device=device)
            labels = torch.zeros(len(groups), width, device=device)
            row = 0
            for i, g in enumerate(groups):
                feats[i, : len(g)] = pooled[row: row + len(g)]
                valid[i, : len(g)] = True
                labels[i, : len(g)] = torch.tensor([float(c.grade >= train_spec.threshold) for c in g])
                row += len(g)
            logits, set_logit = head.forward_full(feats, valid)
            ce = losses.listwise_ce(logits, labels, valid)
            bce = losses.group_bce(head.relevance_logits(logits), labels, valid)
            loss = ce + f.bce_weight * bce
            set_loss = None
            if set_logit is not None:
                any_rel = (labels.bool() & valid).any(1).float()
                set_loss = torch.nn.functional.binary_cross_entropy_with_logits(set_logit, any_rel)
                loss = loss + f.set_weight * set_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            record = {"step": step, "loss": loss.item(), "ce": ce.item(), "bce": bce.item()}
            if set_loss is not None:
                record["set"] = set_loss.item()

            if step % f.eval_every == 0 or step == steps:
                run = score_queries(encoder, head, queries, val_cands, passages, f.eval_depth, cfg.backbone.batch_size)
                val = metrics.evaluate_run(run, val_qrels, val_spec.threshold)
                record |= {"val": val, "elapsed": round(time.time() - start, 1)}
                print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} "
                      f"val MRR@10 {val['mrr@10']:.4f} ({time.time() - start:.0f}s)", flush=True)
                if val["mrr@10"] > best["mrr@10"]:
                    best = val | {"step": step}
                    best_head = copy.deepcopy(head.state_dict())
                    best_lora = lora_state_dict(encoder.model)
            log.write(json.dumps(record) + "\n")
            log.flush()

    torch.save({"state": best_lora, "rank": f.lora_rank, "alpha": f.lora_alpha}, out_dir / "adapter.pt")
    path = out_dir / "best.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_head.items()}, "variant": variant, "objective": "listwise_ce+bce",
        "seed": seed, "in_dim": encoder.hidden_size, "model_cfg": vars(cfg.model),
        "backbone": cfg.backbone.tag, "adapter": "adapter.pt", "finetune_cfg": vars(f) | {"steps": steps},
        "val": best,
    }, path)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    return path
