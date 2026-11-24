"""Rerankers that score raw text, sharing one interface for timing and comparison.

    system.score(query, passages) -> np.ndarray of one score per passage

- TrainedReranker: backbone + trained head, end to end (no feature cache).
- BiEncoder: BGE-small-en-v1.5; query and passages encoded separately, [CLS]
  pooling, cosine similarity. Passages are encoded at query time here, so its
  latency is an upper bound; a deployed bi-encoder would precompute them.
- CrossEncoder: MS MARCO MiniLM-L6-v2 cross-encoder logit.

`rerank_split` runs any system over a split's BM25 pool and records latency
and peak memory with the same depth and batch settings for every system.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rerank.config import Config
from rerank.data import io
from rerank.data.splits import split_spec
from rerank.models.backbone import PairEncoder, load_backbone, pick_device, pick_dtype

BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MINILM = "cross-encoder/ms-marco-MiniLM-L6-v2"


class TrainedReranker:
    def __init__(self, cfg: Config, ckpt: Path, batch_size: int | None = None) -> None:
        from rerank.train.trainer import load_checkpoint

        from rerank.models.lora import add_lora, load_lora

        self.name = ckpt.parent.name
        self.device = pick_device()
        bb = cfg.backbone
        self.encoder = PairEncoder(load_backbone(bb), bb.max_length, bb.max_query_tokens,
                                   self.device, pick_dtype(bb.dtype, self.device))
        self.head, meta = load_checkpoint(ckpt, self.device)
        if meta.get("adapter"):
            adapter = torch.load(ckpt.parent / meta["adapter"], map_location="cpu")
            add_lora(self.encoder.model, adapter["rank"], adapter["alpha"], 0.0)
            load_lora(self.encoder.model, adapter["state"])
            self.encoder.model.to(self.device).eval()
        self.batch_size = batch_size or bb.batch_size

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        vecs = torch.cat([self.encoder.encode([query] * len(passages[i:i + self.batch_size]),
                                              passages[i:i + self.batch_size])
                          for i in range(0, len(passages), self.batch_size)])
        feats = vecs.to(self.device)[None]
        valid = torch.ones(1, len(passages), dtype=torch.bool, device=self.device)
        scores, set_logit = self.head.forward_full(feats, valid)
        self.last_set_logit = None if set_logit is None else float(set_logit[0])
        return self.head.relevance_logits(scores)[0].float().cpu().numpy()


class BiEncoder:
    def __init__(self, model: str = BGE, batch_size: int = 128) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.name = model.split("/")[-1]
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1))
        return torch.cat(out)

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        q = self._embed([BGE_QUERY_PREFIX + query])
        return (self._embed(passages) @ q[0]).float().cpu().numpy()


class CrossEncoder:
    def __init__(self, model: str = MINILM, batch_size: int = 128) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model.split("/")[-1]
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(passages), self.batch_size):
            chunk = passages[i:i + self.batch_size]
            enc = self.tok([query] * len(chunk), chunk, padding=True, truncation="only_second",
                           max_length=512, return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits[:, 0].float().cpu())
        return torch.cat(out).numpy()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb(device: torch.device) -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 1024   # bytes on macOS, KiB on Linux
    out = {"peak_rss_mb": round(rss_mb)}
    if device.type == "cuda":
        out["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    return out


def rerank_split(cfg: Config, system, split: str, depth: int = 100, tag: str | None = None,
                 limit: int | None = None) -> Path:
    candidates = io.read_candidates(cfg.candidates_dir / f"{split}.tsv")
    if limit is not None:
        candidates = dict(list(candidates.items())[:limit])
    queries = split_spec(cfg, split).queries()
    passages = io.load_collection(cfg.data.collection, {c.pid for cs in candidates.values() for c in cs})
    device = getattr(system, "device", torch.device("cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    run, times, set_logits = {}, [], {}
    for qid, cands in candidates.items():
        cands = cands[:depth]
        if not cands:
            run[qid] = []
            continue
        _sync(device)
        t0 = time.perf_counter()
        scores = system.score(queries[qid], [passages[c.pid] for c in cands])
        _sync(device)
        times.append(time.perf_counter() - t0)
        run[qid] = sorted(zip((c.pid for c in cands), scores.tolist()), key=lambda t: -t[1])
        if getattr(system, "last_set_logit", None) is not None:
            set_logits[qid] = system.last_set_logit

    tag = tag or system.name
    if set_logits:          # the [SET] head's "any relevant passage" logit, for abstention and counts
        sp = cfg.results_dir / "set_logits" / f"{tag}.{split}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(set_logits))
    path = cfg.runs_dir / f"{tag}.{split}.run"
    io.write_run(run, path, tag)
    ms = 1000 * np.asarray(times[1:] or times)      # drop the warm-up query
    timing = {"system": tag, "split": split, "depth": depth, "device": device.type,
              "ms_per_query_mean": round(float(ms.mean()), 2),
              "ms_per_query_p50": round(float(np.percentile(ms, 50)), 2),
              "ms_per_query_p95": round(float(np.percentile(ms, 95)), 2)} | peak_memory_mb(device)
    out = cfg.results_dir / "timings" / f"{tag}.{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timing, indent=2))
    return path
