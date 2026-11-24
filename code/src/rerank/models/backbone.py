"""Backbone loading and query-passage pair encoding.

Laya's pinned English checkpoint ships a ModernBERT-large encoder config
(`encoder/config.json`), its tokenizer (`tokenizer/`), and one combined
`model.safetensors` holding the decision-trained backbone *and* Laya's own
decision head. We build a bare `ModernBertModel` from the config and copy in
only the backbone tensors, found by locating the key prefix under which the
encoder's parameters are stored. Laya's original head is discarded; our new
relevance head replaces it. Any missing backbone tensor is a hard error, so a
silently random-initialized backbone can never reach training.

`kind = "hf"` loads any standard AutoModel checkpoint (the ModernBERT fallback,
or unadapted ModernBERT-large for the decision-pretraining ablation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from rerank.config import BackboneConfig

ANCHOR = "embeddings.tok_embeddings.weight"   # a key every ModernBertModel has
# Project rule: the Mac GPU (MPS) is never used, at any cost. Every model and tensor in this project is
# placed with pick_device(), which returns only CUDA or CPU; no library chooses a device on its own
# (no automatic device maps, no pipelines).


@dataclass
class Backbone:
    model: torch.nn.Module
    tokenizer: object
    hidden_size: int
    tag: str


def pick_device() -> torch.device:
    """CUDA, else CPU; RERANK_DEVICE overrides. The Mac GPU (MPS) is never used, by project rule."""
    requested = os.environ.get("RERANK_DEVICE")
    if requested:
        if requested.startswith("mps"):
            raise RuntimeError("the Mac GPU (MPS) is off limits for this project; use cuda or cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def remap_backbone_state(
    state: dict[str, torch.Tensor], expected: Iterable[str]
) -> dict[str, torch.Tensor]:
    """Return the subset of `state` that fills `expected`, with its prefix stripped.

    The checkpoint may store the encoder under any prefix ("", "model.",
    "encoder.model.", ...). Every key ending in the anchor names a candidate
    prefix; the one that supplies every expected key wins.
    """
    expected = list(expected)
    prefixes = sorted({k[: -len(ANCHOR)] for k in state if k.endswith(ANCHOR)}, key=len)
    best_missing: list[str] = expected
    for prefix in prefixes:
        missing = [k for k in expected if prefix + k not in state]
        if not missing:
            return {k: state[prefix + k] for k in expected}
        if len(missing) < len(best_missing):
            best_missing = missing
    raise KeyError(
        f"no prefix in the checkpoint supplies the full backbone; candidate prefixes {prefixes}; "
        f"{len(best_missing)} keys missing at best, e.g. {best_missing[:5]}"
    )


def _prepare_config(config) -> None:
    # ModernBERT can torch.compile its MLP by default; that breaks on MPS/CPU and
    # is unnecessary for inference.
    if hasattr(config, "reference_compile"):
        config.reference_compile = False


def load_laya(cfg: BackboneConfig) -> Backbone:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    root = Path(snapshot_download(
        cfg.repo, revision=cfg.revision,
        allow_patterns=["encoder/config.json", "tokenizer/*", "model.safetensors"],
    ))
    config = AutoConfig.from_pretrained(root / "encoder")
    _prepare_config(config)
    model = AutoModel.from_config(config, attn_implementation="sdpa")
    expected = model.state_dict().keys()
    state = remap_backbone_state(load_file(root / "model.safetensors"), expected)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    return Backbone(model, tokenizer, config.hidden_size, cfg.tag)


def load_hf(cfg: BackboneConfig) -> Backbone:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(cfg.fallback)
    _prepare_config(config)
    model = AutoModel.from_pretrained(cfg.fallback, config=config, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(cfg.fallback)
    return Backbone(model, tokenizer, config.hidden_size, cfg.tag)


def load_backbone(cfg: BackboneConfig) -> Backbone:
    if cfg.kind == "laya":
        return load_laya(cfg)
    if cfg.kind == "hf":
        return load_hf(cfg)
    raise ValueError(f"unknown backbone kind {cfg.kind!r}")


class PairEncoder:
    """Encode (query, passage) pairs jointly and return the final [CLS] state.

    Inputs are `[CLS] query [SEP] passage [SEP]`. Queries are cut to
    `max_query_tokens` first; the passage gets whatever budget remains, so the
    query is never truncated to make room for passage text.
    """

    def __init__(self, backbone: Backbone, max_length: int = 256, max_query_tokens: int = 64,
                 device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        if max_query_tokens > max_length - 3:
            raise ValueError("max_query_tokens leaves no room for the passage")
        self.device = device or pick_device()
        self.dtype = dtype or pick_dtype("auto", self.device)
        self.model = backbone.model.to(self.device, self.dtype).eval()
        self.tok = backbone.tokenizer
        self.hidden_size = backbone.hidden_size
        self.max_length = max_length
        self.max_query_tokens = max_query_tokens
        self.cls_id = self.tok.cls_token_id
        self.sep_id = self.tok.sep_token_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def build_inputs(self, queries: list[str], passages: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        q_ids = self.tok(queries, add_special_tokens=False)["input_ids"]
        p_ids = self.tok(passages, add_special_tokens=False)["input_ids"]
        seqs = []
        for q, p in zip(q_ids, p_ids, strict=True):
            q = q[: self.max_query_tokens]
            room = self.max_length - len(q) - 3
            seqs.append([self.cls_id, *q, self.sep_id, *p[:room], self.sep_id])
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids, mask

    def pooled(self, queries: list[str], passages: list[str]) -> torch.Tensor:
        """Final [CLS] states on the model's device, float32, with autograd left on."""
        ids, mask = self.build_inputs(queries, passages)
        out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        return out.last_hidden_state[:, 0].float()

    @torch.inference_mode()
    def encode(self, queries: list[str], passages: list[str]) -> torch.Tensor:
        return self.pooled(queries, passages).cpu()
