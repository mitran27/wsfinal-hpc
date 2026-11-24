"""Minimal LoRA for the ModernBERT backbone.

Each targeted nn.Linear W becomes W x + (alpha / r) * B A x, with W frozen,
A (r x in) Kaiming-initialised and B (out x r) zero-initialised, so training
starts exactly at the pretrained backbone. ModernBERT's linear layers are
attn.Wqkv, attn.Wo, mlp.Wi and mlp.Wo; the default targets all four.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TARGETS = ("Wqkv", "Wo", "Wi")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)
        return self.base(x) + (self.drop(x) @ a.t() @ b.t()) * self.scale


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05,
             targets=DEFAULT_TARGETS) -> list[str]:
    """Freeze `model` and wrap every nn.Linear whose attribute name is in `targets`."""
    model.requires_grad_(False)
    wrapped = []
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"{parent_name}.{name}" if parent_name else name)
    if not wrapped:
        raise ValueError(f"no nn.Linear named {targets} found")
    return wrapped


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight (W += scale * B A) and drop the wrapper."""
    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                delta = (child.lora_B.float() @ child.lora_A.float()) * child.scale
                base.weight.add_(delta.to(base.weight.dtype))
                setattr(parent, name, base)
                merged += 1
    return merged


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = [k for k in state if k not in model.state_dict()]
    if missing:
        raise KeyError(f"adapter keys not in model, e.g. {missing[:3]}")
    model.load_state_dict(state, strict=False)
