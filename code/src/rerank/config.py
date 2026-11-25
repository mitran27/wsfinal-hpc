"""Typed configuration loaded from a TOML file.

Every stage reads one `Config`. Relative paths in the TOML resolve against the
TOML file's own directory, so commands behave the same from any working directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    collection: Path
    queries_train: Path
    qrels_train: Path
    queries_dev: Path
    qrels_dev: Path
    queries_trec: Path
    qrels_dl19: Path
    qrels_dl20: Path
    work_dir: Path = Path("work")
    collection_train: Path | None = None   # full MS MARCO collection (8.8M) for training passages; None: `collection`


@dataclass
class RetrievalConfig:
    k1: float = 1.2
    b: float = 0.75
    depth: int = 100


@dataclass
class SplitConfig:
    train_queries: int = 20000
    val_queries: int = 1000
    calib_queries: int = 1000
    seed: int = 13


@dataclass
class GroupConfig:
    group_size: int = 32
    negative_depth: int = 100
    seed: int = 13


@dataclass
class BackboneConfig:
    kind: str = "laya"
    repo: str = "convaiinnovations/laya"
    revision: str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
    fallback: str = "answerdotai/ModernBERT-base"
    max_length: int = 256
    max_query_tokens: int = 64
    batch_size: int = 64
    dtype: str = "auto"

    @property
    def tag(self) -> str:
        """Short name for the feature cache, so caches from different backbones never mix."""
        if self.kind == "laya":
            return f"laya-{self.revision[:7]}-L{self.max_length}"
        name = self.fallback.split("/")[-1].lower()
        return f"{name}-L{self.max_length}"


@dataclass
class ModelConfig:
    width: int = 256
    heads: int = 4
    ff_width: int = 1024
    blocks: int = 1
    dropout: float = 0.1
    set_token: bool = False      # [SET] token + set head: P(any passage relevant) (candidate variant only)
    ordinal: bool = False        # graded head: ordered thresholds θ₁ < θ₂ < θ₃ on one score
    rel_grade: int = 2           # binary labels supervise P(g ≥ rel_grade) (TREC "relevant" = grade 2)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_groups: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    eval_every: int = 250
    ranknet_weight: float = 1.0
    rl_weight: float = 0.5
    rl_samples: int = 4
    rl_cutoff: int = 10
    freeze_projection: bool = False
    set_weight: float = 0.5       # weight of the [SET] head's BCE (when the model has one)
    empty_rate: float = 0.15      # share of groups whose judged positives are hidden to make empty pools
    seed: int = 13


@dataclass
class FinetuneConfig:
    """End-to-end training: LoRA adapters on the backbone plus the relevance head."""

    steps: int = 2000
    queries_per_step: int = 2
    group_size: int = 16            # 1+ positive and hard negatives per query
    negative_depth: int = 100
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    bce_weight: float = 0.5         # listwise CE drives ranking; BCE keeps per-passage probabilities
    set_weight: float = 0.5         # [SET] head BCE (when the model has one)
    empty_rate: float = 0.15        # share of groups whose judged positives are dropped to make empty pools
    eval_every: int = 250
    eval_queries: int = 200         # validation queries scored at each evaluation
    eval_depth: int = 100
    grad_checkpointing: bool = True
    train_split: str = "train"      # training candidates file under candidates/ ("train58": all eligible queries)
    seed: int = 13


@dataclass
class Config:
    data: DataConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    # ------------------------------------------------------------ work layout
    @property
    def work(self) -> Path:
        return self.data.work_dir

    @property
    def index_dir(self) -> Path:
        return self.work / "index"

    @property
    def splits_dir(self) -> Path:
        return self.work / "splits"

    @property
    def candidates_dir(self) -> Path:
        return self.work / "candidates"

    @property
    def feature_dir(self) -> Path:
        return self.work / "features" / self.backbone.tag

    @property
    def models_dir(self) -> Path:
        return self.work / "models"

    @property
    def runs_dir(self) -> Path:
        return self.work / "runs"

    @property
    def scores_dir(self) -> Path:
        return self.work / "scores"

    @property
    def results_dir(self) -> Path:
        return self.work / "results"


_SECTIONS = {
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "splits": SplitConfig,
    "groups": GroupConfig,
    "backbone": BackboneConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "finetune": FinetuneConfig,
}


def _override(raw: dict, item: str) -> None:
    """Apply one `section.key=value` override; the value is parsed as TOML (bare words stay strings)."""
    key, _, value = item.partition("=")
    section, _, name = key.strip().partition(".")
    if not section or not name or not _:
        raise ValueError(f"override {item!r} is not of the form section.key=value")
    try:
        parsed = tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value.strip()
    raw.setdefault(section, {})[name] = parsed


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Parse the TOML file, apply `section.key=value` overrides, resolve data paths, and reject unknown keys."""
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    for item in overrides or []:
        _override(raw, item)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        values = dict(raw.get(name, {}))
        allowed = {f.name for f in fields(cls)}
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"unknown keys in [{name}]: {sorted(extra)}")
        if cls is DataConfig:
            values = {k: (path.parent / v).resolve() for k, v in values.items()}
        sections[name] = cls(**values)
    return Config(**sections)
