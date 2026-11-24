"""Training groups: up to `group_size` BM25 candidates per training query.

Each group keeps every retrieved judged positive and fills the rest with
negatives sampled uniformly from BM25 ranks 1..negative_depth. MS MARCO has
almost no judged negatives, so most negatives are unjudged *proxy* negatives;
their grade stays -1 in the output, which is the flag every later stage reads.
Queries with no retrieved positive are skipped and counted.
"""

from __future__ import annotations

import json
import random

from rerank.config import Config
from rerank.data import io


def sample_group(
    cands: list[io.Candidate], size: int, negative_depth: int, rng: random.Random, threshold: int = 1
) -> list[io.Candidate] | None:
    positives = [c for c in cands if c.grade >= threshold][: size - 1]
    if not positives:
        return None
    negatives = [c for c in cands if c.grade < threshold and c.rank <= negative_depth]
    chosen = rng.sample(negatives, min(size - len(positives), len(negatives)))
    return sorted(positives + chosen, key=lambda c: c.rank)


def build_training_groups(cfg: Config) -> dict:
    g = cfg.groups
    candidates = io.read_candidates(cfg.candidates_dir / "train.tsv")
    rng = random.Random(g.seed)
    groups: dict[str, list[io.Candidate]] = {}
    skipped = 0
    for qid in sorted(candidates, key=io.qid_sort_key):
        group = sample_group(candidates[qid], g.group_size, g.negative_depth, rng)
        if group is None:
            skipped += 1
        else:
            groups[qid] = group
    io.write_candidates(groups, cfg.candidates_dir / "train.groups.tsv")

    rows = [c for grp in groups.values() for c in grp]
    stats = {
        "queries": len(candidates),
        "groups": len(groups),
        "skipped_no_retrieved_positive": skipped,
        "skipped_fraction": round(skipped / max(len(candidates), 1), 4),
        "pairs": len(rows),
        "positives": sum(c.grade >= 1 for c in rows),
        "judged_negatives": sum(c.grade == 0 for c in rows),
        "proxy_negatives": sum(c.grade == io.UNJUDGED for c in rows),
    }
    (cfg.candidates_dir / "train.groups.stats.json").write_text(json.dumps(stats, indent=2))
    return stats
