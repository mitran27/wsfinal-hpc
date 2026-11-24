"""File formats shared by every stage and by both team members.

Candidate files are the contract between retrieval, feature caching, training and
the baselines. Each row is

    qid <tab> pid <tab> bm25_rank <tab> bm25_score <tab> grade

where `grade` is the judged relevance grade, or -1 (`UNJUDGED`) when the passage
has no judgment. Rows for one query are contiguous and in BM25 rank order.
Run files use the standard six-column TREC format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

UNJUDGED = -1
CANDIDATE_HEADER = "qid\tpid\trank\tbm25\tgrade"


@dataclass(frozen=True)
class Candidate:
    pid: str
    rank: int
    score: float
    grade: int = UNJUDGED

    @property
    def judged(self) -> bool:
        return self.grade != UNJUDGED


def qid_sort_key(qid: str):
    """Numeric order for numeric ids, so output files diff cleanly."""
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def load_queries(path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and (wanted is None or parts[0] in wanted):
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read 3-column (course dev) or 4-column (TREC) judgments as {qid: {pid: grade}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
            elif len(parts) == 3:
                qid, pid, grade = parts
            else:
                continue
            qrels.setdefault(qid, {})[pid] = int(grade)
    return qrels


def load_collection(path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    """Read `pid <tab> text`. Pass `wanted` to keep memory to the passages in use."""
    passages: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            pid = line[:tab]
            if wanted is None or pid in wanted:
                passages[pid] = line[tab + 1:].rstrip("\n")
    return passages


def collection_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line[: line.find("\t")] for line in f if "\t" in line}


def read_qids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_qids(qids: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{q}\n" for q in qids))


def write_candidates(candidates: dict[str, list[Candidate]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(CANDIDATE_HEADER + "\n")
        for qid, cands in candidates.items():
            for c in cands:
                f.write(f"{qid}\t{c.pid}\t{c.rank}\t{c.score:.6f}\t{c.grade}\n")


def read_candidates(path: Path) -> dict[str, list[Candidate]]:
    """Read a candidate file, preserving query order and row order."""
    out: dict[str, list[Candidate]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != CANDIDATE_HEADER:
            raise ValueError(f"{path}: expected header {CANDIDATE_HEADER!r}, got {header!r}")
        for line in f:
            qid, pid, rank, score, grade = line.rstrip("\n").split("\t")
            out.setdefault(qid, []).append(Candidate(pid, int(rank), float(score), int(grade)))
    return out


def write_run(results: dict[str, list[tuple[str, float]]], path: Path, tag: str) -> None:
    """Write `qid Q0 pid rank score tag`; `results` lists (pid, score) best first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(results, key=qid_sort_key):
            for rank, (pid, score) in enumerate(results[qid], start=1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {tag}\n")


def read_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    results: dict[str, list[tuple[str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, pid, _, score, _ = line.split()
            results.setdefault(qid, []).append((pid, float(score)))
    return results
