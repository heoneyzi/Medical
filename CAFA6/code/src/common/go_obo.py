from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}  # MF, BP, CC roots


def parse_go_parents(obo_path: str | Path) -> dict[str, set[str]]:
    """Parse GO parents from an OBO file (go-basic.obo).

    Captures `is_a` and `relationship: part_of` edges.
    """
    obo_path = Path(obo_path)
    term_parents: dict[str, set[str]] = defaultdict(set)
    cur_id: str | None = None

    with obo_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "[Term]":
                cur_id = None
                continue

            if line.startswith("id: "):
                cur_id = line.split("id: ", 1)[1].strip()
                continue

            if cur_id is None:
                continue

            if line.startswith("is_a: "):
                # is_a: GO:0000001 ! some name
                parent = line.split()[1].strip()
                term_parents[cur_id].add(parent)
                continue

            if line.startswith("relationship: part_of "):
                # relationship: part_of GO:0000001 ! some name
                parts = line.split()
                if len(parts) >= 3:
                    term_parents[cur_id].add(parts[2].strip())
                continue

    return dict(term_parents)


def make_ancestor_getter(term_parents: dict[str, set[str]]):
    @lru_cache(maxsize=None)
    def get_ancestors(term: str) -> set[str]:
        parents = term_parents.get(term, set())
        out = set(parents)
        for p in parents:
            out |= get_ancestors(p)
        return out

    return get_ancestors


def propagate_scores_to_parents(
    scores: dict[str, float],
    *,
    term_parents: dict[str, set[str]],
) -> dict[str, float]:
    """Propagate child scores to parents using max(child) semantics."""
    get_ancestors = make_ancestor_getter(term_parents)

    out = dict(scores)
    for term, score in scores.items():
        for parent in get_ancestors(term):
            prev = out.get(parent)
            if prev is None or score > prev:
                out[parent] = score

    return out


def cap_top_terms(
    scores: dict[str, float],
    *,
    max_terms: int,
    min_score: float,
) -> list[tuple[str, float]]:
    kept = [(t, s) for t, s in scores.items() if s > 0 and s >= min_score]
    kept.sort(key=lambda x: (-x[1], x[0]))
    return kept[:max_terms]


def format_score_3sig(score: float) -> str:
    score = float(score)
    if score <= 0:
        return "0"
    if score > 1.0:
        score = 1.0
    # Up to 3 significant figures (as per task spec).
    return format(score, ".3g")


def merge_score_dicts(dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            prev = merged.get(k)
            if prev is None or v > prev:
                merged[k] = v
    return merged
