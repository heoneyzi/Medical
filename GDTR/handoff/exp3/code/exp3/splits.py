"""Genomic split units and leakage checks for EXP3."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class GenomicUnit:
    unit_id: str
    chromosome: str
    start: int
    end: int
    dependency_keys: tuple[str, ...]
    family: str = "unlabelled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id or not self.chromosome or self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid genomic unit {self}")
        if not self.dependency_keys:
            raise ValueError(f"unit {self.unit_id!r} needs dependency keys")


def assert_no_genomic_overlap(left: Sequence[GenomicUnit],
                              right: Sequence[GenomicUnit], *,
                              buffer_bp: int = 0) -> None:
    """Reject coordinate overlap and hidden homolog/repeat/donor dependence."""
    if buffer_bp < 0:
        raise ValueError("buffer_bp must be non-negative")
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    left_keys: dict[str, str] = {}
    for unit in left:
        intervals.setdefault(unit.chromosome, []).append(
            (unit.start, unit.end, unit.unit_id))
        for key in unit.dependency_keys:
            left_keys[key] = unit.unit_id
    for unit in right:
        for start, end, owner in intervals.get(unit.chromosome, ()):
            if unit.start < end + buffer_bp and start - buffer_bp < unit.end:
                raise RuntimeError(
                    f"split leakage: {owner} overlaps {unit.unit_id} on "
                    f"{unit.chromosome} within buffer {buffer_bp}")
        shared = sorted(set(unit.dependency_keys) & set(left_keys))
        if shared:
            owners = sorted({left_keys[key] for key in shared})
            raise RuntimeError(
                f"split leakage: {unit.unit_id} shares dependency keys "
                f"{shared[:5]} with {owners[:5]}")

