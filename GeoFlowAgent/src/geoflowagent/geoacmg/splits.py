"""Three generalisation axes, kept apart because they answer different questions.

``variant``  held-out variants inside genes the model has seen
``gene``     held-out genes: whole connected components move together
``temporal`` curations the frozen encoders could not have read, because the panel
             had not published them yet

Splitting on the gene is not a refinement of splitting on the variant, it is a
different experiment, and averaging them would hide whichever is worse.  The
temporal axis is orthogonal to both and is applied as a filter, not a partition.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from geoflowagent.geoacmg.tasks import Task
from geoflowagent.utils.io import sha256_text

QWEN25_RELEASE = date(2024, 9, 15)
"""Qwen2.5 was published on this date, so a curation approved afterwards cannot be
in its pretraining corpus. The other frozen encoders are older, so this is the
conservative cut for the whole encoder set."""


@dataclass(frozen=True)
class SplitPlan:
    axis: str
    assignment: dict[str, str]
    report: dict[str, Any]

    def counts(self) -> dict[str, int]:
        return dict(collections.Counter(self.assignment.values()))


def _stable_order(keys: Sequence[str], seed: int) -> list[str]:
    """Deterministic shuffle that does not depend on the Python hash seed."""

    return sorted(keys, key=lambda key: sha256_text(f"{seed}:{key}"))


def by_gene(
    tasks: Sequence[Task],
    *,
    fractions: dict[str, float] | None = None,
    seed: int = 17,
    by: str = "gene",
) -> SplitPlan:
    """Assign whole genes, balancing on task count rather than gene count.

    One gene can hold a quarter of the corpus, so assigning genes uniformly would
    produce splits whose sizes differ by an order of magnitude.  Genes are placed
    largest-first into whichever split is furthest below its target share, which
    keeps the split sizes close without ever separating two variants in one gene.
    """

    weights = fractions or {"train": 0.6, "dev": 0.2, "test": 0.2}
    total_weight = sum(weights.values())
    weights = {name: value / total_weight for name, value in weights.items()}
    def key_of(task: Task) -> str:
        if by == "panel":
            return task.record.expert_panel or "unassigned_panel"
        return task.record.gene

    by_gene_tasks: dict[str, list[Task]] = {}
    for task in tasks:
        by_gene_tasks.setdefault(key_of(task), []).append(task)
    targets = {name: share * len(tasks) for name, share in weights.items()}
    filled = dict.fromkeys(weights, 0.0)
    order = sorted(
        _stable_order(list(by_gene_tasks), seed),
        key=lambda gene: -len(by_gene_tasks[gene]),
    )
    gene_split: dict[str, str] = {}
    for gene in order:
        deficit = {name: targets[name] - filled[name] for name in weights}
        chosen = max(deficit, key=lambda name: (deficit[name], name))
        gene_split[gene] = chosen
        filled[chosen] += len(by_gene_tasks[gene])
    assignment = {task.task_id: gene_split[key_of(task)] for task in tasks}
    return SplitPlan(
        axis=by,
        assignment=assignment,
        report={
            "groups_per_split": dict(collections.Counter(gene_split.values())),
            "group_kind": by,
            "tasks_per_split": dict(collections.Counter(assignment.values())),
            "targets": targets,
            "note": "whole genes move together, so no paralog or panel leaks across splits",
        },
    )


def by_variant(
    tasks: Sequence[Task],
    *,
    fractions: dict[str, float] | None = None,
    seed: int = 17,
) -> SplitPlan:
    """Assign variants, keeping every gene represented in every split.

    This axis deliberately *does* leak the gene, because the question it asks is
    whether the model generalises to a new variant in a gene it has seen. Reading
    it as a substitute for the gene split would overstate generalisation.
    """

    weights = fractions or {"train": 0.6, "dev": 0.2, "test": 0.2}
    total_weight = sum(weights.values())
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name in sorted(weights):
        running += weights[name] / total_weight
        cumulative.append((name, running))
    assignment: dict[str, str] = {}
    for task in tasks:
        draw = int(sha256_text(f"{seed}:{task.task_id}")[:8], 16) / 0xFFFFFFFF
        for name, edge in cumulative:
            if draw <= edge:
                assignment[task.task_id] = name
                break
        else:  # pragma: no cover - floating point tail
            assignment[task.task_id] = cumulative[-1][0]
    return SplitPlan(
        axis="variant",
        assignment=assignment,
        report={
            "tasks_per_split": dict(collections.Counter(assignment.values())),
            "note": (
                "genes appear in every split by design; this axis measures "
                "variant-level generalisation only"
            ),
        },
    )


def temporal(tasks: Sequence[Task], *, cutoff: date = QWEN25_RELEASE) -> SplitPlan:
    """Mark each task as before or after the encoder cut-off.

    Tasks with no approval date go to ``before``, which is the conservative
    direction: it can only make the contamination-controlled half smaller, never
    let a possibly-contaminated record into it.
    """

    assignment: dict[str, str] = {}
    undated = 0
    for task in tasks:
        approved = task.record.approval_date
        if approved is None:
            undated += 1
            assignment[task.task_id] = "before"
        else:
            assignment[task.task_id] = "after" if approved >= cutoff else "before"
    counts = collections.Counter(assignment.values())
    genes_after = {
        task.record.gene for task in tasks if assignment[task.task_id] == "after"
    }
    return SplitPlan(
        axis="temporal",
        assignment=assignment,
        report={
            "cutoff": cutoff.isoformat(),
            "cutoff_rationale": (
                "Qwen2.5 release date; the other frozen encoders are older, so a "
                "curation approved after this date is outside every encoder's corpus"
            ),
            "tasks_per_side": dict(counts),
            "genes_after_cutoff": len(genes_after),
            "undated_records_placed_before": undated,
            "limitation": (
                "this controls database contamination, not literature contamination: a "
                "variant absent from a database may still have been discussed in papers "
                "the encoders read. Report alongside the gene split, not instead of it"
            ),
        },
    )
