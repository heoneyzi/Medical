"""Does this benchmark contain a planning problem at all?

These checks run before any model is trained, and they are the ones that can
disqualify the study.  A benchmark where one query answers everything, or where
the contract already picks the action, measures nothing about planning no matter
how carefully the planner is built.

Every number here is reported as a comparison between two measured quantities
with an interval, never as a pass mark.  "Single-family ceiling is 0.30" means
nothing on its own; "the full registry beats the best single family by 0.70
[0.67, 0.73] per gene" is a statement about the benchmark.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from typing import Any

from geoflowagent.geoacmg.claims import Finding, Role
from geoflowagent.geoacmg.inference import paired_contrast
from geoflowagent.geoacmg.tasks import Task
from geoflowagent.geoacmg.toolmap import ToolFamily

DIAGNOSTIC_CLAIM = "B0"
"""Benchmark well-posedness. Registered as a claim so its findings have a home,
but its role is DIAGNOSTIC: it describes the instrument, it does not support a
scientific conclusion."""


def _solvable_with(task: Task, allowed: frozenset[ToolFamily]) -> bool:
    """Could a policy restricted to ``allowed`` ever satisfy this task's verifier?"""

    return task.required_families <= allowed


def single_family_ceiling(tasks: Sequence[Task]) -> dict[str, Any]:
    """Best accuracy reachable by a policy allowed to call only one kind of tool.

    The quantity that matters is the *gap* to the full registry.  If a single
    family gets you almost everything, the benchmark is a lookup with extra
    steps, and no amount of planning machinery will show a difference.
    """

    families = sorted({f for task in tasks for f in task.families}, key=lambda f: f.value)
    per_family = {
        family.value: sum(_solvable_with(task, frozenset({family})) for task in tasks) / len(tasks)
        for family in families
    }
    best = max(per_family.values()) if per_family else 0.0
    best_family = max(per_family, key=per_family.get) if per_family else ""
    return {
        "per_family": per_family,
        "best_single_family": best_family,
        "best_single_family_accuracy": best,
        "full_registry_accuracy": 1.0,
        "gap": 1.0 - best,
        "note": (
            "full-registry accuracy is 1.0 by construction: a task is only built when "
            "every code the panel applied is decidable by the registry. The gap is "
            "therefore exactly the share of tasks that need more than the single most "
            "useful tool family."
        ),
    }


def single_family_gap_finding(
    tasks: Sequence[Task],
    *,
    seed: int = 17,
    resamples: int = 2000,
    role: Role = Role.DIAGNOSTIC,
) -> Finding:
    """The single-family gap as a paired, gene-clustered interval.

    ``role`` defaults to diagnostic because that is what this number is when it
    is read off a data card.  The preregistration names
    ``benchmark_needs_more_than_one_tool_family`` among its primary findings, so
    the run that is meant to adjudicate B0 asks for ``Role.PRIMARY`` explicitly
    rather than getting it by accident.
    """

    ceiling = single_family_ceiling(tasks)
    best = ceiling["best_single_family"]
    best_family = next(f for f in ToolFamily if f.value == best)
    full = [1.0 for _ in tasks]
    restricted = [
        1.0 if _solvable_with(task, frozenset({best_family})) else 0.0 for task in tasks
    ]
    return paired_contrast(
        claim_id=DIAGNOSTIC_CLAIM,
        name="benchmark_needs_more_than_one_tool_family",
        left=full,
        right=restricted,
        clusters=[task.record.gene for task in tasks],
        unit="gene",
        role=role,
        resamples=resamples,
        seed=seed,
        detail={
            "best_single_family": best,
            "per_family": ceiling["per_family"],
            # The left arm is 1.0 for every task by construction; a task is only
            # built when the registry can decide every code the panel applied.
            # The interval therefore measures the share of tasks that need more
            # than one tool family, and a reader has to be told that.
            "note": ceiling["note"],
            "tasks": len(tasks),
        },
    )


def decision_freedom(tasks: Sequence[Task]) -> dict[str, Any]:
    """How much choice the agent actually has.

    ``forced`` counts tasks where every available family is also required, so
    there is nothing to decide beyond ordering.  ``discardable`` counts the
    families an optimal policy must learn to skip -- the panel looked at them and
    got nothing, and the agent has to work that out without being told.
    """

    forced = 0
    discardable: list[int] = []
    for task in tasks:
        available = frozenset(task.families)
        extra = available - task.required_families
        discardable.append(len(extra))
        if not extra:
            forced += 1
    counts = collections.Counter(discardable)
    return {
        "tasks": len(tasks),
        "tasks_with_no_choice": forced,
        "tasks_with_no_choice_fraction": forced / len(tasks) if tasks else 0.0,
        "discardable_families": {
            "histogram": {str(k): v for k, v in sorted(counts.items())},
            "mean": sum(discardable) / len(discardable) if discardable else 0.0,
        },
        "note": (
            "a task with no discardable family is an enumeration, not a decision; a "
            "benchmark made only of those would measure ordering and nothing else"
        ),
    }


def ordering_multiplicity(tasks: Sequence[Task]) -> dict[str, Any]:
    """How many correct orders of the required work exist.

    This is the object a distribution over plans is asked to cover.  A greedy
    policy commits to one order and cannot represent the others even in
    principle, so tasks with multiplicity one cannot distinguish the two.
    """

    counts = collections.Counter(task.optimal_orderings() for task in tasks)
    multi = sum(v for k, v in counts.items() if k > 1)
    return {
        "histogram": {str(k): v for k, v in sorted(counts.items())},
        "tasks_with_multiple_orderings": multi,
        "fraction": multi / len(tasks) if tasks else 0.0,
        "mean_log2_orderings": (
            sum(math.log2(task.optimal_orderings()) for task in tasks) / len(tasks)
            if tasks
            else 0.0
        ),
    }


def target_balance(tasks: Sequence[Task]) -> dict[str, Any]:
    """Class balance, and the accuracy of always guessing the commonest answer.

    Reported so that no downstream accuracy number can be read without its
    majority-class reference.
    """

    counts = collections.Counter(task.target.value for task in tasks)
    total = sum(counts.values())
    majority = max(counts.values()) / total if total else 0.0
    return {
        "counts": dict(counts.most_common()),
        "majority_class_rate": majority,
        "note": "a classifier that always answers the commonest label scores this",
    }


def effective_cluster_curve(
    tasks: Sequence[Task], caps: Sequence[int] = (25, 50, 100, 200, 400)
) -> dict[str, Any]:
    """Effective independent sample size as a function of a per-gene cap.

    One gene can dominate a curation corpus by an order of magnitude, and under
    cluster resampling that collapses the effective sample far below the gene
    count.  Capping trades tasks for independence.  The curve is reported rather
    than a cap being chosen here, because choosing one is a pre-registration
    decision, not a library default.
    """

    sizes = collections.Counter(task.record.gene for task in tasks)
    rows = []
    for cap in [None, *caps]:
        capped = [min(n, cap) if cap else n for n in sizes.values()]
        total = sum(capped)
        effective = (total ** 2) / sum(n ** 2 for n in capped) if total else 0.0
        rows.append({"cap": cap, "tasks": total, "effective_clusters": effective})
    return {
        "curve": rows,
        "note": (
            "pre-register a cap before opening any test split; the trade is tasks "
            "against independent evidence, and it changes every interval in the study"
        ),
    }


def corpus_card(tasks: Sequence[Task]) -> dict[str, Any]:
    """Everything a reader needs to judge the instrument, in one object."""

    genes = collections.Counter(task.record.gene for task in tasks)
    panels = collections.Counter(task.record.expert_panel for task in tasks)
    sizes = sorted(genes.values())
    total = sum(sizes) or 1
    # Kish effective sample size under cluster resampling: with one gene holding a
    # large share of the tasks, the number of genes overstates how much independent
    # evidence there is, and every interval should be read against this instead.
    effective = (total ** 2) / sum(size ** 2 for size in sizes) if sizes else 0.0
    return {
        "tasks": len(tasks),
        "genes": len(genes),
        "expert_panels": len(panels),
        "tasks_per_gene": {
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "largest_gene": genes.most_common(1)[0][0] if genes else "",
            "largest_gene_share": (max(sizes) / total) if sizes else 0.0,
            "effective_number_of_clusters": effective,
            "note": (
                "the gene is the independent unit. When one gene holds a large share of "
                "the tasks, the gene count overstates the independent evidence; read "
                "every interval against effective_number_of_clusters, not against genes"
            ),
        },
        "effective_cluster_curve": effective_cluster_curve(tasks),
        "target_balance": target_balance(tasks),
        "single_family_ceiling": single_family_ceiling(tasks),
        "decision_freedom": decision_freedom(tasks),
        "ordering_multiplicity": ordering_multiplicity(tasks),
        "required_families_histogram": {
            str(k): v
            for k, v in sorted(
                collections.Counter(len(task.required_families) for task in tasks).items()
            )
        },
    }
