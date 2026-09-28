"""Turning a curated variant into a task the agent can actually be run on.

One task is: *reproduce this expert panel's classification of this variant, using
evidence you have to go and fetch.*

Three design decisions are worth stating, because each one could have been made
the easy way and the easy way would have produced a benchmark that flatters any
method run on it.

**A probe is a query, not a criterion.**  One gnomAD call tells a curator about
PM2, BA1 and BS1 at once, so the action space is one probe per *tool family*, not
one per code.  Cost therefore attaches to the query, which is where it attaches in
reality, and the agent's decision is which kind of evidence to go looking for --
which is the decision a curator actually makes.

**Codes the panel rejected are part of the task.**  A probe returns whatever the
panel found, including nothing.  A family whose codes were all ``Not Met`` costs a
call and moves the score by zero.  Without that, a policy that only ever queries
the families that end up mattering would look optimal, and it is not, because
which families matter is exactly what is unknown before the call.

**The benchmark declares its own domain of validity.**  A record enters only if
our tool registry can decide every code the panel applied, and if the point total
lands in the panel's own band.  Both filters are about whether the task is
well-posed for an automated agent; both are applied before any method is run, and
both are counted in the report.  Tasks excluded by the second filter are the ones
where the panel knew something the point scale does not encode.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from geoflowagent.geoacmg.clingen import CuratedVariant
from geoflowagent.geoacmg.evidence import (
    Classification,
    EvidenceCode,
    band,
    citations,
    points_to_band,
    score,
)
from geoflowagent.geoacmg.toolmap import BINDINGS, FAMILY_SOURCES, Decidability, ToolFamily
from geoflowagent.utils.io import canonical_json, sha256_text

GENERATOR = "geoacmg-v1"
EVIDENCE_ROOT = "evidence"
SOURCE_ROOT = "evidence_sources"
REPORT_ROOT = "report"


def _condition(field_path: str, op: str, value: Any = None) -> dict[str, Any]:
    row: dict[str, Any] = {"field": field_path, "op": op}
    if value is not None:
        row["value"] = value
    return row


@dataclass(frozen=True)
class ProbeOutcome:
    """What one family probe returns in the true world.

    ``met`` and ``not_met`` are the codes whose *primary* query is this family and
    which the panel applied or rejected.  A criterion belongs to exactly one
    probe, so its points are counted exactly once -- but it only becomes
    creditable once its prerequisite families have been queried too, which is
    evaluated at the task level by :meth:`Task.points_after`.
    """

    family: ToolFamily
    met: tuple[EvidenceCode, ...]
    not_met: tuple[EvidenceCode, ...]

    @property
    def points(self) -> int:
        """Points this probe can contribute once its prerequisites are satisfied."""

        return score(self.met)

    @property
    def informative(self) -> bool:
        return bool(self.met)

    @property
    def label(self) -> str:
        """The discretized observation bound into typed state."""

        if self.met:
            return "+".join(sorted(code.label for code in self.met))
        return "none_met"


@dataclass(frozen=True)
class Task:
    """A built task plus everything needed to score and expand it."""

    task_id: str
    record: CuratedVariant
    outcomes: tuple[ProbeOutcome, ...]
    split: str = "unassigned"

    @property
    def families(self) -> tuple[ToolFamily, ...]:
        return tuple(outcome.family for outcome in self.outcomes)

    @property
    def target(self) -> Classification:
        return self.record.assertion

    @property
    def total_points(self) -> int:
        return self.points_after(self.families)

    @property
    def informative_families(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.informative)

    @property
    def required_families(self) -> frozenset[ToolFamily]:
        """Every family that at least one applied criterion depends on."""

        needed: set[ToolFamily] = set()
        for outcome in self.outcomes:
            for code in outcome.met:
                needed.update(BINDINGS[code.base].families)
        return frozenset(needed)

    def points_after(self, queried: Iterable[ToolFamily]) -> int:
        """Score with only these families queried.

        A criterion counts when its primary query has been made *and* every
        prerequisite has: a truncating consequence is worth nothing until the
        gene's mechanism is known.
        """

        chosen = frozenset(queried)
        total = 0
        for outcome in self.outcomes:
            if outcome.family not in chosen:
                continue
            for code in outcome.met:
                if BINDINGS[code.base].resolved_by(chosen):
                    total += code.points
        return total

    def remaining_to_target(self, queried: Iterable[ToolFamily]) -> int:
        """Evidence still missing, in points, to reach the panel's band.

        This is the quantity the goal-conditioned distance head represents. It is
        not an invented cost: it is how much more evidence a curator still needs.
        """

        return points_to_band(self.points_after(queried), self.target)

    def is_sufficient(self, queried: Iterable[ToolFamily]) -> bool:
        """May the target classification be committed after querying these families?

        Two conditions, and both are the panel's own standard rather than ours:
        the evidence gathered must score into the panel's band, and it must
        include every line of evidence the panel actually applied.  The second
        condition is what "show your work" means here, and it is what stops the
        degenerate policy of committing a label without looking -- a policy that
        would otherwise be optimal for any agent that already knew the answer.

        Note what it does *not* require: the families whose codes the panel
        considered and rejected.  Those are available to call and cost the same,
        and knowing which ones to skip is the whole decision problem.
        """

        chosen = frozenset(queried)
        if not self.required_families <= chosen:
            return False
        return band(self.points_after(chosen)) is self.target

    def optimal_orderings(self) -> int:
        """How many distinct orders of the required families are equally correct.

        The *set* of evidence a curation needs is determined; the *order* is not.
        That multiplicity is the object a distribution over plans is supposed to
        cover, and a greedy policy cannot represent it at all.
        """

        n = len(self.required_families)
        result = 1
        for k in range(2, n + 1):
            result *= k
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "gene": self.record.gene,
            "variant": self.record.primary_hgvs,
            "disease": self.record.disease,
            "target": self.target.value,
            "total_points": self.total_points,
            "families": [f.value for f in self.families],
            "informative_families": self.informative_families,
            "required_families": sorted(f.value for f in self.required_families),
            "optimal_orderings": self.optimal_orderings(),
        }


@dataclass
class BuildReport:
    """Why each record was kept or dropped. Every filter is counted."""

    considered: int = 0
    built: int = 0
    dropped: collections.Counter = field(default_factory=collections.Counter)
    multi_path: int = 0
    """Tasks where more than one order of the required work is equally correct."""
    families_histogram: collections.Counter = field(default_factory=collections.Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "built": self.built,
            "dropped": dict(self.dropped),
            "tasks_with_multiple_optimal_orderings": self.multi_path,
            "informative_families_histogram": {
                str(k): v for k, v in sorted(self.families_histogram.items())
            },
            "filters": {
                "fully_decidable": "every applied code is bound to a tool we can call",
                "point_scale_reconstructs": (
                    "the point total lands in the panel's own band; records where it does "
                    "not are ones where the panel used knowledge the point scale does not "
                    "encode, and no evidence-gathering policy could reach the target"
                ),
            },
            "citations": citations(),
        }


def _outcomes_for(record: CuratedVariant) -> tuple[ProbeOutcome, ...]:
    met_by_family: dict[ToolFamily, list[EvidenceCode]] = collections.defaultdict(list)
    not_met_by_family: dict[ToolFamily, list[EvidenceCode]] = collections.defaultdict(list)
    for code in record.met:
        met_by_family[BINDINGS[code.base].primary].append(code)
    for code in record.not_met:
        binding = BINDINGS[code.base]
        if binding.decidability is not Decidability.DECIDABLE:
            continue
        for family in binding.families:
            not_met_by_family[family].append(code)
    prerequisites: set[ToolFamily] = set()
    for code in record.met:
        prerequisites.update(BINDINGS[code.base].requires)
    families = sorted(
        set(met_by_family) | set(not_met_by_family) | prerequisites, key=lambda f: f.value
    )
    return tuple(
        ProbeOutcome(
            family=family,
            met=tuple(sorted(met_by_family.get(family, ()), key=lambda c: c.label)),
            not_met=tuple(sorted(not_met_by_family.get(family, ()), key=lambda c: c.label)),
        )
        for family in families
    )


def build_tasks(
    records: Sequence[CuratedVariant],
    *,
    require_reconstruction: bool = True,
) -> tuple[list[Task], BuildReport]:
    """Build the well-posed subset of the corpus."""

    report = BuildReport()
    tasks: list[Task] = []
    for record in records:
        report.considered += 1
        if any(
            BINDINGS[code.base].decidability is not Decidability.DECIDABLE
            for code in record.met
        ):
            report.dropped["not_fully_decidable"] += 1
            continue
        if require_reconstruction and not record.reconstructs:
            report.dropped["point_scale_disagrees_with_panel"] += 1
            continue
        outcomes = _outcomes_for(record)
        if not any(outcome.informative for outcome in outcomes):
            report.dropped["no_informative_probe"] += 1
            continue
        task_id = f"{record.gene}:{record.uuid[:12]}"
        task = Task(task_id=task_id, record=record, outcomes=outcomes)
        if band(task.total_points) is not task.target:
            # Defensive: family binding must not change the score.
            report.dropped["family_binding_changed_score"] += 1
            continue
        tasks.append(task)
        report.built += 1
        report.families_histogram[task.informative_families] += 1
        if task.optimal_orderings() > 1:
            report.multi_path += 1
    return tasks, report


# --------------------------------------------------------------------- corpus

def probe_tool(family: ToolFamily, cost: float) -> dict[str, Any]:
    """Contract for one family probe, in the repository's search-corpus schema."""

    path = f"{EVIDENCE_ROOT}.{family.value}"
    return {
        "tool_id": f"probe_{family.value}",
        "name": f"{family.value.replace('_', ' ').title()} query",
        "description": (
            f"Retrieve {family.value.replace('_', ' ')} evidence for this variant from "
            + (", ".join(FAMILY_SOURCES[family]) or "a curated source")
            + ". Returns whichever ACMG criteria this evidence line supports, which may "
            "be none."
        ),
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": ["variant"],
            "properties": {"variant": {"type": "string"}},
            "additionalProperties": False,
        },
        "argument_bindings": {"variant": "hgvs"},
        "output_schema": {
            "type": "object",
            "required": ["label", "source_release"],
            "properties": {
                "label": {"type": "string"},
                "source_release": {"type": "string"},
                "payload": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "output_bindings": {"label": path, "source_release": f"{SOURCE_ROOT}.{family.value}"},
        "preconditions": [_condition(path, "missing")],
        "effects": [],
        "tags": [family.value, "probe"],
        "search_cost": float(cost),
        "provenance": {"generator": GENERATOR, "family": family.value},
    }


def report_tool(classification: Classification, cost: float) -> dict[str, Any]:
    slug = classification.value.lower().replace(" ", "_")
    return {
        "tool_id": f"report_{slug}",
        "name": f"Report {classification.value}",
        "description": f"Commit to the classification {classification.value}.",
        "execution_mode": "symbolic",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "preconditions": [_condition(f"{REPORT_ROOT}.ready", "missing")],
        "effects": [
            {"field": f"{REPORT_ROOT}.label", "op": "set", "value": classification.value},
            {"field": f"{REPORT_ROOT}.ready", "op": "set", "value": True},
        ],
        "tags": ["report", "terminal"],
        "search_cost": float(cost),
        "provenance": {"generator": GENERATOR, "classification": classification.value},
    }


def initial_state(task: Task) -> dict[str, Any]:
    """Model-visible state at t=0. No evidence, by construction."""

    record = task.record
    return {
        "record_id": task.task_id,
        "hgvs": record.primary_hgvs,
        "gene_symbol": record.gene,
        "disease": record.disease,
        "mondo_id": record.mondo_id,
        "clinvar_id": record.clinvar_id,
        "allele_registry_id": record.allele_registry_id,
        EVIDENCE_ROOT: {},
        SOURCE_ROOT: {},
    }


def private_verifier(task: Task) -> dict[str, Any]:
    """Ground truth: the panel's classification. Never model-visible."""

    conditions = [
        _condition(f"{REPORT_ROOT}.ready", "eq", True),
        _condition(f"{REPORT_ROOT}.label", "eq", task.target.value),
    ]
    for family in sorted(task.required_families, key=lambda f: f.value):
        conditions.append(_condition(f"{EVIDENCE_ROOT}.{family.value}", "exists"))
    return {"conditions": conditions}


def public_goal() -> dict[str, Any]:
    """What the agent can see: that a classification was committed, not which."""

    return {"conditions": [_condition(f"{REPORT_ROOT}.ready", "eq", True)]}


def snapshots_for(
    task: Task,
    source_release: str,
    *,
    registry: Sequence[ToolFamily] | None = None,
) -> list[dict[str, Any]]:
    """Exact outcome of every probe in the true world, for the whole registry.

    Every task gets a snapshot for *every* family in the registry, not only for
    the families its panel used.  Two reasons, and the second one is the
    important one.

    Mechanically, the search graph has to be complete: an action with no snapshot
    is an unresolved action, and the oracle refuses to turn those into negatives.

    Scientifically, restricting a task's action space to the families that happen
    to matter for it would hand the agent the answer.  "Here are the four tools
    you are allowed to call" is most of the decision already made.  Instead every
    task offers the same registry, and a query the panel had no use for returns
    ``none_met``: it costs a call and supports nothing, which is what such a
    query does in practice.
    """

    families = list(registry) if registry is not None else [o.family for o in task.outcomes]
    known = {outcome.family: outcome for outcome in task.outcomes}
    outcomes = [
        known.get(family, ProbeOutcome(family=family, met=(), not_met=()))
        for family in families
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        tool_id = f"probe_{outcome.family.value}"
        arguments = {"variant": task.record.primary_hgvs}
        rows.append(
            {
                "snapshot_id": sha256_text(
                    canonical_json({"tool": tool_id, "arguments": arguments})
                )[:24],
                "status": "success",
                "source_revision": source_release,
                "tool_id": tool_id,
                "arguments": arguments,
                "output": {
                    "label": outcome.label,
                    "source_release": source_release,
                    "payload": {
                        "family": outcome.family.value,
                        "met": [c.to_dict() for c in outcome.met],
                        "not_met": [c.to_dict() for c in outcome.not_met],
                        "points": outcome.points,
                        "sources": list(FAMILY_SOURCES[outcome.family]),
                    },
                },
                "provenance": {
                    "task_id": task.task_id,
                    "split": task.split,
                    "generator": GENERATOR,
                },
            }
        )
    return rows


def cap_per_gene(tasks: Sequence[Task], limit: int) -> list[Task]:
    """Keep at most ``limit`` tasks per gene, deterministically.

    Selection is by a stable hash of the task id, so the kept subset does not
    depend on corpus order or on the Python hash seed, and re-running the build
    reproduces it exactly.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    by_gene: dict[str, list[Task]] = {}
    for task in tasks:
        by_gene.setdefault(task.record.gene, []).append(task)
    kept: list[Task] = []
    for gene in sorted(by_gene):
        ordered = sorted(by_gene[gene], key=lambda t: sha256_text(t.task_id))
        kept.extend(ordered[:limit])
    return sorted(kept, key=lambda t: t.task_id)


def fingerprint(tasks: Sequence[Task]) -> str:
    return sha256_text(canonical_json([task.to_dict() for task in tasks]))[:16]
