"""Adaptive, audit-preserving orchestration for EXP3.

Scientific non-support is a result, not an execution failure.  This module
therefore separates three concepts which are often accidentally conflated:

* :class:`EvidenceStatus` records what the data say about a hypothesis;
* :class:`ValidityReport` records whether that statement may be interpreted;
* :class:`ExecutionState` records what the program can run next.

Every *valid* scientific status completes an experiment and can activate a
compatible next analysis.  Only invalid measurement, provenance, or split
isolation can hard-fail a node.  Runtime/programming errors are retryable and
are never silently converted into scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import ArtifactRef, ArtifactStore

from .foundation import FIXED_EXP2_FOUNDATION_CLAIMS


_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class EvidenceStatus(str, Enum):
    """Exhaustive outcomes for a valid scientific comparison."""

    SUPPORTED = "supported"
    EQUIVALENT = "equivalent"
    MIXED = "mixed"
    UNRESOLVED = "unresolved"
    REFUTED = "refuted"


ALL_EVIDENCE_STATUSES = frozenset(EvidenceStatus)

EXP3_PHASES = (
    "phase1_roles_plateau",
    "phase2_causal_factorization",
    "phase3_reverse_trace",
    "phase4_motif_validation",
    "phase5_learned_transport",
    "phase6_applications",
    "phase7_generalization",
    "phase8_optional_features",
)

class ExecutionState(str, Enum):
    """Operational state, deliberately independent of scientific outcome."""

    DORMANT = "dormant"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    RETRY_REQUIRED = "retry_required"
    HARD_FAILED = "hard_failed"
    BLOCKED_INVALID = "blocked_invalid"


class InvalidityKind(str, Enum):
    """The only conditions allowed to hard-fail an EXP3 experiment."""

    MEASUREMENT = "measurement"
    PROVENANCE = "provenance"
    LEAKAGE = "leakage"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _json_safe(value: Any, *, path: str = "value") -> None:
    """Reject diagnostics that cannot be faithfully persisted.

    Allowing ``default=str`` here would make tensor/object representations part
    of a scientific record without preserving the actual artifact.  Large
    arrays belong in a hashed artifact store and should be referenced by ID.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Enum):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            _json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains non-JSON value {type(value).__name__}; persist it as "
        "a hashed artifact and record only its reference"
    )


@dataclass(frozen=True)
class ValidityReport:
    """Preconditions for interpreting a scientific decision.

    A failed power calculation, wide confidence interval, or negative result
    does *not* belong here.  Those map to ``unresolved`` or ``refuted``.
    """

    measurement_valid: bool = True
    provenance_valid: bool = True
    leakage_free: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _json_safe(self.details, path="validity.details")

    @property
    def invalidities(self) -> tuple[InvalidityKind, ...]:
        out: list[InvalidityKind] = []
        if not self.measurement_valid:
            out.append(InvalidityKind.MEASUREMENT)
        if not self.provenance_valid:
            out.append(InvalidityKind.PROVENANCE)
        if not self.leakage_free:
            out.append(InvalidityKind.LEAKAGE)
        return tuple(out)

    @property
    def valid(self) -> bool:
        return not self.invalidities


@dataclass(frozen=True)
class ScientificDecision:
    """A valid experiment's interpretation and machine-readable branch code."""

    status: EvidenceStatus
    conclusion_code: str
    interpretation: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    alternatives: tuple[str, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))
        if not _SLUG.match(self.conclusion_code):
            raise ValueError(
                "conclusion_code must be a lowercase machine-readable slug"
            )
        if not self.interpretation.strip():
            raise ValueError("interpretation is required")
        _json_safe(self.diagnostics, path="decision.diagnostics")
        if any(not item.strip() for item in self.alternatives):
            raise ValueError("alternative interpretations cannot be blank")
        if any(not isinstance(item, ArtifactRef) for item in self.artifact_refs):
            raise TypeError("artifact_refs must contain full hash-sealed ArtifactRef objects")
        for item in self.artifact_refs:
            ArtifactStore.verify(item)


@dataclass(frozen=True)
class ExperimentOutcome:
    decision: ScientificDecision
    validity: ValidityReport = field(default_factory=ValidityReport)


@dataclass(frozen=True)
class RetryDirective:
    """A technical retry/adaptation request, never a scientific verdict."""

    reason: str
    adjustments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("retry reason is required")
        _json_safe(self.adjustments, path="retry.adjustments")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    purpose: str
    phase: str = "phase1_roles_plateau"
    depends_on: tuple[str, ...] = ()
    initially_active: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SLUG.match(self.experiment_id):
            raise ValueError(f"invalid experiment_id {self.experiment_id!r}")
        if not self.purpose.strip():
            raise ValueError("experiment purpose is required")
        if self.phase not in EXP3_PHASES:
            raise ValueError(f"unknown EXP3 phase {self.phase!r}")
        if self.experiment_id in self.depends_on:
            raise ValueError("an experiment cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on contains duplicates")


@dataclass(frozen=True)
class OutcomeMatcher:
    """Conjunctive matcher for one upstream result.

    Empty ``conclusion_codes`` means that any code with a matching status is
    accepted.  This lets a rule mean either "continue after any valid answer"
    or "follow specifically the second-writer interpretation".
    """

    statuses: frozenset[EvidenceStatus] = ALL_EVIDENCE_STATUSES
    conclusion_codes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        statuses = frozenset(EvidenceStatus(value) for value in self.statuses)
        object.__setattr__(self, "statuses", statuses)
        if not statuses:
            raise ValueError("OutcomeMatcher needs at least one status")
        for code in self.conclusion_codes:
            if not _SLUG.match(code):
                raise ValueError(f"invalid conclusion code {code!r}")

    def matches(self, decision: ScientificDecision) -> bool:
        return (
            decision.status in self.statuses
            and (
                not self.conclusion_codes
                or decision.conclusion_code in self.conclusion_codes
            )
        )


@dataclass(frozen=True)
class RouteRule:
    """Activate targets when every source result matches its condition."""

    rule_id: str
    when: Mapping[str, OutcomeMatcher]
    activate: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not _SLUG.match(self.rule_id):
            raise ValueError(f"invalid rule_id {self.rule_id!r}")
        if not self.when or not self.activate:
            raise ValueError("a route rule needs source conditions and targets")
        if not self.rationale.strip():
            raise ValueError("route rationale is required")
        normalized = {
            key: value if isinstance(value, OutcomeMatcher)
            else OutcomeMatcher(**value)
            for key, value in self.when.items()
        }
        object.__setattr__(self, "when", normalized)


@dataclass
class ExperimentRecord:
    spec: ExperimentSpec
    state: ExecutionState
    activated_by: list[str] = field(default_factory=list)
    decision: ScientificDecision | None = None
    validity: ValidityReport | None = None
    retry_history: list[RetryDirective] = field(default_factory=list)
    invalidities: tuple[InvalidityKind, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    experiment_id: str
    attempts: int
    upstream: Mapping[str, ScientificDecision]
    adaptation_history: tuple[RetryDirective, ...]


ExperimentRunner = Callable[
    [ExecutionContext], ExperimentOutcome | ScientificDecision | RetryDirective
]


class AdaptiveRouter:
    """Run an outcome-adaptive experiment graph with an auditable history."""

    def __init__(self, run_id: str):
        if not run_id.strip():
            raise ValueError("run_id is required")
        self.run_id = run_id
        self.records: dict[str, ExperimentRecord] = {}
        self.rules: dict[str, RouteRule] = {}
        self.fired_rules: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.ledger_head_sha256 = ""
        self.started = False

    @property
    def foundation_claims(self) -> tuple[str, ...]:
        """Accepted EXP2 premises; intentionally has no setter."""
        return FIXED_EXP2_FOUNDATION_CLAIMS

    # -- Registration and audit -------------------------------------------------
    def _append(self, event: str, payload: Mapping[str, Any]) -> None:
        _json_safe(payload, path="ledger.payload")
        body = {
            "event": event,
            "payload": dict(payload),
            "timestamp": _now(),
            "previous_sha256": self.ledger_head_sha256,
        }
        body["record_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        self.events.append(body)
        self.ledger_head_sha256 = body["record_sha256"]

    def verify_ledger(self) -> None:
        previous = ""
        for row in self.events:
            body = dict(row)
            observed = body.pop("record_sha256", "")
            if body.get("previous_sha256") != previous:
                raise RuntimeError("adaptive ledger chain is broken")
            expected = hashlib.sha256(_canonical(body)).hexdigest()
            if observed != expected:
                raise RuntimeError("adaptive ledger record was modified")
            previous = observed
        if previous != self.ledger_head_sha256:
            raise RuntimeError("adaptive ledger head does not match records")

    def add_experiment(
        self, spec: ExperimentSpec, *, adaptation_reason: str | None = None
    ) -> None:
        if spec.experiment_id in self.records:
            raise KeyError(f"duplicate experiment {spec.experiment_id!r}")
        missing = sorted(set(spec.depends_on) - set(self.records))
        if missing:
            raise KeyError(f"unknown dependencies for {spec.experiment_id}: {missing}")
        if self.started and not (adaptation_reason and adaptation_reason.strip()):
            raise RuntimeError(
                "post-start experiments require an adaptation_reason for the audit trail"
            )
        state = ExecutionState.READY if spec.initially_active and not spec.depends_on \
            else ExecutionState.DORMANT
        self.records[spec.experiment_id] = ExperimentRecord(
            spec=spec, state=state,
            activated_by=["initial_plan"] if spec.initially_active else [],
        )
        self._append("add_experiment", {
            "experiment_id": spec.experiment_id,
            "depends_on": list(spec.depends_on),
            "phase": spec.phase,
            "initially_active": spec.initially_active,
            "adaptive": self.started,
            "adaptation_reason": adaptation_reason,
        })
        self._refresh_states()

    def add_rule(
        self, rule: RouteRule, *, adaptation_reason: str | None = None
    ) -> None:
        if rule.rule_id in self.rules:
            raise KeyError(f"duplicate rule {rule.rule_id!r}")
        unknown = (set(rule.when) | set(rule.activate)) - set(self.records)
        if unknown:
            raise KeyError(f"route {rule.rule_id} refers to unknown nodes {sorted(unknown)}")
        if self.started and not (adaptation_reason and adaptation_reason.strip()):
            raise RuntimeError(
                "post-start routes require an adaptation_reason for the audit trail"
            )
        self.rules[rule.rule_id] = rule
        self._append("add_rule", {
            "rule_id": rule.rule_id,
            "sources": sorted(rule.when),
            "targets": list(rule.activate),
            "adaptive": self.started,
            "adaptation_reason": adaptation_reason,
        })
        self._apply_routes()

    def add_exhaustive_status_routes(
        self,
        source: str,
        targets_by_status: Mapping[EvidenceStatus, Sequence[str]],
        *,
        rule_prefix: str,
        rationale: str,
        adaptation_reason: str | None = None,
    ) -> tuple[str, ...]:
        """Attach an explicit branch for each of the five scientific outcomes.

        This helper makes it difficult to accidentally recreate a success-only
        gate.  Multiple statuses may point to the same compatible experiment,
        but omitting a status is rejected.  When called after execution has
        started, ``adaptation_reason`` is required and copied into every route
        registration event.
        """
        normalized = {
            EvidenceStatus(status): tuple(targets)
            for status, targets in targets_by_status.items()
        }
        missing = ALL_EVIDENCE_STATUSES - set(normalized)
        extra = set(normalized) - ALL_EVIDENCE_STATUSES
        if missing or extra:
            raise ValueError(
                "exhaustive routes must cover exactly all EvidenceStatus values; "
                f"missing={sorted(item.value for item in missing)}, "
                f"extra={sorted(item.value for item in extra)}"
            )
        if not _SLUG.match(rule_prefix):
            raise ValueError("rule_prefix must be a lowercase machine-readable slug")
        if not rationale.strip():
            raise ValueError("rationale is required")
        if source not in self.records:
            raise KeyError(f"unknown route source {source!r}")
        unknown_targets: set[str] = set()
        for status, targets in normalized.items():
            if not targets:
                raise ValueError(f"status {status.value} has no compatible target")
            unknown_targets.update(set(targets) - set(self.records))
        if unknown_targets:
            raise KeyError(f"unknown route targets {sorted(unknown_targets)}")
        added: list[str] = []
        for status in EvidenceStatus:
            targets = normalized[status]
            rule_id = f"{rule_prefix}.{status.value}"
            self.add_rule(
                RouteRule(
                    rule_id=rule_id,
                    when={source: OutcomeMatcher(frozenset({status}))},
                    activate=targets,
                    rationale=f"{rationale} Outcome={status.value}.",
                ),
                adaptation_reason=adaptation_reason,
            )
            added.append(rule_id)
        return tuple(added)

    def activate(self, experiment_id: str, *, reason: str) -> None:
        """Explicitly schedule an audit-labelled adaptive experiment."""
        if experiment_id not in self.records:
            raise KeyError(experiment_id)
        if not reason.strip():
            raise ValueError("activation reason is required")
        record = self.records[experiment_id]
        if record.state in {
            ExecutionState.COMPLETED, ExecutionState.HARD_FAILED,
            ExecutionState.BLOCKED_INVALID,
        }:
            raise RuntimeError(f"cannot activate {experiment_id} from {record.state.value}")
        record.activated_by.append(f"manual:{reason}")
        self._append("activate", {"experiment_id": experiment_id, "reason": reason})
        self._refresh_states()

    # -- State transitions ------------------------------------------------------
    def _refresh_states(self) -> None:
        changed = True
        while changed:
            changed = False
            for record in self.records.values():
                if not record.activated_by or record.state not in {
                    ExecutionState.DORMANT, ExecutionState.BLOCKED_INVALID,
                }:
                    continue
                deps = [self.records[name] for name in record.spec.depends_on]
                if any(dep.state in {
                    ExecutionState.HARD_FAILED, ExecutionState.BLOCKED_INVALID,
                } for dep in deps):
                    if record.state is not ExecutionState.BLOCKED_INVALID:
                        record.state = ExecutionState.BLOCKED_INVALID
                        changed = True
                elif all(dep.state is ExecutionState.COMPLETED for dep in deps):
                    record.state = ExecutionState.READY
                    changed = True

    def _apply_routes(self) -> None:
        changed = True
        while changed:
            changed = False
            for rule_id, rule in self.rules.items():
                if rule_id in self.fired_rules:
                    continue
                decisions: dict[str, ScientificDecision] = {}
                ready = True
                for source, matcher in rule.when.items():
                    record = self.records[source]
                    if record.state is not ExecutionState.COMPLETED or record.decision is None:
                        ready = False
                        break
                    if not matcher.matches(record.decision):
                        ready = False
                        break
                    decisions[source] = record.decision
                if not ready:
                    continue
                self.fired_rules.add(rule_id)
                for target in rule.activate:
                    marker = f"rule:{rule_id}"
                    if marker not in self.records[target].activated_by:
                        self.records[target].activated_by.append(marker)
                self._append("route", {
                    "rule_id": rule_id,
                    "source_conclusions": {
                        key: value.conclusion_code for key, value in decisions.items()
                    },
                    "activated": list(rule.activate),
                    "rationale": rule.rationale,
                })
                self._refresh_states()
                changed = True

    def _context(self, experiment_id: str) -> ExecutionContext:
        record = self.records[experiment_id]
        upstream_names = set(record.spec.depends_on)
        for marker in record.activated_by:
            if not marker.startswith("rule:"):
                continue
            rule_id = marker.removeprefix("rule:")
            rule = self.rules.get(rule_id)
            if rule is not None:
                upstream_names.update(rule.when)
        upstream = {
            name: self.records[name].decision
            for name in sorted(upstream_names)
            if self.records[name].decision is not None
        }
        return ExecutionContext(
            experiment_id=experiment_id,
            attempts=1 + len(record.retry_history),
            upstream=upstream,  # type: ignore[arg-type]
            adaptation_history=tuple(record.retry_history),
        )

    def begin(self, experiment_id: str) -> ExecutionContext:
        record = self.records[experiment_id]
        if record.state is not ExecutionState.READY:
            raise RuntimeError(
                f"{experiment_id} is {record.state.value}, not ready"
            )
        self.started = True
        record.state = ExecutionState.RUNNING
        context = self._context(experiment_id)
        self._append("begin", {
            "experiment_id": experiment_id, "attempt": context.attempts,
        })
        return context

    def record_outcome(self, experiment_id: str, outcome: ExperimentOutcome) -> None:
        record = self.records[experiment_id]
        if record.state is not ExecutionState.RUNNING:
            raise RuntimeError(f"{experiment_id} is not running")
        record.validity = outcome.validity
        if not outcome.validity.valid:
            record.state = ExecutionState.HARD_FAILED
            record.invalidities = outcome.validity.invalidities
            record.error = (
                "invalid scientific record: "
                + ", ".join(item.value for item in record.invalidities)
            )
            # The decision is retained only as the attempted claim; it is never
            # routed or exposed through completed_decisions().
            record.decision = None
            self._append("hard_fail", {
                "experiment_id": experiment_id,
                "invalidities": [item.value for item in record.invalidities],
                "details": dict(outcome.validity.details),
            })
            self._refresh_states()
            return

        for artifact in outcome.decision.artifact_refs:
            ArtifactStore.verify(artifact)
        record.decision = outcome.decision
        record.state = ExecutionState.COMPLETED
        record.error = None
        self._append("scientific_decision", {
            "experiment_id": experiment_id,
            "status": outcome.decision.status.value,
            "conclusion_code": outcome.decision.conclusion_code,
            "interpretation": outcome.decision.interpretation,
            "alternatives": list(outcome.decision.alternatives),
            "artifact_refs": [item.as_dict() for item in outcome.decision.artifact_refs],
            "diagnostics": dict(outcome.decision.diagnostics),
        })
        self._apply_routes()
        self._refresh_states()

    def record_retry(self, experiment_id: str, retry: RetryDirective) -> None:
        record = self.records[experiment_id]
        if record.state is not ExecutionState.RUNNING:
            raise RuntimeError(f"{experiment_id} is not running")
        record.retry_history.append(retry)
        record.state = ExecutionState.RETRY_REQUIRED
        record.error = retry.reason
        self._append("retry_required", {
            "experiment_id": experiment_id,
            "reason": retry.reason,
            "adjustments": dict(retry.adjustments),
        })

    def approve_retry(self, experiment_id: str, *, note: str) -> None:
        record = self.records[experiment_id]
        if record.state is not ExecutionState.RETRY_REQUIRED:
            raise RuntimeError(f"{experiment_id} does not require a retry")
        if not note.strip():
            raise ValueError("retry approval note is required")
        record.state = ExecutionState.READY
        self._append("approve_retry", {"experiment_id": experiment_id, "note": note})

    def run_one(self, experiment_id: str, runner: ExperimentRunner) -> ExecutionState:
        """Execute one ready node.

        An unexpected exception becomes a transparent retry directive.  It
        cannot be misreported as ``unresolved`` and cannot hard-fail the
        scientific program.
        """
        context = self.begin(experiment_id)
        try:
            result = runner(context)
        except Exception as exc:  # intentionally isolated from scientific status
            self.record_retry(experiment_id, RetryDirective(
                reason=f"{type(exc).__name__}: {exc}",
                adjustments={"action": "inspect_and_retry"},
            ))
            return self.records[experiment_id].state
        if isinstance(result, RetryDirective):
            self.record_retry(experiment_id, result)
        elif isinstance(result, ScientificDecision):
            self.record_outcome(experiment_id, ExperimentOutcome(result))
        elif isinstance(result, ExperimentOutcome):
            self.record_outcome(experiment_id, result)
        else:
            self.record_retry(experiment_id, RetryDirective(
                reason=f"runner returned unsupported type {type(result).__name__}",
                adjustments={"action": "fix_runner_contract"},
            ))
        return self.records[experiment_id].state

    def run_ready(
        self, runners: Mapping[str, ExperimentRunner], *, max_nodes: int | None = None
    ) -> tuple[str, ...]:
        """Run ready nodes until quiescence, retry, or ``max_nodes``.

        Newly activated branches are picked up in the same call.  Missing
        runners simply leave nodes ready for an external/HPC implementation.
        """
        executed: list[str] = []
        while max_nodes is None or len(executed) < max_nodes:
            ready = [
                key for key, value in self.records.items()
                if value.state is ExecutionState.READY and key in runners
            ]
            if not ready:
                break
            # Finish the preregistered, independently estimable root panels
            # before following any result-dependent refinement.  Otherwise a
            # lexicographically early branch (for example
            # ``amplifier_replication``) can run before the remaining root
            # panels, which makes a partial HPC invocation depend on names
            # rather than on the scientific design.
            phase_order = {name: index for index, name in enumerate(EXP3_PHASES)}
            experiment_id = min(
                ready,
                key=lambda name: (
                    not self.records[name].spec.initially_active,
                    phase_order[self.records[name].spec.phase],
                    name,
                ),
            )
            self.run_one(experiment_id, runners[experiment_id])
            executed.append(experiment_id)
        return tuple(executed)

    # -- Reporting --------------------------------------------------------------
    def completed_decisions(self) -> dict[str, ScientificDecision]:
        return {
            key: record.decision for key, record in self.records.items()
            if record.state is ExecutionState.COMPLETED and record.decision is not None
        }

    def ready_experiments(self) -> tuple[str, ...]:
        return tuple(sorted(
            key for key, record in self.records.items()
            if record.state is ExecutionState.READY
        ))

    def report(self) -> dict[str, Any]:
        counts = {
            state.value: sum(r.state is state for r in self.records.values())
            for state in ExecutionState
        }
        return {
            "run_id": self.run_id,
            "fixed_exp2_foundation_claims": list(self.foundation_claims),
            "states": counts,
            "ready": list(self.ready_experiments()),
            "decisions": {
                key: {
                    "status": value.status.value,
                    "conclusion_code": value.conclusion_code,
                    "interpretation": value.interpretation,
                }
                for key, value in self.completed_decisions().items()
            },
            "hard_failures": {
                key: [item.value for item in record.invalidities]
                for key, record in self.records.items()
                if record.state is ExecutionState.HARD_FAILED
            },
            "blocked_by_invalidity": sorted(
                key for key, record in self.records.items()
                if record.state is ExecutionState.BLOCKED_INVALID
            ),
            "fired_rules": sorted(self.fired_rules),
            "ledger_head_sha256": self.ledger_head_sha256,
        }

    def _snapshot(self) -> dict[str, Any]:
        def decision_payload(value: ScientificDecision | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "status": value.status.value,
                "conclusion_code": value.conclusion_code,
                "interpretation": value.interpretation,
                "diagnostics": dict(value.diagnostics),
                "alternatives": list(value.alternatives),
                "artifact_refs": [item.as_dict() for item in value.artifact_refs],
            }

        def validity_payload(value: ValidityReport | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "measurement_valid": value.measurement_valid,
                "provenance_valid": value.provenance_valid,
                "leakage_free": value.leakage_free,
                "details": dict(value.details),
            }

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "fixed_exp2_foundation_claims": list(self.foundation_claims),
            "started": self.started,
            "records": {
                key: {
                    "spec": {
                        "experiment_id": record.spec.experiment_id,
                        "purpose": record.spec.purpose,
                        "phase": record.spec.phase,
                        "depends_on": list(record.spec.depends_on),
                        "initially_active": record.spec.initially_active,
                        "tags": list(record.spec.tags),
                    },
                    "state": record.state.value,
                    "activated_by": list(record.activated_by),
                    "decision": decision_payload(record.decision),
                    "validity": validity_payload(record.validity),
                    "retry_history": [
                        {"reason": item.reason, "adjustments": dict(item.adjustments)}
                        for item in record.retry_history
                    ],
                    "invalidities": [item.value for item in record.invalidities],
                    "error": record.error,
                }
                for key, record in self.records.items()
            },
            "rules": {
                key: {
                    "rule_id": rule.rule_id,
                    "when": {
                        source: {
                            "statuses": sorted(item.value for item in matcher.statuses),
                            "conclusion_codes": sorted(matcher.conclusion_codes),
                        }
                        for source, matcher in rule.when.items()
                    },
                    "activate": list(rule.activate),
                    "rationale": rule.rationale,
                }
                for key, rule in self.rules.items()
            },
            "fired_rules": sorted(self.fired_rules),
            "events": self.events,
            "ledger_head_sha256": self.ledger_head_sha256,
        }

    def save(self, path: str | Path) -> Path:
        """Atomically persist all graph state so an HPC run can be resumed."""
        self.verify_ledger()
        for decision in self.completed_decisions().values():
            for artifact in decision.artifact_refs:
                ArtifactStore.verify(artifact)
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._snapshot()
        _json_safe(payload, path="router")
        payload["snapshot_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveRouter":
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        observed_snapshot_hash = raw.pop("snapshot_sha256", "")
        expected_snapshot_hash = hashlib.sha256(_canonical(raw)).hexdigest()
        if observed_snapshot_hash != expected_snapshot_hash:
            raise RuntimeError("adaptive-router snapshot was modified")
        if raw.get("schema_version") != 1:
            raise RuntimeError("unsupported adaptive-router schema")
        if tuple(raw.get("fixed_exp2_foundation_claims", ())) != FIXED_EXP2_FOUNDATION_CLAIMS:
            raise RuntimeError("fixed EXP2 foundation contract changed")
        router = cls(str(raw["run_id"]))
        for key, item in raw["records"].items():
            spec_raw = item["spec"]
            spec = ExperimentSpec(
                experiment_id=spec_raw["experiment_id"],
                purpose=spec_raw["purpose"],
                phase=spec_raw["phase"],
                depends_on=tuple(spec_raw["depends_on"]),
                initially_active=bool(spec_raw["initially_active"]),
                tags=tuple(spec_raw["tags"]),
            )
            decision_raw = item.get("decision")
            decision = None if decision_raw is None else ScientificDecision(
                status=EvidenceStatus(decision_raw["status"]),
                conclusion_code=decision_raw["conclusion_code"],
                interpretation=decision_raw["interpretation"],
                diagnostics=decision_raw.get("diagnostics", {}),
                alternatives=tuple(decision_raw.get("alternatives", ())),
                artifact_refs=tuple(
                    ArtifactRef(**value)
                    for value in decision_raw.get("artifact_refs", ())
                ),
            )
            validity_raw = item.get("validity")
            validity = None if validity_raw is None else ValidityReport(**validity_raw)
            record = ExperimentRecord(
                spec=spec,
                state=ExecutionState(item["state"]),
                activated_by=list(item.get("activated_by", ())),
                decision=decision,
                validity=validity,
                retry_history=[RetryDirective(**value) for value in item.get("retry_history", ())],
                invalidities=tuple(InvalidityKind(value) for value in item.get("invalidities", ())),
                error=item.get("error"),
            )
            if key != spec.experiment_id:
                raise RuntimeError("record key does not match experiment_id")
            router.records[key] = record
        for key, item in raw["rules"].items():
            rule = RouteRule(
                rule_id=item["rule_id"],
                when={
                    source_name: OutcomeMatcher(
                        statuses=frozenset(EvidenceStatus(value) for value in matcher["statuses"]),
                        conclusion_codes=frozenset(matcher.get("conclusion_codes", ())),
                    )
                    for source_name, matcher in item["when"].items()
                },
                activate=tuple(item["activate"]),
                rationale=item["rationale"],
            )
            if key != rule.rule_id:
                raise RuntimeError("rule key does not match rule_id")
            router.rules[key] = rule
        unknown_refs = {
            value
            for record in router.records.values()
            for value in record.spec.depends_on
            if value not in router.records
        }
        for rule in router.rules.values():
            unknown_refs.update((set(rule.when) | set(rule.activate)) - set(router.records))
        if unknown_refs:
            raise RuntimeError(f"snapshot has unknown graph references: {sorted(unknown_refs)}")
        router.fired_rules = set(raw.get("fired_rules", ()))
        if not router.fired_rules <= set(router.rules):
            raise RuntimeError("snapshot lists an unknown fired rule")
        router.events = list(raw.get("events", ()))
        router.ledger_head_sha256 = str(raw.get("ledger_head_sha256", ""))
        router.started = bool(raw.get("started", False))
        router.verify_ledger()
        return router


def _match(
    *statuses: EvidenceStatus, codes: Iterable[str] = ()
) -> OutcomeMatcher:
    return OutcomeMatcher(
        statuses=frozenset(statuses) if statuses else ALL_EVIDENCE_STATUSES,
        conclusion_codes=frozenset(codes),
    )


def build_exp3_research_router(run_id: str = "exp3") -> AdaptiveRouter:
    """Construct the default non-blocking EXP3 decision graph.

    The graph embodies the paper's three Phase-1 extension questions and its
    mandatory scale/content/carrier factorization while avoiding a
    one-way success gate.  Every valid outcome has an explicit scientific next
    step; common integration/application analyses continue after *any* valid
    answer.  Conclusion codes provide finer routing than a bare pass/fail.
    """
    router = AdaptiveRouter(run_id)

    specs = (
        ExperimentSpec(
            "m28_preconditioning",
            "Resolve how incoming m28 prepares the already-established b28 content write.",
            initially_active=True, tags=("foundation_use", "phase1"),
        ),
        ExperimentSpec(
            "b29_decomposition",
            "Fix U28 and compare removal, only, rescue, and independent g29-perp effects.",
            initially_active=True, tags=("mandatory", "0-1"),
        ),
        ExperimentSpec(
            "scale_plateau",
            "Measure dense q=alpha/alpha* radial and equal-norm angular dose curves.",
            initially_active=True, tags=("mandatory", "0-2"),
        ),
        ExperimentSpec(
            "m28_preconditioner_mapping",
            "Map the upstream features that causally prepare the fixed b28 writer.",
            depends_on=("m28_preconditioning",),
        ),
        ExperimentSpec(
            "m28_alternative_inputs",
            "Identify conditional or redundant inputs to b28 without revisiting its writer role.",
            depends_on=("m28_preconditioning",),
        ),
        ExperimentSpec(
            "m28_conditional_null",
            "Bound the m28-specific contribution while retaining the fixed b28 writer conclusion.",
            depends_on=("m28_preconditioning",),
        ),
        ExperimentSpec(
            "m28_identifiability",
            "Refine upstream perturbations or power for the m28-to-b28 preparation question.",
            depends_on=("m28_preconditioning",),
        ),
        ExperimentSpec(
            "amplifier_replication", "Replicate parallel necessity/sufficiency and plateau.",
            depends_on=("b29_decomposition",),
        ),
        ExperimentSpec(
            "second_writer_mapping", "Map independent g29-perp content and motif effects.",
            depends_on=("b29_decomposition",),
        ),
        ExperimentSpec(
            "b29_null_or_equivalence", "Bound b29's smallest effect under equivalence margins.",
            depends_on=("b29_decomposition",),
        ),
        ExperimentSpec(
            "b29_identifiability", "Increase paired clusters or refine U28 stability.",
            depends_on=("b29_decomposition",),
        ),
        ExperimentSpec(
            "normalized_scale_transfer", "Test a shared normalized transition on held-out blocks/models.",
            depends_on=("scale_plateau",),
        ),
        ExperimentSpec(
            "site_specific_scale", "Model heterogeneous block/model thresholds without pooling them.",
            depends_on=("scale_plateau",),
        ),
        ExperimentSpec(
            "nonlinear_scale_mechanism", "Test curvature, saturation, and host-update interactions.",
            depends_on=("scale_plateau",),
        ),
        ExperimentSpec(
            "scale_identifiability", "Refine doses/repeats around uncertain transition intervals.",
            depends_on=("scale_plateau",),
        ),
        ExperimentSpec(
            "m28_target_spec",
            "Seal the bounded upstream trigger specification selected by the m28 branch.",
            depends_on=("m28_preconditioning",),
        ),
        ExperimentSpec(
            "b29_target_spec",
            "Seal the bounded b29 target selected by amplifier/writer/null refinement.",
            depends_on=("b29_decomposition",),
        ),
        ExperimentSpec(
            "scale_target_spec",
            "Seal the shared, site-specific, nonlinear, null, or bounded scale model.",
            depends_on=("scale_plateau",),
        ),
        ExperimentSpec(
            "exp3_mechanism_synthesis",
            "Synthesize m28 preparation, b29 subtype, and scale results on the fixed EXP2 stage foundation.",
            phase="phase1_roles_plateau",
            depends_on=("m28_target_spec", "b29_target_spec", "scale_target_spec"),
        ),
        ExperimentSpec(
            "scale_content_carrier_cube",
            "Orthogonalize g28 scale, g28 content direction, and x31 carrier in a preregistered causal cube.",
            phase="phase2_causal_factorization",
            depends_on=("exp3_mechanism_synthesis",),
        ),
        ExperimentSpec(
            "bidirectional_direction_transfer",
            "Transfer g28 directions both ways against norm, wrong-layer, wrong-pair, and matched-random controls.",
            phase="phase2_causal_factorization",
            depends_on=("exp3_mechanism_synthesis",),
        ),
        ExperimentSpec(
            "cube_b30_mediation",
            "Test whether the cube's content effect is blocked and prospectively rescued at m30.",
            phase="phase2_causal_factorization",
            depends_on=("scale_content_carrier_cube",),
        ),
        ExperimentSpec(
            "causal_factorization_synthesis",
            "Seal the operational scale/carrier/content transport model used by every downstream application.",
            phase="phase2_causal_factorization",
            depends_on=("scale_content_carrier_cube", "cube_b30_mediation",
                        "bidirectional_direction_transfer"),
        ),
        ExperimentSpec(
            "common_reverse_trace", "Trace a shared EXP3 b29/scale target back to sequence.",
            phase="phase3_reverse_trace",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "branch_specific_reverse_trace", "Trace branch-specific b29/scale targets separately.",
            phase="phase3_reverse_trace",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "reverse_trace_identifiability", "Disambiguate unstable or non-identifiable reverse paths.",
            phase="phase3_reverse_trace",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "motif_variant_replay", "Replay reverse-traced motifs and variants with block/rescue controls.",
            phase="phase4_motif_validation",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "learned_transport", "Fit predictive transition models to the selected raw delta target.",
            phase="phase5_learned_transport",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "task_heterogeneity", "Explain why benchmarks differ using frozen mechanism fingerprints.",
            phase="phase6_applications",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "exp1_extension",
            "Apply the fixed EXP2 ontology plus the resolved EXP3 overlay to strengthen EXP1 across tasks.",
            phase="phase6_applications",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "mechanism_application", "Run diagnosis, selective repair, calibration, and steering.",
            phase="phase6_applications",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "cross_chromosome_generalization", "Test frozen conclusions on chr17/chr22 and held-out checkpoints.",
            phase="phase7_generalization",
            depends_on=("causal_factorization_synthesis",),
        ),
        ExperimentSpec(
            "feature_model_on_causal_delta", "Fit SAE/transcoder on the supported or mixed causal delta.",
            phase="phase8_optional_features",
            depends_on=("learned_transport",),
        ),
        ExperimentSpec(
            "feature_model_on_branch_delta", "Fit interpreters to branch-specific b29/scale delta targets.",
            phase="phase8_optional_features",
            depends_on=("learned_transport",),
        ),
        ExperimentSpec(
            "feature_target_refinement", "Establish an identifiable transition target before interpretation.",
            phase="phase8_optional_features",
            depends_on=("learned_transport",),
        ),
    )
    for spec in specs:
        router.add_experiment(spec)

    any_valid = _match()
    rules = (
        RouteRule(
            "m28_preconditioner_branch", {"m28_preconditioning": _match(
                EvidenceStatus.SUPPORTED, codes=("m28_preconditions_writer",)
            )}, ("m28_preconditioner_mapping",),
            "A causal preparation effect is mapped upstream and reused for applications.",
        ),
        RouteRule(
            "m28_alternative_branch", {"m28_preconditioning": _match(
                EvidenceStatus.MIXED, EvidenceStatus.REFUTED
            )}, ("m28_alternative_inputs",),
            "Mixed or contrary m28 evidence redirects to conditional/redundant inputs without revisiting b28's fixed role.",
        ),
        RouteRule(
            "m28_equivalence_branch", {"m28_preconditioning": _match(
                EvidenceStatus.EQUIVALENT
            )}, ("m28_conditional_null",),
            "A bounded m28 null is recorded as a local input-path conclusion.",
        ),
        RouteRule(
            "m28_unresolved_branch", {"m28_preconditioning": _match(
                EvidenceStatus.UNRESOLVED
            )}, ("m28_identifiability",),
            "Uncertain preconditioning evidence triggers only local identification work.",
        ),
        RouteRule(
            "b29_amplifier_branch", {"b29_decomposition": _match(
                EvidenceStatus.SUPPORTED, codes=("b29_amplifier",)
            )}, ("amplifier_replication",),
            "Parallel necessity/sufficiency supports focused amplifier replication.",
        ),
        RouteRule(
            "b29_writer_branch", {"b29_decomposition": _match(
                EvidenceStatus.SUPPORTED, EvidenceStatus.MIXED,
                codes=("b29_second_writer", "b29_editor", "b29_mixed_role"),
            )}, ("second_writer_mapping",),
            "Independent perpendicular effects require mapping, not termination.",
        ),
        RouteRule(
            "b29_refuted_branch", {"b29_decomposition": _match(
                EvidenceStatus.REFUTED
            )}, ("second_writer_mapping",),
            "Refutation of amplification triggers a search for writer/editor/bypass effects.",
        ),
        RouteRule(
            "b29_equivalence_branch", {"b29_decomposition": _match(
                EvidenceStatus.EQUIVALENT
            )}, ("b29_null_or_equivalence",),
            "A bounded null effect becomes an estimand for an equivalence analysis.",
        ),
        RouteRule(
            "b29_unresolved_branch", {"b29_decomposition": _match(
                EvidenceStatus.UNRESOLVED
            )}, ("b29_identifiability",),
            "Uncertainty triggers an identifiable power/subspace refinement.",
        ),
        RouteRule(
            "scale_shared_branch", {"scale_plateau": _match(
                EvidenceStatus.SUPPORTED, codes=("shared_normalized_transition",)
            )}, ("normalized_scale_transfer",),
            "Curve collapse supports testing the shared transition out of sample.",
        ),
        RouteRule(
            "scale_heterogeneous_branch", {"scale_plateau": _match(
                EvidenceStatus.MIXED, codes=("site_specific_transition", "model_specific_transition")
            )}, ("site_specific_scale",),
            "Heterogeneous thresholds should be explained instead of averaged away.",
        ),
        RouteRule(
            "scale_refuted_branch", {"scale_plateau": _match(
                EvidenceStatus.REFUTED
            )}, ("nonlinear_scale_mechanism",),
            "Failure of the normalized plateau motivates nonlinear host-update tests.",
        ),
        RouteRule(
            "scale_unresolved_branch", {"scale_plateau": _match(
                EvidenceStatus.UNRESOLVED
            )}, ("scale_identifiability",),
            "An imprecise transition triggers denser local dose/repeat allocation.",
        ),
        RouteRule(
            "scale_equivalence_branch", {"scale_plateau": _match(
                EvidenceStatus.EQUIVALENT
            )}, ("site_specific_scale",),
            "Equivalent radial/angular curves shift the target to site-specific or null scale effects.",
        ),
        RouteRule(
            "m28_map_to_target", {"m28_preconditioner_mapping": any_valid},
            ("m28_target_spec",), "Map result defines the bounded m28 target.",
        ),
        RouteRule(
            "m28_alternative_to_target", {"m28_alternative_inputs": any_valid},
            ("m28_target_spec",), "Alternative inputs define the bounded m28 target.",
        ),
        RouteRule(
            "m28_null_to_target", {"m28_conditional_null": any_valid},
            ("m28_target_spec",), "A bounded null is itself a usable target specification.",
        ),
        RouteRule(
            "m28_identifiability_to_target", {"m28_identifiability": any_valid},
            ("m28_target_spec",), "Identifiability work produces a bounded target or explicit null.",
        ),
        RouteRule(
            "b29_amplifier_to_target", {"amplifier_replication": any_valid},
            ("b29_target_spec",), "Replication result defines the amplifier target.",
        ),
        RouteRule(
            "b29_writer_to_target", {"second_writer_mapping": any_valid},
            ("b29_target_spec",), "Writer/editor mapping defines the b29 target.",
        ),
        RouteRule(
            "b29_null_to_target", {"b29_null_or_equivalence": any_valid},
            ("b29_target_spec",), "A bounded b29 null remains a usable target specification.",
        ),
        RouteRule(
            "b29_identifiability_to_target", {"b29_identifiability": any_valid},
            ("b29_target_spec",), "Identifiability work produces a bounded b29 target or null.",
        ),
        RouteRule(
            "scale_shared_to_target", {"normalized_scale_transfer": any_valid},
            ("scale_target_spec",), "Transfer result defines the shared scale model.",
        ),
        RouteRule(
            "scale_site_to_target", {"site_specific_scale": any_valid},
            ("scale_target_spec",), "Site-specific analysis defines the heterogeneous scale model.",
        ),
        RouteRule(
            "scale_nonlinear_to_target", {"nonlinear_scale_mechanism": any_valid},
            ("scale_target_spec",), "Nonlinear analysis defines the alternative scale model.",
        ),
        RouteRule(
            "scale_identifiability_to_target", {"scale_identifiability": any_valid},
            ("scale_target_spec",), "Identifiability work produces a bounded scale target or null.",
        ),
        RouteRule(
            "integrate_refined_phase1_targets", {
                "m28_target_spec": any_valid,
                "b29_target_spec": any_valid,
                "scale_target_spec": any_valid,
            }, ("exp3_mechanism_synthesis",),
            "Synthesis waits for the selected refinement branch, then uses all three bounded target specifications.",
        ),
        RouteRule(
            "factorization_panels_after_target_synthesis",
            {"exp3_mechanism_synthesis": any_valid},
            ("scale_content_carrier_cube", "bidirectional_direction_transfer"),
            "Every bounded Phase-1 target is tested by the same causal-use panels.",
        ),
        RouteRule(
            "mediation_after_any_cube_result",
            {"scale_content_carrier_cube": any_valid},
            ("cube_b30_mediation",),
            "A valid cube result defines the content contrast whose b30 mediation is tested.",
        ),
        RouteRule(
            "synthesize_complete_factorization", {
                "scale_content_carrier_cube": any_valid,
                "cube_b30_mediation": any_valid,
                "bidirectional_direction_transfer": any_valid,
            }, ("causal_factorization_synthesis",),
            "Downstream use waits for cube, m30 mediation, and reciprocal transfer decisions.",
        ),
        RouteRule(
            "common_trace_branch", {"causal_factorization_synthesis": _match(
                EvidenceStatus.SUPPORTED,
                codes=("scale_carrier_content_factorization",)
            )}, ("common_reverse_trace",),
            "A supported operational factorization is reverse-traced without re-testing the EXP2 scaffold.",
        ),
        RouteRule(
            "branch_specific_trace_branch", {"causal_factorization_synthesis": _match(
                EvidenceStatus.MIXED, EvidenceStatus.EQUIVALENT,
                EvidenceStatus.REFUTED,
                codes=("conditional_scale_carrier_content_factorization",
                       "factorization_bounded_null",
                       "alternative_transport_architecture"),
            )}, ("branch_specific_reverse_trace",),
            "A conditional, bounded, or contrary factorization is traced with branch-specific targets.",
        ),
        RouteRule(
            "uncertain_trace_branch", {"causal_factorization_synthesis": _match(
                EvidenceStatus.UNRESOLVED
            )}, ("reverse_trace_identifiability",),
            "Reverse attribution waits on a better-defined causal-use estimand, not a success gate.",
        ),
        RouteRule(
            "applications_after_any_factorization", {"causal_factorization_synthesis": any_valid},
            ("learned_transport", "task_heterogeneity", "exp1_extension",
             "mechanism_application", "cross_chromosome_generalization"),
            "Applications use the fixed EXP2 stage account plus the explicitly resolved EXP3 factorization overlay.",
        ),
        RouteRule(
            "motif_after_common_trace", {"common_reverse_trace": _match(
                EvidenceStatus.SUPPORTED, EvidenceStatus.MIXED
            )}, ("motif_variant_replay",),
            "Motif replay opens only after a reverse-trace candidate survives consensus.",
        ),
        RouteRule(
            "motif_after_branch_trace", {"branch_specific_reverse_trace": _match(
                EvidenceStatus.SUPPORTED, EvidenceStatus.MIXED
            )}, ("motif_variant_replay",),
            "A branch-specific consensus candidate can enter causal motif replay.",
        ),
        RouteRule(
            "motif_after_trace_refinement", {"reverse_trace_identifiability": _match(
                EvidenceStatus.SUPPORTED, EvidenceStatus.MIXED
            )}, ("motif_variant_replay",),
            "Successful target refinement may open causal motif replay.",
        ),
        RouteRule(
            "features_after_predicted_rescue", {"learned_transport": _match(
                EvidenceStatus.SUPPORTED, codes=("predicted_delta_rescue",)
            )}, ("feature_model_on_causal_delta",),
            "A feature model opens only after held-out predicted-delta rescue.",
        ),
        RouteRule(
            "features_after_branch_rescue", {"learned_transport": _match(
                EvidenceStatus.MIXED, codes=("branch_specific_predicted_delta",)
            )}, ("feature_model_on_branch_delta",),
            "A branch-specific rescued delta opens only the matching interpreter.",
        ),
        RouteRule(
            "features_after_transport_non_support", {"learned_transport": _match(
                EvidenceStatus.EQUIVALENT, EvidenceStatus.REFUTED,
                EvidenceStatus.UNRESOLVED
            )}, ("feature_target_refinement",),
            "Transport non-support is recorded locally; SAE stays closed until a rescued target exists.",
        ),
        RouteRule(
            "features_after_target_refinement", {"feature_target_refinement": _match(
                EvidenceStatus.SUPPORTED, codes=("predicted_delta_rescue",)
            )}, ("feature_model_on_branch_delta",),
            "A refined target opens an interpreter only after predicted-delta rescue.",
        ),
    )
    for rule in rules:
        router.add_rule(rule)
    return router


__all__ = [
    "ALL_EVIDENCE_STATUSES", "EXP3_PHASES", "FIXED_EXP2_FOUNDATION_CLAIMS",
    "AdaptiveRouter", "EvidenceStatus",
    "ExecutionContext", "ExecutionState", "ExperimentOutcome",
    "ExperimentRecord", "ExperimentRunner", "ExperimentSpec", "InvalidityKind",
    "OutcomeMatcher", "RetryDirective", "RouteRule", "ScientificDecision",
    "ValidityReport", "build_exp3_research_router",
]
