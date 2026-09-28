"""Canonical conversion and schema checks for EXP3 scientific decisions.

Router branch codes are part of the preregistered estimand.  Providers may not
invent a status/code pair after seeing locked results.  Mechanism verdicts
therefore expose deterministic ``status`` and ``conclusion_code`` fields, and
production orchestration validates every branch-defining decision here.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .adaptive import (
    EvidenceStatus, ExperimentOutcome, ScientificDecision, ValidityReport,
)
from .config import Exp3Config
from .artifacts import ArtifactRef


DECISION_SCHEMA: dict[str, dict[EvidenceStatus, frozenset[str]]] = {
    "m28_preconditioning": {
        EvidenceStatus.SUPPORTED: frozenset({"m28_preconditions_writer"}),
        EvidenceStatus.EQUIVALENT: frozenset({"m28_bounded_local_null"}),
        EvidenceStatus.MIXED: frozenset({"m28_redundant_inputs"}),
        EvidenceStatus.REFUTED: frozenset({"m28_alternative_input_path"}),
        EvidenceStatus.UNRESOLVED: frozenset({"m28_unresolved"}),
    },
    "b29_decomposition": {
        EvidenceStatus.SUPPORTED: frozenset({
            "b29_amplifier", "b29_second_writer", "b29_editor"}),
        EvidenceStatus.EQUIVALENT: frozenset({"b29_bounded_null"}),
        EvidenceStatus.MIXED: frozenset({"b29_mixed_role"}),
        EvidenceStatus.REFUTED: frozenset({"amplifier_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"b29_unresolved"}),
    },
    "scale_plateau": {
        EvidenceStatus.SUPPORTED: frozenset({"shared_normalized_transition"}),
        EvidenceStatus.EQUIVALENT: frozenset({"no_common_scale_effect"}),
        EvidenceStatus.MIXED: frozenset({
            "site_specific_transition", "model_specific_transition"}),
        EvidenceStatus.REFUTED: frozenset({"no_shared_plateau"}),
        EvidenceStatus.UNRESOLVED: frozenset({
            "wide_interval", "scale_measurement_incomplete"}),
    },
    "exp3_mechanism_synthesis": {
        EvidenceStatus.SUPPORTED: frozenset({"common_b29_scale_target"}),
        EvidenceStatus.MIXED: frozenset({
            "branch_specific_b29_scale", "mixed_b29_scale"}),
        EvidenceStatus.EQUIVALENT: frozenset({"no_common_b29_scale"}),
        EvidenceStatus.REFUTED: frozenset({"no_common_b29_scale"}),
        EvidenceStatus.UNRESOLVED: frozenset({"exp3_target_unresolved"}),
    },
    "scale_content_carrier_cube": {
        EvidenceStatus.SUPPORTED: frozenset({"axes_causally_separable"}),
        EvidenceStatus.MIXED: frozenset({"axes_conditionally_separable"}),
        EvidenceStatus.EQUIVALENT: frozenset({"cube_axis_effects_bounded_null"}),
        EvidenceStatus.REFUTED: frozenset({"cube_factorization_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"cube_factorization_unresolved"}),
    },
    "cube_b30_mediation": {
        EvidenceStatus.SUPPORTED: frozenset({"b30_mediates_content"}),
        EvidenceStatus.MIXED: frozenset({"partial_b30_mediation"}),
        EvidenceStatus.EQUIVALENT: frozenset({"b30_mediation_bounded_null"}),
        EvidenceStatus.REFUTED: frozenset({"b30_mediation_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"b30_mediation_unresolved"}),
    },
    "bidirectional_direction_transfer": {
        EvidenceStatus.SUPPORTED: frozenset({"specific_bidirectional_transfer"}),
        EvidenceStatus.MIXED: frozenset({"partial_direction_transfer"}),
        EvidenceStatus.EQUIVALENT: frozenset({"direction_transfer_bounded_null"}),
        EvidenceStatus.REFUTED: frozenset({"direction_transfer_nonspecific"}),
        EvidenceStatus.UNRESOLVED: frozenset({"direction_transfer_unresolved"}),
    },
    "causal_factorization_synthesis": {
        EvidenceStatus.SUPPORTED: frozenset({"scale_carrier_content_factorization"}),
        EvidenceStatus.MIXED: frozenset({
            "conditional_scale_carrier_content_factorization"}),
        EvidenceStatus.EQUIVALENT: frozenset({"factorization_bounded_null"}),
        EvidenceStatus.REFUTED: frozenset({"alternative_transport_architecture"}),
        EvidenceStatus.UNRESOLVED: frozenset({"factorization_unresolved"}),
    },
    "common_reverse_trace": {
        EvidenceStatus.SUPPORTED: frozenset({"reverse_trace_consensus"}),
        EvidenceStatus.MIXED: frozenset({"partial_reverse_trace_consensus"}),
        EvidenceStatus.EQUIVALENT: frozenset({"no_reverse_trace_signal"}),
        EvidenceStatus.REFUTED: frozenset({"reverse_trace_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"reverse_trace_unresolved"}),
    },
    "branch_specific_reverse_trace": {
        EvidenceStatus.SUPPORTED: frozenset({"branch_reverse_trace_consensus"}),
        EvidenceStatus.MIXED: frozenset({"partial_branch_trace_consensus"}),
        EvidenceStatus.EQUIVALENT: frozenset({"no_branch_trace_signal"}),
        EvidenceStatus.REFUTED: frozenset({"branch_trace_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"branch_trace_unresolved"}),
    },
    "reverse_trace_identifiability": {
        EvidenceStatus.SUPPORTED: frozenset({"reverse_trace_consensus"}),
        EvidenceStatus.MIXED: frozenset({"partial_reverse_trace_consensus"}),
        EvidenceStatus.EQUIVALENT: frozenset({"no_reverse_trace_signal"}),
        EvidenceStatus.REFUTED: frozenset({"reverse_trace_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"reverse_trace_unresolved"}),
    },
    "learned_transport": {
        EvidenceStatus.SUPPORTED: frozenset({"predicted_delta_rescue"}),
        EvidenceStatus.MIXED: frozenset({"branch_specific_predicted_delta"}),
        EvidenceStatus.EQUIVALENT: frozenset({"bounded_null_transport"}),
        EvidenceStatus.REFUTED: frozenset({"transport_model_refuted"}),
        EvidenceStatus.UNRESOLVED: frozenset({"transport_unresolved"}),
    },
}


def validate_decision_schema(experiment_id: str,
                             decision: ScientificDecision) -> None:
    schema = DECISION_SCHEMA.get(experiment_id)
    if schema is None:
        return
    allowed = schema.get(decision.status, frozenset())
    if decision.conclusion_code not in allowed:
        raise ValueError(
            f"{experiment_id}: conclusion {decision.status.value}/"
            f"{decision.conclusion_code} is outside the preregistered schema; "
            f"allowed={sorted(allowed)}")


def outcome_from_verdict(
    experiment_id: str,
    verdict: Mapping[str, Any],
    *,
    validity: ValidityReport,
    config: Exp3Config,
    runtime_provenance: Mapping[str, str],
    interpretation: str | None = None,
    artifact_refs: Sequence[ArtifactRef] = (),
) -> ExperimentOutcome:
    """Convert a deterministic verdict dictionary into an auditable outcome."""
    if "status" not in verdict or "conclusion_code" not in verdict:
        raise ValueError(
            "verdict lacks canonical status/conclusion_code; do not choose a "
            "router branch manually")
    status = EvidenceStatus(str(verdict["status"]))
    code = str(verdict["conclusion_code"])
    text = interpretation or str(verdict.get("claim", code))
    diagnostics = {
        key: value for key, value in verdict.items()
        if key not in {"status", "conclusion_code", "claim"}
    }
    diagnostics["config_sha256"] = config.config_sha256
    diagnostics["preregistered_margins"] = asdict(config.margins)
    diagnostics["runtime_provenance"] = dict(runtime_provenance)
    decision = ScientificDecision(
        status=status,
        conclusion_code=code,
        interpretation=text,
        diagnostics=diagnostics,
        artifact_refs=tuple(artifact_refs),
    )
    validate_decision_schema(experiment_id, decision)
    return ExperimentOutcome(decision=decision, validity=validity)


__all__ = [
    "DECISION_SCHEMA", "outcome_from_verdict", "validate_decision_schema",
]
