"""Checkpoint-free integration smoke test for the standalone EXP3 program.

No EXP2 experiment is imported or executed.  The dry-run creates a tiny
hash-sealed stand-in for already-completed EXP2 artifacts, asserts all five
accepted foundation claims, and executes two deterministic EXP3-only paths:

1. b29 amplifier + shared normalized scale transition;
2. amplifier refuted + site-specific scale transition.

The numbers are wiring fixtures and carry no biological interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from .adaptive import (
    EvidenceStatus, ExperimentOutcome, ScientificDecision, ValidityReport,
)
from .inputs import (
    REQUIRED_EXP2_FOUNDATIONS,
    FrozenInput,
    create_handoff,
    sha256_file,
)
from .run import Exp3RunContext, execute_program


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ScenarioProvider:
    def __init__(self, scenario: str):
        self.scenario = scenario

    def __call__(self, context: Exp3RunContext) -> ExperimentOutcome:
        experiment_id = context.experiment_id
        if self.scenario == "common":
            planned = {
                "m28_preconditioning": (
                    EvidenceStatus.SUPPORTED, "m28_preconditions_writer",
                    "The fixture selects upstream mapping for the fixed b28 writer.",
                ),
                "b29_decomposition": (
                    EvidenceStatus.SUPPORTED, "b29_amplifier",
                    "The fixture selects the amplifier-compatible EXP3 route.",
                ),
                "scale_plateau": (
                    EvidenceStatus.SUPPORTED, "shared_normalized_transition",
                    "The fixture selects the shared normalized transition route.",
                ),
                "exp3_mechanism_synthesis": (
                    EvidenceStatus.SUPPORTED, "common_b29_scale_target",
                    "The fixture defines a common reverse-tracing target.",
                ),
                "scale_content_carrier_cube": (
                    EvidenceStatus.SUPPORTED, "axes_causally_separable",
                    "The fixture resolves all three preregistered causal axes.",
                ),
                "cube_b30_mediation": (
                    EvidenceStatus.SUPPORTED, "b30_mediates_content",
                    "The fixture sends the content contrast through b30.",
                ),
                "bidirectional_direction_transfer": (
                    EvidenceStatus.SUPPORTED, "specific_bidirectional_transfer",
                    "The fixture direction beats all matched control families.",
                ),
                "causal_factorization_synthesis": (
                    EvidenceStatus.SUPPORTED,
                    "scale_carrier_content_factorization",
                    "The fixture seals the operational causal factorization.",
                ),
                "common_reverse_trace": (
                    EvidenceStatus.SUPPORTED, "reverse_trace_consensus",
                    "Two independent attribution methods agree on a fixture candidate.",
                ),
                "learned_transport": (
                    EvidenceStatus.SUPPORTED, "predicted_delta_rescue",
                    "A held-out fixture delta rescue opens optional interpretation.",
                ),
            }
        elif self.scenario == "alternative":
            planned = {
                "m28_preconditioning": (
                    EvidenceStatus.MIXED, "m28_redundant_inputs",
                    "A mixed local result selects conditional upstream mapping.",
                ),
                "b29_decomposition": (
                    EvidenceStatus.REFUTED, "amplifier_refuted",
                    "A valid negative selects writer/editor mapping.",
                ),
                "scale_plateau": (
                    EvidenceStatus.MIXED, "site_specific_transition",
                    "Heterogeneous transitions select a site-specific model.",
                ),
                "exp3_mechanism_synthesis": (
                    EvidenceStatus.REFUTED, "no_common_b29_scale",
                    "No common overlay selects branch-specific reverse tracing.",
                ),
                "scale_content_carrier_cube": (
                    EvidenceStatus.MIXED, "axes_conditionally_separable",
                    "The fixture resolves a conditional causal cube.",
                ),
                "cube_b30_mediation": (
                    EvidenceStatus.MIXED, "partial_b30_mediation",
                    "The fixture supports only partial b30 mediation.",
                ),
                "bidirectional_direction_transfer": (
                    EvidenceStatus.MIXED, "partial_direction_transfer",
                    "The fixture supports branch-specific direction transfer.",
                ),
                "causal_factorization_synthesis": (
                    EvidenceStatus.MIXED,
                    "conditional_scale_carrier_content_factorization",
                    "The fixture seals a conditional factorization overlay.",
                ),
                "branch_specific_reverse_trace": (
                    EvidenceStatus.SUPPORTED, "branch_reverse_trace_consensus",
                    "A branch-specific fixture candidate passes consensus.",
                ),
                "learned_transport": (
                    EvidenceStatus.MIXED, "branch_specific_predicted_delta",
                    "Only the branch-specific fixture delta is predictively rescued.",
                ),
            }
        else:  # pragma: no cover - construction is internal
            raise ValueError(self.scenario)

        status, code, interpretation = planned.get(
            experiment_id,
            (
                EvidenceStatus.EQUIVALENT,
                "software_fixture_completed",
                "The EXP3-only software fixture completed this routed analysis.",
            ),
        )
        diagnostics = {
            "software_fixture": True,
            "attempt": context.execution.attempts,
            "config_sha256": context.config.config_sha256,
            "runtime_provenance": dict(context.expected_runtime_provenance),
        }
        if experiment_id in {"b29_decomposition", "scale_plateau"}:
            diagnostics["alpha_doses"] = list(context.config.alpha_doses)
        if experiment_id == "scale_plateau":
            diagnostics["required_models"] = list(
                context.config.comparison_models)
            diagnostics["required_sites"] = {
                model: list(sites)
                for model, sites in context.config.required_scale_sites
            }
        return ExperimentOutcome(
            decision=ScientificDecision(
                status=status,
                conclusion_code=code,
                interpretation=interpretation,
                diagnostics=diagnostics,
            ),
            validity=ValidityReport(
                measurement_valid=True,
                provenance_valid=True,
                leakage_free=True,
                details={"software_fixture": True},
            ),
        )


def _create_fixture_handoff(root: Path) -> Path:
    tensors = {
        "u28": (
            torch.eye(4, 2, dtype=torch.float64),
            "b28_causal_subspace"),
        "carrier_axis_x31": (
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            "x31_carrier_axis"),
        "b30_subspace": (
            torch.eye(4, 2, dtype=torch.float64),
            "b30_mediation_subspace"),
        "reference_directions": (
            torch.eye(4, 2, dtype=torch.float64),
            "late_stack_reference_directions"),
    }
    frozen_inputs = {}
    for name, (tensor, semantic_role) in tensors.items():
        artifact = root / f"frozen_{name}.pt"
        torch.save(tensor, artifact)
        frozen_inputs[name] = FrozenInput(
            name=name, path=str(artifact), sha256=sha256_file(artifact),
            kind="tensor", source_split="toy-chr22-discovery",
            source_role="discovery",
            metadata={
                "software_fixture": True,
                "semantic_role": semantic_role,
            },
        )
    handoff = create_handoff(
        handoff_id="exp3-dryrun-handoff",
        exp2_run_id="accepted-exp2-fixture",
        checkpoint="fixture-only",
        checkpoint_sha256=_digest("checkpoint"),
        architecture_sha256=_digest("architecture"),
        code_sha256=_digest("exp2-code"),
        layer_roles={"writer": 28, "b29_candidate": 29, "reencoder": 30,
                     "readout_gate": 31},
        accepted_exp2_foundations={name: True for name in REQUIRED_EXP2_FOUNDATIONS},
        inputs=frozen_inputs,
        scientific_margins={
            "effect": 0.05, "equivalence": 0.02,
            "rescue": 0.05, "specificity": 0.05,
        },
        notes="Software-only handoff; never use as scientific evidence.",
    )
    path = root / "handoff.json"
    handoff.save(path)
    return path


def run_dryrun(output: str | Path) -> dict[str, Any]:
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp3-dryrun-") as directory:
        root = Path(directory)
        handoff_path = _create_fixture_handoff(root)
        scenarios: dict[str, Any] = {}
        for name in ("common", "alternative"):
            report_path, report, router = execute_program(
                handoff_path=handoff_path,
                output_dir=root / "runs",
                run_id=f"dryrun-{name}",
                callback=_ScenarioProvider(name),
                software_fixture=True,
            )
            active = {
                experiment_id: record.state.value
                for experiment_id, record in router.records.items()
                if record.activated_by
            }
            phases = sorted({
                record.spec.phase
                for record in router.records.values()
                if record.activated_by
            })
            scenarios[name] = {
                "status": report["status"],
                "report_existed": report_path.is_file(),
                "exp2_was_executed": report["exp2_was_executed"],
                "active_states": active,
                "active_phases": phases,
                "decisions": report["router"]["decisions"],
            }

        common = scenarios["common"]
        alternative = scenarios["alternative"]
        checks = {
            "exp2_not_executed": not common["exp2_was_executed"]
                and not alternative["exp2_was_executed"],
            "common_route_completed": common["status"] == "complete"
                and all(value == "completed" for value in common["active_states"].values()),
            "negative_route_completed": alternative["status"] == "complete"
                and all(value == "completed" for value in alternative["active_states"].values()),
            "amplifier_route_selected": "amplifier_replication" in common["active_states"],
            "alternative_b29_route_selected": "second_writer_mapping"
                in alternative["active_states"],
            "site_specific_route_selected": "site_specific_scale"
                in alternative["active_states"],
            "branch_trace_selected": "branch_specific_reverse_trace"
                in alternative["active_states"],
            "all_exp3_phases_reached": len(set(common["active_phases"])) == 8
                and len(set(alternative["active_phases"])) == 8,
        }
        result = {
            "schema_version": 1,
            "program": "EXP3 dry-run",
            "scientific_evidence": False,
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "scenarios": scenarios,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
        temporary.replace(target)
        return result


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m exp3.dryrun")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    report = run_dryrun(arguments.output)
    print(
        f"[exp3 dry-run] {report['status']}; "
        f"{sum(report['checks'].values())}/{len(report['checks'])} checks passed"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_dryrun", "main"]
