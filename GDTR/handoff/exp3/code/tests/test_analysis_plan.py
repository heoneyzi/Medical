"""Tests for the Phase 3--8 pre-locked analysis contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp3.adaptive import (
    EvidenceStatus,
    ExperimentOutcome,
    ScientificDecision,
    ValidityReport,
    build_exp3_research_router,
)
from exp3.analysis_plan import (
    AnalysisDeclaration,
    CONFIRMATORY_PHASES,
    ConfirmatoryAnalysisPlan,
)
from exp3.config import Exp3Config
from exp3.data import TokenPanel, TokenUnit
from exp3.dryrun import _create_fixture_handoff
from exp3.inputs import Exp2Handoff
from exp3.run import (
    Exp3RunContext,
    _validate_confirmatory_analysis,
    callback_contract,
    execute_program,
)
from exp3.runtime import create_runtime_manifest


def _confirmatory_ids() -> tuple[str, ...]:
    router = build_exp3_research_router("plan-test")
    return tuple(sorted(
        name for name, record in router.records.items()
        if record.spec.phase in CONFIRMATORY_PHASES
    ))


def _declaration(experiment_id: str) -> AnalysisDeclaration:
    return AnalysisDeclaration(
        experiment_id=experiment_id,
        estimand=f"pre-locked estimand for {experiment_id}",
        decision_rule="Apply the declared interval rule without retuning.",
        thresholds={"two_sided_alpha": 0.05},
        hyperparameters={"bootstrap_replicates": 1000},
        control_families=("matched_negative",),
        control_plan="Run the matched negative control in the same batch.",
        data_roles={
            "selection": "chr22",
            "development": "chr21",
            "confirmation": "chr17",
        },
    )


def _plan(config: Exp3Config, *, omit: str | None = None) -> ConfirmatoryAnalysisPlan:
    return ConfirmatoryAnalysisPlan(
        plan_id="exp3-confirmatory-test",
        config_sha256=config.config_sha256,
        locked_split=config.locked_chromosome,
        declarations=tuple(
            _declaration(name) for name in _confirmatory_ids()
            if name != omit
        ),
        registration_reference="test-only",
    )


def _token_panel_paths(root: Path, config: Exp3Config) -> tuple[Path, ...]:
    tokenizer = lambda sequence: [ord(base) for base in sequence]
    paths = []
    for offset, (role, chromosome) in enumerate((
        ("discovery", config.discovery_chromosome),
        ("development", config.development_chromosome),
        ("locked", config.locked_chromosome),
    )):
        unit_id = f"{role}:unit"
        panel = TokenPanel.from_sequences(
            [(TokenUnit(
                unit_id, chromosome, offset * 100, offset * 100 + 4,
                dependency_keys=(f"{role}:cluster",)), "ACGT")],
            tokenizer=tokenizer,
        )
        path = root / f"panel-{role}"
        panel.save(
            path, panel_name=f"{role}-panel",
            source_split=f"{chromosome}-{role}", source_role=role,
            created_by="test_analysis_plan",
        )
        paths.append(path)
    return tuple(paths)


def _root_scientific_callback(context: Exp3RunContext) -> ExperimentOutcome:
    """Minimal real-mode callback used only for contract integration."""
    margins = context.config.margins
    diagnostics = {
        "config_sha256": context.config.config_sha256,
        "runtime_provenance": dict(context.expected_runtime_provenance),
        "alpha_doses": list(context.config.alpha_doses),
        "margins_used": {
            "effect_margin": margins.b29_effect,
            "equivalence_margin": margins.equivalence,
            "perpendicular_fraction_margin": margins.b29_perpendicular_fraction,
            "independent_effect_margin": margins.b29_independent_effect,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
            "parallel_plateau": {
                "effect_margin": margins.b29_effect,
                "increment_equivalence_margin": margins.b29_plateau_increment,
                "log_q_boundary_margin": margins.b29_log_q_boundary,
                "plateau_remaining_fraction": (
                    margins.b29_plateau_remaining_fraction),
                "plateau_boundary_spread": margins.b29_plateau_boundary_spread,
                "min_clusters": margins.min_dependency_clusters,
                "n_boot": margins.bootstrap_replicates,
                "alpha": margins.alpha,
            },
        },
    }
    return ExperimentOutcome(
        ScientificDecision(
            EvidenceStatus.SUPPORTED,
            "b29_amplifier",
            "Synthetic contract-only root outcome.",
            diagnostics=diagnostics,
        ),
        ValidityReport(),
    )


class AnalysisPlanTests(unittest.TestCase):
    def test_round_trip_and_nested_mutation_detection(self) -> None:
        config = Exp3Config()
        plan = _plan(config)
        with tempfile.TemporaryDirectory() as directory:
            path = plan.save(Path(directory) / "plan.json")
            loaded = ConfirmatoryAnalysisPlan.load(path)
            loaded.validate(
                required_experiment_ids=_confirmatory_ids(),
                expected_config_sha256=config.config_sha256,
                expected_locked_split=config.locked_chromosome,
            )
            self.assertEqual(loaded.plan_sha256, plan.plan_sha256)
            loaded.declarations[0].thresholds["two_sided_alpha"] = 0.10
            with self.assertRaisesRegex(RuntimeError, "changed after sealing"):
                loaded.validate()

    def test_plan_requires_every_possible_phase3_to_8_branch(self) -> None:
        config = Exp3Config()
        missing = _confirmatory_ids()[0]
        plan = _plan(config, omit=missing)
        with self.assertRaisesRegex(RuntimeError, "missing=.*" + missing):
            plan.validate(required_experiment_ids=_confirmatory_ids())

    def test_empty_choices_require_explicit_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "no_thresholds_reason"):
            AnalysisDeclaration(
                experiment_id="common_reverse_trace",
                estimand="fixed estimand",
                decision_rule="fixed rule",
                thresholds={},
                hyperparameters={"rank": 1},
                control_plan="fixed matched controls",
                data_roles={"confirmation": "chr17"},
            )

    def test_wrong_downstream_attestation_is_rejected(self) -> None:
        config = Exp3Config()
        plan = _plan(config)
        experiment_id = "learned_transport"
        expected = plan.attestation(experiment_id)
        expected["thresholds_used"]["two_sided_alpha"] = 0.10
        outcome = ExperimentOutcome(
            ScientificDecision(
                EvidenceStatus.SUPPORTED,
                "predicted_delta_rescue",
                "Synthetic downstream result.",
                diagnostics={"confirmatory_analysis": expected},
            ),
            ValidityReport(),
        )
        with self.assertRaisesRegex(ValueError, "did not use the sealed"):
            _validate_confirmatory_analysis(experiment_id, outcome, plan)

    def test_execute_program_binds_plan_into_scientific_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff_path = _create_fixture_handoff(root)
            handoff = Exp2Handoff.load(handoff_path, verify_files=True)
            config = Exp3Config()
            plan_path = _plan(config).save(root / "analysis_plan.json")
            token_panel_paths = _token_panel_paths(root, config)
            identity = callback_contract(_root_scientific_callback)
            manifest = create_runtime_manifest(
                provider_id=str(identity["provider_id"]),
                checkpoint_sha256=handoff.checkpoint_sha256,
                architecture_sha256=handoff.architecture_sha256,
                exp2_code_sha256=handoff.code_sha256,
                provider_source_sha256=str(identity["source_sha256"]),
            )
            manifest_path = manifest.save(root / "runtime.json")

            with self.assertRaisesRegex(RuntimeError, "analysis plan"):
                execute_program(
                    handoff_path=handoff_path,
                    output_dir=root / "missing",
                    callback=_root_scientific_callback,
                    runtime_manifest_path=manifest_path,
                    max_nodes=1,
                )

            _, report, _ = execute_program(
                handoff_path=handoff_path,
                output_dir=root / "out",
                callback=_root_scientific_callback,
                runtime_manifest_path=manifest_path,
                analysis_plan_path=plan_path,
                token_panel_paths=token_panel_paths,
                max_nodes=1,
            )
            contract = json.loads(Path(report["run_contract_path"]).read_text())
            self.assertEqual(
                contract["confirmatory_analysis_plan_sha256"],
                report["confirmatory_analysis_plan"]["plan_sha256"],
            )
            self.assertEqual(len(contract["token_panels"]), 3)
            self.assertEqual(
                {item["source_role"] for item in contract["token_panels"]},
                {"discovery", "development", "locked"},
            )
            self.assertEqual(report["status"], "ready")


if __name__ == "__main__":
    unittest.main()
