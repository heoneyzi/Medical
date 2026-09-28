"""Regression tests for EXP3-only execution and the EXP2 boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp3.adaptive import (
    EvidenceStatus,
    ExecutionState,
    ExperimentOutcome,
    ScientificDecision,
    ValidityReport,
)
from exp3.callbacks import dispatch, make_callback
from exp3.dryrun import _ScenarioProvider, _create_fixture_handoff, run_dryrun
from exp3.run import Exp3RunContext, execute_program


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "exp3"


class ImportBoundaryTests(unittest.TestCase):
    def test_exp3_never_imports_exp2_step_modules(self) -> None:
        violations: list[str] = []
        for source in sorted(PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name == "exp2" or name.startswith("exp2."):
                        if source.name != "exp2_api.py" or name.startswith("exp2.step"):
                            violations.append(f"{source.name}:{node.lineno}:{name}")
        self.assertEqual(violations, [])

    def test_exp2_allow_list_contains_no_step_import(self) -> None:
        source = PACKAGE_ROOT / "exp2_api.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name.startswith("exp2.step") for name in imported))


class OrchestrationTests(unittest.TestCase):
    def test_dryrun_exercises_common_and_valid_negative_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dryrun.json"
            report = run_dryrun(path)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(path.is_file())
            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(
                report["scenarios"]["alternative"]["exp2_was_executed"]
            )

    def test_valid_refutation_routes_and_finishes_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)
            report_path, report, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="negative-is-evidence",
                callback=_ScenarioProvider("alternative"),
                software_fixture=True,
            )
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report_path.is_file())
            self.assertEqual(
                router.records["b29_decomposition"].state,
                ExecutionState.COMPLETED,
            )
            self.assertEqual(
                router.records["b29_decomposition"].decision.status,
                EvidenceStatus.REFUTED,
            )
            self.assertEqual(
                router.records["second_writer_mapping"].state,
                ExecutionState.COMPLETED,
            )
            self.assertEqual(
                router.records["mechanism_application"].state,
                ExecutionState.COMPLETED,
            )

    def test_invalid_measurement_hard_fails_but_is_not_called_refutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)

            def callback(context: Exp3RunContext):
                decision = ScientificDecision(
                    EvidenceStatus.REFUTED,
                    "attempted_claim",
                    "This attempted claim is invalid and must not be routed.",
                )
                if context.experiment_id == "b29_decomposition":
                    return ExperimentOutcome(
                        decision,
                        ValidityReport(
                            measurement_valid=False,
                            details={"reason": "tap identity mismatch"},
                        ),
                    )
                code = {
                    "m28_preconditioning": "m28_redundant_inputs",
                    "scale_plateau": "site_specific_transition",
                }.get(context.experiment_id, "software_fixture_completed")
                diagnostics = {
                    "software_fixture": True,
                    "config_sha256": context.config.config_sha256,
                    "runtime_provenance": dict(
                        context.expected_runtime_provenance),
                }
                if context.experiment_id == "scale_plateau":
                    diagnostics["alpha_doses"] = list(
                        context.config.alpha_doses)
                    diagnostics["required_models"] = list(
                        context.config.comparison_models)
                    diagnostics["required_sites"] = {
                        model: list(sites) for model, sites
                        in context.config.required_scale_sites
                    }
                return ExperimentOutcome(
                    ScientificDecision(
                        EvidenceStatus.MIXED,
                        code,
                        "Independent node completed with explicit validity.",
                        diagnostics=diagnostics,
                    ),
                    ValidityReport(),
                )

            _, report, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="invalid-measurement",
                callback=callback,
                software_fixture=True,
            )
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(
                router.records["b29_decomposition"].state,
                ExecutionState.HARD_FAILED,
            )
            self.assertIsNone(router.records["b29_decomposition"].decision)
            # The independent scale experiment still executes; invalidity is
            # localized instead of being rewritten as global scientific null.
            self.assertEqual(
                router.records["scale_plateau"].state,
                ExecutionState.COMPLETED,
            )

    def test_implementation_error_is_retryable_not_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)

            def callback(context: Exp3RunContext):
                if context.experiment_id == "b29_decomposition":
                    raise RuntimeError("synthetic CUDA interruption")
                code = {
                    "m28_preconditioning": "m28_redundant_inputs",
                    "scale_plateau": "site_specific_transition",
                }.get(context.experiment_id, "software_fixture_completed")
                return ExperimentOutcome(
                    ScientificDecision(
                        EvidenceStatus.MIXED, code, "Independent node completed.",
                        diagnostics={
                            "software_fixture": True,
                            "config_sha256": context.config.config_sha256,
                            "runtime_provenance": dict(
                                context.expected_runtime_provenance),
                            **({"alpha_doses": list(context.config.alpha_doses)}
                               if context.experiment_id == "scale_plateau" else {}),
                            **({
                                "required_models": list(
                                    context.config.comparison_models),
                                "required_sites": {
                                    model: list(sites) for model, sites
                                    in context.config.required_scale_sites},
                            } if context.experiment_id == "scale_plateau" else {}),
                        },
                    ),
                    ValidityReport(),
                )

            _, report, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="retry-technical-error",
                callback=callback,
                software_fixture=True,
            )
            self.assertEqual(report["status"], "retry_required")
            self.assertEqual(
                router.records["b29_decomposition"].state,
                ExecutionState.RETRY_REQUIRED,
            )
            self.assertIsNone(router.records["b29_decomposition"].decision)

    def test_wrong_runtime_attestation_is_hard_failed_not_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)

            def callback(context: Exp3RunContext):
                provenance = dict(context.expected_runtime_provenance)
                provenance["checkpoint_sha256"] = "0" * 64
                return ExperimentOutcome(
                    ScientificDecision(
                        EvidenceStatus.SUPPORTED,
                        "b29_amplifier",
                        "This decision deliberately claims the wrong runtime.",
                        diagnostics={
                            "software_fixture": True,
                            "config_sha256": context.config.config_sha256,
                            "runtime_provenance": provenance,
                            "alpha_doses": list(context.config.alpha_doses),
                        },
                    ),
                    ValidityReport(),
                )

            _, report, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="wrong-runtime",
                callback=callback,
                max_nodes=1,
                software_fixture=True,
            )
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(
                router.records["b29_decomposition"].state,
                ExecutionState.HARD_FAILED,
            )
            self.assertFalse(
                router.records["b29_decomposition"].retry_history)

    def test_resume_preserves_hash_chained_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)
            _, first, first_router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="resume",
                callback=_ScenarioProvider("common"),
                max_nodes=1,
                software_fixture=True,
            )
            self.assertEqual(len(first["executed_this_invocation"]), 1)
            state_path = first["state_path"]
            _, second, second_router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="resume",
                callback=_ScenarioProvider("common"),
                resume_path=state_path,
                software_fixture=True,
            )
            self.assertEqual(second["status"], "complete")
            self.assertGreater(
                len(second_router.events), len(first_router.events)
            )

    def test_resume_rejects_a_different_exp2_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_handoff = _create_fixture_handoff(first_root)
            second_handoff = _create_fixture_handoff(second_root)
            from exp3.inputs import Exp2Handoff
            changed = Exp2Handoff.load(second_handoff, verify_files=True)
            changed.exp2_run_id = "a-different-accepted-exp2-run"
            changed.handoff_sha256 = ""
            changed.seal()
            changed.save(second_handoff)

            _, first, _ = execute_program(
                handoff_path=first_handoff,
                output_dir=root / "out",
                run_id="resume-contract",
                callback=_ScenarioProvider("common"),
                max_nodes=1,
                software_fixture=True,
            )
            with self.assertRaisesRegex(RuntimeError, "resume contract"):
                execute_program(
                    handoff_path=second_handoff,
                    output_dir=root / "out",
                    run_id="resume-contract",
                    callback=_ScenarioProvider("common"),
                    resume_path=first["state_path"],
                    software_fixture=True,
                )

    def test_tampered_fixed_foundation_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)
            raw = json.loads(handoff.read_text())
            raw["accepted_exp2_foundations"]["b28_content_writer"] = False
            handoff.write_text(json.dumps(raw))
            called = False

            def callback(_: Exp3RunContext):
                nonlocal called
                called = True
                return ScientificDecision(
                    EvidenceStatus.SUPPORTED, "should_not_run", "should not run"
                )

            with self.assertRaisesRegex(RuntimeError, "foundation"):
                execute_program(
                    handoff_path=handoff,
                    output_dir=root / "out",
                    run_id="bad-foundation",
                    callback=callback,
                )
            self.assertFalse(called)

    def test_callback_cannot_mutate_the_fixed_handoff_into_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)

            def callback(context: Exp3RunContext):
                context.handoff.accepted_exp2_foundations["b28_content_writer"] = False
                return ScientificDecision(
                    EvidenceStatus.SUPPORTED,
                    "would_be_false_evidence",
                    "This result must be discarded because provenance changed.",
                )

            _, report, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="callback-mutation",
                callback=callback,
                max_nodes=1,
                software_fixture=True,
            )
            self.assertEqual(report["status"], "invalid")
            record = router.records["b29_decomposition"]
            self.assertEqual(record.state, ExecutionState.HARD_FAILED)
            self.assertIsNone(record.decision)


class CallbackTests(unittest.TestCase):
    def test_mapping_and_bound_provider_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = _create_fixture_handoff(root)
            captured: list[str] = []

            def factory_callback(context: Exp3RunContext):
                captured.append(context.experiment_id)
                code = {
                    "b29_decomposition": "b29_amplifier",
                    "m28_preconditioning": "m28_preconditions_writer",
                    "scale_plateau": "shared_normalized_transition",
                }[context.experiment_id]
                return ExperimentOutcome(
                    ScientificDecision(
                        EvidenceStatus.SUPPORTED, code,
                        "The callback boundary completed with explicit validity.",
                        diagnostics={
                            "software_fixture": True,
                            "config_sha256": context.config.config_sha256,
                            "runtime_provenance": dict(
                                context.expected_runtime_provenance),
                            **({"alpha_doses": list(context.config.alpha_doses)}
                               if context.experiment_id in {
                                   "b29_decomposition", "scale_plateau"} else {}),
                            **({
                                "required_models": list(
                                    context.config.comparison_models),
                                "required_sites": {
                                    model: list(sites) for model, sites
                                    in context.config.required_scale_sites},
                            } if context.experiment_id == "scale_plateau" else {}),
                        },
                    ),
                    ValidityReport(),
                )

            callback = make_callback({
                "b29_decomposition": factory_callback,
                "m28_preconditioning": factory_callback,
                "scale_plateau": factory_callback,
            })
            _, _, router = execute_program(
                handoff_path=handoff,
                output_dir=root / "out",
                run_id="mapping-provider",
                callback=callback,
                max_nodes=3,
                software_fixture=True,
            )
            self.assertEqual(
                set(captured), {
                    "b29_decomposition", "m28_preconditioning", "scale_plateau"
                }
            )
            self.assertTrue(all(
                router.records[name].state is ExecutionState.COMPLETED
                for name in captured
            ))


if __name__ == "__main__":
    unittest.main()
