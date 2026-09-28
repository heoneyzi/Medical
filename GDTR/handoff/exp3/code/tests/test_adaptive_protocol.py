from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from exp3.adaptive import (
    ALL_EVIDENCE_STATUSES,
    EXP3_PHASES,
    AdaptiveRouter,
    EvidenceStatus,
    ExecutionState,
    ExperimentOutcome,
    ExperimentSpec,
    OutcomeMatcher,
    RetryDirective,
    RouteRule,
    ScientificDecision,
    ValidityReport,
    build_exp3_research_router,
    FIXED_EXP2_FOUNDATION_CLAIMS,
)
from exp3.artifacts import ArtifactStore


def decision(
    status: EvidenceStatus,
    code: str = "test_conclusion",
    text: str = "A predeclared scientific interpretation.",
) -> ScientificDecision:
    return ScientificDecision(status, code, text, diagnostics={"estimate": 0.25})


def complete(
    router: AdaptiveRouter,
    experiment_id: str,
    status: EvidenceStatus,
    code: str = "test_conclusion",
    validity: ValidityReport | None = None,
) -> None:
    router.begin(experiment_id)
    router.record_outcome(
        experiment_id,
        ExperimentOutcome(decision(status, code), validity or ValidityReport()),
    )


class AdaptiveProtocolTests(unittest.TestCase):
    def test_artifact_references_are_structured_immutable_and_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            ref = store.put_json(
                "phase1/table", {"value": 1}, source_split="chr22",
                source_role="discovery", created_by="unit-test")
            with self.assertRaises(TypeError):
                ScientificDecision(
                    EvidenceStatus.SUPPORTED, "bad_ref", "bad reference",
                    artifact_refs=("phase1/table",),  # type: ignore[arg-type]
                )
            with self.assertRaises(FileExistsError):
                store.put_json(
                    "phase1/table", {"value": 2}, source_split="chr22",
                    source_role="discovery", created_by="unit-test")

            router = AdaptiveRouter("artifact-integrity")
            router.add_experiment(ExperimentSpec(
                "source", "artifact test", initially_active=True))
            router.begin("source")
            router.record_outcome("source", ExperimentOutcome(
                ScientificDecision(
                    EvidenceStatus.SUPPORTED, "artifact_sealed",
                    "The table is committed by full metadata and content hash.",
                    artifact_refs=(ref,),
                )))
            state = router.save(root / "state.json")
            Path(ref.path).write_text('{"value": 9}')
            with self.assertRaises(RuntimeError):
                AdaptiveRouter.load(state)

    def test_all_five_scientific_outcomes_complete_instead_of_fail(self) -> None:
        for status in EvidenceStatus:
            router = AdaptiveRouter(f"run-{status.value}")
            router.add_experiment(ExperimentSpec("source", "test", initially_active=True))
            router.add_experiment(ExperimentSpec("next", "continue", depends_on=("source",)))
            router.add_rule(RouteRule(
                "always_continue",
                {"source": OutcomeMatcher(ALL_EVIDENCE_STATUSES)},
                ("next",),
                "Every valid scientific answer is informative.",
            ))
            complete(router, "source", status)
            self.assertEqual(router.records["source"].state, ExecutionState.COMPLETED)
            self.assertEqual(router.records["next"].state, ExecutionState.READY)

    def test_default_plan_integrates_any_valid_mandatory_answers(self) -> None:
        for b29_status in EvidenceStatus:
            for scale_status in EvidenceStatus:
                router = build_exp3_research_router(
                    f"pair-{b29_status.value}-{scale_status.value}"
                )
                complete(
                    router, "m28_preconditioning", EvidenceStatus.EQUIVALENT,
                    "m28_bounded_local_null",
                )
                b29_code = {
                    EvidenceStatus.SUPPORTED: "b29_amplifier",
                    EvidenceStatus.EQUIVALENT: "b29_bounded_null",
                    EvidenceStatus.MIXED: "b29_mixed_role",
                    EvidenceStatus.UNRESOLVED: "b29_unresolved",
                    EvidenceStatus.REFUTED: "amplifier_refuted",
                }[b29_status]
                scale_code = {
                    EvidenceStatus.SUPPORTED: "shared_normalized_transition",
                    EvidenceStatus.EQUIVALENT: "no_common_scale_effect",
                    EvidenceStatus.MIXED: "site_specific_transition",
                    EvidenceStatus.UNRESOLVED: "wide_interval",
                    EvidenceStatus.REFUTED: "no_shared_plateau",
                }[scale_status]
                complete(router, "b29_decomposition", b29_status, b29_code)
                complete(router, "scale_plateau", scale_status, scale_code)
                followup = ScientificDecision(
                    EvidenceStatus.EQUIVALENT, "bounded_followup",
                    "The selected refinement produced a bounded target specification.",
                )
                router.run_ready({
                    name: (lambda _, value=followup: value)
                    for name in router.records if name != "exp3_mechanism_synthesis"
                })
                self.assertEqual(
                    router.records["exp3_mechanism_synthesis"].state,
                    ExecutionState.READY,
                    (b29_status, scale_status),
                )

    def test_amplifier_and_second_writer_have_distinct_routes(self) -> None:
        amp = build_exp3_research_router("amplifier")
        complete(amp, "b29_decomposition", EvidenceStatus.SUPPORTED, "b29_amplifier")
        self.assertEqual(amp.records["amplifier_replication"].state, ExecutionState.READY)
        self.assertEqual(amp.records["second_writer_mapping"].state, ExecutionState.DORMANT)

        writer = build_exp3_research_router("writer")
        complete(writer, "b29_decomposition", EvidenceStatus.MIXED, "b29_mixed_role")
        self.assertEqual(writer.records["second_writer_mapping"].state, ExecutionState.READY)
        self.assertEqual(writer.records["amplifier_replication"].state, ExecutionState.DORMANT)

    def test_m28_refines_inputs_without_routing_exp2_foundations(self) -> None:
        positive = build_exp3_research_router("m28-positive")
        complete(
            positive, "m28_preconditioning", EvidenceStatus.SUPPORTED,
            "m28_preconditions_writer",
        )
        self.assertEqual(
            positive.records["m28_preconditioner_mapping"].state,
            ExecutionState.READY,
        )

        mixed = build_exp3_research_router("m28-mixed")
        complete(
            mixed, "m28_preconditioning", EvidenceStatus.MIXED,
            "m28_redundant_inputs",
        )
        self.assertEqual(
            mixed.records["m28_alternative_inputs"].state,
            ExecutionState.READY,
        )
        self.assertFalse(set(mixed.records) & set(FIXED_EXP2_FOUNDATION_CLAIMS))

    def test_target_spec_receives_the_selected_refinement_result(self) -> None:
        router = build_exp3_research_router("routed-upstream")
        complete(
            router, "m28_preconditioning", EvidenceStatus.EQUIVALENT,
            "m28_bounded_local_null",
        )
        complete(
            router, "m28_conditional_null", EvidenceStatus.EQUIVALENT,
            "m28_null_bound_estimated",
        )
        context = router.begin("m28_target_spec")
        self.assertEqual(
            set(context.upstream),
            {"m28_preconditioning", "m28_conditional_null"},
        )

    def test_refuted_and_unresolved_have_actionable_routes(self) -> None:
        refuted = build_exp3_research_router("refuted")
        complete(refuted, "b29_decomposition", EvidenceStatus.REFUTED, "amplifier_refuted")
        self.assertEqual(refuted.records["second_writer_mapping"].state, ExecutionState.READY)

        unresolved = build_exp3_research_router("unresolved")
        complete(unresolved, "scale_plateau", EvidenceStatus.UNRESOLVED, "wide_interval")
        self.assertEqual(unresolved.records["scale_identifiability"].state, ExecutionState.READY)

    def test_mixed_scale_routes_to_site_specific_model(self) -> None:
        router = build_exp3_research_router("heterogeneous")
        complete(
            router, "scale_plateau", EvidenceStatus.MIXED, "site_specific_transition"
        )
        self.assertEqual(router.records["site_specific_scale"].state, ExecutionState.READY)
        self.assertEqual(
            router.records["normalized_scale_transfer"].state, ExecutionState.DORMANT
        )

    def test_valid_refutation_continues_to_applications_and_exp1_extension(self) -> None:
        router = build_exp3_research_router("valid-refutation")
        complete(
            router, "m28_preconditioning", EvidenceStatus.MIXED,
            "m28_redundant_inputs",
        )
        complete(router, "b29_decomposition", EvidenceStatus.REFUTED, "amplifier_refuted")
        complete(router, "scale_plateau", EvidenceStatus.REFUTED, "no_shared_plateau")
        followup = ScientificDecision(
            EvidenceStatus.EQUIVALENT, "bounded_followup",
            "The selected alternative branch produced a bounded target.",
        )
        router.run_ready({
            name: (lambda _, value=followup: value)
            for name in router.records if name != "exp3_mechanism_synthesis"
        })
        complete(
            router, "exp3_mechanism_synthesis", EvidenceStatus.REFUTED,
            "branch_specific_b29_scale",
        )
        complete(
            router, "scale_content_carrier_cube", EvidenceStatus.MIXED,
            "axes_conditionally_separable",
        )
        complete(
            router, "bidirectional_direction_transfer", EvidenceStatus.MIXED,
            "partial_direction_transfer",
        )
        complete(
            router, "cube_b30_mediation", EvidenceStatus.MIXED,
            "partial_b30_mediation",
        )
        complete(
            router, "causal_factorization_synthesis", EvidenceStatus.REFUTED,
            "alternative_transport_architecture",
        )
        expected = {
            "branch_specific_reverse_trace", "learned_transport",
            "task_heterogeneity", "exp1_extension", "mechanism_application",
            "cross_chromosome_generalization",
        }
        self.assertTrue(expected <= set(router.ready_experiments()))

    def test_only_measurement_provenance_or_leakage_hard_fail(self) -> None:
        invalid_reports = (
            ValidityReport(measurement_valid=False, details={"reason": "NaN logits"}),
            ValidityReport(provenance_valid=False, details={"reason": "hash mismatch"}),
            ValidityReport(leakage_free=False, details={"reason": "shared donor family"}),
        )
        for index, report in enumerate(invalid_reports):
            router = AdaptiveRouter(f"invalid-{index}")
            router.add_experiment(ExperimentSpec("source", "test", initially_active=True))
            router.add_experiment(ExperimentSpec("child", "dependent", depends_on=("source",)))
            router.add_rule(RouteRule(
                "continue", {"source": OutcomeMatcher()}, ("child",), "continue",
            ))
            complete(router, "source", EvidenceStatus.SUPPORTED, validity=report)
            self.assertEqual(router.records["source"].state, ExecutionState.HARD_FAILED)
            self.assertIsNone(router.records["source"].decision)
            self.assertEqual(router.records["child"].state, ExecutionState.DORMANT)

    def test_invalid_dependency_blocks_only_activated_descendant(self) -> None:
        router = AdaptiveRouter("scoped-invalidity")
        router.add_experiment(ExperimentSpec("bad", "bad stream", initially_active=True))
        router.add_experiment(ExperimentSpec("independent", "good stream", initially_active=True))
        router.add_experiment(ExperimentSpec(
            "child", "dependent", depends_on=("bad",), initially_active=True,
        ))
        complete(
            router, "bad", EvidenceStatus.SUPPORTED,
            validity=ValidityReport(provenance_valid=False),
        )
        self.assertEqual(router.records["child"].state, ExecutionState.BLOCKED_INVALID)
        self.assertEqual(router.records["independent"].state, ExecutionState.READY)

    def test_runtime_exception_is_retryable_not_hard_failure(self) -> None:
        router = AdaptiveRouter("retry")
        router.add_experiment(ExperimentSpec("source", "test", initially_active=True))

        def broken(_):
            raise RuntimeError("CUDA OOM")

        state = router.run_one("source", broken)
        self.assertEqual(state, ExecutionState.RETRY_REQUIRED)
        self.assertEqual(router.report()["hard_failures"], {})
        router.approve_retry("source", note="reduce microbatch only; estimand unchanged")
        self.assertEqual(router.records["source"].state, ExecutionState.READY)
        router.run_one("source", lambda _: decision(EvidenceStatus.UNRESOLVED, "wide_ci"))
        self.assertEqual(router.records["source"].state, ExecutionState.COMPLETED)

    def test_post_start_adaptation_requires_and_records_reason(self) -> None:
        router = AdaptiveRouter("adaptive-registration")
        router.add_experiment(ExperimentSpec("source", "test", initially_active=True))
        complete(router, "source", EvidenceStatus.MIXED)
        with self.assertRaises(RuntimeError):
            router.add_experiment(ExperimentSpec("new", "adaptive test"))
        router.add_experiment(
            ExperimentSpec("new", "adaptive test", depends_on=("source",)),
            adaptation_reason="Mixed response suggests two regimes.",
        )
        router.activate("new", reason="Estimate each regime separately")
        self.assertEqual(router.records["new"].state, ExecutionState.READY)

    def test_conclusion_code_controls_route_within_same_status(self) -> None:
        router = AdaptiveRouter("codes")
        router.add_experiment(ExperimentSpec("source", "test", initially_active=True))
        router.add_experiment(ExperimentSpec("amp", "amplifier", depends_on=("source",)))
        router.add_experiment(ExperimentSpec("writer", "writer", depends_on=("source",)))
        router.add_rule(RouteRule(
            "amp_route", {"source": OutcomeMatcher(
                frozenset({EvidenceStatus.SUPPORTED}), frozenset({"amplifier"})
            )}, ("amp",), "amplifier route",
        ))
        router.add_rule(RouteRule(
            "writer_route", {"source": OutcomeMatcher(
                frozenset({EvidenceStatus.SUPPORTED}), frozenset({"writer"})
            )}, ("writer",), "writer route",
        ))
        complete(router, "source", EvidenceStatus.SUPPORTED, "writer")
        self.assertEqual(router.records["writer"].state, ExecutionState.READY)
        self.assertEqual(router.records["amp"].state, ExecutionState.DORMANT)

    def test_run_ready_follows_newly_activated_nodes(self) -> None:
        router = AdaptiveRouter("auto")
        router.add_experiment(ExperimentSpec("a", "first", initially_active=True))
        router.add_experiment(ExperimentSpec("b", "second", depends_on=("a",)))
        router.add_rule(RouteRule(
            "a_to_b", {"a": OutcomeMatcher()}, ("b",), "continue after any answer",
        ))
        runners = {
            "a": lambda _: decision(EvidenceStatus.EQUIVALENT, "bounded_null"),
            "b": lambda _: decision(EvidenceStatus.REFUTED, "revised_model"),
        }
        self.assertEqual(router.run_ready(runners), ("a", "b"))
        self.assertTrue(all(
            router.records[name].state is ExecutionState.COMPLETED for name in ("a", "b")
        ))

    def test_exhaustive_route_helper_rejects_success_only_gate(self) -> None:
        router = AdaptiveRouter("exhaustive")
        router.add_experiment(ExperimentSpec("source", "source", initially_active=True))
        router.add_experiment(ExperimentSpec("positive", "positive", depends_on=("source",)))
        router.add_experiment(ExperimentSpec("refine", "refine", depends_on=("source",)))
        with self.assertRaisesRegex(ValueError, "cover exactly all"):
            router.add_exhaustive_status_routes(
                "source", {EvidenceStatus.SUPPORTED: ("positive",)},
                rule_prefix="source_outcome", rationale="Route every result",
            )
        mapping = {
            status: (("positive",) if status is EvidenceStatus.SUPPORTED else ("refine",))
            for status in EvidenceStatus
        }
        added = router.add_exhaustive_status_routes(
            "source", mapping, rule_prefix="source_outcome",
            rationale="Route every result",
        )
        self.assertEqual(len(added), 5)
        complete(router, "source", EvidenceStatus.REFUTED, "negative_answer")
        self.assertEqual(router.records["refine"].state, ExecutionState.READY)
        self.assertEqual(router.records["positive"].state, ExecutionState.DORMANT)

    def test_snapshot_round_trip_and_tamper_detection(self) -> None:
        router = build_exp3_research_router("snapshot")
        complete(router, "b29_decomposition", EvidenceStatus.MIXED, "b29_mixed_role")
        with tempfile.TemporaryDirectory() as tmp:
            path = router.save(Path(tmp) / "router.json")
            loaded = AdaptiveRouter.load(path)
            self.assertEqual(loaded.report(), router.report())
            raw = json.loads(path.read_text())
            raw["records"]["b29_decomposition"]["state"] = "hard_failed"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RuntimeError, "snapshot was modified"):
                AdaptiveRouter.load(path)

    def test_ledger_tampering_is_detected(self) -> None:
        router = AdaptiveRouter("ledger")
        router.add_experiment(ExperimentSpec("source", "test", initially_active=True))
        router.events[0]["payload"]["experiment_id"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "modified"):
            router.verify_ledger()

    def test_non_json_or_non_finite_diagnostics_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ScientificDecision(
                EvidenceStatus.SUPPORTED, "bad", "bad diagnostics", {"tensor": object()}
            )
        with self.assertRaises(ValueError):
            ScientificDecision(
                EvidenceStatus.SUPPORTED, "bad", "bad diagnostics", {"effect": float("nan")}
            )

    def test_default_graph_uses_only_exp3_phase_names(self) -> None:
        router = build_exp3_research_router("phases")
        used = {record.spec.phase for record in router.records.values()}
        self.assertTrue(used <= set(EXP3_PHASES))
        self.assertEqual(set(EXP3_PHASES), used)
        self.assertFalse(any("step" in record.spec.experiment_id for record in router.records.values()))

    def test_exp2_foundation_is_fixed_metadata_not_a_routable_hypothesis(self) -> None:
        router = build_exp3_research_router("foundation")
        self.assertEqual(router.foundation_claims, FIXED_EXP2_FOUNDATION_CLAIMS)
        with self.assertRaises(AttributeError):
            router.foundation_claims = ("replace_foundation",)
        self.assertFalse(set(FIXED_EXP2_FOUNDATION_CLAIMS) & set(router.records))
        self.assertFalse(any(
            source in FIXED_EXP2_FOUNDATION_CLAIMS
            for rule in router.rules.values() for source in rule.when
        ))


if __name__ == "__main__":
    unittest.main()
