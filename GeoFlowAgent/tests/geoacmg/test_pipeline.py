"""Corpus to tasks, and the properties that make the benchmark well posed."""

from __future__ import annotations

from datetime import date

import pytest

from geoflowagent.data.contracts import ContractEngine
from geoflowagent.geoacmg import splits
from geoflowagent.geoacmg.clingen import read_corpus, temporal_split
from geoflowagent.geoacmg.diagnostics import corpus_card, single_family_gap_finding
from geoflowagent.geoacmg.evidence import Classification
from geoflowagent.geoacmg.tasks import (
    build_tasks,
    initial_state,
    private_verifier,
    probe_tool,
    public_goal,
    report_tool,
    snapshots_for,
)
from geoflowagent.geoacmg.toolmap import BINDINGS, Decidability, ToolFamily, coverage


@pytest.fixture
def corpus(corpus_tsv):
    return read_corpus(corpus_tsv)


class TestIngest:
    def test_every_exclusion_is_counted(self, corpus):
        records, report = corpus
        assert report.rows_read == 6
        assert report.skipped["retracted"] == 1
        assert report.skipped["no_applied_code"] == 1
        assert report.kept == len(records) == 4

    def test_truncated_spelling_is_repaired_and_the_repair_is_reported(self, corpus):
        _, report = corpus
        assert any("Very" in key for key in report.normalisations)

    def test_point_scale_fidelity_is_reported_not_enforced(self, corpus):
        records, report = corpus
        assert 0.0 <= report.reconstruction_rate <= 1.0
        # A record whose points disagree with the panel is still ingested.
        assert report.kept == len(records)

    def test_undated_records_fall_on_the_conservative_side(self, corpus):
        records, _ = corpus
        before, after = temporal_split(records, date(2024, 9, 15))
        assert all(r.approval_date is None or r.approval_date < date(2024, 9, 15) for r in before)


class TestToolBinding:
    def test_every_criterion_has_a_binding(self):
        from geoflowagent.geoacmg.evidence import CRITERIA

        assert set(BINDINGS) == set(CRITERIA)

    def test_criteria_needing_a_paper_or_a_pedigree_are_declared_out_of_scope(self):
        for base in ("PS3", "BS3", "PP1", "PS2", "PS4"):
            assert BINDINGS[base].decidability is Decidability.OUT_OF_SCOPE

    def test_pvs1_requires_the_gene_mechanism_as_well_as_the_consequence(self):
        """PVS1 is not 'truncating', it is 'truncating in a loss-of-function gene'."""

        binding = BINDINGS["PVS1"]
        assert binding.primary is ToolFamily.MOLECULAR_CONSEQUENCE
        assert ToolFamily.GENE_MECHANISM in binding.requires
        assert not binding.resolved_by({ToolFamily.MOLECULAR_CONSEQUENCE})
        assert binding.resolved_by({ToolFamily.MOLECULAR_CONSEQUENCE, ToolFamily.GENE_MECHANISM})

    def test_coverage_reports_scope_rather_than_silently_narrowing_it(self, corpus):
        records, _ = corpus
        report = coverage(records)
        assert report["records"] == len(records)
        assert 0.0 <= report["records_fully_decidable_fraction"] <= 1.0
        assert "bindings" in report


class TestTaskConstruction:
    def test_out_of_scope_evidence_excludes_a_record(self, corpus):
        records, _ = corpus
        tasks, report = build_tasks(records)
        assert report.dropped["not_fully_decidable"] >= 1
        assert all("u-ps3" not in task.record.uuid for task in tasks)

    def test_a_criterion_counts_once_and_only_once_its_prerequisites_are_met(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        task = next(t for t in tasks if any(c.base == "PVS1" for o in t.outcomes for c in o.met))
        consequence_only = task.points_after([ToolFamily.MOLECULAR_CONSEQUENCE])
        with_mechanism = task.points_after(
            [ToolFamily.MOLECULAR_CONSEQUENCE, ToolFamily.GENE_MECHANISM]
        )
        assert consequence_only == 0, "PVS1 must not credit before the mechanism is known"
        assert with_mechanism == 8
        # And the full score never double-counts a multi-family criterion.
        assert task.total_points == task.points_after(task.families)

    def test_committing_without_evidence_cannot_satisfy_the_verifier(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        for task in tasks:
            assert not task.is_sufficient([])
            assert task.is_sufficient(task.required_families)

    def test_required_evidence_is_a_strict_subset_of_what_can_be_called(self, corpus):
        """If everything available is required there is no decision, only ordering."""

        records, _ = corpus
        tasks, _ = build_tasks(records)
        assert any(set(t.families) - t.required_families for t in tasks)


class TestCorpusIsExecutable:
    """The emitted corpus must run under the repository's own contract engine."""

    @staticmethod
    def _engine(task, source_release="test-release"):
        families = sorted(set(task.families), key=lambda f: f.value)
        tools = [probe_tool(f, cost=1.0) for f in families]
        tools += [report_tool(label, cost=0.5) for label in Classification]
        return ContractEngine(tools, snapshots_for(task, source_release)), tools

    def test_probes_bind_evidence_and_only_the_right_report_verifies(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        for task in tasks:
            engine, tools = self._engine(task)
            verifier = private_verifier(task)
            slug = task.target.value.lower().replace(" ", "_")
            state = initial_state(task)

            bare = engine.execute(dict(state), f"report_{slug}").state
            assert engine.goal_satisfied(bare, public_goal())
            assert not engine.goal_satisfied(bare, verifier), (
                "a policy that commits the right label without looking must fail"
            )

            for family in sorted(task.required_families, key=lambda f: f.value):
                state = engine.execute(state, f"probe_{family.value}").state
            assert not engine.goal_satisfied(state, verifier)
            done = engine.execute(state, f"report_{slug}").state
            assert engine.goal_satisfied(done, verifier)

            for other in Classification:
                if other is task.target:
                    continue
                wrong_slug = other.value.lower().replace(" ", "_")
                wrong = engine.execute(state, f"report_{wrong_slug}").state
                assert not engine.goal_satisfied(wrong, verifier)

    def test_initial_state_contains_no_evidence(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        for task in tasks:
            state = initial_state(task)
            assert state["evidence"] == {}
            assert state["evidence_sources"] == {}
            assert "report" not in state

    def test_no_private_material_reaches_the_visible_state(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        for task in tasks:
            visible = repr(initial_state(task)) + repr(public_goal())
            assert task.target.value not in visible
            for outcome in task.outcomes:
                for code in outcome.met:
                    assert code.label not in visible


class TestSplits:
    def test_a_gene_never_straddles_a_gene_split(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        plan = splits.by_gene(tasks)
        seen: dict[str, str] = {}
        for task in tasks:
            assigned = plan.assignment[task.task_id]
            assert seen.setdefault(task.record.gene, assigned) == assigned

    def test_temporal_split_states_its_limitation(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        plan = splits.temporal(tasks)
        assert "literature contamination" in plan.report["limitation"]

    def test_splits_are_deterministic(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        assert splits.by_gene(tasks).assignment == splits.by_gene(tasks).assignment
        assert splits.by_variant(tasks).assignment == splits.by_variant(tasks).assignment


class TestDiagnostics:
    def test_card_reports_the_majority_class_and_the_single_tool_ceiling(self, corpus):
        records, _ = corpus
        tasks, _ = build_tasks(records)
        card = corpus_card(tasks)
        assert 0.0 < card["target_balance"]["majority_class_rate"] <= 1.0
        assert card["single_family_ceiling"]["gap"] >= 0.0
        assert card["tasks_per_gene"]["effective_number_of_clusters"] <= card["genes"]

    def test_the_single_family_gap_is_a_diagnostic_not_a_claim(self, corpus):
        from geoflowagent.geoacmg.claims import Role

        records, _ = corpus
        tasks, _ = build_tasks(records)
        finding = single_family_gap_finding(tasks, resamples=200)
        assert finding.role is Role.DIAGNOSTIC
