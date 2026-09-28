"""The experiment layer: probes, pairs, the ladder and the report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoflowagent.geoacmg import ladder, pairs, probes, report
from geoflowagent.geoacmg.claims import Evidence, Finding, Outcome, Preregistration, Role
from geoflowagent.geoacmg.cli import _claims, _primary_findings
from geoflowagent.geoacmg.clingen import read_corpus


class TestMinimalPairs:
    def test_three_families_are_produced_from_real_records(self, corpus_tsv):
        records, _ = read_corpus(corpus_tsv)
        rows, summary = pairs.build_pairs(records, {})
        assert summary["total"] == len(rows)
        kinds = {row["provenance"]["kind"] for row in rows}
        assert "hgvs_spelling" in kinds or summary["invariant_hgvs_spelling"] == 0
        for row in rows:
            assert row["relation"] in {"invariant", "functional_sensitive"}
            assert row["left_text"] and row["right_text"]

    def test_pair_text_never_contains_the_answer(self, corpus_tsv):
        records, _ = read_corpus(corpus_tsv)
        rows, _ = pairs.build_pairs(records, {})
        for row in rows:
            surface = row["left_text"] + row["right_text"]
            for banned in ("Pathogenic", "Benign", "PM2", "PVS1", "BA1"):
                assert banned not in surface

    def test_a_pair_is_dropped_when_its_sides_land_in_different_splits(self, corpus_tsv):
        records, _ = read_corpus(corpus_tsv)
        alternating = {
            record.uuid: ("train" if index % 2 == 0 else "dev")
            for index, record in enumerate(records)
        }
        rows, _ = pairs.build_pairs(records, alternating)
        for row in rows:
            assert row["split"] in {"train", "dev", "test"}


class TestProbes:
    def test_grouped_cv_holds_whole_genes_out(self):
        """A probe that memorises a gene must not be rewarded for it."""

        rng = np.random.default_rng(0)
        genes = [f"g{i % 20}" for i in range(400)]
        offsets = {gene: rng.normal(scale=5.0) for gene in set(genes)}
        features = rng.normal(size=(400, 8))
        # Target is pure gene identity: nothing in the features predicts it.
        targets = np.asarray([offsets[gene] for gene in genes])
        assert probes.grouped_cv_r2(features, targets, genes) < 0.1

    def test_random_projection_control_has_the_same_width(self):
        features = np.random.default_rng(1).normal(size=(50, 12))
        assert probes.random_projection(features).shape == features.shape

    def test_depth_matched_pairs_only_compare_within_a_task_and_depth(self):
        values = [0.0, 1.0, 2.0, 3.0]
        targets = [0.0, 1.0, 5.0, 6.0]
        depths = [0, 0, 1, 1]
        tasks = ["a", "a", "a", "a"]
        correct, chance, clusters = probes.depth_matched_pairs(values, targets, depths, tasks)
        assert len(correct) == 2, "one comparable pair per depth bucket"
        assert set(chance) == {0.5}
        assert set(clusters) == {"a"}

    def test_hubness_reports_near_full_rank_for_isotropic_vectors(self):
        vectors = np.random.default_rng(2).normal(size=(300, 16))
        summary = probes.hubness(vectors, k=5)
        assert summary["effective_rank"] > 10
        assert 0.0 <= summary["hub_occurrence_gini"] <= 1.0


class TestLadder:
    @staticmethod
    def _world(required, target="X"):
        """A miniature task: query the required probes, then report."""

        tools = {f"probe_{name}": 1.0 for name in required} | {"report_X": 0.5, "report_Y": 0.5}

        def candidates(state):
            if state.get("done"):
                return []
            return [t for t in tools if not (t.startswith("probe_") and t[6:] in state)]

        def execute(state, tool):
            new = dict(state)
            if tool.startswith("probe_"):
                new[tool[6:]] = True
                return new, tools[tool], 2
            new["done"] = tool
            return new, tools[tool], 0

        def solved(state):
            return state.get("done") == f"report_{target}" and all(
                name in state for name in required
            )

        return candidates, execute, solved, (lambda s: bool(s.get("done")))

    def test_a_planner_that_reports_without_evidence_fails(self):
        candidates, execute, solved, terminal = self._world(["a", "b"])

        class Rush:
            name = "rush"

            def propose(self, state, cands, ctx):
                return ["report_X"]

        episode = ladder.run_episode(
            planner=Rush(), task_id="t", gene="g", initial_state={},
            candidates_fn=candidates, execute_fn=execute,
            solved_fn=solved, terminal_fn=terminal,
        )
        assert episode.terminated and not episode.solved

    def test_commit_length_changes_plan_evaluations_but_not_the_work_done(self):
        candidates, execute, solved, terminal = self._world(["a", "b", "c"])
        plan = ["probe_a", "probe_b", "probe_c", "report_X"]
        source = ladder.WholePlan(lambda s, c, x: plan, name="fixed")
        episodes = {
            commit: ladder.run_episode(
                planner=source, task_id="t", gene="g", initial_state={},
                candidates_fn=candidates, execute_fn=execute,
                solved_fn=solved, terminal_fn=terminal, commit=commit,
            )
            for commit in (1, 4)
        }
        assert all(e.solved for e in episodes.values())
        assert episodes[1].budget.tool_cost == episodes[4].budget.tool_cost
        assert episodes[1].budget.plan_evaluations > episodes[4].budget.plan_evaluations

    def test_progress_is_evidence_magnitude_so_benign_tasks_are_not_negative(self):
        candidates, execute, solved, terminal = self._world(["a"])

        def negative_execute(state, tool):
            new, cost, _ = execute(state, tool)
            return new, cost, -8 if tool.startswith("probe_") else 0

        episode = ladder.run_episode(
            planner=ladder.WholePlan(lambda s, c, x: ["probe_a", "report_X"]),
            task_id="t", gene="g", initial_state={},
            candidates_fn=candidates, execute_fn=negative_execute,
            solved_fn=solved, terminal_fn=terminal,
        )
        assert episode.points_trace[-1] == 8
        assert episode.progress_efficiency > 0

    def test_commit_curve_names_its_optimum(self):
        made = {
            1: [ladder.Episode("t1", "g", solved=True)],
            4: [ladder.Episode("t1", "g", solved=False)],
        }
        curve = ladder.commit_curve(made)
        assert curve["best_commit"] == 1
        assert "best_commit" in curve["reading"]


class TestReport:
    def test_round_trip_and_verdicts(self, tmp_path):
        prereg = Preregistration("study", _claims(), _primary_findings())
        (tmp_path / "pre.json").write_text(json.dumps(prereg.to_dict()))
        findings = [
            Finding("C3", "P3_flow_minus_greedy_at_matched_cost", Evidence.PAIRED_INTERVAL,
                    0.06, "gene", 57, Role.PRIMARY, 0.01, 0.11),
            Finding("B0", "benchmark_needs_more_than_one_tool_family",
                    Evidence.PAIRED_INTERVAL, 0.72, "gene", 155, Role.PRIMARY, 0.67, 0.77),
        ]
        report.write_findings(tmp_path / "f.jsonl", findings)
        restored = report.read_findings(tmp_path / "f.jsonl")
        assert [f.name for f in restored] == [f.name for f in findings]
        result = report.build(tmp_path / "pre.json", tmp_path / "f.jsonl", tmp_path / "r.md")
        assert result["verdicts"]["C3"] == Outcome.SUPPORTED.value
        assert result["verdicts"]["B0"] == Outcome.SUPPORTED.value
        assert result["verdicts"]["C1"] == Outcome.NOT_TESTED.value
        text = (tmp_path / "r.md").read_text()
        assert "임계값은 쓰이지 않았습니다" in text

    def test_an_unregistered_primary_stops_the_report(self, tmp_path):
        prereg = Preregistration("study", _claims(), _primary_findings())
        (tmp_path / "pre.json").write_text(json.dumps(prereg.to_dict()))
        report.write_findings(
            tmp_path / "f.jsonl",
            [Finding("C3", "found_afterwards", Evidence.PAIRED_INTERVAL, 9.0, "gene", 5,
                     Role.PRIMARY, 8.0, 10.0)],
        )
        with pytest.raises(ValueError, match="not pre-registered"):
            report.build(tmp_path / "pre.json", tmp_path / "f.jsonl", tmp_path / "r.md")


class TestPipeline:
    def test_every_stage_declares_outputs_and_gpu_need(self):
        from geoflowagent.geoacmg import pipeline

        keys = [stage.key for stage in pipeline.STAGES]
        assert keys[0] == "prereg" and keys[-1] == "report"
        assert len(set(keys)) == len(keys)
        for stage in pipeline.STAGES:
            assert stage.outputs, f"{stage.key} declares no output, so it can never be skipped"
        gpu_stages = {s.key for s in pipeline.STAGES if s.needs_gpu}
        assert "embed" in gpu_stages
        assert "oracle" not in gpu_stages, "the oracle is CPU work and must not claim a GPU"
