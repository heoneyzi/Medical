from __future__ import annotations

from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.procedural_search import WORKFLOW_FAMILIES, generate_search_fixture
from geoflowagent.data.procedural_search_hard import generate_hard_search_fixture
from geoflowagent.search import SearchLimits, WeightedSearchOracle
from geoflowagent.utils.io import read_jsonl


def test_fixture_covers_every_family_and_split_without_group_leakage(tmp_path) -> None:
    manifest = generate_search_fixture(tmp_path, task_count=21, seed=7)
    tasks = read_jsonl(tmp_path / "tasks.jsonl")

    assert manifest["research_evidence"] is False
    assert manifest["counts"]["split_tasks"] == {"train": 7, "dev": 7, "test": 7}
    for split in ("train", "dev", "test"):
        assert {row["category"] for row in tasks if row["split"] == split} == set(
            WORKFLOW_FAMILIES
        )
    for group_key in ("entity_group", "template_group", "source", "source_revision"):
        groups = {
            split: {
                row["provenance"][group_key] for row in tasks if row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        assert groups["train"].isdisjoint(groups["dev"])
        assert groups["train"].isdisjoint(groups["test"])
        assert groups["dev"].isdisjoint(groups["test"])


def test_degraded_source_is_a_complete_known_dead_end_not_an_unknown_loop(tmp_path) -> None:
    generate_search_fixture(tmp_path, task_count=21)
    tools = read_jsonl(tmp_path / "tools.jsonl")
    tasks = read_jsonl(tmp_path / "tasks.jsonl")
    snapshots = read_jsonl(tmp_path / "snapshots.jsonl")
    verifiers = {
        row["task_id"]: row["private_verifier"]
        for row in read_jsonl(tmp_path / "verifiers.private.jsonl")
    }
    task = next(
        row for row in tasks if row["provenance"]["primary_annotation_degraded"]
    )
    result = WeightedSearchOracle(
        ContractEngine(tools, snapshots),
        limits=SearchLimits(max_depth=14, max_states=20_000, top_k_paths=4),
    ).search(
        task["initial_state"],
        combine_verification_goal(task["goal"], verifiers[task["task_id"]]),
        task["available_tools"],
    )

    assert result.diagnostics.graph_complete
    assert result.initial_value.value_known
    assert result.paths and result.paths[0].certified_optimal
    assert "annotate_alternate" in result.paths[0].actions
    assert all(snapshot["status"] == "success" for snapshot in snapshots)


def test_hard_fixture_holds_out_compositions_and_has_source_fallbacks(tmp_path) -> None:
    manifest = generate_hard_search_fixture(tmp_path, task_count=60, seed=7)
    tasks = read_jsonl(tmp_path / "tasks.jsonl")
    tools = read_jsonl(tmp_path / "tools.jsonl")

    protocol = manifest["composition_protocol"]
    assert not (set(protocol["train"]) & set(protocol["dev"]))
    assert not (set(protocol["train"]) & set(protocol["test"]))
    assert not (set(protocol["dev"]) & set(protocol["test"]))
    assert manifest["invariants"]["held_out_finish_tools"] is True
    assert manifest["counts"]["split_tasks"] == {"train": 36, "dev": 12, "test": 12}
    assert len({row["provenance"]["template_group"] for row in tasks}) == 60
    assert any(
        row["provenance"]["primary_degraded"]["annotation"] for row in tasks
    )
    tool_ids = {row["tool_id"] for row in tools}
    assert "lookup_disease" in tool_ids
    assert "lookup_disease_alternate" in tool_ids
    for task in tasks:
        assert "quality_check_report" in task["available_tools"]
        assert len(task["initial_state"]["sequence_context"]) == 96
    assert len({task["initial_state"]["sequence_context"] for task in tasks}) == len(tasks)
    assert manifest["counts"]["unique_sequence_contexts"] == len(tasks)
    assert manifest["invariants"]["nonempty_task_unique_sequence_contexts"] is True
