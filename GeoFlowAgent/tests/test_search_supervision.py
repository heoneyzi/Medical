from __future__ import annotations

import copy
from pathlib import Path

import pytest

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.data.procedural_search import generate_search_fixture
from geoflowagent.data.search_supervision import (
    audit_group_leakage,
    compile_search_supervision,
    model_input_view,
    read_search_dataset,
    task_balanced_split,
    validate_preassigned_splits,
    write_search_dataset,
)
from geoflowagent.search import EdgeCostConfig, SearchLimits, WeightedSearchOracle
from geoflowagent.utils.io import canonical_json, read_jsonl


def _tool(
    tool_id: str,
    *,
    preconditions: list[dict] | None = None,
    effects: list[dict] | None = None,
    search_cost: float | None = None,
) -> dict:
    row = {
        "tool_id": tool_id,
        "name": tool_id.replace("_", " ").title(),
        "description": f"Exact fixture action {tool_id}.",
        "preconditions": preconditions or [],
        "effects": effects or [],
    }
    if search_cost is not None:
        row["search_cost"] = search_cost
    return row


def _branching_fixture() -> tuple[dict, list[dict]]:
    tools = [
        _tool(
            "collect_a",
            preconditions=[{"field": "a", "op": "missing"}],
            effects=[{"field": "a", "op": "set", "value": True}],
        ),
        _tool(
            "collect_b",
            preconditions=[{"field": "b", "op": "missing"}],
            effects=[{"field": "b", "op": "set", "value": True}],
        ),
        _tool(
            "detour",
            preconditions=[{"field": "detour", "op": "missing"}],
            effects=[{"field": "detour", "op": "set", "value": True}],
            search_cost=2.0,
        ),
        _tool(
            "finish",
            preconditions=[
                {"field": "a", "op": "eq", "value": True},
                {"field": "b", "op": "eq", "value": True},
                {"field": "complete", "op": "missing"},
            ],
            effects=[{"field": "complete", "op": "set", "value": True}],
        ),
    ]
    task = {
        "task_id": "weighted-branching",
        "query": "Collect both evidence types and finish the report.",
        "initial_state": {},
        "goal": {"complete": True},
        "available_tools": [tool["tool_id"] for tool in tools],
        "split": "train",
        "group_ids": {
            "entity": "entity-1",
            "template": "template-1",
            "workflow": "workflow-1",
            "source": "source-1",
            "release": "release-1",
        },
        "provenance": {"source": "unit-test", "source_revision": "release-1"},
    }
    return task, tools


def _search(task: dict, tools: list[dict]):
    return WeightedSearchOracle(
        ContractEngine(tools),
        cost_config=EdgeCostConfig(call_cost=1.0),
        limits=SearchLimits(max_depth=4, max_states=100, top_k_paths=8),
    ).search(task["initial_state"], task["goal"], task["available_tools"])


def test_compiler_accepts_search_result_and_emits_cache_compatible_rows() -> None:
    task, tools = _branching_fixture()
    search_result = _search(task, tools)

    examples = compile_search_supervision(task, tools, [], search_result.to_dict())
    root = next(row for row in examples if row["state_id"] == search_result.initial_state_id)

    # Existing cache/training field contract remains available without a trajectory file.
    for field in (
        "query",
        "goal",
        "state",
        "next_state",
        "history",
        "candidate_tools",
        "contract_valid_tools",
        "valid_next_tools",
        "gold_next_tool",
        "gold_suffix_tool_ids",
        "background_ids",
        "provenance",
    ):
        assert field in root
    assert root["value_star"] == pytest.approx(3.0)
    assert root["valid_next_tools"] == ["collect_a", "collect_b"]
    assert root["gold_next_tool"] in root["valid_next_tools"]
    assert root["gold_suffix_tool_ids"][-1] == "finish"
    assert STOP_TOOL_ID not in root["gold_suffix_tool_ids"]
    assert len(root["top_k_paths"]) >= 2

    actions = {row["tool_id"]: row for row in root["action_supervision"]}
    assert actions["collect_a"]["q_star"] == pytest.approx(3.0)
    assert actions["collect_a"]["regret"] == pytest.approx(0.0)
    assert actions["detour"]["q_star"] == pytest.approx(6.0)
    assert actions["detour"]["regret"] == pytest.approx(3.0)
    assert actions["finish"]["contract_applicable"] is False
    assert root["action_known_mask"]["finish"] is False
    assert root["action_regret_mask"]["finish"] is True  # legacy invalid-action signal
    assert set(root["action_q"]) == set(task["available_tools"])
    assert set(root["action_successor_ids"]) == set(task["available_tools"])

    terminal = next(row for row in examples if row["terminal"])
    assert terminal["gold_next_tool"] == STOP_TOOL_ID
    assert terminal["gold_suffix_tool_ids"] == []
    assert terminal["value_star"] == 0.0


def test_unobserved_snapshot_is_null_and_masked_even_when_a_fallback_exists() -> None:
    snapshot_tool = {
        "tool_id": "snapshot_lookup",
        "name": "Snapshot lookup",
        "description": "Requires an exact offline response.",
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": ["case_id"],
            "properties": {"case_id": {"type": "string"}},
        },
        "argument_bindings": {"case_id": "case_id"},
        "output_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
        "output_bindings": {"answer": "answer"},
        "preconditions": [{"field": "case_id", "op": "nonempty"}],
        "effects": [],
    }
    fallback = _tool(
        "safe_fallback",
        preconditions=[{"field": "done", "op": "missing"}],
        effects=[{"field": "done", "op": "set", "value": True}],
    )
    task = {
        "task_id": "unknown-snapshot",
        "query": "Resolve this case.",
        "initial_state": {"case_id": "not-cached"},
        "goal": {"done": True},
        "available_tools": ["snapshot_lookup", "safe_fallback"],
        "split": "train",
    }
    result = WeightedSearchOracle(
        ContractEngine([snapshot_tool, fallback]),
        limits=SearchLimits(max_depth=2, top_k_paths=2),
    ).search(task["initial_state"], task["goal"], task["available_tools"])

    rows = compile_search_supervision(
        task,
        [snapshot_tool, fallback],
        [],
        result.to_dict(),
        include_uncertain=True,
    )
    root = next(row for row in rows if row["state_id"] == result.initial_state_id)
    unknown = next(
        action for action in root["action_supervision"] if action["tool_id"] == "snapshot_lookup"
    )

    assert root["training_mask"] is False
    assert root["value_known_mask"] is False
    assert unknown["known"] is False
    assert unknown["label_mask"] is False
    assert unknown["transition_known"] is False
    for field in (
        "executable",
        "successor",
        "successor_id",
        "cost",
        "q_star",
        "regret",
        "is_optimal",
        "reachable",
    ):
        assert unknown[field] is None
    assert root["action_known_mask"]["snapshot_lookup"] is False
    # A certified fallback Q remains usable even though root V*/regret are unknown.
    fallback_row = next(
        action for action in root["action_supervision"] if action["tool_id"] == "safe_fallback"
    )
    assert fallback_row["q_star"] == pytest.approx(1.0)
    assert fallback_row["regret"] is None


def test_compiler_checks_bellman_and_regret_consistency() -> None:
    task, tools = _branching_fixture()
    payload = _search(task, tools).to_dict()
    root_id = payload["initial_state_id"]
    root_value = next(row for row in payload["values"] if row["state_id"] == root_id)
    collect_a = next(row for row in root_value["actions"] if row["tool_id"] == "collect_a")
    collect_a["q_star"] += 0.5

    with pytest.raises(ValueError, match="Bellman inconsistency|Regret inconsistency"):
        compile_search_supervision(task, tools, [], payload)


def test_private_labels_are_not_copied_into_model_input() -> None:
    task, tools = _branching_fixture()
    payload = _search(task, tools).to_dict()
    payload["private_verifier"] = {
        "conditions": [{"field": "private_gene", "op": "eq", "value": "NEVER_COPY_ME"}]
    }
    payload["expected_answer"] = "NEVER_COPY_ME"

    rows = compile_search_supervision(task, tools, [], payload)
    assert all("NEVER_COPY_ME" not in canonical_json(model_input_view(row)) for row in rows)
    assert all("private_verifier" not in canonical_json(model_input_view(row)) for row in rows)

    leaked = copy.deepcopy(payload)
    leaked["states"][0]["state"]["private_verifier"] = {"answer": "leaked"}
    with pytest.raises(ValueError, match="evaluation-only"):
        compile_search_supervision(task, tools, [], leaked)


def test_group_audit_and_task_balanced_split_never_split_prefixes() -> None:
    rows: list[dict] = []
    for task_index, row_count in enumerate((1, 7, 2, 9, 3, 1)):
        entity = "shared-entity" if task_index in {0, 1} else f"entity-{task_index}"
        for state_index in range(row_count):
            rows.append(
                {
                    "example_id": f"task-{task_index}:state-{state_index}",
                    "task_id": f"task-{task_index}",
                    "group_ids": {
                        "case": [f"case-{task_index}"],
                        "entity": [entity],
                        "template": [f"template-{task_index}"],
                        "workflow": ["workflow"],
                        "source": ["source"],
                        "release": ["release"],
                    },
                    "split": None,
                }
            )

    assigned = task_balanced_split(
        rows,
        seed=23,
        ratios={"train": 0.5, "dev": 0.25, "test": 0.25},
        group_kinds=("case", "entity"),
    )
    splits_by_task: dict[str, set[str]] = {}
    for row in assigned:
        splits_by_task.setdefault(row["task_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_task.values())
    assert splits_by_task["task-0"] == splits_by_task["task-1"]
    assert audit_group_leakage(assigned, group_kinds=("case", "entity"))["ok"] is True

    conflicting = [
        {"task_id": "left", "split": "train", "group_ids": {"entity": ["same"]}},
        {"task_id": "right", "split": "test", "group_ids": {"entity": ["same"]}},
    ]
    with pytest.raises(ValueError, match="split leakage"):
        validate_preassigned_splits(conflicting, group_kinds=("entity",))


def test_write_read_manifest_is_deterministic_and_keeps_verifier_private(
    tmp_path: Path,
) -> None:
    task, tools = _branching_fixture()
    examples = compile_search_supervision(task, tools, [], _search(task, tools).to_dict())
    verifier = {
        "task_id": task["task_id"],
        "private_verifier": {
            "conditions": [{"field": "private.answer", "op": "eq", "value": "SECRET"}]
        },
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = write_search_dataset(
        first, [task], tools, examples, private_verifiers=[verifier], seed=17
    )
    manifest_b = write_search_dataset(
        second, [task], tools, examples, private_verifiers=[verifier], seed=17
    )

    assert manifest_a == manifest_b
    assert manifest_a["counts"]["private_verifiers"] == 1
    assert "verifiers.private.jsonl" in manifest_a["processed_files"]
    assert verify_processed_dataset(first) == manifest_a
    loaded = read_search_dataset(first)
    assert loaded["examples"] == examples
    assert loaded["private_verifiers"] == [verifier]
    assert all("SECRET" not in canonical_json(model_input_view(row)) for row in examples)


def test_procedural_fixture_search_compile_write_and_verify(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    generate_search_fixture(raw_dir, task_count=21, seed=17)
    tools = read_jsonl(raw_dir / "tools.jsonl")
    tasks = read_jsonl(raw_dir / "tasks.jsonl")
    snapshots = read_jsonl(raw_dir / "snapshots.jsonl")
    verifiers = read_jsonl(raw_dir / "verifiers.private.jsonl")
    background = read_jsonl(raw_dir / "background.jsonl")
    # Pick a finite, non-degraded case so the exact oracle can certify V*/Q*.
    task = next(
        row
        for row in tasks
        if row["category"] == "annotation"
        and row["provenance"]["primary_annotation_degraded"] is False
    )
    verifier = next(row for row in verifiers if row["task_id"] == task["task_id"])
    relevant_snapshots = [
        row for row in snapshots if row["snapshot_id"].startswith(f"{task['task_id']}:")
    ]
    verification_goal = combine_verification_goal(task["goal"], verifier["private_verifier"])
    result = WeightedSearchOracle(
        ContractEngine(tools, relevant_snapshots),
        limits=SearchLimits(max_depth=10, max_states=20_000, top_k_paths=4),
    ).search(task["initial_state"], verification_goal, task["available_tools"])
    assert result.diagnostics.graph_complete is True

    examples = compile_search_supervision(
        task, tools, relevant_snapshots, result.to_dict()
    )
    output_dir = tmp_path / "processed"
    write_search_dataset(
        output_dir,
        [task],
        tools,
        examples,
        relevant_snapshots,
        background=background,
        private_verifiers=[verifier],
    )
    loaded = read_search_dataset(output_dir)
    assert loaded["manifest"]["dataset_kind"] == "weighted_search_supervision"
    assert loaded["manifest"]["counts"]["known_action_labels"] > 0
    assert not (output_dir / "trajectories.jsonl").exists()

