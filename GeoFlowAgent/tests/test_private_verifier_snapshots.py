from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.preprocess import prepare_dataset, verify_processed_dataset
from geoflowagent.data.serialize import serialize_goal, serialize_history, serialize_state
from geoflowagent.evaluation.environment import SymbolicGenomicsEnvironment
from geoflowagent.utils.io import read_json, read_jsonl, write_jsonl


def _write_snapshot_fixture(raw: Path, *, expected_code: str = "ANSWER-7") -> None:
    raw.mkdir()
    write_jsonl(
        raw / "tools.jsonl",
        [
            {
                "tool_id": "pinned_lookup",
                "name": "Pinned lookup",
                "description": "Read a deterministic reviewed fixture.",
                "execution_mode": "snapshot",
                "input_schema": {
                    "type": "object",
                    "required": ["case_id"],
                    "properties": {"case_id": {"type": "string"}},
                    "additionalProperties": False,
                },
                "argument_bindings": {"case_id": "case_id"},
                "output_schema": {
                    "type": "object",
                    "required": ["answer_code"],
                    "properties": {"answer_code": {"type": "string"}},
                    "additionalProperties": False,
                },
                "output_bindings": {"answer_code": "result.answer_code"},
                "preconditions": [{"field": "report_ready", "op": "neq", "value": True}],
                "effects": [{"field": "report_ready", "op": "set", "value": True}],
            },
            {
                "tool_id": "unobserved_alternative",
                "name": "Unobserved alternative",
                "description": "An applicable edge with no trusted counterfactual outcome.",
                "preconditions": [{"field": "report_ready", "op": "neq", "value": True}],
                "effects": [{"field": "report_ready", "op": "set", "value": True}],
            },
        ],
    )
    write_jsonl(
        raw / "tasks.jsonl",
        [
            {
                "task_id": "private-case",
                "query": "Retrieve the reviewed result.",
                "initial_state": {"case_id": "case-7", "report_ready": False},
                "goal": {"report_ready": True},
                "available_tools": ["pinned_lookup", "unobserved_alternative"],
                "split": "test",
            }
        ],
    )
    write_jsonl(
        raw / "trajectories.jsonl",
        [{"task_id": "private-case", "tool_ids": ["pinned_lookup"]}],
    )
    write_jsonl(
        raw / "verifiers.private.jsonl",
        [
            {
                "task_id": "private-case",
                "private_verifier": {"result.answer_code": expected_code},
            }
        ],
    )
    write_jsonl(
        raw / "snapshots.jsonl",
        [
            {
                "snapshot_id": "lookup-case-7-v1",
                "tool_id": "pinned_lookup",
                "arguments": {"case_id": "case-7"},
                "status": "success",
                "output": {"answer_code": "ANSWER-7"},
                "source_revision": "synthetic-fixture-v1",
            }
        ],
    )


def test_private_verifier_drives_replay_but_never_enters_public_goal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _write_snapshot_fixture(raw)

    def forbidden_counterfactual(*args: object, **kwargs: object) -> object:
        raise AssertionError("observed_only must not invoke counterfactual graph search")

    monkeypatch.setattr(ContractEngine, "action_analysis", forbidden_counterfactual)
    manifest = prepare_dataset(
        raw,
        processed,
        counterfactual_policy="observed_only",
        require_optimal_demo=False,
    )
    examples = read_jsonl(processed / "examples.jsonl")

    assert manifest["counts"]["private_verifiers"] == 1
    assert manifest["counts"]["snapshots"] == 1
    assert "verifiers.private.jsonl" in manifest["processed_files"]
    assert examples[0]["goal"] == {"report_ready": True}
    assert examples[-1]["goal"] == {"report_ready": True}
    assert "private_verifier" not in json.dumps(examples)
    initial_model_text = "\n".join(
        [
            examples[0]["query"],
            serialize_goal(examples[0]["goal"]),
            serialize_state(examples[0]["state"]),
            serialize_history(examples[0]["history"]),
        ]
    )
    assert "ANSWER-7" not in initial_model_text
    assert examples[0]["action_outcome_status"]["unobserved_alternative"] == "unobserved"
    assert examples[0]["action_regret"]["unobserved_alternative"] is None
    assert examples[0]["action_regret_mask"]["unobserved_alternative"] is False
    assert examples[-1]["state"]["result"]["answer_code"] == "ANSWER-7"
    assert examples[-1]["history"][0]["status"] == "success"
    assert examples[-1]["history"][0]["output"] == {"answer_code": "ANSWER-7"}
    assert "ANSWER-7" in serialize_history(examples[-1]["history"])
    assert (
        read_json(processed / "manifest.json")["invariants"]["private_verifier_in_model_input"]
        is False
    )


def test_private_verifier_mismatch_rejects_publicly_complete_trajectory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_snapshot_fixture(raw, expected_code="DIFFERENT-ANSWER")

    with pytest.raises(ValueError, match="does not satisfy goal"):
        prepare_dataset(
            raw,
            tmp_path / "processed",
            counterfactual_policy="observed_only",
            require_optimal_demo=False,
        )


def test_reusing_processed_directory_cannot_retain_a_stale_private_verifier(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _write_snapshot_fixture(raw)
    options = {"counterfactual_policy": "observed_only", "require_optimal_demo": False}
    prepare_dataset(raw, processed, **options)
    assert (processed / "verifiers.private.jsonl").exists()

    (raw / "verifiers.private.jsonl").unlink()
    manifest = prepare_dataset(raw, processed, **options)

    assert not (processed / "verifiers.private.jsonl").exists()
    assert "verifiers.private.jsonl" not in manifest["processed_files"]

    write_jsonl(
        processed / "verifiers.private.jsonl",
        [{"task_id": "private-case", "private_verifier": {"result.answer_code": "STALE"}}],
    )
    with pytest.raises(ValueError, match="unbound generated files"):
        verify_processed_dataset(processed)


def test_evaluation_stop_uses_combined_private_verification_goal() -> None:
    task = {
        "task_id": "verify-stop",
        "query": "Finish the public workflow.",
        "initial_state": {"report_ready": True, "result": {"answer_code": "WRONG"}},
        "goal": {"report_ready": True},
        "available_tools": ["unused"],
    }
    tool = {
        "tool_id": "unused",
        "name": "Unused",
        "description": "Unused symbolic tool.",
        "preconditions": [],
        "effects": [],
    }
    verification_goal = combine_verification_goal(task["goal"], {"result.answer_code": "RIGHT"})
    environment = SymbolicGenomicsEnvironment(task, [tool], verification_goal=verification_goal)

    result = environment.step(STOP_TOOL_ID)

    assert result.goal_satisfied is False
    assert result.valid_action is False
    assert result.execution_success is False
    assert result.observation["summary"] == "premature stop"
