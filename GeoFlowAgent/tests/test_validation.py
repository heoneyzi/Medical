from __future__ import annotations

from pathlib import Path

import pytest

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.preprocess import prepare_dataset
from geoflowagent.data.schema import validate_minimal_pair, validate_task
from geoflowagent.utils.io import read_jsonl, write_jsonl


def test_model_facing_task_rejects_expected_answer_leakage() -> None:
    with pytest.raises(ValueError, match="model-leaking"):
        validate_task(
            {
                "task_id": "leaky",
                "query": "Find a gene",
                "initial_state": {"ready": False},
                "goal": {"ready": True},
                "available_tools": ["lookup"],
                "provenance": {"expected_answer": "PRIVATE_GOLD"},
            }
        )


def test_minimal_pair_requires_an_explicit_research_split() -> None:
    with pytest.raises(ValueError, match="missing required field 'split'"):
        validate_minimal_pair(
            {
                "pair_id": "pair-without-split",
                "relation": "invariant",
                "changed_fields": ["query.wording"],
                "left_text": "left",
                "right_text": "right",
            }
        )


def test_prepare_rejects_unknown_background_reference(
    smoke_raw_dir: Path,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("tools.jsonl", "trajectories.jsonl", "minimal_pairs.jsonl", "background.jsonl"):
        write_jsonl(raw / name, read_jsonl(smoke_raw_dir / name))
    tasks = read_jsonl(smoke_raw_dir / "tasks.jsonl")
    tasks[0]["background_ids"] = ["does-not-exist"]
    write_jsonl(raw / "tasks.jsonl", tasks)

    with pytest.raises(ValueError, match="unknown background"):
        prepare_dataset(raw, tmp_path / "processed")


def test_initially_satisfied_task_compiles_to_stop_only(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_jsonl(
        raw / "tools.jsonl",
        [
            {
                "tool_id": "set_ready",
                "name": "Set ready",
                "description": "Synthetic tool that is unnecessary for this task.",
                "preconditions": [{"field": "ready", "op": "neq", "value": True}],
                "effects": [{"field": "ready", "op": "set", "value": True}],
            }
        ],
    )
    write_jsonl(
        raw / "tasks.jsonl",
        [
            {
                "task_id": "already-ready",
                "query": "Stop because the typed goal is already met.",
                "initial_state": {"ready": True},
                "goal": {"ready": True},
                "available_tools": ["set_ready"],
                "split": "test",
            }
        ],
    )
    write_jsonl(raw / "trajectories.jsonl", [{"task_id": "already-ready", "tool_ids": []}])

    manifest = prepare_dataset(raw, tmp_path / "processed")
    examples = read_jsonl(tmp_path / "processed/examples.jsonl")

    assert manifest["counts"]["examples"] == 1
    assert examples[0]["terminal"] is True
    assert examples[0]["gold_next_tool"] == STOP_TOOL_ID
    assert examples[0]["gold_suffix_tool_ids"] == []
