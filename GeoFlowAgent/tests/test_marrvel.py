from __future__ import annotations

import json
from pathlib import Path

from geoflowagent.data.marrvel import finalize_curation, import_marrvel
from geoflowagent.utils.io import read_jsonl, write_jsonl


def test_marrvel_import_creates_curation_queue_without_inventing_a_plan(tmp_path: Path) -> None:
    source = tmp_path / "marrvel.json"
    source.write_text(
        json.dumps(
            [
                {
                    "index": 7,
                    "name": "Synthetic offline row",
                    "category": "Gene and Variant Utilities",
                    "input": "Which synthetic gene?",
                    "expected": "SYNTH_GENE",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "curation"
    manifest = import_marrvel(output, input_path=str(source))
    queue = read_jsonl(output / "marrvel_curation_queue.jsonl")

    assert manifest["rows"] == 1
    assert queue[0]["expected_answer"] == "SYNTH_GENE"
    assert queue[0]["initial_state"] == {}
    assert queue[0]["private_verifier"] == {}
    assert queue[0]["gold_tool_ids"] == []
    assert queue[0]["curation_status"].startswith("needs_")


def test_finalize_keeps_reference_answer_out_of_model_facing_task(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    raw_dir = tmp_path / "raw"
    write_jsonl(
        queue_path,
        [
            {
                "task_id": "terminal-case",
                "query": "The typed goal is already satisfied.",
                "category": "synthetic",
                "initial_state": {"ready": True},
                "goal": {"ready": True},
                "private_verifier": {"answer_code": "PRIVATE_ANSWER"},
                "available_tools": ["unused_tool"],
                "gold_tool_ids": [],
                "expected_answer": "PRIVATE_ANSWER",
                "curation_status": "validated",
                "provenance": {"source_index": 1},
            }
        ],
    )

    result = finalize_curation(queue_path, raw_dir)
    task = read_jsonl(raw_dir / "tasks.jsonl")[0]
    trajectory = read_jsonl(raw_dir / "trajectories.jsonl")[0]
    reference = read_jsonl(raw_dir / "reference_answers.private.jsonl")[0]
    verifier = read_jsonl(raw_dir / "verifiers.private.jsonl")[0]

    assert result["tasks"] == 1
    assert "expected_answer" not in json.dumps(task)
    assert trajectory["tool_ids"] == []
    assert reference["expected_answer"] == "PRIVATE_ANSWER"
    assert reference["visibility"] == "evaluation_only_not_model_input"
    assert verifier["private_verifier"] == {"answer_code": "PRIVATE_ANSWER"}
    assert "private_verifier" not in task


def test_finalize_requires_private_verifier_for_every_validated_row(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "task_id": "missing-verifier",
                "query": "Find the answer.",
                "category": "synthetic",
                "initial_state": {"ready": False},
                "goal": {"ready": True},
                "available_tools": ["lookup"],
                "gold_tool_ids": ["lookup"],
                "expected_answer": "PRIVATE",
                "curation_status": "validated",
            }
        ],
    )

    import pytest

    with pytest.raises(ValueError, match="private_verifier"):
        finalize_curation(queue_path, tmp_path / "raw")
