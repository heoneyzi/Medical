from __future__ import annotations

from pathlib import Path

import pytest

from geoflowagent.cli import _quality_thresholds
from geoflowagent.data.quality import (
    DatasetQualityError,
    DatasetQualityThresholds,
    audit_dataset_quality,
    require_dataset_quality,
    summarize_search_supervision,
)
from geoflowagent.utils.io import read_yaml, sha256_file, write_json, write_jsonl


def _tool() -> dict:
    return {
        "tool_id": "finish",
        "name": "Finish",
        "description": "Finish a checked workflow.",
        "execution_mode": "symbolic",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "preconditions": [{"field": "ready", "op": "eq", "value": True}],
        "effects": [{"field": "done", "op": "set", "value": True}],
    }


def _task(index: int, split: str, *, query: str | None = None, entity: str | None = None) -> dict:
    return {
        "task_id": f"task-{index}",
        "query": query or f"Complete independently reviewed case {index}.",
        "initial_state": {"ready": True, "case": index},
        "goal": {"done": True},
        "available_tools": ["finish"],
        "split": split,
        "split_group": entity or f"entity-{index}",
        "provenance": {
            "synthetic": True,
            "workflow_family": f"family-{index}",
            "entity_group": entity or f"entity-{index}",
            "template_group": f"template-{index}",
            "topology_group": f"topology-{index}",
            "source": f"source-{index}",
            "source_revision": f"release-{index}",
        },
    }


def _verifier(index: int, value: str) -> dict:
    return {
        "task_id": f"task-{index}",
        "private_verifier": {
            "conditions": [{"field": "answer", "op": "eq", "value": value}]
        },
    }


def _write_raw(root: Path, tasks: list[dict], verifiers: list[dict]) -> None:
    write_jsonl(root / "tools.jsonl", [_tool()])
    write_jsonl(root / "tasks.jsonl", tasks)
    write_jsonl(root / "verifiers.private.jsonl", verifiers)


def test_quality_gate_passes_declared_tiny_fixture_and_flags_synthetic(tmp_path: Path) -> None:
    tasks = [_task(0, "train"), _task(1, "dev"), _task(2, "test")]
    for index, task in enumerate(tasks):
        task["initial_state"]["sequence_context"] = "ACGT" + "ACGT"[index]
    _write_raw(tmp_path, tasks, [_verifier(index, f"HIDDEN-{index}") for index in range(3)])
    thresholds = DatasetQualityThresholds(
        min_tasks=3,
        min_tasks_per_split={"train": 1, "dev": 1, "test": 1},
        min_workflow_families=3,
        min_entities=3,
        min_sequence_contexts=3,
        min_templates=3,
        min_topologies=3,
        min_sources=3,
        min_source_releases=3,
        min_verifier_coverage=1.0,
        split_group_keys=("case", "entity", "template", "source", "release"),
        require_complete_split_group_metadata=True,
    )

    report = audit_dataset_quality(tmp_path, thresholds=thresholds)

    assert report["passed"] is True
    assert report["diversity"]["sequence_contexts"]["count"] == 3
    assert report["split_integrity"]["groups"]["missing_group_ids"] == {}
    assert report["split_integrity"]["groups"]["require_complete_metadata"] is True
    assert report["evidence_status"]["pipeline_validation_only"] is True
    assert report["hashes"]["audit_input_sha256"]
    assert any(
        warning["code"] == "synthetic_pipeline_validation_only"
        for warning in report["warnings"]
    )
    require_dataset_quality(report)


def test_quality_gate_rejects_missing_sequence_diversity(tmp_path: Path) -> None:
    tasks = [_task(0, "train"), _task(1, "dev")]
    for task in tasks:
        task["initial_state"]["sequence_context"] = "ACGT"
    _write_raw(tmp_path, tasks, [_verifier(index, f"HIDDEN-{index}") for index in range(2)])

    report = audit_dataset_quality(
        tmp_path,
        thresholds=DatasetQualityThresholds(min_sequence_contexts=2),
    )

    assert report["passed"] is False
    failure = next(
        item
        for item in report["failures"]
        if item.get("details", {}).get("dimension") == "sequence_contexts"
    )
    assert failure["code"] == "insufficient_diversity"


def test_strict_split_group_metadata_fails_closed_but_permissive_mode_warns(
    tmp_path: Path,
) -> None:
    task = _task(0, "train")
    del task["provenance"]["template_group"]
    _write_raw(tmp_path, [task], [_verifier(0, "HIDDEN")])
    group_keys = ("case", "entity", "template", "source", "release")

    strict_report = audit_dataset_quality(
        tmp_path,
        thresholds=DatasetQualityThresholds(
            split_group_keys=group_keys,
            require_complete_split_group_metadata=True,
        ),
    )

    assert strict_report["passed"] is False
    failure = next(
        item
        for item in strict_report["failures"]
        if item["code"] == "split_group_metadata_missing"
    )
    assert failure["details"]["missing"] == {"template": ["task-0"]}
    assert strict_report["split_integrity"]["groups"]["ok"] is False

    permissive_report = audit_dataset_quality(
        tmp_path,
        thresholds=DatasetQualityThresholds(split_group_keys=group_keys),
    )

    assert permissive_report["passed"] is True
    assert any(
        warning["code"] == "split_group_metadata_missing"
        for warning in permissive_report["warnings"]
    )
    assert permissive_report["split_integrity"]["groups"]["ok"] is True


def test_real_search_config_requires_every_canonical_split_group() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "search_geomarrvel.yaml"
    thresholds = _quality_thresholds(read_yaml(config_path))

    assert thresholds.split_group_keys == (
        "case",
        "entity",
        "template",
        "source",
        "release",
    )
    assert thresholds.require_complete_split_group_metadata is True


def test_strict_group_metadata_audits_raw_inventory_when_processed_is_subset(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    tasks = [_task(0, "train"), _task(1, "dev"), _task(2, "test")]
    del tasks[2]["provenance"]["source_revision"]
    _write_raw(
        raw_dir,
        tasks,
        [_verifier(index, f"HIDDEN-{index}") for index in range(3)],
    )
    # Simulate a sealed development artifact that intentionally omits test rows.
    write_jsonl(processed_dir / "tasks.jsonl", tasks[:2])
    write_json(
        processed_dir / "manifest.json",
        {
            "processed_files": {
                "tasks.jsonl": sha256_file(processed_dir / "tasks.jsonl")
            }
        },
    )

    report = audit_dataset_quality(
        raw_dir,
        processed_dir,
        thresholds=DatasetQualityThresholds(
            split_group_keys=("case", "entity", "template", "source", "release"),
            require_complete_split_group_metadata=True,
        ),
    )

    group_audit = report["split_integrity"]["groups"]
    assert group_audit["inventory_source"] == "raw_tasks"
    assert group_audit["inventory_task_count"] == 3
    assert group_audit["missing_group_ids"] == {"release": ["task-2"]}


def test_quality_gate_catches_group_content_and_private_literal_leakage(tmp_path: Path) -> None:
    query = "The hidden answer is SECRET-GENE."
    left = _task(0, "train", query=query, entity="shared-entity")
    right = _task(1, "test", query=query, entity="shared-entity")
    # Preserve identical public content while IDs and declared splits differ.
    right["initial_state"] = dict(left["initial_state"])
    right["goal"] = dict(left["goal"])
    _write_raw(tmp_path, [left, right], [_verifier(0, "SECRET-GENE"), _verifier(1, "SECRET-GENE")])

    report = audit_dataset_quality(tmp_path)
    codes = {failure["code"] for failure in report["failures"]}

    assert report["passed"] is False
    assert "split_group_leakage" in codes
    assert "duplicate_content_across_splits" in codes
    assert "private_label_leakage" in codes


def test_search_summary_never_counts_unknown_action_as_negative() -> None:
    row = {
        "terminal": False,
        "top_k_paths": [
            {"tool_ids": ["good"], "total_cost": 1.0},
            {"tool_ids": ["unknown", "good"], "total_cost": 2.0},
        ],
        "action_supervision": [
            {
                "tool_id": "good",
                "known": True,
                "label_mask": True,
                "contract_applicable": True,
                "q_star": 1.0,
                "regret": 0.0,
                "is_optimal": True,
                "reachable": True,
            },
            {
                "tool_id": "unknown",
                "known": False,
                "label_mask": False,
                "contract_applicable": True,
                "q_star": None,
                "regret": None,
                "is_optimal": None,
                "reachable": None,
            },
            {
                "tool_id": "invalid",
                "known": True,
                "label_mask": True,
                "contract_applicable": False,
                "q_star": None,
                "regret": None,
                "is_optimal": False,
                "reachable": False,
            },
        ],
    }

    summary = summarize_search_supervision([row])

    assert summary["applicable_actions"] == 2
    assert summary["known_applicable_actions"] == 1
    assert summary["unknown_applicable_actions"] == 1
    assert summary["known_unreachable_applicable_actions"] == 0
    assert summary["unknown_is_not_negative"] is True
    assert summary["path_labeled_states"] == 1
    assert summary["multi_path_states"] == 1
    assert summary["multi_path_state_fraction"] == 1.0


def test_require_quality_raises_with_failure_codes(tmp_path: Path) -> None:
    _write_raw(tmp_path, [_task(0, "train")], [_verifier(0, "HIDDEN")])
    report = audit_dataset_quality(
        tmp_path,
        thresholds=DatasetQualityThresholds(min_tasks=10),
    )

    with pytest.raises(DatasetQualityError, match="too_few_tasks"):
        require_dataset_quality(report)


def test_protected_external_source_is_forced_to_test_split(tmp_path: Path) -> None:
    protected = _task(0, "train")
    protected["provenance"]["dataset_id"] = "hjeong84/marrvel-mcp-benchmark-data"
    _write_raw(tmp_path, [protected], [_verifier(0, "HIDDEN")])

    report = audit_dataset_quality(
        tmp_path,
        thresholds=DatasetQualityThresholds(
            test_only_sources=("hjeong84/marrvel-mcp-benchmark-data",)
        ),
    )

    assert report["passed"] is False
    assert any(
        failure["code"] == "test_only_source_leakage"
        for failure in report["failures"]
    )
    assert report["split_integrity"]["test_only_source_violations"] == [
        {
            "task_id": "task-0",
            "source": "hjeong84/marrvel-mcp-benchmark-data",
            "split": "train",
        }
    ]
