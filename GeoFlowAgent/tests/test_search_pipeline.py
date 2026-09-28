from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from geoflowagent.cli import audit_search_data
from geoflowagent.data.procedural_search import generate_search_fixture
from geoflowagent.data.search_pipeline import (
    _deduplicate_exact_snapshots,
    prepare_search_dataset,
)
from geoflowagent.embeddings.cache import build_embedding_cache
from geoflowagent.search import WeightedSearchOracle
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.utils.io import read_json, read_jsonl, write_jsonl


def _snapshot(snapshot_id: str) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "tool_id": "lookup",
        "arguments": {"assembly": "GRCh38", "variant": "1-100-A-G"},
        "status": "success",
        "output": {"gene": "GENE1"},
        "source_revision": "fixture-v1",
    }


def test_exact_snapshot_dedup_uses_canonical_call_and_rejects_conflicts() -> None:
    first = _snapshot("snapshot-a")
    duplicate = copy.deepcopy(first)
    duplicate["snapshot_id"] = "snapshot-b"
    # Mapping insertion order must not create a second executor key.
    duplicate["arguments"] = {"variant": "1-100-A-G", "assembly": "GRCh38"}
    duplicate["summary"] = "Equivalent normalized record from another case."

    assert _deduplicate_exact_snapshots([duplicate, first]) == [first]

    conflicting = copy.deepcopy(duplicate)
    conflicting["output"] = {"gene": "GENE2"}
    with pytest.raises(ValueError, match="Conflicting exact snapshots"):
        _deduplicate_exact_snapshots([first, conflicting])


def _write_known_dead_end_corpus(raw_dir: Path) -> None:
    tool = {
        "tool_id": "collect_dead_end",
        "name": "Collect dead-end evidence",
        "description": "Produces known evidence that cannot satisfy the requested goal.",
        "preconditions": [{"field": "checked", "op": "missing"}],
        "effects": [{"field": "checked", "op": "set", "value": True}],
    }
    task = {
        "task_id": "known-dead-end",
        "query": "Find a verified answer if one exists.",
        "initial_state": {},
        "goal": {"done": True},
        "available_tools": [tool["tool_id"]],
        "split": "train",
        "group_ids": {
            "case": "case-dead-end",
            "entity": "entity-dead-end",
            "template": "template-dead-end",
            "source": "source-dead-end",
            "release": "release-dead-end",
        },
        "provenance": {"source": "unit-test", "source_revision": "fixture-v1"},
    }
    verifier = {
        "task_id": task["task_id"],
        "private_verifier": {
            "conditions": [{"field": "verified", "op": "eq", "value": True}]
        },
    }
    write_jsonl(raw_dir / "tools.jsonl", [tool])
    write_jsonl(raw_dir / "tasks.jsonl", [task])
    write_jsonl(raw_dir / "verifiers.private.jsonl", [verifier])


def test_pipeline_preserves_known_unreachable_null_values_for_reachability(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_known_dead_end_corpus(raw_dir)

    manifest = prepare_search_dataset(
        raw_dir,
        processed_dir,
        max_depth=2,
        max_states=16,
        require_reachable_root=False,
        include_unreachable=True,
    )
    examples = read_jsonl(processed_dir / "examples.jsonl")

    assert manifest["oracle_metadata"]["all_graphs_complete"] is True
    assert len(examples) == 2
    assert all(row["value_known_mask"] is True for row in examples)
    assert all(row["goal_reachable"] is False for row in examples)
    assert all(row["value_star"] is None for row in examples)
    assert all(row["training_mask"] is False for row in examples)

    # Exercise target tensorization without constructing an embedding cache. A
    # known dead end has no scalar V* regression target, but it remains a valid
    # negative example for the separately masked state-reachability head.
    store = SearchFeatureStore.__new__(SearchFeatureStore)
    store.examples = examples
    store.tool_ids = ["collect_dead_end"]
    store.tool_index = {"collect_dead_end": 0}
    store._state_index = store._build_state_index()
    store._state_payload_index = store._build_state_payload_index()
    targets = store._tensorize_targets()

    assert not targets["value_mask"].any()
    assert torch.isnan(targets["value_target"]).all()
    assert targets["state_reachability_mask"].all()
    assert torch.equal(
        targets["state_reachability_target"], torch.zeros(len(examples))
    )


def test_exact_search_task_shards_resume_without_repeating_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    progress_dir = tmp_path / "oracle-progress"
    _write_known_dead_end_corpus(raw_dir)
    kwargs = {
        "max_depth": 2,
        "max_states": 16,
        "require_reachable_root": False,
        "include_unreachable": True,
        "progress_dir": progress_dir,
    }
    prepare_search_dataset(raw_dir, tmp_path / "processed-first", **kwargs)

    def fail_if_repeated(*args, **kwargs):
        del args, kwargs
        raise AssertionError("completed task oracle should have been reused")

    monkeypatch.setattr(WeightedSearchOracle, "search", fail_if_repeated)
    prepare_search_dataset(raw_dir, tmp_path / "processed-resumed", **kwargs)

    assert read_jsonl(tmp_path / "processed-first" / "examples.jsonl") == read_jsonl(
        tmp_path / "processed-resumed" / "examples.jsonl"
    )


def test_pipeline_requires_exact_private_verifier_coverage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_known_dead_end_corpus(raw_dir)
    write_jsonl(raw_dir / "verifiers.private.jsonl", [])

    with pytest.raises(ValueError, match="coverage must be exactly one per task"):
        prepare_search_dataset(
            raw_dir,
            tmp_path / "processed",
            require_reachable_root=False,
        )


def test_sealed_development_artifacts_physically_exclude_test_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    cache_dir = tmp_path / "cache"
    generate_search_fixture(raw_dir, task_count=21, seed=17)

    manifest = prepare_search_dataset(
        raw_dir,
        processed_dir,
        max_depth=14,
        top_k_paths=4,
        active_splits=("train", "dev"),
    )
    config_path = tmp_path / "config.yaml"
    quality_report_path = tmp_path / "reports" / "quality.json"
    config = {
        "project_root": ".",
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "cache_dir": str(cache_dir),
            "quality_report": str(quality_report_path),
            "run_summary": str(tmp_path / "run_summary.json"),
        },
        "data_quality": {
            "min_tasks": 21,
            "min_tasks_per_split": {"train": 7, "dev": 7, "test": 7},
            "min_verifier_coverage": 1.0,
            "require_search_supervision": True,
            "allow_synthetic": True,
            "scope_overrides": {
                "sealed_development": {
                    "min_tasks": 14,
                    "min_tasks_per_split": {"train": 7, "dev": 7},
                }
            },
        },
        "embedding": {
            "batch_size": 128,
            "storage_dtype": "float32",
            "views": {
                "frozen_hash": {
                    "kind": "prototype",
                    "query_encoder": {
                        "backend": "hash",
                        "dim": 16,
                        "salt": "sealed-unit-test",
                    },
                    "example_fields": ["query", "goal", "state", "history"],
                    "tool_fields": ["description", "contract"],
                }
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cache_manifest = build_embedding_cache(processed_dir, cache_dir, config_path)
    quality_report = audit_search_data(config_path)
    tasks = read_jsonl(processed_dir / "tasks.jsonl")
    examples = read_jsonl(processed_dir / "examples.jsonl")
    snapshots = read_jsonl(processed_dir / "snapshots.jsonl")

    assert manifest["oracle_metadata"]["active_splits"] == ["train", "dev"]
    assert manifest["oracle_metadata"]["raw_inventory"]["split_tasks"]["test"] == 7
    assert {row["split"] for row in tasks} == {"train", "dev"}
    assert {row["split"] for row in examples} == {"train", "dev"}
    assert all(row["provenance"]["split"] != "test" for row in snapshots)
    assert manifest["counts"]["split_tasks"]["test"] == 0
    assert cache_manifest["counts"]["examples"] == len(examples)
    assert cache_manifest["indices"]["examples"] == [row["example_id"] for row in examples]
    assert all("search-test-" not in value for value in cache_manifest["indices"]["examples"])
    assert read_json(cache_dir / "manifest.json") == cache_manifest
    assert quality_report["audit_scope"] == {
        "active_splits": ["train", "dev"],
        "test_unsealed": False,
        "search_metrics_exclude_inactive_splits": True,
    }
    assert quality_report["counts"]["split_tasks"]["test"] == 0
    assert quality_report["raw_inventory"]["counts"]["split_tasks"]["test"] == 7
    assert quality_report["search_supervision"]["states"] == len(examples)
