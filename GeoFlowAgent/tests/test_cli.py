from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import geoflowagent.cli as cli
import geoflowagent.training.flow as flow_training
import geoflowagent.training.metric as metric_training
from geoflowagent.cli import _append_test_access, build_parser, run_search_all
from geoflowagent.data.quality import DatasetQualityError
from geoflowagent.evaluation.embedding_metrics import evaluate_embeddings
from geoflowagent.training.features import FeatureStore
from geoflowagent.training.flow import train_flow_model
from geoflowagent.training.metric import train_metric_model


def test_exploration_commands_keep_test_sealed_by_default() -> None:
    parser = build_parser()

    embedding = parser.parse_args(["evaluate-embeddings", "--config", "config.yaml"])
    metric = parser.parse_args(["train-metric", "--config", "config.yaml"])
    flow = parser.parse_args(["train-flow", "--config", "config.yaml"])
    agent = parser.parse_args(["evaluate-agent", "--config", "config.yaml"])
    run_all = parser.parse_args(["run-all", "--config", "config.yaml"])
    value = parser.parse_args(["train-value", "--config", "config.yaml"])
    search_agent = parser.parse_args(["evaluate-search-agent", "--config", "config.yaml"])
    run_search = parser.parse_args(["run-search", "--config", "config.yaml"])
    prepare_search = parser.parse_args(["prepare-search", "--config", "config.yaml"])

    assert not embedding.include_test
    assert not metric.include_test
    assert not flow.include_test
    assert agent.split == "dev"
    assert run_all.evaluation_split == "dev"
    assert not value.include_test
    assert search_agent.split == "dev"
    assert run_search.evaluation_split == "dev"
    assert not prepare_search.include_test
    assert inspect.signature(evaluate_embeddings).parameters["include_test"].default is False
    assert inspect.signature(train_metric_model).parameters["include_test"].default is False
    assert inspect.signature(train_flow_model).parameters["include_test"].default is False


def test_test_unsealing_is_explicit_in_cli() -> None:
    parser = build_parser()

    embedding = parser.parse_args(
        ["evaluate-embeddings", "--config", "config.yaml", "--include-test"]
    )
    agent = parser.parse_args(["evaluate-agent", "--config", "config.yaml", "--split", "test"])
    run_all = parser.parse_args(
        ["run-all", "--config", "config.yaml", "--evaluation-split", "test"]
    )
    value = parser.parse_args(["train-value", "--config", "config.yaml", "--include-test"])
    search_agent = parser.parse_args(
        ["evaluate-search-agent", "--config", "config.yaml", "--split", "test"]
    )
    run_search = parser.parse_args(
        ["run-search", "--config", "config.yaml", "--evaluation-split", "test"]
    )
    prepare_search = parser.parse_args(
        ["prepare-search", "--config", "config.yaml", "--include-test"]
    )

    assert embedding.include_test
    assert agent.split == "test"
    assert run_all.evaluation_split == "test"
    assert value.include_test
    assert search_agent.split == "test"
    assert run_search.evaluation_split == "test"
    assert prepare_search.include_test


def test_search_command_surface_and_geometry_overrides() -> None:
    parser = build_parser()

    for command in ("generate-search-fixture", "prepare-search", "audit-search-data"):
        args = parser.parse_args([command, "--config", "config.yaml"])
        assert args.command == command
    geometry = parser.parse_args(
        [
            "compare-value-geometries",
            "--config",
            "config.yaml",
            "--energies",
            "cosine",
            "poincare",
            "--seeds",
            "17",
            "29",
        ]
    )
    assert geometry.energies == ["cosine", "poincare"]
    assert geometry.seeds == [17, 29]


def test_audit_search_data_persists_failure_before_failing(monkeypatch, tmp_path) -> None:
    report_path = tmp_path / "reports" / "quality.json"
    paths = {
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "quality_report": report_path,
    }
    report = {
        "passed": False,
        "failures": [{"code": "too_few_tasks", "message": "not enough tasks"}],
    }
    monkeypatch.setattr(cli, "_paths", lambda _: ({"data_quality": {}}, paths))
    (paths["processed_dir"]).mkdir(parents=True)
    (paths["processed_dir"] / "manifest.json").write_text(
        json.dumps({"oracle_metadata": {"active_splits": ["train", "dev"]}}),
        encoding="utf-8",
    )

    complete_report = {
        **report,
        "counts": {},
        "evidence_status": {},
        "split_integrity": {},
        "hashes": {"audit_input_sha256": "audit-input"},
    }
    monkeypatch.setattr(
        cli, "audit_dataset_quality", lambda *args, **kwargs: deepcopy(complete_report)
    )

    with pytest.raises(DatasetQualityError, match="too_few_tasks"):
        cli.audit_search_data(tmp_path / "config.yaml")

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["passed"] is False
    assert persisted["failures"][0] == report["failures"][0]
    assert persisted["audit_scope"]["test_unsealed"] is False
    assert persisted["raw_inventory"]["passed"] is False


def test_run_search_all_orders_gate_before_training_and_hashes_artifacts(
    monkeypatch, tmp_path
) -> None:
    paths = {
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "cache_dir": tmp_path / "cache",
        "quality_report": tmp_path / "reports" / "quality.json",
        "value_output_dir": tmp_path / "checkpoints" / "value",
        "search_agent_report_dir": tmp_path / "reports" / "agent",
        "run_summary": tmp_path / "run_summary.json",
    }
    config = {"synthetic_fixture": {"enabled": True}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic_fixture:\n  enabled: true\n", encoding="utf-8")
    calls: list[str] = []

    def write(path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(_):
        calls.append("fixture")
        result = {
            "research_evidence": False,
            "counts": {"tasks": 21},
            "content_sha256": "fixture-content",
        }
        write(paths["raw_dir"] / "fixture_manifest.json", result)
        return result

    def prepare(_, *, active_splits):
        calls.append(f"prepare:{','.join(active_splits)}")
        result = {"counts": {"tasks": 14, "examples": 67}}
        write(paths["processed_dir"] / "manifest.json", result)
        return result

    def audit(_, *, active_splits):
        calls.append(f"quality:{','.join(active_splits)}")
        result = {
            "passed": True,
            "counts": {"tasks": 14},
            "evidence_status": {"research_evidence": False},
        }
        write(paths["quality_report"], result)
        return result

    def embed(*_):
        calls.append("embed")
        result = {"counts": {"examples": 100}}
        write(paths["cache_dir"] / "manifest.json", result)
        return result

    def evaluate_embeddings_stub(*_, include_test=False):
        calls.append(f"embedding-eval:{include_test}")
        result = {
            "reported_splits": ["train", "dev"],
            "views": {"general": {}},
            "auxiliary_views": {"sequence_aux": {}},
            "cka": [],
        }
        output_dir = paths["quality_report"].parent / "embedding"
        write(output_dir / "embedding_report.json", result)
        return result

    def train(*_, energy_override=None, include_test=False):
        calls.append(f"train:{energy_override}:{include_test}")
        result = {
            "energy": energy_override or "directed_quasimetric",
            "cache_content_sha256": "cache-content",
            "best_epoch": 3,
            "best_dev_selection_score": 0.75,
            "parameter_count": 123,
            "selected_thresholds": {"completion": 0.5, "action_reachability": 0.5},
            "metrics": {"dev": {"joint_stop_action_accuracy": 0.8}},
        }
        checkpoint_dir = paths["value_output_dir"]
        write(checkpoint_dir / "value_geometry.pt", {"checkpoint": True})
        write(checkpoint_dir / "value_metrics.json", result)
        return result

    def evaluate(*_, split="dev"):
        calls.append(f"evaluate:{split}")
        result = {
            "conditions": [{"condition": "learned_policy", "success": {"mean": 1.0}}],
            "run_provenance": {"device": "cpu"},
        }
        write(paths["search_agent_report_dir"] / f"{split}_details.jsonl", result)
        write(paths["search_agent_report_dir"] / f"{split}_summary.json", result)
        return result

    monkeypatch.setattr(cli, "_paths", lambda _: (config, paths))
    monkeypatch.setattr(cli, "generate_search_fixture_from_config", fixture)
    monkeypatch.setattr(cli, "prepare_search_from_config", prepare)
    monkeypatch.setattr(cli, "audit_search_data", audit)
    monkeypatch.setattr(cli, "build_embedding_cache", embed)
    monkeypatch.setattr(cli, "evaluate_embeddings", evaluate_embeddings_stub)
    monkeypatch.setattr(cli, "train_value_geometry", train)
    monkeypatch.setattr(cli, "evaluate_search_agent", evaluate)

    result = run_search_all(config_path, energy="cosine")

    assert calls == [
        "fixture",
        "prepare:train,dev",
        "quality:train,dev",
        "embed",
        "embedding-eval:False",
        "train:cosine:False",
        "evaluate:dev",
    ]
    assert result["quality"]["passed"] is True
    assert result["selected_energy"] == "cosine"
    assert result["test_unsealed"] is False
    assert result["test_access"] is None
    assert set(result["artifacts"]) == {
        "fixture_manifest_sha256",
        "processed_manifest_sha256",
        "quality_report_sha256",
        "cache_manifest_sha256",
        "embedding_report_sha256",
        "value_checkpoint_sha256",
        "value_report_sha256",
        "agent_details_sha256",
        "agent_report_sha256",
    }
    assert json.loads(paths["run_summary"].read_text(encoding="utf-8")) == result


def test_test_access_ledger_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 17\n", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "tasks.jsonl").write_text('{"task_id":"held-out"}\n', encoding="utf-8")
    paths = {
        "raw_dir": raw_dir,
        "run_summary": tmp_path / "artifacts" / "run_summary.json",
    }

    first = _append_test_access(
        config_path, paths, command="run-search", reason="registered final evaluation"
    )
    second = _append_test_access(
        config_path, paths, command="evaluate-search-agent", reason="error analysis"
    )
    rows = [
        json.loads(line)
        for line in Path(first["ledger_path"]).read_text(encoding="utf-8").splitlines()
    ]

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert rows[0]["previous_event_sha256"] is None
    assert rows[1]["previous_event_sha256"] == rows[0]["event_sha256"]
    assert rows[0]["raw_tasks_sha256"] is not None
    assert rows[0]["event"] == "test_unsealed"


def _store_without_test_rows(processed_dir, cache_dir) -> FeatureStore:
    store = FeatureStore(processed_dir, cache_dir)
    original_indices = store.indices

    def sealed_indices(split: str, *, include_terminal: bool) -> list[int]:
        if split == "test":
            return []
        return original_indices(split, include_terminal=include_terminal)

    store.indices = sealed_indices  # type: ignore[method-assign]
    return store


def test_metric_training_rejects_explicitly_unsealed_empty_test_split(
    processed_dir, cache_dir, repository_root, monkeypatch, tmp_path
) -> None:
    store = _store_without_test_rows(processed_dir, cache_dir)
    monkeypatch.setattr(metric_training, "FeatureStore", lambda *args, **kwargs: store)

    with pytest.raises(ValueError, match="explicitly unsealed.*nonterminal test rows"):
        train_metric_model(
            processed_dir,
            cache_dir,
            tmp_path / "metric",
            repository_root / "configs" / "smoke.yaml",
            include_test=True,
        )


def test_flow_training_rejects_explicitly_unsealed_empty_test_split(
    processed_dir, cache_dir, repository_root, monkeypatch, tmp_path
) -> None:
    store = _store_without_test_rows(processed_dir, cache_dir)
    checkpoint = {
        "model_config": {"structured_dim": store.structured_dim, "shared_dim": 32},
    }
    monkeypatch.setattr(flow_training, "FeatureStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(
        flow_training,
        "load_metric_checkpoint",
        lambda *args, **kwargs: (torch.nn.Identity(), checkpoint),
    )

    with pytest.raises(ValueError, match="explicitly unsealed.*no test rows"):
        train_flow_model(
            processed_dir,
            cache_dir,
            tmp_path / "metric.pt",
            tmp_path / "flow",
            repository_root / "configs" / "smoke.yaml",
            include_test=True,
        )
