from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from geoflowagent.data.marrvel import finalize_curation, import_marrvel
from geoflowagent.data.preprocess import prepare_dataset
from geoflowagent.data.procedural_search import generate_search_fixture
from geoflowagent.data.procedural_search_hard import generate_hard_search_fixture
from geoflowagent.data.quality import (
    DatasetQualityThresholds,
    audit_dataset_quality,
    require_dataset_quality,
)
from geoflowagent.data.search_pipeline import prepare_search_dataset
from geoflowagent.embeddings.cache import build_embedding_cache
from geoflowagent.evaluation.agent import evaluate_agent
from geoflowagent.evaluation.embedding_metrics import evaluate_embeddings
from geoflowagent.evaluation.search_agent import evaluate_search_agent
from geoflowagent.training.dagger import run_search_dagger
from geoflowagent.training.flow import train_flow_model
from geoflowagent.training.metric import train_metric_model
from geoflowagent.training.state_flow import (
    evaluate_search_state_flow_checkpoint,
    train_search_state_flow,
)
from geoflowagent.training.value import (
    compare_existing_value_run_groups,
    compare_value_capacities,
    compare_value_geometries,
    compare_value_input_ablations,
    train_value_geometry,
)
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_path,
    sha256_file,
    sha256_text,
    write_json,
)

SEARCH_SPLITS = ("train", "dev", "test")
SEALED_DEVELOPMENT_SPLITS = ("train", "dev")
TEST_ACCESS_LEDGER_VERSION = "geoflowagent.test-access.v1"


def _paths(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = Path(config_path).resolve()
    config = read_yaml(config_path)
    root = resolve_path(config_path.parent, config.get("project_root", ".")).resolve()
    paths = {key: resolve_path(root, value) for key, value in config.get("paths", {}).items()}
    return config, paths


def _embedding_report_dir(paths: Mapping[str, Path]) -> Path:
    configured = paths.get("embedding_report_dir")
    if configured is not None:
        return configured
    quality_report = paths.get("quality_report")
    if quality_report is not None:
        return quality_report.parent / "embedding"
    return paths["cache_dir"].parent / "reports" / "embedding"


def _search_prepare_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the checked-in search config into the explicit search API."""

    section = config.get("search_preprocessing", {})
    if not isinstance(section, Mapping):
        raise ValueError("search_preprocessing must be a mapping")
    costs = section.get("costs", {})
    if not isinstance(costs, Mapping):
        raise ValueError("search_preprocessing.costs must be a mapping")
    status_costs = costs.get("status", {})
    if not isinstance(status_costs, Mapping):
        raise ValueError("search_preprocessing.costs.status must be a mapping")
    split_group_kinds = section.get(
        "split_group_kinds", ("case", "entity", "template", "source", "release")
    )
    if isinstance(split_group_kinds, str) or not isinstance(split_group_kinds, Sequence):
        raise ValueError("search_preprocessing.split_group_kinds must be a sequence")
    return {
        "seed": int(config.get("seed", 17)),
        "max_depth": int(section.get("max_depth", 16)),
        "max_states": int(section.get("max_states", 20_000)),
        "top_k_paths": int(section.get("top_k_paths", 8)),
        "near_optimal_slack": float(section.get("near_optimal_slack", 0.0)),
        "call_cost": float(costs.get("call", section.get("call_cost", 1.0))),
        "failure_cost": float(costs.get("failure", section.get("failure_cost", 1.0))),
        "redundancy_cost": float(costs.get("redundancy", section.get("redundancy_cost", 0.25))),
        "status_costs": {str(key): float(value) for key, value in status_costs.items()},
        "include_unreachable": bool(section.get("include_unreachable", True)),
        "require_complete_graph": bool(section.get("require_complete_graph", True)),
        "require_reachable_root": bool(section.get("require_reachable_root", True)),
        "split_group_kinds": tuple(str(value) for value in split_group_kinds),
    }


def _quality_thresholds(
    config: Mapping[str, Any],
    *,
    scope: str | None = None,
) -> DatasetQualityThresholds:
    section = config.get("data_quality", {})
    if not isinstance(section, Mapping):
        raise ValueError("data_quality must be a mapping")
    values = dict(section)
    raw_overrides = values.pop("scope_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("data_quality.scope_overrides must be a mapping")
    if scope is not None:
        override = raw_overrides.get(scope, {})
        if not isinstance(override, Mapping):
            raise ValueError(f"data_quality.scope_overrides.{scope} must be a mapping")
        values.update(dict(override))
    for field_name in ("split_group_keys", "test_only_sources"):
        if field_name not in values:
            continue
        sequence = values[field_name]
        if isinstance(sequence, str) or not isinstance(sequence, Sequence):
            raise ValueError(f"data_quality.{field_name} must be a sequence")
        values[field_name] = tuple(str(value) for value in sequence)
    try:
        return DatasetQualityThresholds(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid data_quality setting: {exc}") from exc


def _inventory_thresholds(config: Mapping[str, Any]) -> DatasetQualityThresholds:
    """Keep corpus-level gates while disabling unavailable search-state metrics."""

    thresholds = _quality_thresholds(config)
    return replace(
        thresholds,
        require_search_supervision=False,
        min_known_applicable_action_fraction=0.0,
        max_unknown_applicable_action_fraction=1.0,
        min_branch_state_fraction=0.0,
        min_branch_label_coverage=0.0,
        min_nontrivial_regret_fraction=0.0,
        min_multi_path_state_fraction=0.0,
        max_oracle_contract_collapse_fraction=1.0,
        min_search_complete_fraction=0.0,
        max_incomplete_or_unknown_state_fraction=1.0,
    )


def _scope_for_splits(active_splits: Sequence[str]) -> str:
    normalized = tuple(dict.fromkeys(str(value) for value in active_splits))
    if normalized == SEALED_DEVELOPMENT_SPLITS:
        return "sealed_development"
    if normalized == SEARCH_SPLITS:
        return "test_unsealed"
    return "_".join(normalized)


def _ledger_path(paths: Mapping[str, Path]) -> Path:
    configured = paths.get("test_access_ledger")
    return configured or (paths["run_summary"].parent / "test_access_ledger.jsonl")


def _append_test_access(
    config_path: str | Path,
    paths: Mapping[str, Path],
    *,
    command: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Append one verified hash-chain entry before any test artifact is opened."""

    ledger_path = _ledger_path(paths)
    rows = read_jsonl(ledger_path) if ledger_path.exists() else []
    expected_previous = None
    for index, row in enumerate(rows):
        supplied_hash = row.get("event_sha256")
        payload = {key: value for key, value in row.items() if key != "event_sha256"}
        if row.get("schema_version") != TEST_ACCESS_LEDGER_VERSION:
            raise ValueError(f"Unsupported test access ledger row {index}")
        if row.get("sequence") != index + 1:
            raise ValueError(f"Test access ledger sequence is broken at row {index + 1}")
        if row.get("previous_event_sha256") != expected_previous:
            raise ValueError(f"Test access ledger hash chain is broken at row {index + 1}")
        if supplied_hash != sha256_text(canonical_json(payload)):
            raise ValueError(f"Test access ledger event hash is invalid at row {index + 1}")
        expected_previous = str(supplied_hash)

    resolved_config = Path(config_path).resolve()
    raw_tasks = paths.get("raw_dir", Path()) / "tasks.jsonl"
    normalized_reason = (reason or "explicit final test evaluation").strip()
    if not normalized_reason:
        raise ValueError("test access reason cannot be empty")
    event: dict[str, Any] = {
        "schema_version": TEST_ACCESS_LEDGER_VERSION,
        "sequence": len(rows) + 1,
        "event_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "test_unsealed",
        "command": command,
        "reason": normalized_reason,
        "active_splits": list(SEARCH_SPLITS),
        "experiment_config_sha256": sha256_file(resolved_config),
        "raw_tasks_sha256": sha256_file(raw_tasks) if raw_tasks.is_file() else None,
        "previous_event_sha256": expected_previous,
    }
    event["event_sha256"] = sha256_text(canonical_json(event))
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (canonical_json(event) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "ledger_path": str(ledger_path),
        "sequence": event["sequence"],
        "event_id": event["event_id"],
        "event_sha256": event["event_sha256"],
    }


def generate_search_fixture_from_config(config_path: str | Path) -> dict[str, Any]:
    config, paths = _paths(config_path)
    section = config.get("synthetic_fixture", {})
    if not isinstance(section, Mapping):
        raise ValueError("synthetic_fixture must be a mapping")
    difficulty = str(section.get("difficulty", "standard_v1"))
    generator = {
        "standard_v1": generate_search_fixture,
        "hard_v2": generate_hard_search_fixture,
    }.get(difficulty)
    if generator is None:
        raise ValueError(
            "synthetic_fixture.difficulty must be 'standard_v1' or 'hard_v2'"
        )
    return generator(
        paths["raw_dir"],
        task_count=int(section.get("task_count", 42)),
        seed=int(section.get("seed", config.get("seed", 17))),
    )


def prepare_search_from_config(
    config_path: str | Path,
    *,
    active_splits: Sequence[str] = SEALED_DEVELOPMENT_SPLITS,
) -> dict[str, Any]:
    config, paths = _paths(config_path)
    return prepare_search_dataset(
        paths["raw_dir"],
        paths["processed_dir"],
        active_splits=active_splits,
        progress_backup_path=paths.get("search_prepare_progress_backup"),
        **_search_prepare_kwargs(config),
    )


def audit_search_data(
    config_path: str | Path,
    *,
    active_splits: Sequence[str] = SEALED_DEVELOPMENT_SPLITS,
) -> dict[str, Any]:
    """Audit raw inventory separately from active search-supervision rows."""

    config, paths = _paths(config_path)
    normalized = tuple(dict.fromkeys(str(value) for value in active_splits))
    manifest = read_json(paths["processed_dir"] / "manifest.json")
    prepared_splits = tuple(manifest.get("oracle_metadata", {}).get("active_splits", []))
    if prepared_splits != normalized:
        raise ValueError(
            f"Processed active_splits={list(prepared_splits)} do not match requested "
            f"active_splits={list(normalized)}"
        )

    # Inventory audit may inspect task/split metadata and source integrity for the
    # whole registered corpus, but it has no processed examples and therefore no
    # V*, Q*, regret, or top-k values from sealed test tasks.
    inventory = audit_dataset_quality(
        paths["raw_dir"],
        None,
        thresholds=_inventory_thresholds(config),
    )
    # Search-label statistics are computed from the physically filtered active
    # package only. Passing it as both raw and processed roots prevents fallback
    # to excluded raw tasks/verifiers/snapshots inside the generic auditor.
    report = audit_dataset_quality(
        paths["processed_dir"],
        paths["processed_dir"],
        thresholds=_quality_thresholds(config, scope=_scope_for_splits(normalized)),
    )
    report["audit_scope"] = {
        "active_splits": list(normalized),
        "test_unsealed": "test" in normalized,
        "search_metrics_exclude_inactive_splits": True,
    }
    report["raw_inventory"] = {
        "passed": inventory["passed"],
        "counts": inventory["counts"],
        "failures": inventory["failures"],
        "split_integrity": inventory["split_integrity"],
        "audit_input_sha256": inventory["hashes"]["audit_input_sha256"],
    }
    if not inventory["passed"]:
        report["passed"] = False
        report["failures"].append(
            {
                "code": "raw_inventory_gate_failed",
                "message": "Full-corpus task/split/source inventory failed before active search use",
                "details": {
                    "failure_codes": sorted(str(row.get("code")) for row in inventory["failures"])
                },
            }
        )
    write_json(paths["quality_report"], report)
    require_dataset_quality(report)
    return report


def run_search_all(
    config_path: str | Path,
    *,
    energy: str | None = None,
    evaluation_split: str = "dev",
    test_access_reason: str | None = None,
) -> dict[str, Any]:
    """Run the fail-closed search-distillation pipeline and bind all artifacts."""

    if evaluation_split not in {"dev", "test"}:
        raise ValueError("evaluation_split must be 'dev' or explicitly unsealed 'test'")
    resolved_config_path = Path(config_path).resolve()
    config, paths = _paths(resolved_config_path)
    dagger_section = config.get("search_dagger", {})
    if not isinstance(dagger_section, Mapping):
        raise ValueError("search_dagger must be a mapping")
    if evaluation_split == "test" and bool(dagger_section.get("enabled", False)):
        raise ValueError(
            "A DAgger-selected study must not retrain inside run-search test mode. "
            "Materialize a separate final processed/cache with prepare-search and embed, "
            "then call evaluate-search-agent --split test --checkpoint with the frozen "
            "development checkpoint."
        )
    fixture_section = config.get("synthetic_fixture", {})
    if not isinstance(fixture_section, Mapping):
        raise ValueError("synthetic_fixture must be a mapping")
    fixture_manifest = None
    if bool(fixture_section.get("enabled", False)):
        fixture_manifest = generate_search_fixture_from_config(resolved_config_path)

    active_splits = SEARCH_SPLITS if evaluation_split == "test" else SEALED_DEVELOPMENT_SPLITS
    test_access = (
        _append_test_access(
            resolved_config_path,
            paths,
            command="run-search",
            reason=test_access_reason,
        )
        if evaluation_split == "test"
        else None
    )
    prepare_manifest = prepare_search_from_config(resolved_config_path, active_splits=active_splits)
    quality_report = audit_search_data(resolved_config_path, active_splits=active_splits)
    cache_manifest = build_embedding_cache(
        paths["processed_dir"], paths["cache_dir"], resolved_config_path
    )
    embedding_report_dir = _embedding_report_dir(paths)
    embedding_report = evaluate_embeddings(
        paths["processed_dir"],
        paths["cache_dir"],
        embedding_report_dir,
        include_test=evaluation_split == "test",
    )
    value_result = train_value_geometry(
        paths["processed_dir"],
        paths["cache_dir"],
        paths["value_output_dir"],
        resolved_config_path,
        energy_override=energy,
        include_test=evaluation_split == "test",
    )
    checkpoint_path = paths["value_output_dir"] / "value_geometry.pt"
    dagger_result = None
    if bool(dagger_section.get("enabled", False)):
        if "dagger_output_dir" not in paths:
            raise ValueError("Enabled search_dagger requires paths.dagger_output_dir")
        dagger_result = run_search_dagger(
            paths["processed_dir"],
            paths["cache_dir"],
            checkpoint_path,
            paths["dagger_output_dir"],
            resolved_config_path,
        )
        checkpoint_path = paths["dagger_output_dir"] / "value_geometry.pt"
    agent_result = evaluate_search_agent(
        paths["processed_dir"],
        paths["cache_dir"],
        checkpoint_path,
        paths["search_agent_report_dir"],
        resolved_config_path,
        split=evaluation_split,
    )

    details_path = paths["search_agent_report_dir"] / f"{evaluation_split}_details.jsonl"
    agent_summary_path = paths["search_agent_report_dir"] / f"{evaluation_split}_summary.json"
    artifact_hashes = {
        "processed_manifest_sha256": sha256_file(paths["processed_dir"] / "manifest.json"),
        "quality_report_sha256": sha256_file(paths["quality_report"]),
        "cache_manifest_sha256": sha256_file(paths["cache_dir"] / "manifest.json"),
        "embedding_report_sha256": sha256_file(embedding_report_dir / "embedding_report.json"),
        "value_checkpoint_sha256": sha256_file(checkpoint_path),
        "value_report_sha256": sha256_file(paths["value_output_dir"] / "value_metrics.json"),
        "agent_details_sha256": sha256_file(details_path),
        "agent_report_sha256": sha256_file(agent_summary_path),
    }
    if dagger_result is not None:
        artifact_hashes.update(
            {
                "dagger_checkpoint_sha256": sha256_file(checkpoint_path),
                "dagger_report_sha256": sha256_file(
                    paths["dagger_output_dir"] / "dagger_metrics.json"
                ),
            }
        )
    fixture_path = paths["raw_dir"] / "fixture_manifest.json"
    if fixture_manifest is not None:
        artifact_hashes["fixture_manifest_sha256"] = sha256_file(fixture_path)
    summary = {
        "run_kind": "search_distilled_value_geometry",
        "evaluation_split": evaluation_split,
        "test_unsealed": evaluation_split == "test",
        "test_access": test_access,
        "experiment_config_sha256": sha256_file(resolved_config_path),
        "selected_energy": value_result["energy"],
        "cache_content_sha256": value_result["cache_content_sha256"],
        "run_provenance": agent_result["run_provenance"],
        "synthetic_fixture": (
            {
                "enabled": True,
                "research_evidence": fixture_manifest["research_evidence"],
                "counts": fixture_manifest["counts"],
            }
            if fixture_manifest is not None
            else {"enabled": False}
        ),
        "prepared": prepare_manifest["counts"],
        "quality": {
            "passed": quality_report["passed"],
            "counts": quality_report["counts"],
            "evidence_status": quality_report["evidence_status"],
            "search_supervision": quality_report.get("search_supervision", {}),
        },
        "cache": cache_manifest["counts"],
        "embedding": {
            "reported_splits": embedding_report["reported_splits"],
            "prototype_views": sorted(embedding_report["views"]),
            "auxiliary_views": sorted(embedding_report["auxiliary_views"]),
            "cka_pairs": len(embedding_report["cka"]),
        },
        "value": {
            "fit_split": "train",
            "selection_split": "dev",
            "test_in_loss_or_selection": False,
            "best_epoch": value_result["best_epoch"],
            "best_dev_selection_score": value_result["best_dev_selection_score"],
            "parameter_count": value_result["parameter_count"],
            "selected_thresholds": value_result["selected_thresholds"],
            "metrics": value_result["metrics"],
        },
        "search_dagger": dagger_result,
        "agent": agent_result["conditions"],
        "artifacts": artifact_hashes,
    }
    write_json(paths["run_summary"], summary)
    return summary


def run_all(
    config_path: str | Path,
    *,
    distance: str | None = None,
    evaluation_split: str = "dev",
) -> dict[str, Any]:
    if evaluation_split not in {"dev", "test"}:
        raise ValueError("evaluation_split must be 'dev' or explicitly unsealed 'test'")
    include_test = evaluation_split == "test"
    config, paths = _paths(config_path)
    preprocessing = config.get("preprocessing", {})
    prepare_manifest = prepare_dataset(
        paths["raw_dir"],
        paths["processed_dir"],
        seed=int(config.get("seed", 17)),
        max_search_depth=int(preprocessing.get("max_search_depth", 12)),
        require_optimal_demo=bool(preprocessing.get("require_optimal_demo", True)),
        counterfactual_policy=str(preprocessing.get("counterfactual_policy", "symbolic")),
    )
    cache_manifest = build_embedding_cache(paths["processed_dir"], paths["cache_dir"], config_path)
    embedding_report = evaluate_embeddings(
        paths["processed_dir"],
        paths["cache_dir"],
        paths["embedding_report_dir"],
        include_test=include_test,
    )
    metric_result = train_metric_model(
        paths["processed_dir"],
        paths["cache_dir"],
        paths["metric_output_dir"],
        config_path,
        distance_override=distance,
        include_test=include_test,
    )
    metric_checkpoint = paths["metric_output_dir"] / "metric_model.pt"
    flow_result = train_flow_model(
        paths["processed_dir"],
        paths["cache_dir"],
        metric_checkpoint,
        paths["flow_output_dir"],
        config_path,
        include_test=include_test,
    )
    flow_checkpoint = paths["flow_output_dir"] / "flow_model.pt"
    agent_result = evaluate_agent(
        paths["processed_dir"],
        paths["cache_dir"],
        config_path,
        metric_checkpoint,
        flow_checkpoint,
        paths["agent_report_dir"],
        config_path,
        split=evaluation_split,
    )
    artifact_hashes = {
        "processed_manifest_sha256": sha256_file(paths["processed_dir"] / "manifest.json"),
        "cache_manifest_sha256": sha256_file(paths["cache_dir"] / "manifest.json"),
        "embedding_report_sha256": sha256_file(
            paths["embedding_report_dir"] / "embedding_report.json"
        ),
        "metric_checkpoint_sha256": sha256_file(metric_checkpoint),
        "metric_report_sha256": sha256_file(paths["metric_output_dir"] / "metric_metrics.json"),
        "flow_checkpoint_sha256": sha256_file(flow_checkpoint),
        "flow_report_sha256": sha256_file(paths["flow_output_dir"] / "flow_metrics.json"),
        "agent_details_sha256": sha256_file(paths["agent_report_dir"] / "agent_details.jsonl"),
        "agent_report_sha256": sha256_file(paths["agent_report_dir"] / "agent_summary.json"),
    }
    summary = {
        "evaluation_split": evaluation_split,
        "experiment_config_sha256": sha256_file(config_path),
        "run_provenance": agent_result["run_provenance"],
        "selected_distance": metric_result["distance"],
        "cache_content_sha256": metric_result["cache_content_sha256"],
        "artifacts": artifact_hashes,
        "prepared": prepare_manifest["counts"],
        "cache": cache_manifest["counts"],
        "embedding_views": {
            "prototype": sorted(embedding_report["views"]),
            "auxiliary": sorted(embedding_report["auxiliary_views"]),
        },
        "metric": metric_result["metrics"],
        "flow": flow_result["metrics"],
        "agent": agent_result["conditions"],
    }
    write_json(paths["run_summary"], summary)
    return summary


def compare_distances(config_path: str | Path, distances: list[str]) -> list[dict[str, Any]]:
    if not distances:
        raise ValueError("At least one distance family is required")
    _, paths = _paths(config_path)
    root = paths["metric_output_dir"].parent / "distance_comparison"
    rows = []
    for distance in distances:
        distance_output = root / distance
        result = train_metric_model(
            paths["processed_dir"],
            paths["cache_dir"],
            distance_output,
            config_path,
            distance_override=distance,
        )
        rows.append(
            {
                "distance": distance,
                "parameter_count": result["parameter_count"],
                "training_config_sha256": result["training_config_sha256"],
                "metric_checkpoint_sha256": result["metric_checkpoint_sha256"],
                "metric_report_sha256": sha256_file(distance_output / "metric_metrics.json"),
                "dev": result["metrics"]["dev"],
            }
        )
    write_json(
        root / "comparison.json",
        {
            "selection_split": "dev",
            "test_sealed": True,
            "experiment_config_sha256": sha256_file(config_path),
            "cache_content_sha256": result["cache_content_sha256"],
            "run_provenance": result["run_provenance"],
            "results": rows,
        },
    )
    return rows


def _console_summary(command: str, result: Any) -> Any:
    """Keep CLI output useful without dumping manifests containing every file/ID."""

    if command == "prepare":
        return {"counts": result["counts"], "schema_version": result["schema_version"]}
    if command == "embed":
        return {
            "counts": result["counts"],
            "cache_version": result["cache_version"],
            "config_sha256": result["config_sha256"],
        }
    if command == "evaluate-embeddings":
        return {
            "reported_splits": result["reported_splits"],
            "prototype_views": sorted(result["views"]),
            "auxiliary_views": sorted(result["auxiliary_views"]),
            "cka_pairs": len(result["cka"]),
        }
    if command in {"train-metric", "train-flow"}:
        return {
            "best_epoch": result["best_epoch"],
            "parameter_count": result["parameter_count"],
            "metrics": result["metrics"],
        }
    if command == "evaluate-agent":
        return {"conditions": result["conditions"]}
    if command == "generate-search-fixture":
        return {
            "research_evidence": result["research_evidence"],
            "counts": result["counts"],
            "content_sha256": result["content_sha256"],
        }
    if command == "prepare-search":
        return {
            "counts": result["counts"],
            "active_splits": result.get("oracle_metadata", {}).get("active_splits", []),
            "dataset_content_sha256": result["dataset_content_sha256"],
        }
    if command == "audit-search-data":
        return {
            "passed": result["passed"],
            "counts": result["counts"],
            "evidence_status": result["evidence_status"],
            "audit_scope": result.get("audit_scope"),
            "failures": result["failures"],
        }
    if command == "train-value":
        return {
            "energy": result["energy"],
            "best_epoch": result["best_epoch"],
            "best_dev_selection_score": result["best_dev_selection_score"],
            "parameter_count": result["parameter_count"],
            "metrics": result["metrics"],
        }
    if command == "train-dagger":
        return {
            "selected_round": result["selected_round"],
            "best_dev_selection_score": result["best_dev_selection_score"],
            "test_sealed": result["test_sealed"],
            "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        }
    if command == "train-state-flow":
        return {
            "best_epoch": result["best_epoch"],
            "best_dev_validation_loss": result["best_dev_validation_loss"],
            "selected_stop_threshold": result["selected_stop_threshold"],
            "stop_calibration": result["stop_calibration"],
            "parameter_count": result["trainable_parameter_count"],
            "test_reported": result["test_reported"],
            "metrics": result["metrics"],
        }
    if command == "evaluate-state-flow":
        return {
            "evaluation_split": result["evaluation_split"],
            "test_unsealed": result["test_unsealed"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics": result["metrics"],
        }
    if command == "compare-value-geometries":
        return {
            "selection_split": result["selection_split"],
            "test_sealed": result["test_sealed"],
            "selected_energy": result["selected_energy"],
            "aggregate": result["aggregate"],
        }
    if command == "compare-value-inputs":
        return {
            "selection_split": result["selection_split"],
            "test_sealed": result["test_sealed"],
            "capacity_matched": result["capacity_matched"],
            "aggregate": result["aggregate"],
            "effects": result["effects"],
        }
    if command == "compare-value-capacities":
        return {
            "selection_split": result["selection_split"],
            "test_sealed": result["test_sealed"],
            "aggregate": result["aggregate"],
            "half_x_minus_configured_dev_joint_accuracy": result[
                "half_x_minus_configured_dev_joint_accuracy"
            ],
            "two_x_minus_configured_dev_joint_accuracy": result[
                "two_x_minus_configured_dev_joint_accuracy"
            ],
        }
    if command == "compare-value-run-groups":
        return {
            "selection_split": result["selection_split"],
            "test_sealed": result["test_sealed"],
            "comparison": result["comparison"],
            "parameter_match": result["parameter_match"],
            "paired_task_macro_effects": result["paired_task_macro_effects"],
        }
    if command == "evaluate-search-agent":
        return {
            "evaluation_split": result["evaluation_split"],
            "test_unsealed": result["test_unsealed"],
            "test_access": result.get("test_access"),
            "conditions": result["conditions"],
        }
    if command == "run-search":
        return {
            "evaluation_split": result["evaluation_split"],
            "test_unsealed": result["test_unsealed"],
            "test_access": result.get("test_access"),
            "selected_energy": result["selected_energy"],
            "prepared": result["prepared"],
            "quality_passed": result["quality"]["passed"],
            "cache": result["cache"],
            "embedding": result["embedding"],
            "value": result["value"],
            "agent": result["agent"],
            "artifacts": result["artifacts"],
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoflow", description="GeoFlowAgent experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "embed",
        "evaluate-embeddings",
        "train-metric",
        "train-flow",
        "evaluate-agent",
        "run-all",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        if command in {"train-metric", "run-all"}:
            subparser.add_argument("--distance")
        if command in {"evaluate-embeddings", "train-metric", "train-flow"}:
            subparser.add_argument(
                "--include-test",
                action="store_true",
                help="Explicitly unseal test metrics after dev-based selection.",
            )
        if command == "evaluate-agent":
            subparser.add_argument(
                "--split",
                choices=["dev", "test"],
                default="dev",
                help="Defaults to dev; selecting test explicitly unseals final evaluation.",
            )
        if command == "run-all":
            subparser.add_argument(
                "--evaluation-split",
                choices=["dev", "test"],
                default="dev",
                help="Use test only for a final or synthetic smoke run.",
            )
    compare = subparsers.add_parser("compare-distances")
    compare.add_argument("--config", required=True)
    compare.add_argument(
        "--distances",
        nargs="+",
        default=["euclidean", "cosine", "diagonal_mahalanobis", "bilinear"],
    )
    importer = subparsers.add_parser("import-marrvel")
    importer.add_argument("--output-dir", required=True)
    importer.add_argument("--input")
    importer.add_argument("--revision")
    finalizer = subparsers.add_parser("finalize-marrvel")
    finalizer.add_argument("--queue", required=True)
    finalizer.add_argument("--raw-dir", required=True)

    for command in ("generate-search-fixture", "prepare-search", "audit-search-data"):
        search_parser = subparsers.add_parser(command)
        search_parser.add_argument("--config", required=True)
        if command in {"prepare-search", "audit-search-data"}:
            search_parser.add_argument(
                "--include-test",
                action="store_true",
                help="Explicitly unseal test preprocessing/audit and record access.",
            )
            search_parser.add_argument("--test-access-reason")

    value = subparsers.add_parser("train-value")
    value.add_argument("--config", required=True)
    value.add_argument("--energy")
    value.add_argument("--seed", type=int)
    value.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unseal test metrics after dev-based checkpoint selection.",
    )

    dagger = subparsers.add_parser("train-dagger")
    dagger.add_argument("--config", required=True)
    dagger.add_argument(
        "--initial-checkpoint",
        help="Defaults to paths.value_output_dir/value_geometry.pt.",
    )

    state_flow = subparsers.add_parser("train-state-flow")
    state_flow.add_argument("--config", required=True)
    state_flow.add_argument("--seed", type=int)
    state_flow.add_argument("--output-dir")
    state_flow.add_argument(
        "--root-only-evaluation",
        action="store_true",
        help="Evaluate one independent root per task; useful for multi-seed replication.",
    )
    state_flow.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unseal test metrics after train/dev fitting and selection.",
    )
    state_flow.add_argument("--test-access-reason")

    state_flow_eval = subparsers.add_parser("evaluate-state-flow")
    state_flow_eval.add_argument("--config", required=True)
    state_flow_eval.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="dev",
        help="Defaults to dev; selecting test explicitly unseals final evaluation.",
    )
    state_flow_eval.add_argument("--checkpoint")
    state_flow_eval.add_argument("--output")
    state_flow_eval.add_argument("--test-access-reason")
    state_flow_eval.add_argument(
        "--root-only",
        action="store_true",
        help="Evaluate only one independent root snapshot per task.",
    )

    geometry = subparsers.add_parser("compare-value-geometries")
    geometry.add_argument("--config", required=True)
    geometry.add_argument("--output-dir")
    geometry.add_argument(
        "--energies",
        nargs="+",
        help="Defaults to geometry_comparison.energies in the config.",
    )
    geometry.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Defaults to geometry_comparison.seeds in the config.",
    )

    input_ablation = subparsers.add_parser("compare-value-inputs")
    input_ablation.add_argument("--config", required=True)
    input_ablation.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Defaults to geometry_comparison.seeds in the config.",
    )

    capacity = subparsers.add_parser("compare-value-capacities")
    capacity.add_argument("--config", required=True)
    capacity.add_argument("--output-dir")
    capacity.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Defaults to geometry_comparison.seeds in the config.",
    )

    run_groups = subparsers.add_parser("compare-value-run-groups")
    run_groups.add_argument("--config", required=True)
    run_groups.add_argument("--left-dir", required=True)
    run_groups.add_argument("--right-dir", required=True)
    run_groups.add_argument("--left-label", required=True)
    run_groups.add_argument("--right-label", required=True)
    run_groups.add_argument("--output", required=True)
    run_groups.add_argument("--seeds", nargs="+", type=int, required=True)

    search_eval = subparsers.add_parser("evaluate-search-agent")
    search_eval.add_argument("--config", required=True)
    search_eval.add_argument(
        "--checkpoint",
        help="Defaults to paths.value_output_dir/value_geometry.pt.",
    )
    search_eval.add_argument(
        "--output-dir",
        help="Optional report directory, useful for checkpoint/ablation comparisons.",
    )
    search_eval.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Defaults to dev; selecting test explicitly unseals final evaluation.",
    )
    search_eval.add_argument("--test-access-reason")

    search_run = subparsers.add_parser("run-search")
    search_run.add_argument("--config", required=True)
    search_run.add_argument("--energy")
    search_run.add_argument(
        "--evaluation-split",
        choices=["dev", "test"],
        default="dev",
        help="Defaults to dev; use test only for a final or synthetic smoke run.",
    )
    search_run.add_argument("--test-access-reason")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "import-marrvel":
        print(import_marrvel(args.output_dir, input_path=args.input, revision=args.revision))
        return
    if args.command == "finalize-marrvel":
        print(finalize_curation(args.queue, args.raw_dir))
        return
    if args.command == "generate-search-fixture":
        result = generate_search_fixture_from_config(args.config)
        print(_console_summary(args.command, result))
        return
    if args.command == "prepare-search":
        config, paths = _paths(args.config)
        del config
        active_splits = SEARCH_SPLITS if args.include_test else SEALED_DEVELOPMENT_SPLITS
        if args.include_test:
            _append_test_access(
                args.config,
                paths,
                command="prepare-search",
                reason=args.test_access_reason,
            )
        result = prepare_search_from_config(args.config, active_splits=active_splits)
        print(_console_summary(args.command, result))
        return
    if args.command == "audit-search-data":
        config, paths = _paths(args.config)
        del config
        active_splits = SEARCH_SPLITS if args.include_test else SEALED_DEVELOPMENT_SPLITS
        if args.include_test:
            _append_test_access(
                args.config,
                paths,
                command="audit-search-data",
                reason=args.test_access_reason,
            )
        result = audit_search_data(args.config, active_splits=active_splits)
        print(_console_summary(args.command, result))
        return
    if args.command == "run-search":
        result = run_search_all(
            args.config,
            energy=args.energy,
            evaluation_split=args.evaluation_split,
            test_access_reason=args.test_access_reason,
        )
        print(_console_summary(args.command, result))
        return
    config, paths = _paths(args.config)
    if args.command == "prepare":
        section = config.get("preprocessing", {})
        result = prepare_dataset(
            paths["raw_dir"],
            paths["processed_dir"],
            seed=int(config.get("seed", 17)),
            max_search_depth=int(section.get("max_search_depth", 12)),
            require_optimal_demo=bool(section.get("require_optimal_demo", True)),
            counterfactual_policy=str(section.get("counterfactual_policy", "symbolic")),
        )
    elif args.command == "embed":
        result = build_embedding_cache(paths["processed_dir"], paths["cache_dir"], args.config)
    elif args.command == "evaluate-embeddings":
        result = evaluate_embeddings(
            paths["processed_dir"],
            paths["cache_dir"],
            _embedding_report_dir(paths),
            include_test=args.include_test,
        )
    elif args.command == "train-metric":
        result = train_metric_model(
            paths["processed_dir"],
            paths["cache_dir"],
            paths["metric_output_dir"],
            args.config,
            distance_override=args.distance,
            include_test=args.include_test,
        )
    elif args.command == "train-flow":
        result = train_flow_model(
            paths["processed_dir"],
            paths["cache_dir"],
            paths["metric_output_dir"] / "metric_model.pt",
            paths["flow_output_dir"],
            args.config,
            include_test=args.include_test,
        )
    elif args.command == "evaluate-agent":
        result = evaluate_agent(
            paths["processed_dir"],
            paths["cache_dir"],
            args.config,
            paths["metric_output_dir"] / "metric_model.pt",
            paths["flow_output_dir"] / "flow_model.pt",
            paths["agent_report_dir"],
            args.config,
            split=args.split,
        )
    elif args.command == "compare-distances":
        result = compare_distances(args.config, args.distances)
    elif args.command == "train-value":
        if args.include_test:
            _append_test_access(
                args.config,
                paths,
                command="train-value",
                reason="explicit train-value --include-test metrics",
            )
        result = train_value_geometry(
            paths["processed_dir"],
            paths["cache_dir"],
            paths["value_output_dir"],
            args.config,
            energy_override=args.energy,
            seed_override=args.seed,
            include_test=args.include_test,
        )
    elif args.command == "train-dagger":
        if "dagger_output_dir" not in paths:
            raise ValueError("train-dagger requires paths.dagger_output_dir")
        initial_checkpoint = (
            Path(args.initial_checkpoint)
            if args.initial_checkpoint
            else paths["value_output_dir"] / "value_geometry.pt"
        )
        result = run_search_dagger(
            paths["processed_dir"],
            paths["cache_dir"],
            initial_checkpoint,
            paths["dagger_output_dir"],
            args.config,
        )
    elif args.command == "train-state-flow":
        if args.include_test:
            _append_test_access(
                args.config,
                paths,
                command="train-state-flow",
                reason=args.test_access_reason,
            )
        result = train_search_state_flow(
            paths["processed_dir"],
            paths["cache_dir"],
            Path(args.output_dir) if args.output_dir else paths["state_flow_output_dir"],
            args.config,
            seed_override=args.seed,
            include_test=args.include_test,
            root_only_evaluation=args.root_only_evaluation,
        )
    elif args.command == "evaluate-state-flow":
        if args.split == "test":
            _append_test_access(
                args.config,
                paths,
                command="evaluate-state-flow",
                reason=args.test_access_reason,
            )
        checkpoint = (
            Path(args.checkpoint)
            if args.checkpoint
            else paths["state_flow_output_dir"] / "search_state_flow.pt"
        )
        output = (
            Path(args.output)
            if args.output
            else paths["state_flow_output_dir"] / "evaluations" / f"{args.split}_metrics.json"
        )
        result = evaluate_search_state_flow_checkpoint(
            paths["processed_dir"],
            paths["cache_dir"],
            checkpoint,
            output,
            args.config,
            split=args.split,
            allow_test=args.split == "test",
            root_only=args.root_only,
        )
    elif args.command == "compare-value-geometries":
        comparison = config.get("geometry_comparison", {})
        if not isinstance(comparison, Mapping):
            raise ValueError("geometry_comparison must be a mapping")
        energies = args.energies or comparison.get("energies", [])
        seeds = args.seeds or comparison.get("seeds", [])
        if isinstance(energies, str) or not isinstance(energies, Sequence):
            raise ValueError("geometry comparison energies must be a sequence")
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            raise ValueError("geometry comparison seeds must be a sequence")
        result = compare_value_geometries(
            paths["processed_dir"],
            paths["cache_dir"],
            Path(args.output_dir) if args.output_dir else paths["geometry_comparison_dir"],
            args.config,
            energies=[str(value) for value in energies],
            seeds=[int(value) for value in seeds],
        )
    elif args.command == "compare-value-inputs":
        comparison = config.get("geometry_comparison", {})
        if not isinstance(comparison, Mapping):
            raise ValueError("geometry_comparison must be a mapping")
        seeds = args.seeds or comparison.get("seeds", [])
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            raise ValueError("input-ablation seeds must be a sequence")
        result = compare_value_input_ablations(
            paths["processed_dir"],
            paths["cache_dir"],
            paths["geometry_comparison_dir"].parent / "input_ablation",
            args.config,
            seeds=[int(value) for value in seeds],
        )
    elif args.command == "compare-value-capacities":
        comparison = config.get("geometry_comparison", {})
        if not isinstance(comparison, Mapping):
            raise ValueError("geometry_comparison must be a mapping")
        seeds = args.seeds or comparison.get("seeds", [])
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            raise ValueError("capacity comparison seeds must be a sequence")
        result = compare_value_capacities(
            paths["processed_dir"],
            paths["cache_dir"],
            (
                Path(args.output_dir)
                if args.output_dir
                else paths["geometry_comparison_dir"].parent / "capacity_comparison"
            ),
            args.config,
            seeds=[int(value) for value in seeds],
        )
    elif args.command == "compare-value-run-groups":
        result = compare_existing_value_run_groups(
            args.left_dir,
            args.right_dir,
            args.output,
            left_label=args.left_label,
            right_label=args.right_label,
            seeds=args.seeds,
        )
    elif args.command == "evaluate-search-agent":
        if args.split == "test":
            _append_test_access(
                args.config,
                paths,
                command="evaluate-search-agent",
                reason=args.test_access_reason,
            )
        checkpoint = (
            Path(args.checkpoint)
            if args.checkpoint
            else paths["value_output_dir"] / "value_geometry.pt"
        )
        result = evaluate_search_agent(
            paths["processed_dir"],
            paths["cache_dir"],
            checkpoint,
            Path(args.output_dir) if args.output_dir else paths["search_agent_report_dir"],
            args.config,
            split=args.split,
        )
    else:
        result = run_all(
            args.config,
            distance=args.distance,
            evaluation_split=args.evaluation_split,
        )
    print(_console_summary(args.command, result))


if __name__ == "__main__":
    main()
