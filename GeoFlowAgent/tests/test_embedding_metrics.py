from __future__ import annotations

import shutil

import numpy as np
import pytest

from geoflowagent.data.preprocess import prepare_dataset
from geoflowagent.embeddings.cache import EmbeddingCache, build_embedding_cache
from geoflowagent.evaluation.embedding_metrics import (
    anisotropy,
    evaluate_embeddings,
    fit_whitener,
    goal_progress_metrics,
    linear_cka,
    retrieval_metrics,
    whiten,
)
from geoflowagent.utils.io import read_jsonl, write_jsonl


def test_low_rank_whitener_matches_dense_ridge_transform() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(5, 9)).astype(np.float32)
    values = rng.normal(size=(4, 9)).astype(np.float32)
    ridge = 1e-3

    compact = fit_whitener(train, ridge=ridge)
    actual = whiten(values, compact)

    mean = train.mean(axis=0, keepdims=True)
    centered = train - mean
    covariance = centered.T @ centered / (len(train) - 1)
    scale = float(np.trace(covariance) / covariance.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance + ridge * max(scale, 1e-6) * np.eye(covariance.shape[0])
    )
    dense = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    expected = (values - mean) @ dense

    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_linear_cka_is_one_for_an_orthogonal_feature_rotation() -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(size=(12, 5))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(5, 5)))

    assert np.isclose(linear_cka(values, values @ orthogonal), 1.0, atol=1e-10)


def test_goal_progress_metrics_detect_monotone_within_task_geometry() -> None:
    states = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    goals = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)
    examples = [
        {"task_id": "task-a", "value_star": 0.0, "terminal": True},
        {"task_id": "task-a", "value_star": 1.0, "terminal": False},
        {"task_id": "task-a", "value_star": 2.0, "terminal": False},
    ]

    metrics = goal_progress_metrics(states, goals, examples)

    for distance in metrics.values():
        assert distance["global_distance_value_spearman"] == pytest.approx(1.0)
        assert distance["mean_within_task_distance_value_spearman"] == pytest.approx(1.0)
        assert distance["within_task_pairwise_progress_order_accuracy"] == pytest.approx(1.0)


def test_embedding_evaluator_rejects_cache_from_another_processed_manifest(
    smoke_raw_dir, cache_dir, tmp_path
) -> None:
    other_processed = tmp_path / "other-processed"
    prepare_dataset(smoke_raw_dir, other_processed, seed=999)

    with pytest.raises(ValueError, match="different processed manifest"):
        evaluate_embeddings(other_processed, cache_dir, tmp_path / "report")


def test_embedding_audit_rejects_an_empty_dev_split(
    smoke_raw_dir, repository_root, tmp_path
) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(smoke_raw_dir, raw)
    tasks = read_jsonl(raw / "tasks.jsonl")
    for task in tasks:
        task["split"] = "train"
    write_jsonl(raw / "tasks.jsonl", tasks)
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    prepare_dataset(raw, processed)
    build_embedding_cache(processed, cache, repository_root / "configs" / "smoke.yaml")

    with pytest.raises(ValueError, match="No nonterminal dev rows"):
        evaluate_embeddings(processed, cache, tmp_path / "report")


def test_retrieval_regret_reports_known_label_coverage_and_null() -> None:
    examples = [
        {
            "split": "test",
            "terminal": False,
            "candidate_tools": ["tool-a", "tool-b"],
            "contract_valid_tools": ["tool-a", "tool-b"],
            "valid_next_tools": ["tool-b"],
            "gold_next_tool": "tool-b",
            "action_regret": {"tool-a": None, "tool-b": 0.0},
            "action_regret_mask": {"tool-a": False, "tool-b": True},
        },
        {
            "split": "test",
            "terminal": False,
            "candidate_tools": ["tool-a", "tool-b"],
            "contract_valid_tools": ["tool-a", "tool-b"],
            "valid_next_tools": ["tool-b"],
            "gold_next_tool": "tool-b",
            "action_regret": {"tool-a": None, "tool-b": 0.25},
            "action_regret_mask": {"tool-a": False, "tool-b": True},
        },
    ]
    distances = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    metrics = retrieval_metrics(
        distances, examples, ["tool-a", "tool-b"], split="test", hard_mask=False
    )

    assert metrics["regret@1"] == 0.25
    assert metrics["regret_label_coverage"] == 0.5

    all_unknown = retrieval_metrics(
        distances[:1], examples[:1], ["tool-a", "tool-b"], split="test", hard_mask=False
    )
    assert all_unknown["regret@1"] is None
    assert all_unknown["regret_label_coverage"] == 0.0


def test_embedding_audit_uses_the_background_pooled_downstream_context(
    processed_dir, cache_dir, tmp_path
) -> None:
    report = evaluate_embeddings(processed_dir, cache_dir, tmp_path / "report")
    cache = EmbeddingCache(cache_dir)
    examples = read_jsonl(processed_dir / "examples.jsonl")
    context = cache.context_matrix("general", examples)
    development = np.asarray([row["split"] in {"train", "dev"} for row in examples])

    assert not np.allclose(context, cache.example_matrix("general"))
    assert np.isclose(
        report["views"]["general"]["geometry"]["state_anisotropy"],
        anisotropy(context[development]),
    )


def test_embedding_audit_keeps_test_sealed_unless_explicitly_requested(
    processed_dir, cache_dir, tmp_path
) -> None:
    development = evaluate_embeddings(processed_dir, cache_dir, tmp_path / "development")
    cache = EmbeddingCache(cache_dir)
    retrieval = development["views"]["general"]["retrieval"]["cosine.no_mask"]
    ridge_probe = development["views"]["general"]["ridge_probe"]
    minimal_pairs = development["views"]["general"]["minimal_pairs"]

    assert development["reported_splits"] == ["train", "dev"]
    assert development["cache_version"] == cache.manifest["cache_version"]
    assert development["feature_spec_version"] == cache.manifest["feature_spec_version"]
    assert development["cache_config_sha256"] == cache.manifest["config_sha256"]
    assert development["cache_content_sha256"] == cache.content_sha256
    assert development["source_manifest_sha256"] == cache.manifest["source_manifest_sha256"]
    assert "test" not in retrieval
    assert "test" not in ridge_probe
    assert minimal_pairs["count"] == 0
    assert minimal_pairs["excluded_count"] == minimal_pairs["total_count"] == 5

    final = evaluate_embeddings(
        processed_dir,
        cache_dir,
        tmp_path / "final",
        include_test=True,
    )
    final_retrieval = final["views"]["general"]["retrieval"]["cosine.no_mask"]
    final_pairs = final["views"]["general"]["minimal_pairs"]
    assert final["reported_splits"] == ["train", "dev", "test"]
    assert "test" in final_retrieval
    assert final_pairs["count"] == final_pairs["total_count"] == 5
    assert final_pairs["excluded_count"] == 0
