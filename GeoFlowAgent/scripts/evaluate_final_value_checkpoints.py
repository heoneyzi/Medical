#!/usr/bin/env python3
"""Evaluate frozen value checkpoints on the once-unsealed final package."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoflowagent.cli import _append_test_access, _paths
from geoflowagent.embeddings.cache import EmbeddingCache, verify_evaluation_cache_superset
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import evaluate_value_geometry, load_value_geometry_checkpoint
from geoflowagent.utils.io import sha256_file, write_json
from geoflowagent.utils.reproducibility import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-access-reason", required=True)
    args = parser.parse_args()

    config, paths = _paths(args.config)
    access = _append_test_access(
        args.config,
        paths,
        command="evaluate-final-value-checkpoints",
        reason=args.test_access_reason,
    )
    eval_config = config.get("search_agent_evaluation", {})
    training_cache_dir = eval_config.get("training_cache_dir")
    if training_cache_dir is None:
        raise ValueError("Final config must declare search_agent_evaluation.training_cache_dir")
    device = choose_device(str(config.get("device", "auto")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for checkpoint_value in args.checkpoint:
        checkpoint_path = Path(checkpoint_value)
        model, checkpoint = load_value_geometry_checkpoint(checkpoint_path, device)
        store = SearchFeatureStore(
            paths["processed_dir"],
            paths["cache_dir"],
            structured_dim=int(checkpoint["model_config"]["structured_dim"]),
        )
        store.apply_feature_ablation(checkpoint.get("feature_ablation"))
        if checkpoint["views"] != list(store.views):
            raise ValueError("Checkpoint view order differs from final cache")
        if checkpoint["tool_ids"] != EmbeddingCache(training_cache_dir).tool_ids:
            raise ValueError("Checkpoint tool order differs from declared training cache")
        superset = verify_evaluation_cache_superset(training_cache_dir, store.cache)
        training = checkpoint.get("training_config", {})
        metrics = evaluate_value_geometry(
            model,
            store,
            "test",
            device,
            target_scale=float(checkpoint["target_scale"]),
            batch_size=int(training.get("batch_size", 256)),
            completion_threshold=float(checkpoint["completion_threshold"]),
            reachability_threshold=float(checkpoint["reachability_threshold"]),
            bootstrap_reps=int(training.get("bootstrap_reps", 1000)),
            bootstrap_seed=int(checkpoint["seed"]) + 2,
        )
        metrics.pop("hard_example_indices", None)
        report = {
            "kind": "final_value_checkpoint_evaluation",
            "evaluation_split": "test",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "seed": int(checkpoint["seed"]),
            "energy": checkpoint["energy"],
            "parameter_count": checkpoint["parameter_count"],
            "feature_ablation": checkpoint.get("feature_ablation"),
            "registry_superset_verification": superset,
            "metrics": metrics,
        }
        write_json(
            output_dir / f"{report['checkpoint_sha256'][:12]}_seed-{report['seed']}.json",
            report,
        )
        reports.append(report)
    write_json(
        output_dir / "summary.json",
        {"test_access": access, "config": str(Path(args.config).resolve()), "runs": reports},
    )


if __name__ == "__main__":
    main()
