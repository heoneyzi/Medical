"""Search-DAgger / expert iteration over exact searched state graphs.

The base dataset contains exhaustive or bounded-search oracle labels.  This module
does not fabricate new labels: it rolls the learned policy through the certified
graph, aggregates the train states that the learner actually visits, and retrains
with those same exact labels.  The replay manifest contains stable example IDs and
record hashes only; private verifier literals are never copied into it.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import (
    evaluate_value_geometry,
    load_value_geometry_checkpoint,
    train_value_geometry,
)
from geoflowagent.utils.io import (
    canonical_json,
    read_yaml,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)


def _dev_selection_score(metrics: Mapping[str, Any]) -> float:
    """Use the exact scalarization used by value-training checkpoint selection."""

    completion = metrics.get("completion", {})
    action_reachability = metrics.get("action_reachability", {})
    state_reachability = metrics.get("state_reachability", {})
    return (
        0.25 * float(metrics.get("joint_stop_action_accuracy") or 0.0)
        + 0.25 * float(metrics.get("gated_optimal_set_accuracy") or 0.0)
        + 0.20 * float(completion.get("balanced_accuracy") or 0.0)
        + 0.15 * float(action_reachability.get("balanced_accuracy") or 0.0)
        + 0.15 * float(state_reachability.get("balanced_accuracy") or 0.0)
    )


def _root_indices(store: SearchFeatureStore, split: str) -> list[int]:
    by_task: dict[str, list[int]] = {}
    for index in store.indices(split):
        row = store.examples[index]
        if int(row.get("depth", len(row.get("prefix_tool_ids", [])))) == 0:
            by_task.setdefault(str(row["task_id"]), []).append(index)
    duplicates = sorted(task_id for task_id, rows in by_task.items() if len(rows) != 1)
    if duplicates:
        raise ValueError(f"Search-DAgger requires exactly one root per task: {duplicates}")
    expected = {str(store.examples[index]["task_id"]) for index in store.indices(split)}
    missing = sorted(expected - set(by_task))
    if missing:
        raise ValueError(f"Search-DAgger found no root state for tasks: {missing}")
    return [by_task[task_id][0] for task_id in sorted(by_task)]


@torch.inference_mode()
def rollout_search_graph(
    model: torch.nn.Module,
    store: SearchFeatureStore,
    device: torch.device,
    *,
    split: str = "train",
    completion_threshold: float,
    reachability_threshold: float,
    max_steps: int,
    stop_contract_guard: bool = True,
    hard_reachability_guard: bool = True,
) -> dict[str, Any]:
    """Roll a policy through known successor IDs without consulting private labels."""

    if split != "train":
        raise ValueError("Search-DAgger rollouts are restricted to the train split")
    if max_steps <= 0:
        raise ValueError("Search-DAgger max_steps must be positive")
    model.eval()
    episodes: list[dict[str, Any]] = []
    visited_ids: list[str] = []
    hard_ids: list[str] = []
    for root_index in _root_indices(store, split):
        current = root_index
        trace: list[dict[str, Any]] = []
        failure: str | None = None
        success = False
        seen: set[int] = set()
        for step in range(max_steps + 1):
            if current in seen:
                failure = "graph_cycle"
                break
            seen.add(current)
            batch = store.batch([current], device)
            row = batch["rows"][0]
            example_id = str(row["example_id"])
            visited_ids.append(example_id)
            terminal = bool(row.get("terminal"))
            output = model(
                batch["state_views"],
                batch["goal_views"],
                batch["tool_views"],
                batch["structured_state"],
                batch["structured_goal"],
                batch["structured_tools"],
                candidate_mask=batch["policy_candidate_mask"],
                mask_unreachable=hard_reachability_guard,
                reachability_threshold=reachability_threshold,
            )
            completion_probability = float(output["completion_logit"].sigmoid().item())
            candidates = batch["policy_candidate_mask"][0]
            learned_stop = completion_probability >= completion_threshold
            should_stop = not bool(candidates.any()) or (
                learned_stop and (terminal or not stop_contract_guard)
            )
            if should_stop:
                correct = terminal
                if not correct:
                    hard_ids.append(example_id)
                    failure = "premature_stop"
                else:
                    success = True
                trace.append(
                    {
                        "step": step,
                        "example_id": example_id,
                        "record_sha256": row.get("record_sha256"),
                        "selected_tool": "<STOP>",
                        "completion_probability": completion_probability,
                        "terminal": terminal,
                        "oracle_optimal": correct,
                        "selected_regret": 0.0 if correct else None,
                    }
                )
                break
            if step >= max_steps:
                hard_ids.append(example_id)
                failure = "step_limit"
                break

            score = output["policy_logits"][0]
            allowed = torch.isfinite(score) & candidates
            reachability_fallback = not bool(allowed.any())
            if reachability_fallback:
                score = (
                    -output["q"][0]
                    + torch.nn.functional.logsigmoid(
                        output["action_reachability_logit"][0]
                    )
                )
                allowed = candidates
            tool_index = int(score.masked_fill(~allowed, -torch.inf).argmax().item())
            tool_id = store.tool_ids[tool_index]
            known = bool(batch["known_action_mask"][0, tool_index])
            optimal = bool(batch["optimal_action_mask"][0, tool_index]) if known else None
            regret_tensor = batch["regret"][0, tool_index]
            regret = float(regret_tensor.item()) if bool(torch.isfinite(regret_tensor)) else None
            successor = int(batch["successor_index"][0, tool_index].item())
            hard = optimal is not True
            if hard:
                hard_ids.append(example_id)
            trace.append(
                {
                    "step": step,
                    "example_id": example_id,
                    "record_sha256": row.get("record_sha256"),
                    "selected_tool": tool_id,
                    "completion_probability": completion_probability,
                    "terminal": terminal,
                    "oracle_label_known": known,
                    "oracle_optimal": optimal,
                    "selected_regret": regret,
                    "reachability_guard_fallback": reachability_fallback,
                }
            )
            if not known:
                failure = "unknown_oracle_label"
                break
            if successor < 0:
                failure = "unknown_successor"
                break
            current = successor
        else:  # pragma: no cover - the explicit step limit branch exits first
            failure = "step_limit"
        episodes.append(
            {
                "task_id": str(store.examples[root_index]["task_id"]),
                "success": success,
                "failure": failure,
                "visited_states": len(trace),
                "hard_states": sum(row.get("oracle_optimal") is not True for row in trace),
                "trace": trace,
            }
        )
    failures = Counter(str(row["failure"]) for row in episodes if row["failure"] is not None)
    unique_train = len(store.indices("train"))
    return {
        "split": split,
        "tasks": len(episodes),
        "success_rate": sum(bool(row["success"]) for row in episodes) / max(1, len(episodes)),
        "visited_occurrences": len(visited_ids),
        "unique_visited_states": len(set(visited_ids)),
        "train_state_coverage": len(set(visited_ids)) / max(1, unique_train),
        "hard_occurrences": len(hard_ids),
        "unique_hard_states": len(set(hard_ids)),
        "failure_counts": dict(sorted(failures.items())),
        "visited_example_ids": visited_ids,
        "hard_example_ids": hard_ids,
        "episodes": episodes,
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any], store: SearchFeatureStore
) -> None:
    if checkpoint.get("cache_content_sha256") != store.cache.content_sha256:
        raise ValueError("Search-DAgger checkpoint was trained with a different frozen cache")
    if checkpoint.get("tool_ids") != store.tool_ids:
        raise ValueError("Search-DAgger checkpoint tool order differs from the feature store")
    if checkpoint.get("views") != list(store.views):
        raise ValueError("Search-DAgger checkpoint view order differs from the feature store")
    if checkpoint.get(
        "feature_ablation", {"zero_views": [], "zero_structured": False}
    ) != store.feature_ablation:
        raise ValueError("Search-DAgger checkpoint feature ablation differs from the store")


def run_search_dagger(
    processed_dir: str | Path,
    cache_dir: str | Path,
    initial_checkpoint: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run train-only policy aggregation and select a round on held-out dev."""

    config = read_yaml(config_path)
    dagger = config.get("search_dagger", {})
    if not isinstance(dagger, Mapping):
        raise ValueError("search_dagger must be a mapping")
    rounds = int(dagger.get("rounds", 3))
    replay_fraction = float(dagger.get("replay_fraction", 0.5))
    hard_multiplier = int(dagger.get("hard_example_multiplier", 3))
    max_steps = int(
        dagger.get(
            "max_steps",
            config.get("search_agent_evaluation", {}).get("max_steps", 16),
        )
    )
    if rounds <= 0:
        raise ValueError("search_dagger.rounds must be positive")
    if not 0 < replay_fraction <= 1:
        raise ValueError("search_dagger.replay_fraction must be in (0, 1]")
    if hard_multiplier < 1:
        raise ValueError("search_dagger.hard_example_multiplier must be at least one")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty Search-DAgger output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    structured_dim = int(config.get("value_training", {}).get("structured_dim", 64))
    store = SearchFeatureStore(processed_dir, cache_dir, structured_dim=structured_dim)
    store.apply_feature_ablation(
        config.get("value_training", {}).get("feature_ablation")
    )
    if store.indices("test"):
        raise ValueError(
            "Search-DAgger requires a physically sealed train+dev dataset; test rows are present"
        )

    current_path = Path(initial_checkpoint)
    model, checkpoint = load_value_geometry_checkpoint(current_path, device)
    _validate_checkpoint(checkpoint, store)
    energy = str(checkpoint["energy"])
    batch_size = int(config.get("value_training", {}).get("batch_size", 256))
    baseline_dev = evaluate_value_geometry(
        model,
        store,
        "dev",
        device,
        target_scale=float(checkpoint["target_scale"]),
        batch_size=batch_size,
        completion_threshold=float(checkpoint["completion_threshold"]),
        reachability_threshold=float(checkpoint["reachability_threshold"]),
    )
    best_score = _dev_selection_score(baseline_dev)
    best_round = 0
    best_path = current_path
    round_rows: list[dict[str, Any]] = []
    aggregate_replay: list[str] = []

    for round_index in range(1, rounds + 1):
        rollout = rollout_search_graph(
            model,
            store,
            device,
            completion_threshold=float(checkpoint["completion_threshold"]),
            reachability_threshold=float(checkpoint["reachability_threshold"]),
            max_steps=max_steps,
            stop_contract_guard=bool(dagger.get("stop_contract_guard", True)),
            hard_reachability_guard=bool(dagger.get("hard_reachability_guard", True)),
        )
        aggregate_replay.extend(rollout["visited_example_ids"])
        aggregate_replay.extend(
            rollout["hard_example_ids"] * (hard_multiplier - 1)
        )
        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=False)
        episodes_path = round_dir / "train_rollouts.jsonl"
        write_jsonl(episodes_path, rollout["episodes"])
        training_dir = round_dir / "value"
        training = train_value_geometry(
            processed_dir,
            cache_dir,
            training_dir,
            config_path,
            energy_override=energy,
            seed_override=seed + round_index,
            include_test=False,
            replay_example_ids=aggregate_replay,
            external_replay_fraction=replay_fraction,
        )
        current_path = training_dir / "value_geometry.pt"
        model, checkpoint = load_value_geometry_checkpoint(current_path, device)
        _validate_checkpoint(checkpoint, store)
        score = float(training["best_dev_selection_score"])
        round_summary = {
            "round": round_index,
            "rollout": {key: value for key, value in rollout.items() if key != "episodes"},
            "rollout_file_sha256": sha256_file(episodes_path),
            "aggregate_replay_occurrences": len(aggregate_replay),
            "aggregate_replay_unique_states": len(set(aggregate_replay)),
            "aggregate_replay_sha256": sha256_text(canonical_json(aggregate_replay)),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "best_epoch": training["best_epoch"],
            "dev_selection_score": score,
            "dev_metrics": training["metrics"]["dev"],
        }
        round_rows.append(round_summary)
        write_json(round_dir / "round_summary.json", round_summary)
        if score > best_score + 1e-8:
            best_score = score
            best_round = round_index
            best_path = current_path

    final_checkpoint = output_dir / "value_geometry.pt"
    shutil.copy2(best_path, final_checkpoint)
    final_model, final_payload = load_value_geometry_checkpoint(final_checkpoint, device)
    del final_model
    selected_dev = (
        baseline_dev
        if best_round == 0
        else round_rows[best_round - 1]["dev_metrics"]
    )
    result = {
        "kind": "search_dagger_expert_iteration",
        "algorithm": (
            "learner graph rollouts with exact precomputed oracle relabeling and "
            "aggregate occupancy replay"
        ),
        "test_sealed": True,
        "rounds": rounds,
        "selected_round": best_round,
        "best_dev_selection_score": best_score,
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "selected_checkpoint_sha256": sha256_file(final_checkpoint),
        "selected_checkpoint_training_seed": final_payload.get("seed"),
        "cache_content_sha256": store.cache.content_sha256,
        "processed_manifest_sha256": sha256_file(Path(processed_dir) / "manifest.json"),
        "config_sha256": sha256_file(config_path),
        "replay_fraction": replay_fraction,
        "hard_example_multiplier": hard_multiplier,
        "baseline_dev_selection_score": _dev_selection_score(baseline_dev),
        "baseline_dev_metrics": baseline_dev,
        "selected_dev_metrics": selected_dev,
        "round_results": round_rows,
        "scientific_boundary": (
            "This is exact-graph Search-DAgger over train states already represented in the "
            "immutable frozen cache. Novel runtime states require a separately versioned "
            "snapshot/search/cache rebuild and are never assigned synthetic negative labels."
        ),
        "run_provenance": collect_run_provenance(config_path, device=device),
    }
    write_json(output_dir / "dagger_metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run train-only exact-graph Search-DAgger")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_search_dagger(
        args.processed_dir,
        args.cache_dir,
        args.initial_checkpoint,
        args.output_dir,
        args.config,
    )
    print(
        canonical_json(
            {
                "selected_round": result["selected_round"],
                "best_dev_selection_score": result["best_dev_selection_score"],
                "test_sealed": result["test_sealed"],
            }
        )
    )


if __name__ == "__main__":
    main()
