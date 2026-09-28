from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.embeddings.base import l2_normalize
from geoflowagent.embeddings.cache import EmbeddingCache
from geoflowagent.utils.io import read_jsonl, sha256_file, write_json
from geoflowagent.utils.reporting import markdown_table, write_markdown


def cosine_distance(query: np.ndarray, tools: np.ndarray) -> np.ndarray:
    return 1.0 - l2_normalize(query) @ l2_normalize(tools).T


def squared_euclidean_distance(query: np.ndarray, tools: np.ndarray) -> np.ndarray:
    return (
        np.sum(query * query, axis=1, keepdims=True)
        + np.sum(tools * tools, axis=1)[None, :]
        - 2.0 * query @ tools.T
    ).clip(min=0.0)


@dataclass(frozen=True)
class LowRankWhitener:
    """Ridge whitener represented in the train-data subspace.

    Computing a dense D×D covariance/eigendecomposition is wasteful for frozen
    encoders such as a 3,584-dimensional Qwen when the curated training set has
    far fewer rows.  This representation is algebraically equivalent to ridge
    whitening while costing O(min(N,D)·D) memory.
    """

    mean: np.ndarray
    basis: np.ndarray
    inverse_sqrt: np.ndarray
    residual_inverse_sqrt: float


def fit_whitener(train: np.ndarray, ridge: float = 1e-3) -> LowRankWhitener:
    if train.ndim != 2 or len(train) == 0:
        raise ValueError("Whitening requires a non-empty [rows, dim] training matrix")
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    centered = np.asarray(train - mean, dtype=np.float64)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = np.square(singular_values) / max(1, len(train) - 1)
    scale = float(eigenvalues.sum() / max(1, train.shape[1]))
    ridge_value = ridge * max(scale, 1e-6)
    return LowRankWhitener(
        mean=mean,
        basis=right.T.astype(np.float32),
        inverse_sqrt=(1.0 / np.sqrt(eigenvalues + ridge_value)).astype(np.float32),
        residual_inverse_sqrt=float(1.0 / math.sqrt(ridge_value)),
    )


def whiten(array: np.ndarray, transform: LowRankWhitener) -> np.ndarray:
    centered = np.asarray(array - transform.mean, dtype=np.float32)
    coordinates = centered @ transform.basis
    correction = transform.inverse_sqrt - transform.residual_inverse_sqrt
    return (
        centered * transform.residual_inverse_sqrt + (coordinates * correction) @ transform.basis.T
    )


def effective_rank(array: np.ndarray) -> float:
    centered = array - array.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    probabilities = energy / max(float(energy.sum()), 1e-12)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    return float(np.exp(entropy))


def anisotropy(array: np.ndarray) -> float:
    normalized = l2_normalize(array)
    similarities = normalized @ normalized.T
    if len(array) <= 1:
        return 0.0
    return float((similarities.sum() - np.trace(similarities)) / (len(array) * (len(array) - 1)))


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right):
        raise ValueError("CKA inputs must be two [rows, dim] matrices with equal row counts")
    left = np.asarray(left - left.mean(axis=0, keepdims=True), dtype=np.float64)
    right = np.asarray(right - right.mean(axis=0, keepdims=True), dtype=np.float64)
    rows, left_dim = left.shape
    right_dim = right.shape[1]
    primal_elements = left_dim * right_dim + left_dim**2 + right_dim**2
    dual_elements = 2 * rows**2
    if primal_elements <= dual_elements:
        cross = left.T @ right
        left_self = left.T @ left
        right_self = right.T @ right
        numerator = float(np.sum(cross**2))
        denominator = math.sqrt(float(np.sum(left_self**2)) * float(np.sum(right_self**2)))
    else:
        left_gram = left @ left.T
        right_gram = right @ right.T
        numerator = float(np.sum(left_gram * right_gram))
        denominator = math.sqrt(float(np.sum(left_gram**2)) * float(np.sum(right_gram**2)))
    return numerator / max(denominator, 1e-12)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x = _rankdata(np.asarray(left, dtype=np.float64))
    y = _rankdata(np.asarray(right, dtype=np.float64))
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def goal_progress_metrics(
    states: np.ndarray,
    goals: np.ndarray,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test whether raw frozen geometry orders within-task remaining cost."""

    if states.shape != goals.shape or len(states) != len(examples):
        raise ValueError("Goal-progress inputs must align by example row")
    normalized_states = l2_normalize(states)
    normalized_goals = l2_normalize(goals)
    distances = {
        "cosine": 1.0 - np.sum(normalized_states * normalized_goals, axis=1),
        "squared_euclidean": np.sum(np.square(states - goals), axis=1),
    }
    output: dict[str, Any] = {}
    for name, values in distances.items():
        by_task: dict[str, list[int]] = {}
        eligible = []
        for index, row in enumerate(examples):
            value = row.get("value_star")
            if value is None or not math.isfinite(float(value)):
                continue
            eligible.append(index)
            by_task.setdefault(str(row["task_id"]), []).append(index)
        task_correlations: list[float] = []
        correct_pairs = 0
        ordered_pairs = 0
        for indices in by_task.values():
            remaining = [float(examples[index]["value_star"]) for index in indices]
            correlation = spearman(
                [float(values[index]) for index in indices], remaining
            )
            if correlation is not None:
                task_correlations.append(correlation)
            for left_position, left in enumerate(indices):
                for right in indices[left_position + 1 :]:
                    value_delta = float(examples[left]["value_star"]) - float(
                        examples[right]["value_star"]
                    )
                    if abs(value_delta) <= 1e-9:
                        continue
                    distance_delta = float(values[left] - values[right])
                    correct_pairs += int(distance_delta * value_delta > 0)
                    ordered_pairs += 1
        terminal = [index for index in eligible if bool(examples[index].get("terminal"))]
        nonterminal = [index for index in eligible if not bool(examples[index].get("terminal"))]
        output[name] = {
            "count": len(eligible),
            "global_distance_value_spearman": spearman(
                [float(values[index]) for index in eligible],
                [float(examples[index]["value_star"]) for index in eligible],
            ),
            "mean_within_task_distance_value_spearman": (
                float(np.mean(task_correlations)) if task_correlations else None
            ),
            "within_task_pairwise_progress_order_accuracy": (
                correct_pairs / ordered_pairs if ordered_pairs else None
            ),
            "ordered_pairs": ordered_pairs,
            "mean_terminal_distance": (
                float(np.mean(values[terminal])) if terminal else None
            ),
            "mean_nonterminal_distance": (
                float(np.mean(values[nonterminal])) if nonterminal else None
            ),
        }
    return output


def retrieval_metrics(
    distances: np.ndarray,
    examples: list[dict[str, Any]],
    tool_ids: list[str],
    *,
    split: str,
    hard_mask: bool,
) -> dict[str, float | int | None]:
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    rows = []
    distance_regret: list[float] = []
    target_regret: list[float] = []
    top1_counts = np.zeros(len(tool_ids), dtype=np.int64)
    for row_index, example in enumerate(examples):
        if example["split"] != split or example["terminal"]:
            continue
        candidates = [tool_id for tool_id in example["candidate_tools"] if tool_id in tool_index]
        if hard_mask:
            allowed = set(example["contract_valid_tools"])
            candidates = [tool_id for tool_id in candidates if tool_id in allowed]
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda item: float(distances[row_index, tool_index[item]]))
        valid = set(example["valid_next_tools"]) - {STOP_TOOL_ID}
        gold = example["gold_next_tool"]
        best_valid_rank = min(
            (rank + 1 for rank, tool_id in enumerate(ordered) if tool_id in valid),
            default=len(ordered) + 1,
        )
        prediction = ordered[0]
        top1_counts[tool_index[prediction]] += 1
        known = bool(example.get("action_regret_mask", {}).get(prediction, True))
        regret = example.get("action_regret", {}).get(prediction)
        rows.append(
            {
                "hit1": prediction in valid,
                "hit3": any(tool_id in valid for tool_id in ordered[:3]),
                "mrr": 1.0 / best_valid_rank,
                "single_trace_em": prediction == gold,
                "regret": float(regret) if known and regret is not None else None,
            }
        )
        for tool_id in candidates:
            known = bool(example.get("action_regret_mask", {}).get(tool_id, True))
            regret = example.get("action_regret", {}).get(tool_id)
            if known and regret is not None:
                distance_regret.append(float(distances[row_index, tool_index[tool_id]]))
                target_regret.append(float(regret))
    if not rows:
        return {"count": 0, "regret@1": None, "regret_label_coverage": 0.0}
    used = top1_counts[top1_counts > 0]
    hub_share = float(used.max() / used.sum()) if len(used) else 0.0
    known_regrets = [row["regret"] for row in rows if row["regret"] is not None]
    return {
        "count": len(rows),
        "valid_hit@1": float(np.mean([row["hit1"] for row in rows])),
        "valid_hit@3": float(np.mean([row["hit3"] for row in rows])),
        "valid_mrr": float(np.mean([row["mrr"] for row in rows])),
        "single_trace_em": float(np.mean([row["single_trace_em"] for row in rows])),
        "regret@1": float(np.mean(known_regrets)) if known_regrets else None,
        "regret_label_coverage": len(known_regrets) / len(rows),
        "distance_regret_spearman": spearman(distance_regret, target_regret),
        "top1_hub_share": hub_share,
    }


def ridge_probe(
    matrix: np.ndarray,
    examples: list[dict[str, Any]],
    tool_ids: list[str],
    report_splits: tuple[str, ...] = ("dev",),
    ridge: float = 1.0,
) -> dict[str, Any]:
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    train_indices = [
        index
        for index, row in enumerate(examples)
        if row["split"] == "train" and not row["terminal"] and row["gold_next_tool"] in tool_index
    ]
    if not train_indices:
        return {"error": "no train examples"}
    x_train = matrix[train_indices].astype(np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True).clip(min=1e-6)
    x_train = (x_train - mean) / scale
    y_train = np.zeros((len(train_indices), len(tool_ids)), dtype=np.float64)
    for row_no, example_index in enumerate(train_indices):
        y_train[row_no, tool_index[examples[example_index]["gold_next_tool"]]] = 1.0
    gram = x_train.T @ x_train + ridge * np.eye(x_train.shape[1])
    weights = np.linalg.solve(gram, x_train.T @ y_train)
    output: dict[str, Any] = {}
    for split in report_splits:
        indices = [
            index
            for index, row in enumerate(examples)
            if row["split"] == split and not row["terminal"] and row["gold_next_tool"] in tool_index
        ]
        if not indices:
            output[split] = {"count": 0}
            continue
        scores = ((matrix[indices] - mean) / scale) @ weights
        predictions = scores.argmax(axis=1)
        set_hits = []
        exact_hits = []
        for local_index, example_index in enumerate(indices):
            predicted = tool_ids[int(predictions[local_index])]
            row = examples[example_index]
            set_hits.append(predicted in set(row["valid_next_tools"]))
            exact_hits.append(predicted == row["gold_next_tool"])
        output[split] = {
            "count": len(indices),
            "valid_set_accuracy": float(np.mean(set_hits)),
            "single_trace_accuracy": float(np.mean(exact_hits)),
        }
    return output


def minimal_pair_metrics(
    cache: EmbeddingCache,
    view: str,
    pairs: list[dict[str, Any]],
    *,
    included_splits: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate only pairs explicitly assigned to an unsealed split.

    Pair encodings share one cache array, so split filtering must be applied by
    row index before computing any aggregate.  Missing split provenance is
    deliberately excluded instead of being guessed from pair content.
    """
    selected_indices = [
        index for index, row in enumerate(pairs) if row.get("split") in included_splits
    ]
    summary: dict[str, Any] = {
        "count": len(selected_indices),
        "total_count": len(pairs),
        "excluded_count": len(pairs) - len(selected_indices),
        "reported_splits": list(included_splits),
    }
    if not selected_indices:
        return summary
    left_path = f"{view}/pairs.left.npy"
    right_path = f"{view}/pairs.right.npy"
    if left_path not in cache.manifest["files"] or right_path not in cache.manifest["files"]:
        return {**summary, "count": 0, "reason": "pair encoding disabled for this view"}
    left = np.asarray(cache.array(left_path), dtype=np.float32)[selected_indices]
    right = np.asarray(cache.array(right_path), dtype=np.float32)[selected_indices]
    distance = 1.0 - np.sum(l2_normalize(left) * l2_normalize(right), axis=1)
    grouped: dict[str, list[float]] = {}
    selected_pairs = [pairs[index] for index in selected_indices]
    for value, row in zip(distance, selected_pairs, strict=True):
        grouped.setdefault(str(row["relation"]), []).append(float(value))
    result = summary
    for relation, values in grouped.items():
        result[f"{relation}_mean_cosine_distance"] = float(np.mean(values))
    invariant = grouped.get("invariant", [])
    sensitive = grouped.get("functional_sensitive", [])
    if invariant and sensitive:
        result["sensitive_minus_invariant"] = float(np.mean(sensitive) - np.mean(invariant))
        wins = [float(s > i) for s in sensitive for i in invariant]
        result["pairwise_separation_auc"] = float(np.mean(wins))
    return result


def evaluate_embeddings(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    make_plots: bool = False,
    include_test: bool = False,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    verify_processed_dataset(processed_dir)
    cache = EmbeddingCache(cache_dir)
    if cache.manifest["source_manifest_sha256"] != sha256_file(processed_dir / "manifest.json"):
        raise ValueError("Embedding cache was built from a different processed manifest")
    examples = read_jsonl(processed_dir / "examples.jsonl")
    tools = read_jsonl(processed_dir / "tools.jsonl")
    if [row["example_id"] for row in examples] != cache.example_ids:
        raise ValueError("Processed examples and embedding cache order/IDs differ")
    if [row["tool_id"] for row in tools] != cache.tool_ids:
        raise ValueError("Processed tools and embedding cache order/IDs differ")
    report_splits = ("train", "dev", "test") if include_test else ("train", "dev")
    for required_split in ("train", "dev"):
        if not any(row["split"] == required_split and not row["terminal"] for row in examples):
            raise ValueError(
                f"No nonterminal {required_split} rows; embedding audit cannot be estimated safely"
            )
    if include_test and not any(row["split"] == "test" for row in examples):
        raise ValueError("Test was explicitly unsealed, but the processed dataset has no test rows")
    analysis_mask = np.asarray([row["split"] in report_splits for row in examples])
    analysis_examples = [row for row in examples if row["split"] in report_splits]
    pairs_path = processed_dir / "minimal_pairs.jsonl"
    pairs = read_jsonl(pairs_path) if pairs_path.exists() else []
    report: dict[str, Any] = {
        "reported_splits": list(report_splits),
        "test_unsealed": include_test,
        "cache_version": cache.manifest["cache_version"],
        "feature_spec_version": cache.manifest["feature_spec_version"],
        "cache_config_sha256": cache.manifest["config_sha256"],
        "cache_content_sha256": cache.content_sha256,
        "source_manifest_sha256": cache.manifest["source_manifest_sha256"],
        "views": {},
        "auxiliary_views": {},
        "cka": {},
    }
    table_rows = []
    matrices: dict[str, np.ndarray] = {}
    for view in cache.prototype_views:
        states = cache.context_matrix(view, examples)[analysis_mask]
        tools = cache.tool_matrix(view)
        matrices[view] = states
        train_mask = np.asarray([row["split"] == "train" for row in analysis_examples])
        transform = fit_whitener(states[train_mask])
        distance_functions: dict[str, Callable[[], np.ndarray]] = {
            "cosine": lambda states=states, tools=tools: cosine_distance(states, tools),
            "euclidean": lambda states=states, tools=tools: squared_euclidean_distance(
                states, tools
            ),
            "train_whitened_euclidean": lambda states=states, tools=tools, transform=transform: (
                squared_euclidean_distance(whiten(states, transform), whiten(tools, transform))
            ),
        }
        view_result: dict[str, Any] = {
            "geometry": {
                "state_anisotropy": anisotropy(states),
                "tool_anisotropy": anisotropy(tools),
                "state_effective_rank": effective_rank(states),
                "tool_effective_rank": effective_rank(tools),
            },
            "minimal_pairs": minimal_pair_metrics(
                cache, view, pairs, included_splits=report_splits
            ),
            "goal_progress": {
                split: goal_progress_metrics(
                    states[np.asarray([row["split"] == split for row in analysis_examples])],
                    np.asarray(cache.example_field(view, "goal"), dtype=np.float32)[
                        analysis_mask
                    ][np.asarray([row["split"] == split for row in analysis_examples])],
                    [row for row in analysis_examples if row["split"] == split],
                )
                for split in report_splits
            },
            "ridge_probe": ridge_probe(
                states,
                analysis_examples,
                cache.tool_ids,
                tuple(split for split in report_splits if split != "train"),
            ),
            "retrieval": {},
        }
        for distance_name, function in distance_functions.items():
            distances = function()
            for hard_mask in (False, True):
                mask_name = "exact_mask" if hard_mask else "no_mask"
                key = f"{distance_name}.{mask_name}"
                view_result["retrieval"][key] = {}
                for split in report_splits:
                    metrics = retrieval_metrics(
                        distances,
                        analysis_examples,
                        cache.tool_ids,
                        split=split,
                        hard_mask=hard_mask,
                    )
                    view_result["retrieval"][key][split] = metrics
                    if split != "train":
                        table_rows.append(
                            {
                                "split": split,
                                "view": view,
                                "distance": distance_name,
                                "mask": mask_name,
                                "valid_hit@1": metrics.get("valid_hit@1"),
                                "valid_mrr": metrics.get("valid_mrr"),
                                "regret@1": metrics.get("regret@1"),
                                "regret_label_coverage": metrics.get("regret_label_coverage"),
                            }
                        )
        report["views"][view] = view_result

    # State-only encoders (for example SapBERT or DNABERT-2) cannot retrieve a
    # tool directly because no tool-side prototype exists.  They still receive
    # geometry, minimal-pair, probe, CKA, and PCA audits so their contribution is
    # measured instead of being hidden inside the downstream fusion module.
    for view in cache.state_only_views:
        states = cache.context_matrix(view, examples)[analysis_mask]
        matrices[view] = states
        report["auxiliary_views"][view] = {
            "geometry": {
                "state_anisotropy": anisotropy(states),
                "state_effective_rank": effective_rank(states),
            },
            "minimal_pairs": minimal_pair_metrics(
                cache, view, pairs, included_splits=report_splits
            ),
            "ridge_probe": ridge_probe(
                states,
                analysis_examples,
                cache.tool_ids,
                tuple(split for split in report_splits if split != "train"),
            ),
            "retrieval": {
                "not_applicable": "state-only views intentionally have no tool prototypes"
            },
        }

    view_names = sorted(matrices)
    for left_index, left_name in enumerate(view_names):
        for right_name in view_names[left_index + 1 :]:
            report["cka"][f"{left_name}__{right_name}"] = linear_cka(
                matrices[left_name], matrices[right_name]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "embedding_report.json", report)
    sections = [
        (
            "읽는 법",
            "`valid_hit@1`은 여러 정답 행동 중 하나를 맞히면 성공입니다. `single_trace`는 보조 지표이며, "
            "exact mask가 만든 이득과 dense geometry가 만든 이득을 반드시 분리해 보세요. "
            "기본 실행은 test를 통계에 집계하지 않으며 `--include-test`가 명시된 최종 실행만 "
            "test를 보고합니다.",
        ),
        ("Reported retrieval", markdown_table(table_rows)),
        (
            "Full JSON",
            "세부 anisotropy, effective rank, CKA, minimal-pair, probe 결과는 `embedding_report.json`에 있습니다.",
        ),
    ]
    write_markdown(output_dir / "embedding_report.md", "GeoFlowAgent embedding audit", sections)

    if make_plots and matrices:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Plots require `pip install -e '.[eval]'`") from exc
        for view, matrix in matrices.items():
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            points = centered @ right[:2].T
            figure, axis = plt.subplots(figsize=(7, 5))
            split_colors = {"train": "#2563eb", "dev": "#f59e0b", "test": "#dc2626"}
            for split in report_splits:
                color = split_colors[split]
                mask = np.asarray([row["split"] == split for row in analysis_examples])
                axis.scatter(
                    points[mask, 0], points[mask, 1], s=16, alpha=0.7, label=split, c=color
                )
            axis.set_title(f"{view}: PCA diagnostic (not evidence of validity)")
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_dir / f"{view}.pca.png", dpi=160)
            plt.close(figure)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen embedding geometry")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unseal test statistics after development is complete.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_embeddings(
        args.processed_dir,
        args.cache_dir,
        args.output_dir,
        make_plots=args.plots,
        include_test=args.include_test,
    )
    print(
        {
            "prototype_views": sorted(report["views"]),
            "auxiliary_views": sorted(report["auxiliary_views"]),
            "cka_pairs": len(report["cka"]),
        }
    )


if __name__ == "__main__":
    main()
