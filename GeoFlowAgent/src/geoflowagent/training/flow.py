from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION, STOP_TOOL_ID
from geoflowagent.models.flow import WholePlanFlow
from geoflowagent.training.features import FeatureStore
from geoflowagent.training.metric import load_metric_checkpoint
from geoflowagent.utils.io import canonical_json, read_yaml, sha256_file, sha256_text, write_json
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)


def masked_tool_set_context(prototypes: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    weights = candidate_mask.to(prototypes.dtype)
    return (weights @ prototypes) / weights.sum(dim=1, keepdim=True).clamp_min(1)


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def polyline_geometry(points: torch.Tensor) -> tuple[float, float]:
    """Return normalized embedding-path length and mean turning angle.

    The plan-slot trajectory and the ODE transport trajectory are different axes.
    This helper is used for the former: tool/STOP anchors are first normalized so
    scale drift cannot masquerade as a longer semantic path.  Degenerate segments
    are ignored when calculating the angle.
    """

    if points.ndim != 2:
        raise ValueError(f"Expected [points, dim], received shape={tuple(points.shape)}")
    if points.shape[0] < 2:
        return 0.0, 0.0
    normalized = F.normalize(points, dim=-1)
    segments = normalized[1:] - normalized[:-1]
    norms = torch.linalg.vector_norm(segments, dim=-1)
    length = float(norms.sum().item())
    if segments.shape[0] < 2:
        return length, 0.0
    valid = (norms[:-1] > 1e-8) & (norms[1:] > 1e-8)
    if not valid.any():
        return length, 0.0
    cosine = F.cosine_similarity(segments[:-1][valid], segments[1:][valid], dim=-1)
    angle = torch.acos(cosine.clamp(-1.0, 1.0)).mean()
    return length, float(angle.item())


def ode_turn_angle(transport: list[torch.Tensor]) -> float:
    """Mean angle between consecutive ODE updates over examples and plan slots."""

    if len(transport) < 3:
        return 0.0
    increments = torch.stack(
        [transport[index + 1] - transport[index] for index in range(len(transport) - 1)]
    )
    norms = torch.linalg.vector_norm(increments, dim=-1)
    valid = (norms[:-1] > 1e-8) & (norms[1:] > 1e-8)
    cosine = F.cosine_similarity(increments[:-1], increments[1:], dim=-1)
    if not valid.any():
        return 0.0
    return float(torch.acos(cosine[valid].clamp(-1.0, 1.0)).mean().item())


def flow_selection_key(metrics: dict[str, float | int]) -> tuple[float, float, float]:
    """Lexicographic dev criterion: action validity, STOP balance, then suffix edit."""

    return (
        float(metrics.get("valid_first_action_accuracy", 0.0)),
        float(metrics.get("stop_balanced_accuracy", 0.0)),
        -float(metrics.get("normalized_edit_distance", 1.0)),
    )


def _flow_losses(
    flow: WholePlanFlow,
    step: dict[str, torch.Tensor],
    target_bundle: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    stop_weight: float,
    tool_weight: float,
    separation_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = target_bundle["weights"]
    squared_error = (step["predicted_velocity"] - step["target_velocity"]).square().mean(-1)
    fm_loss = (squared_error * weights).sum() / weights.sum().clamp_min(1)
    plan_mask = target_bundle["plan_mask"]
    stop_logits = flow.stop_head(step["predicted_endpoint"]).squeeze(-1)
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits[plan_mask], target_bundle["stop_target"][plan_mask]
    )
    labels = target_bundle["tool_target"]
    tool_mask = labels >= 0
    if tool_mask.any():
        scores = torch.einsum(
            "bld,td->blt",
            F.normalize(step["predicted_endpoint"], dim=-1),
            F.normalize(prototypes, dim=-1),
        )
        scores = scores.masked_fill(~candidate_mask[:, None, :], torch.finfo(scores.dtype).min)
        tool_loss = F.cross_entropy(scores[tool_mask] / 0.1, labels[tool_mask])
    else:
        tool_loss = fm_loss * 0
    # Do not optimize special anchors against tools that never occur in the
    # current training batch. This keeps a held-out tool from entering the loss
    # merely because its frozen description is present in the global registry.
    visible_prototypes = prototypes[candidate_mask.any(dim=0)]
    separation_loss = flow.special_anchor_separation_loss(visible_prototypes)
    total = (
        fm_loss
        + stop_weight * stop_loss
        + tool_weight * tool_loss
        + separation_weight * separation_loss
    )
    return total, {
        "fm": float(fm_loss.item()),
        "stop": float(stop_loss.item()),
        "tool": float(tool_loss.item()),
        "separation": float(separation_loss.item()),
    }


@torch.inference_mode()
def evaluate_flow_model(
    flow: WholePlanFlow,
    geometry: torch.nn.Module,
    store: FeatureStore,
    split: str,
    device: torch.device,
    *,
    nfe: int,
    pad_weight: float,
    stop_threshold: float,
    seed: int,
) -> dict[str, float | int]:
    indices = store.indices(split, include_terminal=True)
    if not indices:
        return {"count": 0}
    batch = store.batch(indices, device)
    prototypes = geometry.tool_prototypes(batch["tool_views"])
    context = geometry.encode_context(
        batch["state_views"], batch["structured_state"], batch["aux_views"]
    )
    set_context = masked_tool_set_context(prototypes, batch["candidate_mask"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        len(indices),
        flow.max_plan_length,
        flow.dim,
        generator=generator,
        device="cpu",
    ).to(device)
    plan, transport = flow.solve(
        context,
        set_context,
        nfe=nfe,
        noise=noise,
        return_transport=True,
    )
    predicted = flow.decode(
        plan,
        prototypes,
        batch["candidate_mask"],
        stop_threshold=stop_threshold,
    )
    suffixes = [store.suffix_indices(row) for row in batch["rows"]]
    targets = flow.build_targets(prototypes, suffixes, pad_weight=pad_weight)
    endpoint_delta = plan - targets["target"]

    def relative_endpoint_error(mask: torch.Tensor) -> float:
        if not mask.any():
            return 0.0
        expanded = mask.unsqueeze(-1)
        numerator = torch.linalg.vector_norm(endpoint_delta * expanded)
        denominator = torch.linalg.vector_norm(targets["target"] * expanded).clamp_min(1e-8)
        return float((numerator / denominator).item())

    relative_error = relative_endpoint_error(torch.ones_like(targets["plan_mask"]))
    active_error = relative_endpoint_error(targets["plan_mask"])
    tool_slot_error = relative_endpoint_error(targets["tool_target"] >= 0)
    stop_slot_error = relative_endpoint_error(targets["stop_target"].bool())
    pad_slot_error = relative_endpoint_error(~targets["plan_mask"])
    first_valid = []
    first_exact = []
    stop_correct = []
    suffix_exact = []
    normalized_edits = []
    generated_lengths = []
    terminal_stop_hits = []
    premature_stop_hits = []
    predicted_path_lengths = []
    predicted_turn_angles = []
    gold_path_lengths = []
    gold_turn_angles = []
    for row_index, (row, predicted_suffix, gold_suffix) in enumerate(
        zip(batch["rows"], predicted, suffixes, strict=True)
    ):
        predicted_first = predicted_suffix[0] if predicted_suffix else None
        valid_indices = {
            store.tool_index[tool_id]
            for tool_id in row["valid_next_tools"]
            if tool_id in store.tool_index
        }
        if STOP_TOOL_ID in row["valid_next_tools"]:
            first_valid.append(predicted_first is None)
        else:
            first_valid.append(predicted_first in valid_indices)
        gold_first = gold_suffix[0] if gold_suffix else None
        first_exact.append(predicted_first == gold_first)
        stop_correct.append((len(predicted_suffix) == 0) == bool(row["terminal"]))
        if row["terminal"]:
            terminal_stop_hits.append(len(predicted_suffix) == 0)
        else:
            premature_stop_hits.append(len(predicted_suffix) == 0)
        suffix_exact.append(predicted_suffix == gold_suffix)
        normalized_edits.append(
            edit_distance(predicted_suffix, gold_suffix)
            / max(1, len(predicted_suffix), len(gold_suffix))
        )
        generated_lengths.append(len(predicted_suffix))
        # If decoding stopped before L_max, include the slot that emitted STOP.
        # A full-length sequence has no predicted STOP slot to append.
        predicted_points = (
            len(predicted_suffix) + 1
            if len(predicted_suffix) < flow.max_plan_length
            else flow.max_plan_length
        )
        predicted_geometry = polyline_geometry(plan[row_index, :predicted_points])
        gold_geometry = polyline_geometry(targets["target"][row_index, : len(gold_suffix) + 1])
        predicted_path_lengths.append(predicted_geometry[0])
        predicted_turn_angles.append(predicted_geometry[1])
        gold_path_lengths.append(gold_geometry[0])
        gold_turn_angles.append(gold_geometry[1])
    transport_length = 0.0
    if transport is not None:
        increments = [
            torch.linalg.vector_norm(transport[index + 1] - transport[index], dim=-1).mean()
            for index in range(len(transport) - 1)
        ]
        transport_length = float(torch.stack(increments).sum().item())
    ode_curvature = ode_turn_angle(transport or [])
    terminal_stop_recall = float(np.mean(terminal_stop_hits)) if terminal_stop_hits else 0.0
    premature_stop_rate = float(np.mean(premature_stop_hits)) if premature_stop_hits else 0.0
    stop_balanced_accuracy = (
        0.5 * (terminal_stop_recall + (1.0 - premature_stop_rate))
        if terminal_stop_hits and premature_stop_hits
        else 0.0
    )
    return {
        "count": len(indices),
        "valid_first_action_accuracy": float(np.mean(first_valid)),
        "single_trace_first_action_accuracy": float(np.mean(first_exact)),
        "terminal_stop_accuracy": float(np.mean(stop_correct)),
        "terminal_stop_recall": terminal_stop_recall,
        "premature_stop_rate": premature_stop_rate,
        "stop_balanced_accuracy": stop_balanced_accuracy,
        "suffix_exact_match": float(np.mean(suffix_exact)),
        "normalized_edit_distance": float(np.mean(normalized_edits)),
        "relative_endpoint_error": relative_error,
        "relative_endpoint_error_active": active_error,
        "relative_endpoint_error_tool_slots": tool_slot_error,
        "relative_endpoint_error_stop_slots": stop_slot_error,
        "relative_endpoint_error_pad_slots": pad_slot_error,
        "mean_generated_length": float(np.mean(generated_lengths)),
        "mean_predicted_plan_path_length": float(np.mean(predicted_path_lengths)),
        "mean_gold_plan_path_length": float(np.mean(gold_path_lengths)),
        "mean_predicted_plan_turn_angle_radians": float(np.mean(predicted_turn_angles)),
        "mean_gold_plan_turn_angle_radians": float(np.mean(gold_turn_angles)),
        "mean_transport_length": transport_length,
        "mean_ode_turn_angle_radians": ode_curvature,
        "nfe": nfe,
    }


def train_flow_model(
    processed_dir: str | Path,
    cache_dir: str | Path,
    metric_checkpoint: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    include_test: bool = False,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    train_config = dict(config.get("flow_training", {}))
    seed = int(config.get("seed", 17))
    experiment_config_sha256 = sha256_file(config_path)
    training_config_sha256 = sha256_text(canonical_json(train_config))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    run_provenance = collect_run_provenance(config_path, device=device)
    geometry, geometry_checkpoint = load_metric_checkpoint(metric_checkpoint, device)
    for parameter in geometry.parameters():
        parameter.requires_grad_(False)
    geometry.eval()
    store = FeatureStore(
        processed_dir,
        cache_dir,
        structured_dim=int(geometry_checkpoint["model_config"]["structured_dim"]),
    )
    train_indices = store.indices("train", include_terminal=True)
    if not train_indices:
        raise ValueError("No flow training examples")
    if not store.indices("dev", include_terminal=True):
        raise ValueError("No dev rows; cannot select a flow checkpoint safely")
    if include_test and not store.indices("test", include_terminal=True):
        raise ValueError(
            "Test was explicitly unsealed, but there are no test rows for flow evaluation"
        )
    if store.tool_ids != geometry_checkpoint["tool_ids"]:
        raise ValueError("Metric checkpoint tool IDs do not match current processed dataset")
    if store.cache.manifest["config_sha256"] != geometry_checkpoint["cache_config_sha256"]:
        raise ValueError("Metric checkpoint was trained on a different embedding cache config")
    if store.cache.content_sha256 != geometry_checkpoint.get("cache_content_sha256"):
        raise ValueError("Metric checkpoint was trained on different embedding cache contents")
    if store.cache.manifest["source_manifest_sha256"] != geometry_checkpoint.get(
        "source_manifest_sha256"
    ):
        raise ValueError("Metric checkpoint was trained on a different processed dataset")
    if store.cache.manifest["feature_spec_version"] != geometry_checkpoint.get(
        "feature_spec_version"
    ):
        raise ValueError("Metric checkpoint was trained with different feature semantics")
    flow = WholePlanFlow(
        int(geometry_checkpoint["model_config"]["shared_dim"]),
        context_dim=int(geometry_checkpoint["model_config"]["shared_dim"]),
        max_plan_length=int(train_config.get("max_plan_length", 10)),
        hidden_dim=int(train_config.get("hidden_dim", 128)),
        layers=int(train_config.get("layers", 2)),
        heads=int(train_config.get("heads", 4)),
        dropout=float(train_config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(train_config.get("learning_rate", 5e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    epochs = int(train_config.get("epochs", 200))
    batch_size = int(train_config.get("batch_size", 16))
    patience = int(train_config.get("patience", 30))
    pad_weight = float(train_config.get("pad_weight", 0.1))
    nfe = int(train_config.get("nfe", 16))
    stop_threshold = float(train_config.get("stop_threshold", 0.5))
    best_key: tuple[float, float, float] | None = None
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_indices)
        flow.train()
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[start : start + batch_size]
            batch = store.batch(batch_indices, device)
            with torch.no_grad():
                prototypes = geometry.tool_prototypes(batch["tool_views"])
                context = geometry.encode_context(
                    batch["state_views"], batch["structured_state"], batch["aux_views"]
                )
                set_context = masked_tool_set_context(prototypes, batch["candidate_mask"])
            suffixes = [store.suffix_indices(row) for row in batch["rows"]]
            targets = flow.build_targets(prototypes, suffixes, pad_weight=pad_weight)
            step = flow.flow_matching_step(targets["target"], context, set_context)
            loss, _ = _flow_losses(
                flow,
                step,
                targets,
                prototypes,
                batch["candidate_mask"],
                stop_weight=float(train_config.get("stop_weight", 1.0)),
                tool_weight=float(train_config.get("tool_weight", 0.5)),
                separation_weight=float(train_config.get("separation_weight", 0.05)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        flow.eval()
        dev = evaluate_flow_model(
            flow,
            geometry,
            store,
            "dev",
            device,
            nfe=nfe,
            pad_weight=pad_weight,
            stop_threshold=stop_threshold,
            seed=seed + 10_000,
        )
        selection_key = flow_selection_key(dev)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, seen),
                "dev_valid_first_action_accuracy": selection_key[0],
                "dev_terminal_stop_accuracy": dev.get("terminal_stop_accuracy", 0.0),
                "dev_stop_balanced_accuracy": selection_key[1],
                "dev_normalized_edit_distance": dev.get("normalized_edit_distance", 1.0),
            }
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_state = copy.deepcopy(flow.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Flow training did not produce a checkpoint")
    flow.load_state_dict(best_state)
    flow.eval()
    report_splits = ("train", "dev", "test") if include_test else ("train", "dev")
    metrics = {
        split: evaluate_flow_model(
            flow,
            geometry,
            store,
            split,
            device,
            nfe=nfe,
            pad_weight=pad_weight,
            stop_threshold=stop_threshold,
            seed=seed + 10_000,
        )
        for split in report_splits
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "kind": "whole_plan_flow",
        "seed": seed,
        "experiment_config_sha256": experiment_config_sha256,
        "training_config_sha256": training_config_sha256,
        "training_config": train_config,
        "run_provenance": run_provenance,
        "model_config": flow.export_config(),
        "model_state": flow.state_dict(),
        "metric_checkpoint_sha256": sha256_file(metric_checkpoint),
        "feature_spec_version": store.cache.manifest["feature_spec_version"],
        "cache_config_sha256": store.cache.manifest["config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
        "tool_ids": store.tool_ids,
        "nfe": nfe,
        "stop_threshold": stop_threshold,
        "pad_weight": pad_weight,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in flow.parameters()),
    }
    checkpoint_path = output_dir / "flow_model.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    result = {
        "seed": seed,
        "experiment_config_sha256": experiment_config_sha256,
        "training_config_sha256": training_config_sha256,
        "training_config": train_config,
        "run_provenance": run_provenance,
        "cache_content_sha256": store.cache.content_sha256,
        "metric_checkpoint_sha256": checkpoint["metric_checkpoint_sha256"],
        "flow_checkpoint_sha256": checkpoint_sha256,
        "best_epoch": best_epoch,
        "best_dev_valid_first_action_accuracy": best_key[0],
        "best_dev_selection_key": list(best_key),
        "parameter_count": checkpoint["parameter_count"],
        "metrics": metrics,
        "history": history,
        "test_reported": include_test,
    }
    write_json(output_dir / "flow_metrics.json", result)
    return result


def load_flow_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[WholePlanFlow, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported flow checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("kind") != "whole_plan_flow":
        raise ValueError(f"Not a whole-plan flow checkpoint: {path}")
    if checkpoint.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError(
            "Flow checkpoint feature semantics differ from this runtime: "
            f"{checkpoint.get('feature_spec_version')!r} != {FEATURE_SPEC_VERSION!r}"
        )
    model = WholePlanFlow(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train whole-plan rectified flow")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--metric-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unseal and report the test split after dev-based selection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_flow_model(
        args.processed_dir,
        args.cache_dir,
        args.metric_checkpoint,
        args.output_dir,
        args.config,
        include_test=args.include_test,
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
