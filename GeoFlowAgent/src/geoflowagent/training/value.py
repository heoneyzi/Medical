"""Train and compare small goal-conditioned value geometries on search labels."""

from __future__ import annotations

import argparse
import copy
import errno
import math
import os
import random
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.embeddings.cache import EmbeddingCache
from geoflowagent.models.value_geometry import (
    GoalConditionedValueGeometry,
    ValueGeometryLossWeights,
    search_distillation_loss,
)
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_yaml,
    sha256_file,
    sha256_text,
    write_json,
)
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)

VALUE_TRAINING_RESULT_VERSION = 3
VALUE_RESUME_VERSION = "geoflowagent.search-value-resume.v2"


def _capture_global_rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_global_rng_state(resume: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in ("python_rng_state", "numpy_rng_state", "torch_rng_state")
        if key not in resume
    ]
    if missing:
        raise ValueError(
            "Legacy value resume lacks exact stochastic state; restart that partial run "
            f"instead of silently changing its trajectory: {missing}"
        )
    random.setstate(resume["python_rng_state"])
    np.random.set_state(resume["numpy_rng_state"])
    torch.set_rng_state(resume["torch_rng_state"].cpu())
    cuda_rng_states = resume.get("cuda_rng_states")
    if torch.cuda.is_available() and cuda_rng_states is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _write_optional_progress_backup(path: Path | None, value: Any) -> None:
    """Keep training alive if a best-effort persistent progress mirror is full."""

    if path is None:
        return
    try:
        _atomic_write_json(path, value)
    except OSError as error:
        if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
        warnings.warn(
            f"Persistent progress mirror unavailable ({error}); local resume is intact",
            RuntimeWarning,
            stacklevel=2,
        )


def _value_progress_backup_path(
    output_dir: Path,
    *,
    energy: str,
    seed: int,
    training_hash: str,
) -> Path | None:
    """Return an optional persistent progress path without changing run semantics."""

    root = os.environ.get("GEOFLOW_VALUE_PROGRESS_BACKUP_DIR")
    if not root:
        return None
    run_fingerprint = sha256_text(str(output_dir.resolve()))[:10]
    label = f"{energy}__seed-{seed}__{training_hash[:12]}__{run_fingerprint}"
    return Path(root) / f"{label}.json"


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(rows)) if rows else None


def _balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(targets)
    if not mask.any():
        return logits.sum() * 0.0
    labels = targets[mask].to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits[mask], labels, reduction="none")
    positives = labels > 0.5
    positive_count = positives.sum()
    negative_count = (~positives).sum()
    if positive_count and negative_count:
        weights = torch.where(
            positives,
            labels.numel() / (2.0 * positive_count),
            labels.numel() / (2.0 * negative_count),
        )
        losses = losses * weights
    return losses.mean()


def _target_scale(store: SearchFeatureStore, train_indices: Sequence[int]) -> float:
    values = store._targets["q_star"][list(train_indices)]  # noqa: SLF001 - immutable tensor store
    values = values[torch.isfinite(values) & (values > 0)]
    if not values.numel():
        return 1.0
    return max(1.0, float(values.median().item()))


def _scaled_targets(batch: Mapping[str, Any], scale: float) -> dict[str, torch.Tensor]:
    return {
        "q": batch["q_star"] / scale,
        "value": batch["value_target"] / scale,
        "regret": batch["regret"] / scale,
        "edge": batch["edge_cost"] / scale,
        "successor_value": batch["successor_value_target"] / scale,
    }


def _transition_target(
    model: GoalConditionedValueGeometry,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """Project cached successor contexts with a detached deterministic target map."""

    mask = batch["successor_feature_mask"]
    batch_size, action_count = mask.shape
    flat_views = {
        name: torch.nan_to_num(value, nan=0.0).reshape(batch_size * action_count, -1)
        for name, value in batch["successor_context_views"].items()
    }
    flat_structured = torch.nan_to_num(
        batch["structured_successor_state"], nan=0.0
    ).reshape(batch_size * action_count, -1)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        target = model.encode_entity(flat_views, flat_structured).reshape(
            batch_size, action_count, -1
        )
    model.train(was_training)
    return target.masked_fill(~mask.unsqueeze(-1), torch.nan)


def _augment_views(
    state_views: Mapping[str, torch.Tensor],
    goal_views: Mapping[str, torch.Tensor],
    tool_views: Mapping[str, torch.Tensor],
    *,
    view_dropout: float,
    noise_std: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not 0 <= view_dropout < 1:
        raise ValueError("view_dropout must be in [0, 1)")
    if noise_std < 0:
        raise ValueError("embedding_noise_std must be non-negative")
    names = sorted(state_views)
    keep = {name: random.random() >= view_dropout for name in names}
    if names and not any(keep.values()):
        keep[random.choice(names)] = True

    def transform(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for name, value in values.items():
            current = value if keep[name] else torch.zeros_like(value)
            # Zero-filled missing modalities are semantic masks, not noisy
            # observations. Preserve exact zeros for state-only goal/tool views.
            if noise_std and keep[name] and bool(torch.count_nonzero(value)):
                current = current + torch.randn_like(current) * noise_std
            output[name] = current
        return output

    return transform(state_views), transform(goal_views), transform(tool_views)


def _soft_action_targets(
    regret: torch.Tensor,
    known_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("regret_temperature must be positive")
    valid = known_mask.bool() & candidate_mask.bool() & torch.isfinite(regret)
    safe = torch.where(valid, regret, torch.zeros_like(regret))
    weights = torch.where(valid, torch.exp(-safe / temperature), torch.zeros_like(regret))
    return weights


def _loss_for_batch(
    model: GoalConditionedValueGeometry,
    batch: Mapping[str, Any],
    *,
    target_scale: float,
    weights: ValueGeometryLossWeights,
    action_temperature: float,
    regret_temperature: float,
    ranking_margin: float,
    supervision: str,
    view_dropout: float,
    embedding_noise_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    state_views, goal_views, tool_views = _augment_views(
        batch["state_views"],
        batch["goal_views"],
        batch["tool_views"],
        view_dropout=view_dropout,
        noise_std=embedding_noise_std,
    )
    output = model(
        state_views,
        goal_views,
        tool_views,
        batch["structured_state"],
        batch["structured_goal"],
        batch["structured_tools"],
        candidate_mask=batch["policy_candidate_mask"],
    )
    target = _scaled_targets(batch, target_scale)
    known = (
        batch["known_action_mask"]
        | batch["action_reachability_mask"]
        | batch["transition_known_mask"]
    ) & batch["policy_candidate_mask"]
    if supervision == "boltzmann":
        soft_target = _soft_action_targets(
            target["regret"], known, batch["policy_candidate_mask"], regret_temperature
        )
        optimal_target = None
    elif supervision == "optimal_set":
        soft_target = None
        optimal_target = batch["optimal_action_mask"]
    else:
        raise ValueError("value_training.supervision must be boltzmann or optimal_set")
    losses = search_distillation_loss(
        output,
        known,
        candidate_mask=batch["policy_candidate_mask"],
        optimal_action_mask=optimal_target,
        soft_action_target=soft_target,
        q_target=target["q"],
        value_target=target["value"],
        value_known_mask=batch["value_mask"],
        regret_target=target["regret"],
        edge_cost=target["edge"],
        successor_value_target=target["successor_value"],
        transition_target=_transition_target(model, batch),
        action_reachability_target=batch["action_reachability_target"],
        state_reachability_target=batch["state_reachability_target"],
        state_reachability_known_mask=batch["state_reachability_mask"],
        completion_target=batch["completion_target"],
        completion_known_mask=batch["completion_mask"],
        weights=weights,
        action_temperature=action_temperature,
        ranking_margin=ranking_margin,
    )
    # Exact-search graphs commonly have fewer dead ends and terminal states than
    # ordinary states. Balanced BCE prevents those safety labels from disappearing
    # in the average while preserving their explicit masks.
    losses["action_reachability"] = _balanced_bce(
        output["action_reachability_logit"],
        batch["action_reachability_target"],
        batch["action_reachability_mask"] & batch["policy_candidate_mask"],
    )
    losses["state_reachability"] = _balanced_bce(
        output["state_reachability_logit"],
        batch["state_reachability_target"],
        batch["state_reachability_mask"],
    )
    losses["completion"] = _balanced_bce(
        output["completion_logit"], batch["completion_target"], batch["completion_mask"]
    )
    total = (
        weights.action * losses["action"]
        + weights.q * losses["q"]
        + weights.value * losses["value"]
        + weights.regret * losses["regret"]
        + weights.bellman * losses["bellman"]
        + weights.transition * losses["transition"]
        + weights.action_reachability * losses["action_reachability"]
        + weights.state_reachability * losses["state_reachability"]
        + weights.completion * losses["completion"]
    )
    losses["total"] = total
    return total, {name: float(value.detach().item()) for name, value in losses.items()}


def _task_balanced_epoch(
    store: SearchFeatureStore,
    indices: Sequence[int],
    rng: random.Random,
    *,
    max_states_per_task: int | None,
    hard_pool: Sequence[int],
    hard_replay_fraction: float,
    external_replay_pool: Sequence[int] = (),
    external_replay_fraction: float = 0.0,
) -> list[int]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_task[str(store.examples[index]["task_id"])].append(index)
    selected: list[int] = []
    for task_id in sorted(by_task):
        values = list(by_task[task_id])
        rng.shuffle(values)
        if max_states_per_task is not None:
            values = values[:max_states_per_task]
        selected.extend(values)
    if hard_pool and hard_replay_fraction > 0:
        replay_count = round(len(selected) * hard_replay_fraction)
        selected.extend(rng.choices(list(hard_pool), k=replay_count))
    if external_replay_pool and external_replay_fraction > 0:
        replay_count = round(len(selected) * external_replay_fraction)
        selected.extend(rng.choices(list(external_replay_pool), k=replay_count))
    rng.shuffle(selected)
    return selected


def _binary_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    if not len(targets):
        return {"count": 0, "accuracy": None, "balanced_accuracy": None, "brier": None}
    predictions = probabilities >= threshold
    labels = targets >= 0.5
    positive = labels
    negative = ~labels
    sensitivity = float(predictions[positive].mean()) if positive.any() else None
    specificity = float((~predictions[negative]).mean()) if negative.any() else None
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else sensitivity if sensitivity is not None else specificity
    )
    return {
        "count": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": float(np.mean((probabilities - labels.astype(np.float32)) ** 2)),
    }


def _bootstrap_task_mean(
    values: Mapping[str, Sequence[float]], *, reps: int, seed: int
) -> dict[str, float] | None:
    task_ids = sorted(key for key, rows in values.items() if rows)
    if not task_ids:
        return None
    task_values = np.asarray([np.mean(values[key]) for key in task_ids], dtype=np.float64)
    point = float(task_values.mean())
    if reps <= 0 or len(task_values) < 2:
        return {"mean": point, "low": point, "high": point, "tasks": len(task_values)}
    rng = np.random.default_rng(seed)
    samples = task_values[
        rng.integers(0, len(task_values), size=(reps, len(task_values)))
    ].mean(axis=1)
    return {
        "mean": point,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "tasks": len(task_values),
    }


@torch.inference_mode()
def _prepare_evaluation_batches(
    model: GoalConditionedValueGeometry,
    store: SearchFeatureStore,
    split: str,
    device: torch.device,
    batch_size: int,
) -> list[tuple[list[int], dict[str, Any], dict[str, torch.Tensor]]]:
    """Run once and retain only tensors consumed by evaluation.

    ``SearchFeatureStore.batch`` includes every frozen successor view with shape
    ``[B, actions, embedding_dim]``. Keeping those inputs for an entire split can
    occupy tens of GB for real multi-view encoders even though thresholding,
    metrics, and hard-example mining never read them after the forward pass.
    Compacting here preserves identical predictions while bounding retained GPU
    memory by labels and model outputs rather than raw successor embeddings.
    """

    model.eval()
    indices = store.indices(split)
    prepared: list[tuple[list[int], dict[str, Any], dict[str, torch.Tensor]]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = store.batch(batch_indices, device)
        output = model(
            batch["state_views"],
            batch["goal_views"],
            batch["tool_views"],
            batch["structured_state"],
            batch["structured_goal"],
            batch["structured_tools"],
            candidate_mask=batch["policy_candidate_mask"],
        )
        retained_batch_keys = (
            "rows",
            "policy_candidate_mask",
            "optimal_action_mask",
            "completion_target",
            "completion_mask",
            "regret",
            "regret_mask",
            "q_star",
            "q_mask",
            "value_target",
            "value_mask",
            "action_reachability_mask",
            "action_reachability_target",
            "state_reachability_mask",
            "state_reachability_target",
        )
        retained_output_keys = (
            "policy_logits",
            "q",
            "value",
            "action_reachability_logit",
            "state_reachability_logit",
            "completion_logit",
            "state_latent",
            "goal_latent",
        )
        compact_batch = {key: batch[key] for key in retained_batch_keys}
        compact_output = {key: output[key] for key in retained_output_keys}
        prepared.append((batch_indices, compact_batch, compact_output))
    return prepared


def _threshold_selection_score(
    prepared_batches: Sequence[
        tuple[list[int], dict[str, Any], dict[str, torch.Tensor]]
    ],
    *,
    completion_threshold: float,
    reachability_threshold: float,
) -> float:
    """Compute only the five threshold-dependent dev selection components."""

    batches = [batch for _, batch, _ in prepared_batches]
    outputs = [output for _, _, output in prepared_batches]
    candidate = torch.cat([batch["policy_candidate_mask"] for batch in batches])
    optimal = torch.cat([batch["optimal_action_mask"] for batch in batches]) & candidate
    eligible = optimal.any(dim=1)
    policy_logits = torch.cat([output["policy_logits"] for output in outputs])
    policy_index = policy_logits.argmax(dim=1)
    policy_hit = optimal.gather(1, policy_index[:, None]).squeeze(1)
    completion_probability = torch.cat(
        [output["completion_logit"].sigmoid() for output in outputs]
    )
    terminal = torch.cat([batch["completion_target"] for batch in batches]) > 0.5
    stop = completion_probability >= completion_threshold
    decision_mask = eligible | terminal
    decision_hit = torch.where(eligible, (~stop) & policy_hit, stop & terminal)
    joint = float(decision_hit[decision_mask].float().mean().item()) if decision_mask.any() else 0.0

    q_value = torch.cat([output["q"] for output in outputs])
    action_reachability = torch.cat(
        [output["action_reachability_logit"].sigmoid() for output in outputs]
    )
    gated_allowed = candidate & (action_reachability >= reachability_threshold)
    gated_index = q_value.masked_fill(~gated_allowed, torch.inf).argmin(dim=1)
    gated_hit = optimal.gather(1, gated_index[:, None]).squeeze(1) & gated_allowed.any(dim=1)
    gated = float(gated_hit[eligible].float().mean().item()) if eligible.any() else 0.0

    action_mask = torch.cat(
        [batch["action_reachability_mask"] for batch in batches]
    ) & candidate
    action_target = torch.cat(
        [batch["action_reachability_target"] for batch in batches]
    )
    state_probability = torch.cat(
        [output["state_reachability_logit"].sigmoid() for output in outputs]
    )
    state_target = torch.cat(
        [batch["state_reachability_target"] for batch in batches]
    )
    state_mask = torch.cat([batch["state_reachability_mask"] for batch in batches])
    completion_target = torch.cat([batch["completion_target"] for batch in batches])
    completion_mask = torch.cat([batch["completion_mask"] for batch in batches])

    def balanced(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float) -> float:
        selected_probability = probability[mask].detach().cpu().numpy()
        selected_target = target[mask].detach().cpu().numpy()
        return float(
            _binary_metrics(
                selected_probability,
                selected_target,
                threshold=threshold,
            ).get("balanced_accuracy")
            or 0.0
        )

    return (
        0.25 * joint
        + 0.25 * gated
        + 0.20
        * balanced(
            completion_probability,
            completion_target,
            completion_mask,
            completion_threshold,
        )
        + 0.15
        * balanced(
            action_reachability,
            action_target,
            action_mask,
            reachability_threshold,
        )
        + 0.15
        * balanced(
            state_probability,
            state_target,
            state_mask,
            reachability_threshold,
        )
    )


def _hard_example_indices(
    prepared_batches: Sequence[
        tuple[list[int], dict[str, Any], dict[str, torch.Tensor]]
    ],
    *,
    completion_threshold: float = 0.5,
) -> list[int]:
    """Find replay states without recomputing unrelated regression/slice metrics."""

    hard: list[int] = []
    for batch_indices, batch, output in prepared_batches:
        candidate = batch["policy_candidate_mask"]
        optimal = batch["optimal_action_mask"] & candidate
        eligible = optimal.any(dim=1)
        finite_candidate = candidate.any(dim=1)
        terminal = batch["completion_target"] > 0.5
        stop = output["completion_logit"].sigmoid() >= completion_threshold
        policy_index = output["policy_logits"].argmax(dim=1)
        policy_hit = optimal.gather(1, policy_index[:, None]).squeeze(1)
        for local, global_index in enumerate(batch_indices):
            is_hard = (
                bool(eligible[local])
                and (bool(stop[local]) or not bool(policy_hit[local]))
            ) or (
                bool(terminal[local]) and not bool(stop[local])
            ) or (
                not bool(eligible[local])
                and not bool(terminal[local])
                and bool(finite_candidate[local])
                and bool(stop[local])
            )
            if is_hard:
                hard.append(global_index)
    return sorted(set(hard))


@torch.inference_mode()
def _latent_geometry_diagnostics(
    model: GoalConditionedValueGeometry,
    states: Sequence[torch.Tensor],
    goals: Sequence[torch.Tensor],
    *,
    seed: int,
    sample_count: int = 2048,
) -> dict[str, Any]:
    """Empirically describe the learned energy without asserting metric axioms."""

    if not states or not goals:
        return {"sample_count": 0}
    state = torch.cat(list(states), dim=0)
    goal = torch.cat(list(goals), dim=0)
    count = min(len(state), len(goal), sample_count)
    state = state[:count]
    goal = goal[:count]
    forward = model.energy_head(state, goal)
    reverse = model.energy_head(goal, state)
    scale = 0.5 * (forward.abs() + reverse.abs()) + 1e-8

    points = torch.cat([state, goal], dim=0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    triplet_count = min(sample_count, max(1, len(points) * 2))
    indices = torch.randint(
        len(points), (3, triplet_count), generator=generator, device="cpu"
    ).to(points.device)
    left, middle, right = (points[indices[position]] for position in range(3))
    left_right = model.energy_head(left, right)
    left_middle = model.energy_head(left, middle)
    middle_right = model.energy_head(middle, right)
    tolerance = 1e-5 * (
        left_right.abs() + left_middle.abs() + middle_right.abs() + 1.0
    )
    self_energy = model.energy_head(points[:count], points[:count])
    output: dict[str, Any] = {
        "energy_family": model.energy_name,
        "sample_count": count,
        "state_to_goal_mean": float(forward.mean().item()),
        "goal_to_state_mean": float(reverse.mean().item()),
        "signed_forward_minus_reverse_mean": float((forward - reverse).mean().item()),
        "mean_relative_asymmetry": float(((forward - reverse).abs() / scale).mean().item()),
        "nonnegative_energy_fraction": float(
            torch.cat([forward, reverse, left_right]).ge(-1e-7).float().mean().item()
        ),
        "mean_absolute_self_energy": float(self_energy.abs().mean().item()),
        "sampled_triangle_violation_fraction": float(
            (left_right > left_middle + middle_right + tolerance).float().mean().item()
        ),
        "warning": (
            "These are empirical latent-space diagnostics. Squared Euclidean and flexible "
            "compatibility heads need not satisfy the ordinary triangle inequality."
        ),
    }
    energy = model.energy_head
    direction = getattr(energy, "direction", None)
    if callable(direction):
        output["learned_direction_norm"] = float(direction().norm().item())
    weight = getattr(energy, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        output["bilinear_weight_relative_antisymmetry"] = float(
            ((weight - weight.T).norm() / weight.norm().clamp_min(1e-8)).item()
        )
    projection = getattr(energy, "projection", None)
    if hasattr(projection, "weight"):
        singular = torch.linalg.svdvals(projection.weight.float())
        output["projection_singular_value_max"] = float(singular.max().item())
        output["projection_singular_value_min"] = float(singular.min().item())
    return output


@torch.inference_mode()
def evaluate_value_geometry(
    model: GoalConditionedValueGeometry,
    store: SearchFeatureStore,
    split: str,
    device: torch.device,
    *,
    target_scale: float,
    batch_size: int = 256,
    completion_threshold: float = 0.5,
    reachability_threshold: float = 0.5,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 17,
    _prepared_batches: Sequence[
        tuple[list[int], dict[str, Any], dict[str, torch.Tensor]]
    ]
    | None = None,
) -> dict[str, Any]:
    indices = store.indices(split)
    if not indices:
        return {"count": 0, "task_count": 0, "split": split}
    model.eval()
    q_errors: list[float] = []
    value_errors: list[float] = []
    pair_hits: list[float] = []
    action_reach_prob: list[float] = []
    action_reach_target: list[float] = []
    state_reach_prob: list[float] = []
    state_reach_target: list[float] = []
    completion_prob: list[float] = []
    completion_target: list[float] = []
    policy_hits: list[float] = []
    gated_hits: list[float] = []
    q_only_hits: list[float] = []
    decision_hits: list[float] = []
    selected_regrets: list[float] = []
    selected_regret_known: list[float] = []
    selected_unreachable: list[float] = []
    per_task_hits: dict[str, list[float]] = defaultdict(list)
    per_task_decision_hits: dict[str, list[float]] = defaultdict(list)
    per_task_regrets: dict[str, list[float]] = defaultdict(list)
    per_group_hits: dict[str, dict[str, list[float]]] = {
        "workflow_family": defaultdict(list),
        "assembly": defaultdict(list),
        "source_revision": defaultdict(list),
        "requested_evidence_count": defaultdict(list),
        "primary_degraded_count": defaultdict(list),
        "remaining_path_length": defaultdict(list),
        "held_out_finish_tool": defaultdict(list),
    }
    train_candidate_tools = {
        str(tool_id)
        for row in store.examples
        if row.get("split") == "train"
        for tool_id in row.get("candidate_tools", [])
    }
    latent_states: list[torch.Tensor] = []
    latent_goals: list[torch.Tensor] = []
    hard_indices: list[int] = []

    prepared_batches = _prepared_batches or _prepare_evaluation_batches(
        model, store, split, device, batch_size
    )
    flattened = [index for batch_indices, _, _ in prepared_batches for index in batch_indices]
    if flattened != indices:
        raise ValueError("Prepared evaluation batches do not match the requested split/order")
    for batch_indices, batch, output in prepared_batches:
        latent_states.append(output["state_latent"].detach())
        latent_goals.append(output["goal_latent"].detach())
        candidate = batch["policy_candidate_mask"]
        finite_candidate = candidate.any(dim=1)
        optimal = batch["optimal_action_mask"] & candidate
        eligible = optimal.any(dim=1)

        policy_index = output["policy_logits"].argmax(dim=1)
        q_only = output["q"].masked_fill(~candidate, torch.inf).argmin(dim=1)
        gated_allowed = candidate & (
            output["action_reachability_logit"].sigmoid() >= reachability_threshold
        )
        gated_index = output["q"].masked_fill(~gated_allowed, torch.inf).argmin(dim=1)
        stop_probability = output["completion_logit"].sigmoid()
        stop_prediction = stop_probability >= completion_threshold
        terminal = batch["completion_target"] > 0.5

        for local, global_index in enumerate(batch_indices):
            row = batch["rows"][local]
            task_id = str(row["task_id"])
            if bool(eligible[local]):
                policy_hit = float(optimal[local, policy_index[local]].item())
                q_hit = float(optimal[local, q_only[local]].item())
                gated_hit = (
                    float(optimal[local, gated_index[local]].item())
                    if bool(gated_allowed[local].any())
                    else 0.0
                )
                policy_hits.append(policy_hit)
                q_only_hits.append(q_hit)
                gated_hits.append(gated_hit)
                per_task_hits[task_id].append(policy_hit)
                provenance = row.get("provenance", {})
                requested = provenance.get("requested_evidence", [])
                degraded = provenance.get("primary_degraded", {})
                paths = row.get("top_k_paths", [])
                remaining_length = (
                    len(paths[0].get("tool_ids", [])) if paths else -1
                )
                finish_tools = [
                    str(tool_id)
                    for tool_id in row.get("candidate_tools", [])
                    if str(tool_id).startswith("finish_")
                ]
                group_values = {
                    "workflow_family": row.get("category", provenance.get("workflow_family")),
                    "assembly": row.get("state", {}).get("assembly", "missing"),
                    "source_revision": provenance.get("source_revision", "missing"),
                    "requested_evidence_count": (
                        len(requested) if isinstance(requested, list) else "missing"
                    ),
                    "primary_degraded_count": (
                        sum(bool(value) for value in degraded.values())
                        if isinstance(degraded, Mapping)
                        else int(bool(provenance.get("primary_annotation_degraded", False)))
                    ),
                    "remaining_path_length": remaining_length,
                    "held_out_finish_tool": any(
                        tool_id not in train_candidate_tools for tool_id in finish_tools
                    ),
                }
                for group, value in group_values.items():
                    per_group_hits[group][str(value)].append(policy_hit)
                selected = int(policy_index[local].item())
                regret = batch["regret"][local, selected]
                regret_known = bool(torch.isfinite(regret))
                selected_regret_known.append(float(regret_known))
                if regret_known:
                    selected_regrets.append(float(regret.item()))
                    per_task_regrets[task_id].append(float(regret.item()))
                reach_mask = bool(batch["action_reachability_mask"][local, selected])
                reach_target = batch["action_reachability_target"][local, selected]
                selected_unreachable.append(
                    float(reach_mask and bool(torch.isfinite(reach_target)) and reach_target < 0.5)
                )
                joint_hit = float((not bool(stop_prediction[local])) and policy_hit > 0.5)
                decision_hits.append(joint_hit)
                per_task_decision_hits[task_id].append(joint_hit)
                if joint_hit < 0.5:
                    hard_indices.append(global_index)
            elif bool(terminal[local]):
                joint_hit = float(stop_prediction[local])
                decision_hits.append(joint_hit)
                per_task_decision_hits[task_id].append(joint_hit)
                if not bool(stop_prediction[local]):
                    hard_indices.append(global_index)
            elif bool(finite_candidate[local]):
                # Known dead-end state: it trains reachability but has no oracle
                # optimal action, so it is excluded from action accuracy.
                if bool(stop_prediction[local]):
                    hard_indices.append(global_index)

        q_mask = batch["q_mask"]
        if q_mask.any():
            q_errors.extend(
                ((output["q"] * target_scale - batch["q_star"]).abs()[q_mask])
                .cpu()
                .tolist()
            )
        value_mask = batch["value_mask"]
        if value_mask.any():
            value_errors.extend(
                ((output["value"] * target_scale - batch["value_target"]).abs()[value_mask])
                .cpu()
                .tolist()
            )
        regret = batch["regret"]
        rank_mask = batch["regret_mask"] & candidate
        for row_index in range(len(batch_indices)):
            valid = torch.where(rank_mask[row_index])[0]
            for left_position in range(len(valid)):
                for right_position in range(left_position + 1, len(valid)):
                    left, right = valid[left_position], valid[right_position]
                    oracle_delta = regret[row_index, left] - regret[row_index, right]
                    if abs(float(oracle_delta)) <= 1e-7:
                        continue
                    predicted_delta = output["q"][row_index, left] - output["q"][row_index, right]
                    pair_hits.append(float(torch.sign(oracle_delta) == torch.sign(predicted_delta)))

        action_mask = batch["action_reachability_mask"] & candidate
        if action_mask.any():
            action_reach_prob.extend(
                output["action_reachability_logit"].sigmoid()[action_mask].cpu().tolist()
            )
            action_reach_target.extend(
                batch["action_reachability_target"][action_mask].cpu().tolist()
            )
        state_mask = batch["state_reachability_mask"]
        if state_mask.any():
            state_reach_prob.extend(
                output["state_reachability_logit"].sigmoid()[state_mask].cpu().tolist()
            )
            state_reach_target.extend(batch["state_reachability_target"][state_mask].cpu().tolist())
        completion_mask = batch["completion_mask"]
        if completion_mask.any():
            completion_prob.extend(stop_probability[completion_mask].cpu().tolist())
            completion_target.extend(batch["completion_target"][completion_mask].cpu().tolist())

    task_interval = _bootstrap_task_mean(
        per_task_hits, reps=bootstrap_reps, seed=bootstrap_seed
    )
    return {
        "split": split,
        "count": len(indices),
        "task_count": len({store.examples[index]["task_id"] for index in indices}),
        "action_evaluable_states": len(policy_hits),
        "policy_optimal_set_accuracy": _mean(policy_hits),
        "q_only_optimal_set_accuracy": _mean(q_only_hits),
        "gated_optimal_set_accuracy": _mean(gated_hits),
        "joint_stop_action_accuracy": _mean(decision_hits),
        "task_macro_policy_accuracy": task_interval,
        "per_task_policy_accuracy": {
            task_id: float(np.mean(values))
            for task_id, values in sorted(per_task_hits.items())
            if values
        },
        "per_task_joint_accuracy": {
            task_id: float(np.mean(values))
            for task_id, values in sorted(per_task_decision_hits.items())
            if values
        },
        "per_task_regret_at_1": {
            task_id: float(np.mean(values))
            for task_id, values in sorted(per_task_regrets.items())
            if values
        },
        "regret_at_1": _mean(selected_regrets),
        "regret_at_1_label_coverage": _mean(selected_regret_known),
        "selected_known_unreachable_rate": _mean(selected_unreachable),
        "pairwise_regret_order_accuracy": _mean(pair_hits),
        "q_mae": _mean(q_errors),
        "value_mae": _mean(value_errors),
        "action_reachability": _binary_metrics(
            np.asarray(action_reach_prob),
            np.asarray(action_reach_target),
            threshold=reachability_threshold,
        ),
        "state_reachability": _binary_metrics(
            np.asarray(state_reach_prob),
            np.asarray(state_reach_target),
            threshold=reachability_threshold,
        ),
        "completion": _binary_metrics(
            np.asarray(completion_prob),
            np.asarray(completion_target),
            threshold=completion_threshold,
        ),
        "slice_policy_accuracy": {
            group: {value: _mean(rows) for value, rows in sorted(groups.items())}
            for group, groups in per_group_hits.items()
        },
        "latent_geometry": _latent_geometry_diagnostics(
            model, latent_states, latent_goals, seed=bootstrap_seed + 91_003
        ),
        "thresholds": {
            "completion": completion_threshold,
            "action_reachability": reachability_threshold,
        },
        "hard_example_indices": sorted(set(hard_indices)),
    }


def _choose_dev_thresholds(
    model: GoalConditionedValueGeometry,
    store: SearchFeatureStore,
    device: torch.device,
    *,
    target_scale: float,
    batch_size: int,
    grid: Sequence[float],
) -> tuple[float, float, dict[str, Any]]:
    if not grid or any(not 0 < float(value) < 1 for value in grid):
        raise ValueError("threshold_grid must contain probabilities strictly between 0 and 1")
    prepared_batches = _prepare_evaluation_batches(
        model, store, "dev", device, batch_size
    )
    best: tuple[float, float, float, dict[str, Any]] | None = None
    for completion in sorted(set(float(value) for value in grid)):
        for reachability in sorted(set(float(value) for value in grid)):
            score = _threshold_selection_score(
                prepared_batches,
                completion_threshold=completion,
                reachability_threshold=reachability,
            )
            tie = -abs(completion - 0.5) - abs(reachability - 0.5)
            candidate = (score, tie, -completion - reachability)
            if best is None or candidate[:3] > best[:3]:
                best = (*candidate, {})
                selected = (completion, reachability)
    assert best is not None
    metrics = evaluate_value_geometry(
        model,
        store,
        "dev",
        device,
        target_scale=target_scale,
        batch_size=batch_size,
        completion_threshold=selected[0],
        reachability_threshold=selected[1],
        _prepared_batches=prepared_batches,
    )
    metrics["dev_selection_score"] = best[0]
    return selected[0], selected[1], metrics


def _loss_weights(config: Mapping[str, Any]) -> ValueGeometryLossWeights:
    defaults = ValueGeometryLossWeights()
    names = defaults.__dataclass_fields__
    supplied = config.get("loss_weights", {})
    unknown = sorted(set(supplied) - set(names))
    if unknown:
        raise ValueError(f"Unknown value loss weights: {unknown}")
    return ValueGeometryLossWeights(
        **{name: float(supplied.get(name, getattr(defaults, name))) for name in names}
    )


def train_value_geometry(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    energy_override: str | None = None,
    seed_override: int | None = None,
    include_test: bool = False,
    replay_example_ids: Sequence[str] | None = None,
    external_replay_fraction: float | None = None,
    feature_ablation_override: Mapping[str, Any] | None = None,
    model_capacity_override: Mapping[str, int] | None = None,
    _store_override: SearchFeatureStore | None = None,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    train_config = dict(config.get("value_training", {}))
    if model_capacity_override is not None:
        unknown_capacity = sorted(
            set(model_capacity_override) - {"shared_dim", "hidden_dim", "energy_rank"}
        )
        if unknown_capacity:
            raise ValueError(f"Unknown model-capacity overrides: {unknown_capacity}")
        train_config.update(
            {name: int(value) for name, value in model_capacity_override.items()}
        )
    seed = int(seed_override if seed_override is not None else config.get("seed", 17))
    energy = energy_override or str(train_config.get("energy", "directed_quasimetric"))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    run_provenance = collect_run_provenance(config_path, device=device)
    source_tree_sha256 = run_provenance.get("source_tree_sha256")
    if not source_tree_sha256:
        raise RuntimeError("Cannot bind value training to a source-tree fingerprint")
    structured_dim = int(train_config.get("structured_dim", 64))
    if _store_override is None:
        store = SearchFeatureStore(processed_dir, cache_dir, structured_dim=structured_dim)
    else:
        store = _store_override
        if store.structured_dim != structured_dim:
            raise ValueError("Shared feature store has a different structured width")
        if store.processed_dir.resolve() != Path(processed_dir).resolve():
            raise ValueError("Shared feature store uses a different processed dataset")
        if store.cache.root.resolve() != Path(cache_dir).resolve():
            raise ValueError("Shared feature store uses a different embedding cache")
    store.apply_feature_ablation(
        feature_ablation_override
        if feature_ablation_override is not None
        else train_config.get("feature_ablation")
    )
    train_config["feature_ablation"] = store.feature_ablation
    train_indices = store.indices("train")
    dev_indices = store.indices("dev")
    if not train_indices:
        raise ValueError("No training search states")
    if not dev_indices:
        raise ValueError("No dev search states; checkpoint selection cannot use test")
    if include_test and not store.indices("test"):
        raise ValueError("Test was unsealed explicitly but contains no search states")
    replay_ids = [str(value) for value in replay_example_ids or []]
    missing_replay = sorted(set(replay_ids) - set(store.example_index))
    if missing_replay:
        raise ValueError(f"External replay references unknown examples: {missing_replay}")
    train_set = set(train_indices)
    nontrain_replay = sorted(
        {
            example_id
            for example_id in replay_ids
            if store.example_index[example_id] not in train_set
        }
    )
    if nontrain_replay:
        raise ValueError(
            "External replay may contain train examples only; "
            f"nontrain={nontrain_replay}"
        )
    replay_indices = [store.example_index[example_id] for example_id in replay_ids]
    dagger_config = config.get("search_dagger", {})
    if not isinstance(dagger_config, Mapping):
        raise ValueError("search_dagger must be a mapping")
    replay_fraction = float(
        external_replay_fraction
        if external_replay_fraction is not None
        else dagger_config.get("replay_fraction", 0.0) if replay_ids else 0.0
    )
    if not 0 <= replay_fraction <= 1:
        raise ValueError("external_replay_fraction must be in [0, 1]")
    if replay_fraction and not replay_indices:
        raise ValueError("external_replay_fraction is positive but no replay examples were given")
    replay_spec = {
        "count": len(replay_ids),
        "unique_examples": len(set(replay_ids)),
        "ordered_ids_sha256": sha256_text(canonical_json(replay_ids)),
        "fraction": replay_fraction,
        "train_only": True,
    }
    effective_config = {**train_config, "energy": energy, "seed": seed}
    if replay_ids:
        effective_config["external_replay"] = replay_spec
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "value_geometry.pt"
    metrics_path = output_dir / "value_metrics.json"
    resume_path = output_dir / "training_resume.pt"
    progress_path = output_dir / "training_progress.json"
    experiment_hash = sha256_file(config_path)
    training_hash = sha256_text(canonical_json(effective_config))
    progress_backup_path = _value_progress_backup_path(
        output_dir,
        energy=energy,
        seed=seed,
        training_hash=training_hash,
    )
    resume_enabled = bool(train_config.get("resume", True))
    if checkpoint_path.exists() and metrics_path.exists():
        existing = read_json(metrics_path)
        if existing.get("training_result_version") != VALUE_TRAINING_RESULT_VERSION:
            raise RuntimeError(
                "Completed value run predates the current reproducibility contract; "
                f"use a new output directory instead of overwriting {output_dir}"
            )
        else:
            expected = {
                "energy": energy,
                "seed": seed,
                "experiment_config_sha256": experiment_hash,
                "training_config_sha256": training_hash,
                "cache_content_sha256": store.cache.content_sha256,
                "source_tree_sha256": source_tree_sha256,
            }
            mismatches = {
                name: {"expected": value, "actual": existing.get(name)}
                for name, value in expected.items()
                if existing.get(name) != value
            }
            actual_checkpoint_sha = sha256_file(checkpoint_path)
            if existing.get("checkpoint_sha256") != actual_checkpoint_sha:
                mismatches["checkpoint_sha256"] = {
                    "expected": existing.get("checkpoint_sha256"),
                    "actual": actual_checkpoint_sha,
                }
            if mismatches:
                raise RuntimeError(
                    f"Refusing to overwrite a completed incompatible value run: {mismatches}"
                )
            return existing
    scale = _target_scale(store, train_indices)
    model = GoalConditionedValueGeometry(
        store.view_dims,
        structured_dim=structured_dim,
        shared_dim=int(train_config.get("shared_dim", 64)),
        hidden_dim=int(train_config.get("hidden_dim", 128)),
        energy=energy,
        energy_rank=train_config.get("energy_rank"),
        dropout=float(train_config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(train_config.get("epochs", 100)))
    )
    weights = _loss_weights(train_config)
    epochs = int(train_config.get("epochs", 100))
    batch_size = int(train_config.get("batch_size", 64))
    patience = int(train_config.get("patience", 15))
    max_states = train_config.get("max_states_per_task")
    max_states = int(max_states) if max_states is not None else None
    hard_fraction = float(train_config.get("hard_replay_fraction", 0.25))
    hard_warmup = int(train_config.get("hard_replay_warmup", 5))
    if not 0 <= hard_fraction <= 1:
        raise ValueError("hard_replay_fraction must be in [0, 1]")
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    hard_pool: list[int] = []
    history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    start_epoch = 1
    if resume_enabled and resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        expected = {
            "resume_version": VALUE_RESUME_VERSION,
            "training_config_sha256": training_hash,
            "cache_content_sha256": store.cache.content_sha256,
            "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
            "source_tree_sha256": source_tree_sha256,
            "model_config": model.export_config(),
        }
        mismatches = {
            key: {"expected": value, "found": resume.get(key)}
            for key, value in expected.items()
            if resume.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Value-geometry resume checkpoint does not match this immutable run: "
                f"{mismatches}"
            )
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best_score = float(resume["best_score"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        stale = int(resume["stale"])
        hard_pool = [int(value) for value in resume["hard_pool"]]
        history = list(resume["history"])
        rng.setstate(resume["sampling_rng_state"])
        _restore_global_rng_state(resume)
        start_epoch = int(resume["completed_epoch"]) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience:
            break
        epoch_indices = _task_balanced_epoch(
            store,
            train_indices,
            rng,
            max_states_per_task=max_states,
            hard_pool=hard_pool,
            hard_replay_fraction=hard_fraction if epoch > hard_warmup else 0.0,
            external_replay_pool=replay_indices,
            external_replay_fraction=replay_fraction,
        )
        model.train()
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        for start in range(0, len(epoch_indices), batch_size):
            selected = epoch_indices[start : start + batch_size]
            batch = store.batch(selected, device)
            loss, components = _loss_for_batch(
                model,
                batch,
                target_scale=scale,
                weights=weights,
                action_temperature=float(train_config.get("action_temperature", 1.0)),
                regret_temperature=float(train_config.get("regret_temperature", 0.25)),
                ranking_margin=float(train_config.get("ranking_margin", 0.1)),
                supervision=str(train_config.get("supervision", "boltzmann")),
                view_dropout=float(train_config.get("view_dropout", 0.0)),
                embedding_noise_std=float(train_config.get("embedding_noise_std", 0.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config.get("gradient_clip", 1.0))
            )
            optimizer.step()
            for name, value in components.items():
                totals[name] += value * len(selected)
            seen += len(selected)
        scheduler.step()
        epoch_completion, epoch_reachability, dev = _choose_dev_thresholds(
            model,
            store,
            device,
            target_scale=scale,
            batch_size=batch_size,
            grid=train_config.get("threshold_grid", [0.3, 0.5, 0.7]),
        )
        score = float(dev["dev_selection_score"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "train": {name: value / max(1, seen) for name, value in totals.items()},
                "dev_joint_stop_action_accuracy": dev.get("joint_stop_action_accuracy"),
                "dev_selection_score": score,
                "dev_policy_optimal_set_accuracy": dev.get("policy_optimal_set_accuracy"),
                "dev_regret_at_1": dev.get("regret_at_1"),
                "dev_completion_balanced_accuracy": dev["completion"]["balanced_accuracy"],
                "dev_action_reachability_balanced_accuracy": dev["action_reachability"][
                    "balanced_accuracy"
                ],
                "dev_thresholds": {
                    "completion": epoch_completion,
                    "action_reachability": epoch_reachability,
                },
                "hard_replay_pool": len(hard_pool),
                "external_replay_pool": len(replay_indices),
                "external_replay_fraction": replay_fraction,
            }
        )
        if epoch >= hard_warmup:
            train_predictions = _prepare_evaluation_batches(
                model, store, "train", device, batch_size
            )
            hard_pool = _hard_example_indices(train_predictions)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        resume_state = {
            "resume_version": VALUE_RESUME_VERSION,
            "training_config_sha256": training_hash,
            "cache_content_sha256": store.cache.content_sha256,
            "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
            "source_tree_sha256": source_tree_sha256,
            "model_config": model.export_config(),
            "completed_epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale": stale,
            "hard_pool": hard_pool,
            "history": history,
            "sampling_rng_state": rng.getstate(),
            **_capture_global_rng_state(),
        }
        _atomic_torch_save(resume_state, resume_path)
        progress = {
            "status": "early_stopping" if stale >= patience else "running",
            "output_dir": str(output_dir),
            "energy": energy,
            "seed": seed,
            "completed_epoch": epoch,
            "configured_epochs": epochs,
            "best_epoch": best_epoch,
            "best_dev_selection_score": best_score,
            "stale_epochs": stale,
            "patience": patience,
            "training_config_sha256": training_hash,
            "cache_content_sha256": store.cache.content_sha256,
            "resume_checkpoint": str(resume_path),
            "feature_ablation": store.feature_ablation,
            "model_capacity_override": dict(model_capacity_override or {}),
            "latest": history[-1],
        }
        _atomic_write_json(progress_path, progress)
        _write_optional_progress_backup(progress_backup_path, progress)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Value training did not produce a checkpoint")
    model.load_state_dict(best_state)
    threshold_grid = train_config.get("threshold_grid", [0.3, 0.5, 0.7])
    completion_threshold, reachability_threshold, calibrated_dev = _choose_dev_thresholds(
        model,
        store,
        device,
        target_scale=scale,
        batch_size=batch_size,
        grid=threshold_grid,
    )
    report_splits = ("train", "dev", "test") if include_test else ("train", "dev")
    metrics = {
        split: evaluate_value_geometry(
            model,
            store,
            split,
            device,
            target_scale=scale,
            batch_size=batch_size,
            completion_threshold=completion_threshold,
            reachability_threshold=reachability_threshold,
            bootstrap_reps=int(train_config.get("bootstrap_reps", 1000)),
            bootstrap_seed=seed + (0 if split == "train" else 1),
        )
        for split in report_splits
    }
    for split_metrics in metrics.values():
        split_metrics.pop("hard_example_indices", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "kind": "search_value_geometry",
        "seed": seed,
        "experiment_config_sha256": experiment_hash,
        "training_config_sha256": training_hash,
        "training_config": effective_config,
        "model_config": model.export_config(),
        "model_state": model.state_dict(),
        "energy": energy,
        "target_scale": scale,
        "completion_threshold": completion_threshold,
        "reachability_threshold": reachability_threshold,
        "tool_ids": store.tool_ids,
        "views": list(store.views),
        "view_roles": store.view_roles,
        "feature_ablation": store.feature_ablation,
        "feature_spec_version": store.cache.manifest["feature_spec_version"],
        "cache_config_sha256": store.cache.manifest["config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "cache_evaluation_fingerprint_sha256": (
            store.cache.evaluation_fingerprint_sha256
        ),
        "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
        "source_tree_sha256": source_tree_sha256,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    result = {
        "training_result_version": VALUE_TRAINING_RESULT_VERSION,
        "energy": energy,
        "seed": seed,
        "experiment_config_sha256": experiment_hash,
        "training_config_sha256": training_hash,
        "source_tree_sha256": source_tree_sha256,
        "cache_content_sha256": store.cache.content_sha256,
        "cache_evaluation_fingerprint_sha256": (
            store.cache.evaluation_fingerprint_sha256
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "best_dev_selection_score": best_score,
        "selected_thresholds": {
            "completion": completion_threshold,
            "action_reachability": reachability_threshold,
        },
        "calibration_dev": {
            key: value
            for key, value in calibrated_dev.items()
            if key != "hard_example_indices"
        },
        "target_scale": scale,
        "parameter_count": checkpoint["parameter_count"],
        "train_states": len(train_indices),
        "dev_states": len(dev_indices),
        "view_roles": store.view_roles,
        "feature_ablation": store.feature_ablation,
        "metrics": metrics,
        "history": history,
        "test_reported": include_test,
        "run_provenance": run_provenance,
        "external_replay": replay_spec,
    }
    _atomic_write_json(metrics_path, result)
    completed_progress = {
        "status": "complete",
        "output_dir": str(output_dir),
        "energy": energy,
        "seed": seed,
        "completed_epoch": history[-1]["epoch"],
        "configured_epochs": epochs,
        "best_epoch": best_epoch,
        "best_dev_selection_score": best_score,
        "stale_epochs": stale,
        "patience": patience,
        "training_config_sha256": training_hash,
        "cache_content_sha256": store.cache.content_sha256,
        "resume_checkpoint": str(resume_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "metrics": str(metrics_path),
        "feature_ablation": store.feature_ablation,
        "model_capacity_override": dict(model_capacity_override or {}),
        "latest": history[-1],
    }
    _atomic_write_json(progress_path, completed_progress)
    _write_optional_progress_backup(progress_backup_path, completed_progress)
    return result


def load_value_geometry_checkpoint(
    path: str | Path, device: torch.device | str
) -> tuple[GoalConditionedValueGeometry, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {checkpoint.get('checkpoint_version')!r}")
    if checkpoint.get("kind") != "search_value_geometry":
        raise ValueError(f"Not a search value-geometry checkpoint: {path}")
    if checkpoint.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError("Value checkpoint feature semantics differ from this runtime")
    model = GoalConditionedValueGeometry(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def compare_value_geometries(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    energies: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    if not energies or not seeds:
        raise ValueError("Geometry comparison requires at least one energy and seed")
    output_dir = Path(output_dir)
    config = read_yaml(config_path)
    train_config = config.get("value_training", {})
    if not isinstance(train_config, Mapping):
        raise ValueError("value_training must be a mapping")
    shared_store = SearchFeatureStore(
        processed_dir,
        cache_dir,
        structured_dim=int(train_config.get("structured_dim", 64)),
    )
    rows: list[dict[str, Any]] = []
    for energy in energies:
        for seed in seeds:
            run_dir = output_dir / energy / f"seed-{int(seed)}"
            result = train_value_geometry(
                processed_dir,
                cache_dir,
                run_dir,
                config_path,
                energy_override=energy,
                seed_override=int(seed),
                include_test=False,
                _store_override=shared_store,
            )
            rows.append(
                {
                    "energy": energy,
                    "seed": int(seed),
                    "parameter_count": result["parameter_count"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "dev": result["metrics"]["dev"],
                }
            )
    aggregates = []
    for energy in energies:
        selected = [row for row in rows if row["energy"] == energy]
        scores = [
            float(row["dev"].get("joint_stop_action_accuracy") or 0.0) for row in selected
        ]
        regrets = [
            float(row["dev"]["regret_at_1"])
            for row in selected
            if row["dev"].get("regret_at_1") is not None
        ]
        aggregates.append(
            {
                "energy": energy,
                "runs": len(selected),
                "mean_dev_joint_accuracy": float(np.mean(scores)),
                "std_dev_joint_accuracy": float(np.std(scores)),
                "mean_dev_regret_at_1": float(np.mean(regrets)) if regrets else None,
                "parameter_counts": sorted({row["parameter_count"] for row in selected}),
                "mean_latent_geometry": {
                    key: float(np.mean(values))
                    for key in sorted(
                        {
                            key
                            for row in selected
                            for key, value in row["dev"].get(
                                "latent_geometry", {}
                            ).items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                        }
                    )
                    if (
                        values := [
                            float(row["dev"]["latent_geometry"][key])
                            for row in selected
                            if isinstance(
                                row["dev"].get("latent_geometry", {}).get(key),
                                (int, float),
                            )
                            and not isinstance(
                                row["dev"].get("latent_geometry", {}).get(key), bool
                            )
                        ]
                    )
                },
            }
        )
    aggregates.sort(
        key=lambda row: (
            -row["mean_dev_joint_accuracy"],
            (
                row["mean_dev_regret_at_1"]
                if row["mean_dev_regret_at_1"] is not None
                else math.inf
            ),
        )
    )
    selected_energy = aggregates[0]["energy"]

    def paired_task_effect(
        left_energy: str,
        right_energy: str,
        seed: int,
        *,
        metric_key: str = "per_task_policy_accuracy",
    ) -> dict[str, Any]:
        """Bootstrap task-macro metric differences after averaging matched seeds."""

        left_by_seed = {
            int(row["seed"]): row["dev"].get(metric_key, {})
            for row in rows
            if row["energy"] == left_energy
        }
        right_by_seed = {
            int(row["seed"]): row["dev"].get(metric_key, {})
            for row in rows
            if row["energy"] == right_energy
        }
        matched_seeds = sorted(set(left_by_seed) & set(right_by_seed))
        if not matched_seeds:
            return {"mean": None, "low": None, "high": None, "tasks": 0, "seeds": []}
        task_ids = sorted(
            set.intersection(
                *(
                    set(left_by_seed[run_seed]) & set(right_by_seed[run_seed])
                    for run_seed in matched_seeds
                )
            )
        )
        differences = {
            task_id: [
                float(
                    np.mean(
                        [
                            float(left_by_seed[run_seed][task_id])
                            - float(right_by_seed[run_seed][task_id])
                            for run_seed in matched_seeds
                        ]
                    )
                )
            ]
            for task_id in task_ids
        }
        interval = _bootstrap_task_mean(differences, reps=2000, seed=seed)
        output: dict[str, Any] = interval or {
            "mean": None,
            "low": None,
            "high": None,
            "tasks": 0,
        }
        output["seeds"] = matched_seeds
        return output

    euclidean_reference = "euclidean" if "euclidean" in energies else selected_energy
    def paired_family(metric_key: str, seed_offset: int) -> dict[str, Any]:
        return {
            "reference_energy": euclidean_reference,
            "energy_minus_reference": {
                energy: paired_task_effect(
                    energy,
                    euclidean_reference,
                    seed_offset + index,
                    metric_key=metric_key,
                )
                for index, energy in enumerate(energies)
            },
            "selected_energy": selected_energy,
            "selected_minus_energy": {
                energy: paired_task_effect(
                    selected_energy,
                    energy,
                    seed_offset + 10_000 + index,
                    metric_key=metric_key,
                )
                for index, energy in enumerate(energies)
            },
        }

    paired_task_macro_policy = paired_family("per_task_policy_accuracy", 840_000)
    paired_task_macro_policy["interpretation"] = (
        "Positive differences favor the left energy for policy accuracy. Intervals "
        "resample held-out dev tasks after averaging matched-seed differences."
    )
    paired_task_macro = {
        "policy_accuracy": paired_task_macro_policy,
        "joint_stop_action_accuracy": paired_family(
            "per_task_joint_accuracy", 860_000
        ),
        "regret_at_1": paired_family("per_task_regret_at_1", 880_000),
        "interpretation": (
            "Intervals resample held-out dev tasks after averaging paired differences "
            "over matched seeds. Positive accuracy differences and negative regret "
            "differences favor the left energy. A 95% interval excluding zero supports "
            "a task-macro difference; selection remains dev-only until one sealed-test "
            "confirmation."
        ),
    }
    report = {
        "selection_split": "dev",
        "test_sealed": True,
        "energies": list(energies),
        "seeds": [int(seed) for seed in seeds],
        "runs": rows,
        "aggregate": aggregates,
        "selected_energy": selected_energy,
        "paired_task_macro_policy": paired_task_macro_policy,
        "paired_task_macro": paired_task_macro,
        "experiment_config_sha256": sha256_file(config_path),
    }
    write_json(output_dir / "comparison.json", report)
    return report


def compare_existing_value_run_groups(
    left_dir: str | Path,
    right_dir: str | Path,
    output_path: str | Path,
    *,
    left_label: str,
    right_label: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Compare completed value runs without retraining or touching sealed test data.

    Each directory must contain ``seed-N/value_metrics.json``. Effects are first
    paired within seed and independent dev task, then averaged over matched seeds
    before the task bootstrap. Requiring identical task IDs avoids silently
    changing the estimand through an intersection of incomplete evaluations.
    """

    if not left_label or not right_label or left_label == right_label:
        raise ValueError("Run-group labels must be distinct and non-empty")
    matched_seeds = sorted({int(seed) for seed in seeds})
    if not matched_seeds:
        raise ValueError("Run-group comparison requires at least one seed")

    metric_specs = {
        "policy_accuracy": (
            "policy_optimal_set_accuracy",
            "per_task_policy_accuracy",
            "positive",
        ),
        "joint_stop_action_accuracy": (
            "joint_stop_action_accuracy",
            "per_task_joint_accuracy",
            "positive",
        ),
        "regret_at_1": ("regret_at_1", "per_task_regret_at_1", "negative"),
    }

    def load_group(root: Path, label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for seed in matched_seeds:
            path = root / f"seed-{seed}" / "value_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} result for seed {seed}: {path}")
            result = read_json(path)
            if bool(result.get("test_reported", False)):
                raise ValueError(f"Comparison input unexpectedly reports test metrics: {path}")
            if int(result.get("seed", seed)) != seed:
                raise ValueError(f"Seed metadata mismatch in {path}")
            dev = result.get("metrics", {}).get("dev")
            if not isinstance(dev, Mapping) or dev.get("split") not in {None, "dev"}:
                raise ValueError(f"Missing dev-only metrics in {path}")
            rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "metrics_sha256": sha256_file(path),
                    "checkpoint_sha256": result.get("checkpoint_sha256"),
                    "experiment_config_sha256": result.get("experiment_config_sha256"),
                    "training_config_sha256": result.get("training_config_sha256"),
                    "parameter_count": int(result["parameter_count"]),
                    "dev": dev,
                }
            )
        return rows

    left_rows = load_group(Path(left_dir), left_label)
    right_rows = load_group(Path(right_dir), right_label)

    def aggregate(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
        output: dict[str, Any] = {
            "label": label,
            "runs": len(rows),
            "parameter_counts": sorted({int(row["parameter_count"]) for row in rows}),
        }
        for output_key, (scalar_key, _, _) in metric_specs.items():
            values = [float(row["dev"][scalar_key]) for row in rows]
            output[f"mean_dev_{output_key}"] = float(np.mean(values))
            output[f"std_dev_{output_key}"] = float(np.std(values))
        return output

    effects: dict[str, Any] = {}
    for offset, (name, (_, per_task_key, favorable_sign)) in enumerate(metric_specs.items()):
        left_by_seed = {
            int(row["seed"]): row["dev"].get(per_task_key, {}) for row in left_rows
        }
        right_by_seed = {
            int(row["seed"]): row["dev"].get(per_task_key, {}) for row in right_rows
        }
        task_ids: list[str] | None = None
        for seed in matched_seeds:
            left_tasks = set(left_by_seed[seed])
            right_tasks = set(right_by_seed[seed])
            if not left_tasks or left_tasks != right_tasks:
                raise ValueError(
                    f"Dev task IDs differ for {name}, seed {seed}: "
                    f"{len(left_tasks)} left versus {len(right_tasks)} right"
                )
            if task_ids is None:
                task_ids = sorted(left_tasks)
            elif left_tasks != set(task_ids):
                raise ValueError(f"Dev task IDs vary across seeds for {name}")
        assert task_ids is not None
        differences = {
            task_id: [
                float(
                    np.mean(
                        [
                            float(left_by_seed[seed][task_id])
                            - float(right_by_seed[seed][task_id])
                            for seed in matched_seeds
                        ]
                    )
                )
            ]
            for task_id in task_ids
        }
        interval = _bootstrap_task_mean(
            differences,
            reps=5000,
            seed=910_000 + offset,
        )
        assert interval is not None
        effects[name] = {
            "contrast": f"{left_label}_minus_{right_label}",
            "favorable_sign": favorable_sign,
            "matched_seeds": matched_seeds,
            **interval,
        }

    left_parameters = sorted({int(row["parameter_count"]) for row in left_rows})
    right_parameters = sorted({int(row["parameter_count"]) for row in right_rows})
    parameter_match: dict[str, Any] = {
        "left": left_parameters,
        "right": right_parameters,
        "exact": left_parameters == right_parameters,
    }
    if len(left_parameters) == len(right_parameters) == 1:
        parameter_match["relative_left_minus_right"] = (
            (left_parameters[0] - right_parameters[0]) / right_parameters[0]
        )

    report = {
        "comparison": f"{left_label}_versus_{right_label}",
        "selection_split": "dev",
        "test_sealed": True,
        "paired_unit": "independent_task_after_matched_seed_averaging",
        "seeds": matched_seeds,
        "parameter_match": parameter_match,
        "aggregate": [aggregate(left_rows, left_label), aggregate(right_rows, right_label)],
        "paired_task_macro_effects": effects,
        "inputs": left_rows + right_rows,
        "interpretation": (
            "Accuracy effects favor the left group when positive; regret favors it when "
            "negative. A 95% task-bootstrap interval excluding zero supports a dev-only "
            "difference. This comparison does not authorize sealed-test access."
        ),
    }
    _atomic_write_json(Path(output_path), report)
    return report


def compare_value_input_ablations(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Capacity-matched attribution of frozen and structured input channels."""

    if not seeds:
        raise ValueError("Input-ablation comparison requires at least one seed")
    cache = EmbeddingCache(cache_dir)
    view_names = sorted(cache.prototype_views + cache.state_only_views)
    variants: dict[str, dict[str, Any]] = {
        "full": {"zero_views": [], "zero_structured": False},
        "frozen_only": {"zero_views": [], "zero_structured": True},
        "structured_only": {"zero_views": ["*"], "zero_structured": False},
        "contract_only": {"zero_views": ["*"], "zero_structured": True},
    }
    for view in view_names:
        variants[f"view_only__{view}"] = {
            "zero_views": [name for name in view_names if name != view],
            "zero_structured": True,
        }
        variants[f"full_minus__{view}"] = {
            "zero_views": [view],
            "zero_structured": False,
        }
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for variant, ablation in variants.items():
        for seed in seeds:
            run_dir = output_dir / variant / f"seed-{int(seed)}"
            result = train_value_geometry(
                processed_dir,
                cache_dir,
                run_dir,
                config_path,
                seed_override=int(seed),
                include_test=False,
                feature_ablation_override=ablation,
            )
            rows.append(
                {
                    "variant": variant,
                    "feature_ablation": result["feature_ablation"],
                    "seed": int(seed),
                    "parameter_count": result["parameter_count"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "dev": result["metrics"]["dev"],
                }
            )
    aggregates: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        joint = [float(row["dev"]["joint_stop_action_accuracy"]) for row in selected]
        policy = [float(row["dev"]["policy_optimal_set_accuracy"]) for row in selected]
        regret = [float(row["dev"]["regret_at_1"]) for row in selected]
        aggregates.append(
            {
                "variant": variant,
                "runs": len(selected),
                "mean_dev_joint_accuracy": float(np.mean(joint)),
                "std_dev_joint_accuracy": float(np.std(joint)),
                "mean_dev_policy_accuracy": float(np.mean(policy)),
                "mean_dev_regret_at_1": float(np.mean(regret)),
                "parameter_counts": sorted({row["parameter_count"] for row in selected}),
            }
        )
    by_variant = {row["variant"]: row for row in aggregates}
    def paired_task_effect(left_variant: str, right_variant: str, seed: int) -> dict[str, Any]:
        left_runs = [row for row in rows if row["variant"] == left_variant]
        right_runs = [row for row in rows if row["variant"] == right_variant]
        left_tasks = [row["dev"]["per_task_policy_accuracy"] for row in left_runs]
        right_tasks = [row["dev"]["per_task_policy_accuracy"] for row in right_runs]
        task_ids = sorted(
            set.intersection(
                *(set(values) for values in [*left_tasks, *right_tasks])
            )
        )
        differences = {
            task_id: [
                float(np.mean([values[task_id] for values in left_tasks]))
                - float(np.mean([values[task_id] for values in right_tasks]))
            ]
            for task_id in task_ids
        }
        interval = _bootstrap_task_mean(differences, reps=2000, seed=seed)
        return interval or {"mean": None, "low": None, "high": None, "tasks": 0}

    full = by_variant["full"]
    effects = {
        "frozen_beyond_structured_joint_accuracy": (
            full["mean_dev_joint_accuracy"]
            - by_variant["structured_only"]["mean_dev_joint_accuracy"]
        ),
        "structured_beyond_frozen_joint_accuracy": (
            full["mean_dev_joint_accuracy"]
            - by_variant["frozen_only"]["mean_dev_joint_accuracy"]
        ),
        "full_beyond_contract_only_joint_accuracy": (
            full["mean_dev_joint_accuracy"]
            - by_variant["contract_only"]["mean_dev_joint_accuracy"]
        ),
        "paired_task_macro_policy": {
            "full_minus_structured_only": paired_task_effect(
                "full", "structured_only", 810_001
            ),
            "full_minus_frozen_only": paired_task_effect("full", "frozen_only", 810_002),
            "full_minus_contract_only": paired_task_effect(
                "full", "contract_only", 810_003
            ),
        },
        "per_view": {
            view: {
                "view_only_minus_contract_joint_accuracy": (
                    by_variant[f"view_only__{view}"]["mean_dev_joint_accuracy"]
                    - by_variant["contract_only"]["mean_dev_joint_accuracy"]
                ),
                "full_minus_without_view_joint_accuracy": (
                    full["mean_dev_joint_accuracy"]
                    - by_variant[f"full_minus__{view}"]["mean_dev_joint_accuracy"]
                ),
                "view_only_minus_contract_task_macro_policy": paired_task_effect(
                    f"view_only__{view}", "contract_only", 820_000 + index
                ),
                "full_minus_without_view_task_macro_policy": paired_task_effect(
                    "full", f"full_minus__{view}", 830_000 + index
                ),
            }
            for index, view in enumerate(view_names)
        },
    }
    report = {
        "selection_split": "dev",
        "test_sealed": True,
        "capacity_matched": True,
        "views": view_names,
        "seeds": [int(seed) for seed in seeds],
        "variants": variants,
        "runs": rows,
        "aggregate": aggregates,
        "effects": effects,
        "experiment_config_sha256": sha256_file(config_path),
        "interpretation": (
            "Zeroing is applied both to cached and runtime-encoded inputs while model "
            "architecture and parameter count remain fixed. Closed-loop evaluation is still "
            "required before claiming an operational contribution."
        ),
    }
    write_json(output_dir / "comparison.json", report)
    return report


def compare_value_capacities(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Separate representation/task limits from an under-capacity value head."""

    if not seeds:
        raise ValueError("Capacity comparison requires at least one seed")
    config = read_yaml(config_path)
    base = config.get("value_training", {})
    if not isinstance(base, Mapping):
        raise ValueError("value_training must be a mapping")
    small = {
        "shared_dim": int(base.get("shared_dim", 64)),
        "hidden_dim": int(base.get("hidden_dim", 128)),
        "energy_rank": int(base.get("energy_rank", 32)),
    }
    capacities = {
        "half_x": {
            "shared_dim": max(1, small["shared_dim"] // 2),
            "hidden_dim": max(1, small["hidden_dim"] // 2),
            "energy_rank": max(1, small["energy_rank"] // 2),
        },
        "configured": small,
        "two_x": {
            "shared_dim": 2 * small["shared_dim"],
            "hidden_dim": 2 * small["hidden_dim"],
            "energy_rank": 2 * small["energy_rank"],
        },
    }
    output_dir = Path(output_dir)
    shared_store = SearchFeatureStore(
        processed_dir,
        cache_dir,
        structured_dim=int(base.get("structured_dim", 64)),
    )
    rows: list[dict[str, Any]] = []
    for capacity, dimensions in capacities.items():
        for seed in seeds:
            result = train_value_geometry(
                processed_dir,
                cache_dir,
                output_dir / capacity / f"seed-{int(seed)}",
                config_path,
                seed_override=int(seed),
                include_test=False,
                model_capacity_override=dimensions,
                _store_override=shared_store,
            )
            rows.append(
                {
                    "capacity": capacity,
                    "dimensions": dimensions,
                    "seed": int(seed),
                    "parameter_count": result["parameter_count"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "train": result["metrics"]["train"],
                    "dev": result["metrics"]["dev"],
                }
            )
    aggregates: list[dict[str, Any]] = []
    for capacity in capacities:
        selected = [row for row in rows if row["capacity"] == capacity]
        train_joint = [
            float(row["train"]["joint_stop_action_accuracy"]) for row in selected
        ]
        dev_joint = [float(row["dev"]["joint_stop_action_accuracy"]) for row in selected]
        aggregates.append(
            {
                "capacity": capacity,
                "dimensions": capacities[capacity],
                "runs": len(selected),
                "parameter_counts": sorted({row["parameter_count"] for row in selected}),
                "mean_train_joint_accuracy": float(np.mean(train_joint)),
                "mean_dev_joint_accuracy": float(np.mean(dev_joint)),
                "std_dev_joint_accuracy": float(np.std(dev_joint)),
                "mean_train_minus_dev_gap": float(
                    np.mean(np.asarray(train_joint) - np.asarray(dev_joint))
                ),
            }
        )
    aggregate_by_capacity = {row["capacity"]: row for row in aggregates}
    reduced = aggregate_by_capacity["half_x"]
    configured = aggregate_by_capacity["configured"]
    enlarged = aggregate_by_capacity["two_x"]
    report = {
        "selection_split": "dev",
        "test_sealed": True,
        "seeds": [int(seed) for seed in seeds],
        "capacities": capacities,
        "runs": rows,
        "aggregate": aggregates,
        "half_x_minus_configured_dev_joint_accuracy": (
            reduced["mean_dev_joint_accuracy"] - configured["mean_dev_joint_accuracy"]
        ),
        "two_x_minus_configured_dev_joint_accuracy": (
            enlarged["mean_dev_joint_accuracy"] - configured["mean_dev_joint_accuracy"]
        ),
        "experiment_config_sha256": sha256_file(config_path),
        "interpretation": (
            "A capacity diagnosis uses a half/configured/two-x curve. A larger head supports "
            "under-capacity only when it closes a training deficit and improves held-out dev "
            "across seeds; parameter count alone is not evidence."
        ),
    }
    write_json(output_dir / "comparison.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train search-distilled value geometry")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--energy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_value_geometry(
        args.processed_dir,
        args.cache_dir,
        args.output_dir,
        args.config,
        energy_override=args.energy,
        seed_override=args.seed,
        include_test=args.include_test,
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
