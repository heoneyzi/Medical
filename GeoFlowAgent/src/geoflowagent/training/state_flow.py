"""Train and evaluate a search-distilled multi-path State Flow.

The stage implemented here is deliberately narrow and auditable:

* exact ``top_k_paths`` are validated by :class:`SearchPathStore`;
* frozen cache arrays are assembled without fitting dataset statistics;
* a small conditional rectified flow generates successor-state/tool anchors;
* decoding is constrained only by model-visible contract/transition information;
* train/dev labels are loaded for fitting and selection, while test stays sealed;
* both open-loop sampled paths and receding-horizon replanning are reported.

This is a genuine generative path model, but it is still an offline snapshot
experiment.  It does not claim live-tool robustness or reproduce FlowPlan's full
environment/evaluation protocol.
"""

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
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.embeddings.cache import EmbeddingCache, verify_evaluation_cache_superset
from geoflowagent.models.search_state_flow import SearchStateFlow
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.search_paths import SearchPathReference, SearchPathStore
from geoflowagent.utils.io import canonical_json, read_yaml, sha256_file, sha256_text, write_json
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)

STATE_FLOW_FEATURE_VERSION = "geoflowagent.search-state-flow-feature.v1"
STATE_FLOW_RESUME_VERSION = "geoflowagent.search-state-flow-resume.v2"


def _mean(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(selected) if selected else None


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
            "Legacy State Flow resume lacks exact stochastic state; restart that partial "
            f"run instead of silently changing its trajectory: {missing}"
        )
    random.setstate(resume["python_rng_state"])
    np.random.set_state(resume["numpy_rng_state"])
    torch.set_rng_state(resume["torch_rng_state"].cpu())
    cuda_rng_states = resume.get("cuda_rng_states")
    if torch.cuda.is_available() and cuda_rng_states is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])


def _write_optional_progress_backup(path: Path | None, value: Any) -> None:
    """Mirror progress without letting transient persistent-storage pressure kill a run."""

    if path is None:
        return
    try:
        _atomic_write_json(path, value)
    except OSError as error:
        if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        warnings.warn(
            f"Persistent State Flow progress mirror unavailable ({error}); "
            "local resume is intact",
            RuntimeWarning,
            stacklevel=2,
        )


def _state_flow_progress_backup_path(
    output_dir: Path,
    *,
    seed: int,
    training_hash: str,
    configured: str | Path | None,
) -> Path | None:
    root = os.environ.get("GEOFLOW_STATE_FLOW_PROGRESS_BACKUP_DIR")
    if root:
        fingerprint = sha256_text(str(output_dir.resolve()))[:10]
        return Path(root) / (
            f"state_flow__seed-{seed}__{training_hash[:12]}__{fingerprint}.json"
        )
    return Path(configured) if configured else None


def _normalize_block(value: torch.Tensor) -> torch.Tensor:
    """Normalize each row without estimating any corpus-level statistic."""

    return F.normalize(torch.nan_to_num(value.float(), nan=0.0), dim=-1)


class StateFlowFeatureSpace:
    """Label-free concatenation of immutable multi-view cache features."""

    def __init__(self, store: SearchFeatureStore) -> None:
        self.store = store
        ordered_views = tuple(sorted(store.views))
        self.views = ordered_views
        self.snapshot = torch.cat(
            [
                *[_normalize_block(store.state_snapshot_views[view]) for view in ordered_views],
                _normalize_block(store.structured_state),
            ],
            dim=-1,
        )
        self.context = torch.cat(
            [
                *[_normalize_block(store.state_views[view]) for view in ordered_views],
                _normalize_block(store.structured_state),
            ],
            dim=-1,
        )
        self.goal = torch.cat(
            [
                *[_normalize_block(store.goal_views[view]) for view in ordered_views],
                _normalize_block(store.structured_goal),
            ],
            dim=-1,
        )
        self.tool = torch.cat(
            [
                *[_normalize_block(store.tool_views[view]) for view in ordered_views],
                _normalize_block(store.structured_tools),
            ],
            dim=-1,
        )
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}

    def device_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Keep immutable concatenated features and transition indices on CUDA."""

        if device.type != "cuda":
            return {
                "snapshot": self.snapshot,
                "context": self.context,
                "goal": self.goal,
                "tool": self.tool,
                "policy_candidate_mask": self.store._targets[  # noqa: SLF001
                    "policy_candidate_mask"
                ],
                "transition_known_mask": self.store._targets[  # noqa: SLF001
                    "transition_known_mask"
                ],
                "successor_feature_mask": self.store._targets[  # noqa: SLF001
                    "successor_feature_mask"
                ],
                "successor_index": self.store._targets[  # noqa: SLF001
                    "successor_index"
                ],
            }
        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                "snapshot": self.snapshot.to(device),
                "context": self.context.to(device),
                "goal": self.goal.to(device),
                "tool": self.tool.to(device),
                "policy_candidate_mask": self.store._targets[  # noqa: SLF001
                    "policy_candidate_mask"
                ].to(device),
                "transition_known_mask": self.store._targets[  # noqa: SLF001
                    "transition_known_mask"
                ].to(device),
                "successor_feature_mask": self.store._targets[  # noqa: SLF001
                    "successor_feature_mask"
                ].to(device),
                "successor_index": self.store._targets[  # noqa: SLF001
                    "successor_index"
                ].to(device),
            }
            self._device_cache[key] = cached
        return cached

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "snapshot_input_dim": int(self.snapshot.shape[-1]),
            "context_input_dim": int(self.context.shape[-1]),
            "goal_input_dim": int(self.goal.shape[-1]),
            "tool_input_dim": int(self.tool.shape[-1]),
        }

    def condition(
        self,
        model: SearchStateFlow,
        row_indices: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        tensors = self.device_tensors(device)
        indices = torch.as_tensor(row_indices, dtype=torch.long, device=device)
        return model.encode_condition(
            tensors["context"][indices],
            tensors["goal"][indices],
        )

    def transition_sequence(
        self,
        model: SearchStateFlow,
        reference: SearchPathReference,
        device: torch.device,
    ) -> torch.Tensor:
        tensors = self.device_tensors(device)
        successor = torch.as_tensor(
            reference.state_indices[1:], dtype=torch.long, device=device
        )
        tools = torch.as_tensor(reference.tool_indices, dtype=torch.long, device=device)
        return model.transition_anchor(
            tensors["snapshot"][successor],
            tensors["tool"][tools],
        )


def _balanced_stop_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    labels = targets[mask]
    scores = logits[mask]
    if not labels.numel():
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    positive = labels > 0.5
    if positive.any() and (~positive).any():
        return 0.5 * losses[positive].mean() + 0.5 * losses[~positive].mean()
    return losses.mean()


def _candidate_transition_indices(
    store: SearchFeatureStore,
    source_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decodable tool/successor pairs without oracle reachability masks."""

    targets = store._targets  # noqa: SLF001 - immutable, validated tensor store
    mask = (
        targets["policy_candidate_mask"][source_index]
        & targets["transition_known_mask"][source_index]
        & targets["successor_feature_mask"][source_index]
    )
    tool_indices = torch.where(mask)[0]
    successor_indices = targets["successor_index"][source_index, tool_indices]
    if (successor_indices < 0).any():
        raise RuntimeError("A decodable transition is missing its successor state")
    return tool_indices, successor_indices


def _transition_classification_loss(
    model: SearchStateFlow,
    endpoints: torch.Tensor,
    references: Sequence[SearchPathReference],
    features: StateFlowFeatureSpace,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("transition_temperature must be positive")
    # Every search state has the same global tool axis, so all variable-size
    # candidate sets can be represented by one [transition, tool] mask.  The
    # previous implementation launched a separate GPU projection and softmax
    # for every path step; that was exact but left the GPU mostly idle.  This
    # formulation is mathematically identical and makes large path sets usable.
    batch_indices: list[int] = []
    slots: list[int] = []
    source_indices: list[int] = []
    gold_tools: list[int] = []
    for batch_index, reference in enumerate(references):
        for step, (source_index, gold_tool) in enumerate(
            zip(reference.state_indices[:-1], reference.tool_indices, strict=True)
        ):
            batch_indices.append(batch_index)
            slots.append(step)
            source_indices.append(source_index)
            gold_tools.append(gold_tool)
    if not source_indices:
        return endpoints.sum() * 0.0

    device = endpoints.device
    if hasattr(features, "device_tensors"):
        targets = features.device_tensors(device)
    else:  # Lightweight duck-typed feature spaces used by focused tests.
        targets = {
            "snapshot": features.snapshot.to(device),
            "tool": features.tool.to(device),
            **{
                name: value.to(device)
                for name, value in features.store._targets.items()  # noqa: SLF001
            },
        }
    source = torch.as_tensor(source_indices, dtype=torch.long, device=device)
    candidate_mask = (
        targets["policy_candidate_mask"][source]
        & targets["transition_known_mask"][source]
        & targets["successor_feature_mask"][source]
    )
    gold = torch.as_tensor(gold_tools, dtype=torch.long, device=device)
    gold_present = candidate_mask.gather(1, gold[:, None]).squeeze(1)
    if not bool(gold_present.all()):
        bad = int(torch.where(~gold_present)[0][0])
        reference = references[batch_indices[bad]]
        step = slots[bad]
        raise ValueError(
            f"Gold tool {reference.tool_ids[step]!r} is absent among decodable "
            f"actions at {reference.state_ids[step]!r}"
        )

    successor = targets["successor_index"][source]
    if bool((successor[candidate_mask] < 0).any()):
        raise RuntimeError("A decodable transition is missing its successor state")
    successor = successor.clamp_min(0)
    successor_features = targets["snapshot"][successor]
    tool_features = targets["tool"].unsqueeze(0).expand(len(source), -1, -1)
    anchors = model.transition_anchor(successor_features, tool_features)
    predicted = endpoints[
        torch.as_tensor(batch_indices, device=device),
        torch.as_tensor(slots, device=device),
    ]
    logits = F.cosine_similarity(predicted[:, None, :], anchors, dim=-1) / temperature
    logits = logits.masked_fill(~candidate_mask, -torch.inf)
    return F.cross_entropy(logits, gold)


@dataclass(frozen=True)
class StateFlowLossWeights:
    flow_matching: float = 1.0
    endpoint: float = 0.5
    stop: float = 0.5
    transition: float = 0.5


def _loss_weights(config: Mapping[str, Any]) -> StateFlowLossWeights:
    supplied = config.get("loss_weights", {})
    names = StateFlowLossWeights.__dataclass_fields__
    unknown = sorted(set(supplied) - set(names))
    if unknown:
        raise ValueError(f"Unknown state-flow loss weights: {unknown}")
    defaults = StateFlowLossWeights()
    values = {name: float(supplied.get(name, getattr(defaults, name))) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("State-flow loss weights must be finite and non-negative")
    return StateFlowLossWeights(**values)


def _state_flow_loss(
    model: SearchStateFlow,
    references: Sequence[SearchPathReference],
    features: StateFlowFeatureSpace,
    device: torch.device,
    *,
    pad_weight: float,
    transition_temperature: float,
    loss_weights: StateFlowLossWeights,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    row_indices = [reference.row_index for reference in references]
    condition = features.condition(model, row_indices, device)
    sequences = [features.transition_sequence(model, reference, device) for reference in references]
    targets = model.build_targets(sequences, pad_weight=pad_weight)
    step = model.flow_matching_step(targets["target"], condition, time=time, noise=noise)
    squared = (step["predicted_velocity"] - step["target_velocity"]).square().mean(-1)
    fm_loss = (squared * targets["weights"]).sum() / targets["weights"].sum().clamp_min(1)
    transition_mask = targets["transition_mask"]
    if transition_mask.any():
        endpoint_loss = (
            1.0
            - F.cosine_similarity(
                step["predicted_endpoint"][transition_mask],
                targets["target"][transition_mask],
                dim=-1,
            )
        ).mean()
    else:
        endpoint_loss = fm_loss * 0.0
    stop_loss = _balanced_stop_loss(
        model.stop_head(step["predicted_endpoint"]).squeeze(-1),
        targets["stop_target"],
        targets["active_mask"],
    )
    transition_loss = _transition_classification_loss(
        model,
        step["predicted_endpoint"],
        references,
        features,
        temperature=transition_temperature,
    )
    total = (
        loss_weights.flow_matching * fm_loss
        + loss_weights.endpoint * endpoint_loss
        + loss_weights.stop * stop_loss
        + loss_weights.transition * transition_loss
    )
    return total, {
        "total": float(total.detach().item()),
        "flow_matching": float(fm_loss.detach().item()),
        "endpoint": float(endpoint_loss.detach().item()),
        "stop": float(stop_loss.detach().item()),
        "transition": float(transition_loss.detach().item()),
    }


def _fixed_noise_time(
    model: SearchStateFlow,
    batch_size: int,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    time = torch.rand(batch_size, generator=generator)
    noise = torch.randn(
        batch_size,
        model.max_plan_length,
        model.dim,
        generator=generator,
    )
    return time.to(device), noise.to(device)


@torch.inference_mode()
def _calibrate_stop_threshold(
    model: SearchStateFlow,
    references: Sequence[SearchPathReference],
    features: StateFlowFeatureSpace,
    device: torch.device,
    *,
    grid: Sequence[float],
    batch_size: int,
    pad_weight: float,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    """Choose STOP threshold on dev denoising endpoints, never on test.

    Each dev reference supplies its own known transition/STOP positions.  The
    calibration therefore measures the STOP head at rectified-interpolation
    endpoints, while final free-running metrics remain separately reported.
    """

    thresholds = sorted({float(value) for value in grid})
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("stop_threshold_grid must contain probabilities in (0, 1)")
    probabilities: list[float] = []
    labels: list[float] = []
    model.eval()
    for start in range(0, len(references), batch_size):
        batch = references[start : start + batch_size]
        conditions = features.condition(model, [reference.row_index for reference in batch], device)
        sequences = [features.transition_sequence(model, reference, device) for reference in batch]
        targets = model.build_targets(sequences, pad_weight=pad_weight)
        time, noise = _fixed_noise_time(model, len(batch), device, seed=seed + start)
        step = model.flow_matching_step(targets["target"], conditions, time=time, noise=noise)
        mask = targets["active_mask"]
        probabilities.extend(
            model.stop_probability(step["predicted_endpoint"])[mask].cpu().tolist()
        )
        labels.extend(targets["stop_target"][mask].cpu().tolist())
    probability = np.asarray(probabilities, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64) >= 0.5
    if not label.any() or label.all():
        raise ValueError("Dev STOP calibration requires positive and negative slots")
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        prediction = probability >= threshold
        sensitivity = float(prediction[label].mean())
        specificity = float((~prediction[~label]).mean())
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )
    return selected["threshold"], {
        "selection_split": "dev",
        "test_sealed": True,
        "protocol": "teacher_forced_rectified_interpolation_endpoints",
        "count": len(labels),
        "positive_count": int(label.sum()),
        "negative_count": int((~label).sum()),
        "grid": rows,
        "selected": selected,
        "warning": (
            "Calibration endpoints are denoising approximations; free-running STOP "
            "metrics must be interpreted separately."
        ),
    }


@torch.inference_mode()
def _validation_loss(
    model: SearchStateFlow,
    references: Sequence[SearchPathReference],
    features: StateFlowFeatureSpace,
    device: torch.device,
    *,
    batch_size: int,
    pad_weight: float,
    transition_temperature: float,
    loss_weights: StateFlowLossWeights,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    seen = 0
    for start in range(0, len(references), batch_size):
        batch = references[start : start + batch_size]
        time, noise = _fixed_noise_time(model, len(batch), device, seed=seed + start)
        _, components = _state_flow_loss(
            model,
            batch,
            features,
            device,
            pad_weight=pad_weight,
            transition_temperature=transition_temperature,
            loss_weights=loss_weights,
            time=time,
            noise=noise,
        )
        for name, value in components.items():
            totals[name] += value * len(batch)
        seen += len(batch)
    return {name: value / max(1, seen) for name, value in totals.items()}


def _edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
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


@torch.inference_mode()
def _decode_open_loop(
    model: SearchStateFlow,
    plan: torch.Tensor,
    root_index: int,
    features: StateFlowFeatureSpace,
    *,
    stop_threshold: float,
) -> dict[str, Any]:
    store = features.store
    current = int(root_index)
    tool_indices: list[int] = []
    state_indices = [current]
    total_cost = 0.0
    valid = True
    stopped = False
    stop_valid = False
    premature_stop = False
    failure: str | None = None
    stop_probabilities = model.stop_probability(plan.unsqueeze(0))[0]
    for slot in range(model.max_plan_length):
        terminal = bool(store.examples[current].get("terminal"))
        if float(stop_probabilities[slot].item()) >= stop_threshold:
            stopped = True
            stop_valid = terminal
            premature_stop = not terminal
            if premature_stop:
                failure = "premature_stop"
            break
        if terminal:
            failure = "missing_stop_after_goal"
            break
        candidates, successors = _candidate_transition_indices(store, current)
        if not len(candidates):
            valid = False
            failure = "no_decodable_transition"
            break
        tensors = features.device_tensors(plan.device)
        anchors = model.transition_anchor(
            tensors["snapshot"][successors.to(plan.device)],
            tensors["tool"][candidates.to(plan.device)],
        )
        scores = F.cosine_similarity(plan[slot].unsqueeze(0), anchors, dim=-1)
        selected_position = int(scores.argmax().item())
        tool_index = int(candidates[selected_position].item())
        successor = int(successors[selected_position].item())
        cost = store._targets["edge_cost"][current, tool_index]  # noqa: SLF001
        if not torch.isfinite(cost):
            valid = False
            failure = "missing_transition_cost"
            break
        tool_indices.append(tool_index)
        state_indices.append(successor)
        total_cost += float(cost.item())
        current = successor
    goal_reached = bool(store.examples[current].get("terminal"))
    return {
        "tool_indices": tuple(tool_indices),
        "tool_ids": tuple(store.tool_ids[index] for index in tool_indices),
        "state_indices": tuple(state_indices),
        "total_cost": total_cost,
        "valid_chain": valid,
        "goal_reached": goal_reached,
        "stopped": stopped,
        "stop_valid": stop_valid,
        "premature_stop": premature_stop,
        "failure": failure,
    }


@torch.inference_mode()
def _sample_plans(
    model: SearchStateFlow,
    features: StateFlowFeatureSpace,
    row_index: int,
    device: torch.device,
    *,
    samples: int,
    nfe: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if samples <= 0:
        raise ValueError("samples must be positive")
    condition = features.condition(model, [row_index], device)
    noise = torch.randn(
        samples,
        model.max_plan_length,
        model.dim,
        generator=generator,
        device="cpu",
    ).to(device)
    plans, _ = model.solve(condition.expand(samples, -1), nfe=nfe, noise=noise)
    return plans


@torch.inference_mode()
def _sample_plan(
    model: SearchStateFlow,
    features: StateFlowFeatureSpace,
    row_index: int,
    device: torch.device,
    *,
    nfe: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return _sample_plans(
        model,
        features,
        row_index,
        device,
        samples=1,
        nfe=nfe,
        generator=generator,
    )[0]


@torch.inference_mode()
def _replanning_rollout(
    model: SearchStateFlow,
    features: StateFlowFeatureSpace,
    root_index: int,
    device: torch.device,
    *,
    nfe: int,
    stop_threshold: float,
    max_steps: int,
    generator: torch.Generator,
    guard_premature_stop: bool = False,
    feedback: bool = True,
    planner_call_budget: int | None = None,
) -> dict[str, Any]:
    store = features.store
    current = int(root_index)
    visited = {current}
    tools: list[int] = []
    cost = 0.0
    failure: str | None = None
    planner_calls = 0
    for _ in range(max_steps):
        if bool(store.examples[current].get("terminal")):
            break
        if planner_call_budget is not None and planner_calls >= planner_call_budget:
            failure = "planner_call_budget"
            break
        visible = current if feedback else root_index
        plan = _sample_plan(model, features, visible, device, nfe=nfe, generator=generator)
        planner_calls += 1
        if (
            not guard_premature_stop
            and float(model.stop_probability(plan[:1])[0].item()) >= stop_threshold
        ):
            failure = "premature_stop"
            break
        candidates, successors = _candidate_transition_indices(store, visible)
        if not len(candidates):
            failure = "no_decodable_transition"
            break
        tensors = features.device_tensors(device)
        anchors = model.transition_anchor(
            tensors["snapshot"][successors.to(device)],
            tensors["tool"][candidates.to(device)],
        )
        selected_position = int(
            F.cosine_similarity(plan[0].unsqueeze(0), anchors, dim=-1).argmax().item()
        )
        tool_index = int(candidates[selected_position].item())
        actual_candidates, actual_successors = _candidate_transition_indices(store, current)
        actual_match = torch.where(actual_candidates == tool_index)[0]
        if len(actual_match) != 1:
            failure = "blind_invalid_action" if not feedback else "invalid_action"
            break
        successor = int(actual_successors[int(actual_match[0].item())].item())
        edge_cost = store._targets["edge_cost"][current, tool_index]  # noqa: SLF001
        if not torch.isfinite(edge_cost):
            failure = "missing_transition_cost"
            break
        tools.append(tool_index)
        cost += float(edge_cost.item())
        current = successor
        if current in visited:
            failure = "cycle"
            break
        visited.add(current)
    else:
        failure = "max_steps"
    # If a matched blind control fails earlier than its observed-replanning pair,
    # consume the remaining calls without executing their actions. This cannot
    # improve success, but makes planner calls and NFE exactly equal per state.
    while planner_call_budget is not None and planner_calls < planner_call_budget:
        visible = current if feedback else root_index
        _sample_plan(model, features, visible, device, nfe=nfe, generator=generator)
        planner_calls += 1
    return {
        "success": bool(store.examples[current].get("terminal")),
        "tool_indices": tuple(tools),
        "total_cost": cost,
        "failure": failure,
        "planner_calls": planner_calls,
    }


def _bootstrap_effect(values: Sequence[float], *, seed: int, reps: int = 2000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"estimate": None, "low": None, "high": None, "units": 0}
    estimate = float(array.mean())
    if len(array) == 1 or reps <= 0:
        return {"estimate": estimate, "low": estimate, "high": estimate, "units": len(array)}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(reps, len(array)))].mean(axis=1)
    return {
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "units": len(array),
    }


def _paired_task_effect(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    *,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted(set(left) & set(right))
    differences = [fmean(left[task_id]) - fmean(right[task_id]) for task_id in task_ids]
    result = _bootstrap_effect(differences, seed=seed)
    result["unit"] = "independent_task"
    return result


@torch.inference_mode()
def evaluate_search_state_flow(
    model: SearchStateFlow,
    store: SearchFeatureStore,
    path_store: SearchPathStore,
    split: str,
    device: torch.device,
    *,
    samples_per_state: int = 4,
    nfe: int = 12,
    stop_threshold: float = 0.5,
    max_replan_steps: int | None = None,
    seed: int = 17,
    allow_test: bool = False,
    root_only: bool = False,
) -> dict[str, Any]:
    """Evaluate sampled paths and replan rollouts without selecting on test."""

    if split == "test" and not allow_test:
        raise PermissionError("Test evaluation requires explicit allow_test=True")
    if samples_per_state <= 0:
        raise ValueError("samples_per_state must be positive")
    if not 0 < stop_threshold < 1:
        raise ValueError("stop_threshold must be strictly between zero and one")
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    indices = path_store.eligible_indices(split)
    if root_only:
        indices = [index for index in indices if not store.examples[index].get("prefix_tool_ids")]
    if not indices:
        return {"split": split, "eligible_states": 0, "eligible_tasks": 0}
    if split == "test" and path_store.allowed_splits != frozenset({"test"}):
        raise ValueError("Test evaluation requires a test-only SearchPathStore")
    features = StateFlowFeatureSpace(store)
    model.eval()
    max_replan_steps = max_replan_steps or (model.max_plan_length * 2)

    first_hits: list[float] = []
    goal_hits: list[float] = []
    valid_hits: list[float] = []
    stop_hits: list[float] = []
    premature_hits: list[float] = []
    reference_hits: list[float] = []
    optimal_cost_hits: list[float] = []
    excess_costs: list[float] = []
    edit_scores: list[float] = []
    novel_valid_hits: list[float] = []
    reference_coverage: list[float] = []
    unique_paths: list[float] = []
    replan_success: list[float] = []
    replan_excess: list[float] = []
    replan_failures: defaultdict[str, int] = defaultdict(int)
    guarded_success: list[float] = []
    guarded_excess: list[float] = []
    guarded_failures: defaultdict[str, int] = defaultdict(int)
    task_goal_hits: defaultdict[str, list[float]] = defaultdict(list)
    task_replan_hits: defaultdict[str, list[float]] = defaultdict(list)
    task_blind_hits: defaultdict[str, list[float]] = defaultdict(list)
    blind_success: list[float] = []
    blind_excess: list[float] = []
    blind_failures: defaultdict[str, int] = defaultdict(int)
    replan_planner_calls: list[float] = []
    guarded_planner_calls: list[float] = []
    blind_planner_calls: list[float] = []
    matched_call_pairs: list[float] = []
    root_open_loop: list[float] = []
    root_replan: list[float] = []
    root_blind: list[float] = []

    for row_index in indices:
        references = path_store.references(row_index)
        reference_tools = {reference.tool_indices for reference in references}
        observed: set[tuple[int, ...]] = set()
        row = store.examples[row_index]
        task_id = str(row["task_id"])
        row_seed = int(seed) + 10_000 * int(row_index)
        generator = torch.Generator(device="cpu").manual_seed(row_seed)
        value_star = float(row["value_star"])
        optimal = store._targets["optimal_action_mask"][row_index]  # noqa: SLF001
        row_goal_hits: list[float] = []
        plans = _sample_plans(
            model,
            features,
            row_index,
            device,
            samples=samples_per_state,
            nfe=nfe,
            generator=generator,
        )
        for plan in plans:
            decoded = _decode_open_loop(
                model,
                plan,
                row_index,
                features,
                stop_threshold=stop_threshold,
            )
            tools = decoded["tool_indices"]
            observed.add(tools)
            first_hits.append(float(bool(tools) and bool(optimal[tools[0]])))
            goal = float(decoded["goal_reached"])
            goal_hits.append(goal)
            row_goal_hits.append(goal)
            task_goal_hits[task_id].append(goal)
            valid_hits.append(float(decoded["valid_chain"]))
            stop_hits.append(float(decoded["stop_valid"]))
            premature_hits.append(float(decoded["premature_stop"]))
            in_reference = tools in reference_tools
            reference_hits.append(float(in_reference))
            novel_valid_hits.append(
                float(decoded["goal_reached"] and decoded["valid_chain"] and not in_reference)
            )
            if decoded["goal_reached"]:
                excess = max(0.0, float(decoded["total_cost"]) - value_star)
                excess_costs.append(excess)
                optimal_cost_hits.append(float(excess <= 1e-6))
            closest = min(
                _edit_distance(tools, reference.tool_indices)
                / max(1, len(tools), len(reference.tool_indices))
                for reference in references
            )
            edit_scores.append(closest)
        reference_coverage.append(len(observed & reference_tools) / len(reference_tools))
        unique_paths.append(float(len(observed)))

        # One independent receding-horizon rollout per root.  The terminal verifier
        # ends the episode; generated STOP remains a premature-stop safety signal.
        rollout = _replanning_rollout(
            model,
            features,
            row_index,
            device,
            nfe=nfe,
            stop_threshold=stop_threshold,
            max_steps=max_replan_steps,
            generator=torch.Generator(device="cpu").manual_seed(row_seed),
        )
        replan_success.append(float(rollout["success"]))
        replan_planner_calls.append(float(rollout["planner_calls"]))
        task_replan_hits[task_id].append(float(rollout["success"]))
        if rollout["success"]:
            replan_excess.append(max(0.0, float(rollout["total_cost"]) - value_star))
        elif rollout["failure"] is not None:
            replan_failures[str(rollout["failure"])] += 1

        # The guarded policy is a transparent ablation: an external terminal
        # verifier may accept STOP, but a nonterminal snapshot may not.  It
        # isolates transition generation from STOP calibration rather than
        # presenting verifier gating as an intrinsic model capability.
        guarded = _replanning_rollout(
            model,
            features,
            row_index,
            device,
            nfe=nfe,
            stop_threshold=stop_threshold,
            max_steps=max_replan_steps,
            generator=torch.Generator(device="cpu").manual_seed(row_seed),
            guard_premature_stop=True,
        )
        guarded_success.append(float(guarded["success"]))
        guarded_planner_calls.append(float(guarded["planner_calls"]))
        if guarded["success"]:
            guarded_excess.append(max(0.0, float(guarded["total_cost"]) - value_star))
        elif guarded["failure"] is not None:
            guarded_failures[str(guarded["failure"])] += 1

        blind = _replanning_rollout(
            model,
            features,
            row_index,
            device,
            nfe=nfe,
            stop_threshold=stop_threshold,
            max_steps=max_replan_steps,
            generator=torch.Generator(device="cpu").manual_seed(row_seed),
            feedback=False,
            planner_call_budget=int(rollout["planner_calls"]),
        )
        blind_success.append(float(blind["success"]))
        blind_planner_calls.append(float(blind["planner_calls"]))
        matched_call_pairs.append(float(blind["planner_calls"] == rollout["planner_calls"]))
        task_blind_hits[task_id].append(float(blind["success"]))
        if blind["success"]:
            blind_excess.append(max(0.0, float(blind["total_cost"]) - value_star))
        elif blind["failure"] is not None:
            blind_failures[str(blind["failure"])] += 1

        if not row.get("prefix_tool_ids"):
            root_open_loop.append(fmean(row_goal_hits))
            root_replan.append(float(rollout["success"]))
            root_blind.append(float(blind["success"]))

    task_macro = _mean(fmean(values) for values in task_goal_hits.values())
    return {
        "split": split,
        "evaluation_scope": "task_roots" if root_only else "all_eligible_states",
        "eligible_states": len(indices),
        "eligible_tasks": len(task_goal_hits),
        "samples_per_state": samples_per_state,
        "sample_count": len(goal_hits),
        "nfe": nfe,
        "stop_threshold": stop_threshold,
        "open_loop": {
            "first_action_optimal_set_accuracy": _mean(first_hits),
            "valid_transition_chain_rate": _mean(valid_hits),
            "goal_completion_rate": _mean(goal_hits),
            "task_macro_goal_completion_rate": task_macro,
            "valid_stop_after_goal_rate": _mean(stop_hits),
            "premature_stop_rate": _mean(premature_hits),
            "serialized_top_k_path_rate": _mean(reference_hits),
            "novel_valid_goal_path_rate": _mean(novel_valid_hits),
            "optimal_cost_given_goal_rate": _mean(optimal_cost_hits),
            "mean_excess_cost_given_goal": _mean(excess_costs),
            "closest_reference_normalized_edit_distance": _mean(edit_scores),
            "mean_reference_coverage_at_k": _mean(reference_coverage),
            "mean_unique_paths_at_k": _mean(unique_paths),
        },
        "receding_horizon": {
            "rollouts": len(replan_success),
            "goal_completion_rate": _mean(replan_success),
            "mean_excess_cost_given_goal": _mean(replan_excess),
            "mean_planner_calls": _mean(replan_planner_calls),
            "mean_total_nfe": nfe * _mean(replan_planner_calls),
            "failure_counts": dict(sorted(replan_failures.items())),
            "boundary": (
                "Offline exact-snapshot execute-observe-replan; no live APIs or "
                "perturbation recovery are exercised."
            ),
        },
        "receding_horizon_verifier_stop_guard": {
            "rollouts": len(guarded_success),
            "goal_completion_rate": _mean(guarded_success),
            "mean_excess_cost_given_goal": _mean(guarded_excess),
            "mean_planner_calls": _mean(guarded_planner_calls),
            "mean_total_nfe": nfe * _mean(guarded_planner_calls),
            "failure_counts": dict(sorted(guarded_failures.items())),
            "ablation": (
                "STOP is rejected whenever the exact snapshot is nonterminal; this "
                "measures transition generation with an external verifier guard."
            ),
        },
        "compute_matched_blind_replanning": {
            "rollouts": len(blind_success),
            "goal_completion_rate": _mean(blind_success),
            "mean_excess_cost_given_goal": _mean(blind_excess),
            "mean_planner_calls": _mean(blind_planner_calls),
            "mean_total_nfe": nfe * _mean(blind_planner_calls),
            "per_state_planner_call_match_rate": _mean(matched_call_pairs),
            "failure_counts": dict(sorted(blind_failures.items())),
            "control": (
                "Uses the same per-state noise sequence and exactly the observed policy's "
                "planner-call/NFE budget, but never exposes successor observations to the "
                "planner. Calls left after an early blind failure are spent without execution."
            ),
        },
        "feedback_effects": {
            "replanning_minus_one_shot_task_macro": _paired_task_effect(
                task_replan_hits, task_goal_hits, seed=seed + 700_001
            ),
            "replanning_minus_blind_task_macro": _paired_task_effect(
                task_replan_hits, task_blind_hits, seed=seed + 700_002
            ),
            "root_start": {
                "tasks": len(root_replan),
                "one_shot_goal_completion_rate": _mean(root_open_loop),
                "replanning_goal_completion_rate": _mean(root_replan),
                "blind_replanning_goal_completion_rate": _mean(root_blind),
                "replanning_minus_one_shot": _bootstrap_effect(
                    [left - right for left, right in zip(root_replan, root_open_loop, strict=True)],
                    seed=seed + 700_003,
                ),
                "replanning_minus_blind": _bootstrap_effect(
                    [left - right for left, right in zip(root_replan, root_blind, strict=True)],
                    seed=seed + 700_004,
                ),
            },
            "interpretation": (
                "Intervals bootstrap independent tasks (or one root per task), not correlated "
                "search-state prefixes. A positive interval excluding zero supports a feedback "
                "benefit under this synthetic snapshot distribution."
            ),
        },
        "test_unsealed": split == "test",
    }


def _all_references(path_store: SearchPathStore, split: str) -> list[SearchPathReference]:
    return [
        reference
        for row_index in path_store.eligible_indices(split)
        for reference in path_store.references(row_index)
    ]


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Keep the previous checkpoint intact if a worker dies while saving."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def train_search_state_flow(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    include_test: bool = False,
    root_only_evaluation: bool = False,
) -> dict[str, Any]:
    """Fit on train paths, select on dev, and keep test sealed by default."""

    config = read_yaml(config_path)
    train_config = dict(config.get("state_flow_training", {}))
    seed = int(seed_override if seed_override is not None else config.get("seed", 17))
    effective_config = {**train_config, "seed": seed}
    training_config_sha256 = sha256_text(canonical_json(effective_config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "training_resume.pt"
    progress_path = output_dir / "training_progress.json"
    progress_backup_path = _state_flow_progress_backup_path(
        output_dir,
        seed=seed,
        training_hash=training_config_sha256,
        configured=train_config.get("progress_backup_path"),
    )
    resume_enabled = bool(train_config.get("resume", True))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    run_provenance = collect_run_provenance(config_path, device=device)
    source_tree_sha256 = run_provenance.get("source_tree_sha256")
    if not source_tree_sha256:
        raise RuntimeError("Cannot bind State Flow training to a source-tree fingerprint")
    structured_dim = int(
        train_config.get(
            "structured_dim", config.get("value_training", {}).get("structured_dim", 64)
        )
    )
    store = SearchFeatureStore(processed_dir, cache_dir, structured_dim=structured_dim)
    max_plan_length = int(train_config.get("max_plan_length", 12))
    # Only train/dev oracle paths are read here.  Test evaluation, when explicitly
    # requested, creates a separate test-only adapter after fitting is complete.
    path_store = SearchPathStore(
        store,
        beta=float(train_config.get("path_beta", 1.0)),
        max_path_length=max_plan_length - 1,
        allowed_splits=("train", "dev"),
    )
    if not path_store.eligible_indices("train"):
        raise ValueError("No eligible training paths")
    if not path_store.eligible_indices("dev"):
        raise ValueError("No eligible dev paths; selection cannot use test")
    features = StateFlowFeatureSpace(store)
    model = SearchStateFlow(
        **features.dimensions,
        dim=int(train_config.get("dim", 32)),
        max_plan_length=max_plan_length,
        hidden_dim=int(train_config.get("hidden_dim", 64)),
        layers=int(train_config.get("layers", 1)),
        heads=int(train_config.get("heads", 4)),
        dropout=float(train_config.get("dropout", 0.0)),
        projection_seed=int(train_config.get("projection_seed", seed)),
        tool_mix=float(train_config.get("tool_mix", 0.35)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    epochs = int(train_config.get("epochs", 24))
    patience = int(train_config.get("patience", 7))
    batch_size = int(train_config.get("batch_size", 32))
    steps_per_epoch = int(
        train_config.get(
            "steps_per_epoch",
            math.ceil(len(path_store.eligible_indices("train")) / batch_size),
        )
    )
    if epochs <= 0 or patience <= 0 or batch_size <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs, patience, batch_size, and steps_per_epoch must be positive")
    pad_weight = float(train_config.get("pad_weight", 0.1))
    transition_temperature = float(train_config.get("transition_temperature", 0.1))
    loss_weights = _loss_weights(train_config)
    gradient_clip = float(train_config.get("gradient_clip", 1.0))
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    sampling_rng = random.Random(seed)
    dev_references = _all_references(path_store, "dev")

    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_enabled and resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        expected = {
            "resume_version": STATE_FLOW_RESUME_VERSION,
            "training_config_sha256": training_config_sha256,
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
                "State Flow resume checkpoint does not match this immutable run: "
                f"{mismatches}"
            )
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best_loss = float(resume["best_loss"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        stale = int(resume["stale"])
        history = list(resume["history"])
        sampling_rng.setstate(resume["sampling_rng_state"])
        _restore_global_rng_state(resume)
        start_epoch = int(resume["completed_epoch"]) + 1

    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience:
            break
        model.train()
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        for _ in range(steps_per_epoch):
            references = path_store.sample_batch("train", batch_size, sampling_rng)
            loss, components = _state_flow_loss(
                model,
                references,
                features,
                device,
                pad_weight=pad_weight,
                transition_temperature=transition_temperature,
                loss_weights=loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            for name, value in components.items():
                totals[name] += value * len(references)
            seen += len(references)
        scheduler.step()
        dev_loss = _validation_loss(
            model,
            dev_references,
            features,
            device,
            batch_size=batch_size,
            pad_weight=pad_weight,
            transition_temperature=transition_temperature,
            loss_weights=loss_weights,
            seed=seed + 100_000,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "train": {name: value / max(1, seen) for name, value in totals.items()},
                "dev": dev_loss,
            }
        )
        if dev_loss["total"] < best_loss - 1e-8:
            best_loss = dev_loss["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        resume_state = {
            "resume_version": STATE_FLOW_RESUME_VERSION,
            "training_config_sha256": training_config_sha256,
            "cache_content_sha256": store.cache.content_sha256,
            "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
            "source_tree_sha256": source_tree_sha256,
            "model_config": model.export_config(),
            "completed_epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale": stale,
            "history": history,
            "sampling_rng_state": sampling_rng.getstate(),
            **_capture_global_rng_state(),
        }
        _atomic_torch_save(resume_state, resume_path)
        progress = {
            "status": "early_stopping" if stale >= patience else "running",
            "completed_epoch": epoch,
            "configured_epochs": epochs,
            "best_epoch": best_epoch,
            "best_dev_validation_loss": best_loss,
            "stale_epochs": stale,
            "patience": patience,
            "training_config_sha256": training_config_sha256,
            "cache_content_sha256": store.cache.content_sha256,
            "resume_checkpoint": str(resume_path),
            "latest": history[-1],
        }
        _atomic_write_json(progress_path, progress)
        _write_optional_progress_backup(progress_backup_path, progress)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("State Flow training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    samples = int(train_config.get("evaluation_samples_per_state", 4))
    nfe = int(train_config.get("nfe", 12))
    stop_threshold, stop_calibration = _calibrate_stop_threshold(
        model,
        dev_references,
        features,
        device,
        grid=train_config.get("stop_threshold_grid", [0.3, 0.5, 0.7, 0.9]),
        batch_size=batch_size,
        pad_weight=pad_weight,
        seed=seed + 200_000,
    )
    report_train_rollouts = os.environ.get("GEOFLOW_STATE_FLOW_REPORT_TRAIN", "1") != "0"
    metrics = {}
    if report_train_rollouts:
        metrics["train"] = evaluate_search_state_flow(
            model,
            store,
            path_store,
            "train",
            device,
            samples_per_state=1,
            nfe=nfe,
            stop_threshold=stop_threshold,
            seed=seed + 1,
        )
    metrics["dev"] = evaluate_search_state_flow(
        model,
        store,
        path_store,
        "dev",
        device,
        samples_per_state=samples,
        nfe=nfe,
        stop_threshold=stop_threshold,
        seed=seed + 2,
        root_only=root_only_evaluation,
    )
    test_labels_loaded = False
    if include_test:
        test_paths = SearchPathStore(
            store,
            beta=float(train_config.get("path_beta", 1.0)),
            max_path_length=max_plan_length - 1,
            allowed_splits=("test",),
        )
        test_labels_loaded = True
        metrics["test"] = evaluate_search_state_flow(
            model,
            store,
            test_paths,
            "test",
            device,
            samples_per_state=samples,
            nfe=nfe,
            stop_threshold=stop_threshold,
            seed=seed + 3,
            allow_test=True,
            root_only=root_only_evaluation,
        )

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "kind": "search_state_flow",
        "state_flow_feature_version": STATE_FLOW_FEATURE_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "seed": seed,
        "experiment_config_sha256": sha256_file(config_path),
        "training_config_sha256": training_config_sha256,
        "training_config": effective_config,
        "model_config": model.export_config(),
        "model_state": model.state_dict(),
        "views": list(store.views),
        "view_roles": store.view_roles,
        "tool_ids": store.tool_ids,
        "cache_config_sha256": store.cache.manifest["config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "cache_evaluation_fingerprint_sha256": (
            store.cache.evaluation_fingerprint_sha256
        ),
        "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
        "source_tree_sha256": source_tree_sha256,
        "training_splits": ["train"],
        "selection_split": "dev",
        "test_labels_loaded": test_labels_loaded,
        "stop_threshold": stop_threshold,
        "best_epoch": best_epoch,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    checkpoint_path = output_dir / "search_state_flow.pt"
    _atomic_torch_save(checkpoint, checkpoint_path)
    result = {
        "kind": "search_state_flow",
        "seed": seed,
        "experiment_config_sha256": checkpoint["experiment_config_sha256"],
        "training_config_sha256": checkpoint["training_config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "best_dev_validation_loss": best_loss,
        "stop_calibration": stop_calibration,
        "selected_stop_threshold": stop_threshold,
        "trainable_parameter_count": checkpoint["trainable_parameter_count"],
        "path_supervision": {
            "train": path_store.report("train"),
            "dev": path_store.report("dev"),
        },
        "metrics": metrics,
        "train_rollouts_reported": report_train_rollouts,
        "evaluation_scope": (
            "task_roots" if root_only_evaluation else "all_eligible_states"
        ),
        "history": history,
        "test_reported": include_test,
        "test_labels_loaded": test_labels_loaded,
        "scientific_boundary": {
            "is_generative_state_flow": True,
            "legacy_single_trace_flow": False,
            "live_tool_environment": False,
            "claim": (
                "Offline search-distilled multi-path State Flow with constrained "
                "snapshot decoding; real-data and live-environment evidence remain required."
            ),
        },
        "run_provenance": run_provenance,
    }
    write_json(output_dir / "state_flow_metrics.json", result)
    completed_progress = {
        "status": "complete",
        "completed_epoch": history[-1]["epoch"],
        "configured_epochs": epochs,
        "best_epoch": best_epoch,
        "best_dev_validation_loss": best_loss,
        "training_config_sha256": training_config_sha256,
        "cache_content_sha256": store.cache.content_sha256,
        "resume_checkpoint": str(resume_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
    }
    _atomic_write_json(progress_path, completed_progress)
    _write_optional_progress_backup(progress_backup_path, completed_progress)
    return result


def load_search_state_flow_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[SearchStateFlow, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("kind") != "search_state_flow":
        raise ValueError(f"Not a search State Flow checkpoint: {path}")
    if checkpoint.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError("State Flow checkpoint cache-feature semantics differ from runtime")
    if checkpoint.get("state_flow_feature_version") != STATE_FLOW_FEATURE_VERSION:
        raise ValueError("State Flow checkpoint assembly semantics differ from runtime")
    model = SearchStateFlow(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def evaluate_search_state_flow_checkpoint(
    processed_dir: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    split: str = "dev",
    allow_test: bool = False,
    root_only: bool = False,
) -> dict[str, Any]:
    """Evaluate an already selected checkpoint without refitting it."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if split == "test" and not allow_test:
        raise PermissionError("Test evaluation requires explicit allow_test=True")
    config = read_yaml(config_path)
    train_config = dict(config.get("state_flow_training", {}))
    device = choose_device(str(config.get("device", "auto")))
    model, checkpoint = load_search_state_flow_checkpoint(checkpoint_path, device)
    structured_dim = int(
        checkpoint.get("training_config", {}).get(
            "structured_dim",
            config.get("value_training", {}).get("structured_dim", 64),
        )
    )
    store = SearchFeatureStore(processed_dir, cache_dir, structured_dim=structured_dim)
    if checkpoint.get("cache_config_sha256") != store.cache.manifest.get("config_sha256"):
        raise ValueError("Checkpoint and evaluation cache use different embedding configs")
    registry_superset = None
    if checkpoint.get("tool_ids") != store.tool_ids:
        eval_config = config.get("search_agent_evaluation", {})
        training_cache_dir = eval_config.get("training_cache_dir")
        if training_cache_dir is None:
            raise ValueError("Checkpoint and evaluation cache have different tool registries")
        if checkpoint.get("tool_ids") != EmbeddingCache(training_cache_dir).tool_ids:
            raise ValueError("Configured training cache does not match checkpoint tool order")
        registry_superset = verify_evaluation_cache_superset(training_cache_dir, store.cache)
    if checkpoint.get("views") != list(store.views):
        raise ValueError("Checkpoint and evaluation cache expose different frozen views")
    same_training_cache = checkpoint.get("cache_content_sha256") == store.cache.content_sha256
    if not same_training_cache:
        expected_fingerprint = checkpoint.get("cache_evaluation_fingerprint_sha256")
        if expected_fingerprint is None:
            raise ValueError(
                "Legacy State Flow checkpoint may only use its exact training cache"
            )
        if (
            expected_fingerprint != store.cache.evaluation_fingerprint_sha256
            and registry_superset is None
        ):
            raise ValueError(
                "State Flow checkpoint and evaluation cache use different frozen "
                "coordinates or tool prototypes"
            )
    features = StateFlowFeatureSpace(store)
    expected_dimensions = {key: checkpoint["model_config"][key] for key in features.dimensions}
    if expected_dimensions != features.dimensions:
        raise ValueError("Checkpoint and evaluation feature widths differ")
    paths = SearchPathStore(
        store,
        beta=float(checkpoint.get("training_config", {}).get("path_beta", 1.0)),
        max_path_length=model.max_plan_length - 1,
        allowed_splits=(split,),
    )
    result = evaluate_search_state_flow(
        model,
        store,
        paths,
        split,
        device,
        samples_per_state=int(train_config.get("evaluation_samples_per_state", 4)),
        nfe=int(train_config.get("nfe", 12)),
        stop_threshold=float(checkpoint["stop_threshold"]),
        seed=int(checkpoint["seed"]) + {"train": 1, "dev": 2, "test": 3}[split],
        allow_test=allow_test,
        root_only=root_only,
    )
    report = {
        "kind": "search_state_flow_evaluation",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_selection_split": checkpoint["selection_split"],
        "checkpoint_test_labels_loaded": checkpoint["test_labels_loaded"],
        "evaluation_split": split,
        "test_unsealed": split == "test",
        "evaluation_scope": "task_roots" if root_only else "all_eligible_states",
        "cache_config_sha256": store.cache.manifest["config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "checkpoint_training_cache_sha256": checkpoint["cache_content_sha256"],
        "evaluation_cache_matches_training_cache": same_training_cache,
        "registry_superset_verification": registry_superset,
        "cache_evaluation_fingerprint_sha256": (
            store.cache.evaluation_fingerprint_sha256
        ),
        "path_supervision": paths.report(split),
        "metrics": result,
        "scientific_boundary": (
            "Offline exact-snapshot evaluation; verifier-guarded STOP is an ablation, "
            "not learned-policy performance."
        ),
    }
    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train search-distilled multi-path State Flow")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--root-only-evaluation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_search_state_flow(
        args.processed_dir,
        args.cache_dir,
        args.output_dir,
        args.config,
        seed_override=args.seed,
        include_test=args.include_test,
        root_only_evaluation=args.root_only_evaluation,
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()


__all__ = [
    "STATE_FLOW_FEATURE_VERSION",
    "StateFlowFeatureSpace",
    "StateFlowLossWeights",
    "evaluate_search_state_flow",
    "evaluate_search_state_flow_checkpoint",
    "load_search_state_flow_checkpoint",
    "train_search_state_flow",
]
