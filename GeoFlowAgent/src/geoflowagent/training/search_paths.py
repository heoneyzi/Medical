"""Strict multi-path targets for a future search-conditioned Flow planner.

This module is deliberately an adapter, not a Flow trainer.  Search distillation
already provides exact state/action values, while a generative State Flow needs a
second, stronger contract: every reference trajectory must be an executable chain
of known, goal-reachable transitions in the same task and split.  ``SearchPathStore``
checks that contract before exposing any path to a future tool-anchor, state-anchor,
or residual-flow objective.

Rows that are terminal, unreachable, or marked ``training_mask=false`` are excluded
from path generation.  They remain valuable to the value model's completion and
reachability heads, but an empty suffix from such a row must never become a Flow
``STOP`` target.

The sampler is hierarchical.  It chooses a task uniformly, then a state uniformly
within that task, and finally a reference path with

``p(path | state, goal) proportional to exp(-(cost - V*) / beta)``.

Consequently, tasks with large search graphs and states with many equivalent paths
do not silently dominate training.  An explicit ``random.Random`` or
``torch.Generator`` is required so sampling is reproducible and testable.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

import torch

from geoflowagent.training.search_features import SearchFeatureStore

RandomGenerator = random.Random | torch.Generator


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return result


def _draw_unit(generator: RandomGenerator) -> float:
    if isinstance(generator, random.Random):
        return generator.random()
    if isinstance(generator, torch.Generator):
        # Match the generator's device so both CPU and CUDA generators have a
        # deterministic implementation.  The scalar is immediately copied back.
        return float(torch.rand((), generator=generator, device=generator.device).cpu().item())
    raise TypeError("generator must be random.Random or torch.Generator")


def _draw_index(size: int, generator: RandomGenerator) -> int:
    if size <= 0:
        raise ValueError("Cannot sample from an empty collection")
    if isinstance(generator, random.Random):
        return generator.randrange(size)
    if isinstance(generator, torch.Generator):
        return int(
            torch.randint(size, (1,), generator=generator, device=generator.device).cpu().item()
        )
    raise TypeError("generator must be random.Random or torch.Generator")


@dataclass(frozen=True)
class SearchPathReference:
    """One fully validated reference suffix rooted at a processed search state."""

    row_index: int
    example_id: str
    task_id: str
    split: str
    root_state_id: str
    rank: int
    tool_ids: tuple[str, ...]
    tool_indices: tuple[int, ...]
    state_ids: tuple[str, ...]
    state_indices: tuple[int, ...]
    total_cost: float
    value_star: float
    excess_cost: float
    probability: float

    @property
    def horizon(self) -> int:
        return len(self.tool_ids)


class SearchPathStore:
    """Validate and sample multi-path supervision held by ``SearchFeatureStore``.

    This class intentionally does not construct continuous targets.  A future Flow
    implementation can use ``tool_indices`` for a tool-anchor baseline or gather
    frozen/learned state anchors with ``state_indices``.  Keeping this boundary
    independent makes it impossible to claim a trained generative Flow merely
    because top-k search paths were serialized.
    """

    def __init__(
        self,
        store: SearchFeatureStore,
        *,
        beta: float = 1.0,
        tolerance: float = 1e-6,
        max_path_length: int | None = None,
        allowed_splits: tuple[str, ...] | None = None,
    ) -> None:
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if max_path_length is not None and max_path_length <= 0:
            raise ValueError("max_path_length must be positive when provided")
        self.store = store
        self.beta = float(beta)
        self.tolerance = float(tolerance)
        self.max_path_length = max_path_length
        if allowed_splits is not None:
            if not allowed_splits or any(
                not isinstance(item, str) or not item for item in allowed_splits
            ):
                raise ValueError("allowed_splits must contain non-empty split names")
            if len(set(allowed_splits)) != len(allowed_splits):
                raise ValueError("allowed_splits cannot contain duplicates")
            self.allowed_splits = frozenset(allowed_splits)
        else:
            self.allowed_splits = None

        self._state_index = self._index_states()
        self._references: dict[int, tuple[SearchPathReference, ...]] = {}
        self._indices_by_split_task: dict[str, dict[str, tuple[int, ...]]] = {}
        self._exclusion_counts: Counter[str] = Counter()
        self._build()

    def _index_states(self) -> dict[tuple[str, str], int]:
        output: dict[tuple[str, str], int] = {}
        for index, row in enumerate(self.store.examples):
            task_id = row.get("task_id")
            state_id = row.get("state_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Search row {index} needs a non-empty task_id")
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"Search row {index} needs a non-empty state_id")
            key = (task_id, state_id)
            if key in output:
                raise ValueError(f"Duplicate search state key {key!r}")
            output[key] = index
        return output

    @staticmethod
    def _exclusion_reason(row: Mapping[str, Any]) -> str | None:
        if row.get("terminal") is True:
            return "terminal"
        if row.get("goal_reachable") is not True:
            return "unreachable_or_unknown"
        if row.get("training_mask") is not True:
            return "training_mask_false"
        return None

    @staticmethod
    def _actions_by_tool(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        raw = row.get("action_supervision")
        if not isinstance(raw, list):
            raise TypeError(f"Example {row.get('example_id')!r}.action_supervision must be a list")
        output: dict[str, Mapping[str, Any]] = {}
        for action in raw:
            if not isinstance(action, Mapping):
                raise TypeError("action_supervision entries must be objects")
            tool_id = action.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError("action_supervision entries need a non-empty tool_id")
            if tool_id in output:
                raise ValueError(
                    f"Example {row.get('example_id')!r} has duplicate action {tool_id!r}"
                )
            output[tool_id] = action
        return output

    def _validate_reference(
        self,
        row_index: int,
        raw_path: Any,
        *,
        rank: int,
    ) -> SearchPathReference:
        row = self.store.examples[row_index]
        label = f"Example {row['example_id']!r} top_k_paths[{rank - 1}]"
        if not isinstance(raw_path, Mapping):
            raise TypeError(f"{label} must be an object")

        raw_tools = raw_path.get("tool_ids")
        raw_states = raw_path.get("state_ids")
        if not isinstance(raw_tools, list) or not all(
            isinstance(tool_id, str) and tool_id for tool_id in raw_tools
        ):
            raise ValueError(f"{label}.tool_ids must be a list of non-empty strings")
        if not raw_tools:
            raise ValueError(f"{label} cannot be empty for a nonterminal Flow target")
        if not isinstance(raw_states, list) or not all(
            isinstance(state_id, str) and state_id for state_id in raw_states
        ):
            raise ValueError(f"{label}.state_ids must be a list of non-empty strings")
        if len(raw_states) != len(raw_tools) + 1:
            raise ValueError(
                f"{label} has {len(raw_tools)} actions but {len(raw_states)} states; "
                "a complete path needs actions+1 states"
            )
        if self.max_path_length is not None and len(raw_tools) > self.max_path_length:
            raise ValueError(
                f"{label} horizon={len(raw_tools)} exceeds max_path_length="
                f"{self.max_path_length}; do not truncate a reference path"
            )

        task_id = str(row["task_id"])
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError(f"Example {row['example_id']!r} needs a non-empty split")
        root_state_id = str(row["state_id"])
        if raw_states[0] != root_state_id:
            raise ValueError(
                f"{label} starts at {raw_states[0]!r}, expected root {root_state_id!r}"
            )

        state_indices: list[int] = []
        for state_id in raw_states:
            resolved = self._state_index.get((task_id, state_id))
            if resolved is None:
                raise ValueError(f"{label} references state {state_id!r} outside task {task_id!r}")
            state_row = self.store.examples[resolved]
            if state_row.get("task_id") != task_id:
                raise ValueError(f"{label} crosses task boundaries at state {state_id!r}")
            if state_row.get("split") != split:
                raise ValueError(f"{label} crosses split boundaries at state {state_id!r}")
            if state_row.get("goal_reachable") is not True:
                raise ValueError(f"{label} includes non-reachable state {state_id!r}")
            if state_row.get("training_mask") is not True:
                raise ValueError(f"{label} includes non-training state {state_id!r}")
            state_indices.append(resolved)

        total_from_edges = 0.0
        tool_indices: list[int] = []
        for step, tool_id in enumerate(raw_tools):
            if tool_id not in self.store.tool_index:
                raise ValueError(f"{label} uses unknown tool {tool_id!r}")
            source = self.store.examples[state_indices[step]]
            if source.get("terminal") is True:
                raise ValueError(f"{label} continues after terminal state {raw_states[step]!r}")
            action = self._actions_by_tool(source).get(tool_id)
            if action is None:
                raise ValueError(
                    f"{label} action {tool_id!r} is absent at state {raw_states[step]!r}"
                )
            if action.get("transition_known") is not True or action.get("known") is not True:
                raise ValueError(
                    f"{label} action {tool_id!r} at state {raw_states[step]!r} is not known"
                )
            if action.get("executable") is not True:
                raise ValueError(
                    f"{label} action {tool_id!r} at state {raw_states[step]!r} is not executable"
                )
            if action.get("reachable") is not True:
                raise ValueError(
                    f"{label} action {tool_id!r} at state {raw_states[step]!r} is not reachable"
                )
            expected_successor = raw_states[step + 1]
            if action.get("successor_id") != expected_successor:
                raise ValueError(
                    f"{label} chain mismatch at step {step}: action {tool_id!r} points to "
                    f"{action.get('successor_id')!r}, path names {expected_successor!r}"
                )
            total_from_edges += _finite_nonnegative(
                action.get("cost"),
                label=f"{label} action {tool_id!r}.cost",
            )
            tool_indices.append(int(self.store.tool_index[tool_id]))

        final_row = self.store.examples[state_indices[-1]]
        if final_row.get("terminal") is not True:
            raise ValueError(f"{label} does not terminate at a goal-satisfied state")

        total_cost = _finite_nonnegative(raw_path.get("total_cost"), label=f"{label}.total_cost")
        if not math.isclose(
            total_cost,
            total_from_edges,
            rel_tol=0.0,
            abs_tol=self.tolerance,
        ):
            raise ValueError(
                f"{label} cost mismatch: stored={total_cost}, edge_sum={total_from_edges}"
            )
        value_star = _finite_nonnegative(row.get("value_star"), label=f"{label} root V*")
        if total_cost < value_star - self.tolerance:
            raise ValueError(f"{label} total_cost={total_cost} is below certified V*={value_star}")
        excess_cost = max(0.0, total_cost - value_star)
        return SearchPathReference(
            row_index=row_index,
            example_id=str(row["example_id"]),
            task_id=task_id,
            split=split,
            root_state_id=root_state_id,
            rank=rank,
            tool_ids=tuple(raw_tools),
            tool_indices=tuple(tool_indices),
            state_ids=tuple(raw_states),
            state_indices=tuple(state_indices),
            total_cost=total_cost,
            value_star=value_star,
            excess_cost=excess_cost,
            probability=0.0,
        )

    def _build(self) -> None:
        by_split_task: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for row_index, row in enumerate(self.store.examples):
            # This check intentionally happens before reading any oracle path,
            # reachability, or value label.  A trainer can therefore construct a
            # train/dev-only store without silently validating sealed test labels.
            if self.allowed_splits is not None and row.get("split") not in self.allowed_splits:
                self._exclusion_counts["split_not_loaded"] += 1
                continue
            exclusion = self._exclusion_reason(row)
            if exclusion is not None:
                self._exclusion_counts[exclusion] += 1
                continue

            raw_paths = row.get("top_k_paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError(
                    f"Eligible example {row.get('example_id')!r} needs at least one top-k path"
                )
            references = [
                self._validate_reference(row_index, raw, rank=rank)
                for rank, raw in enumerate(raw_paths, start=1)
            ]
            signatures = [reference.tool_ids for reference in references]
            if len(set(signatures)) != len(signatures):
                raise ValueError(
                    f"Eligible example {row.get('example_id')!r} has duplicate tool paths"
                )
            references.sort(key=lambda item: (item.total_cost, item.tool_ids, item.state_ids))
            if not math.isclose(
                references[0].total_cost,
                references[0].value_star,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                raise ValueError(
                    f"Eligible example {row.get('example_id')!r} has no path attaining V*: "
                    f"best={references[0].total_cost}, V*={references[0].value_star}"
                )

            unnormalized = [
                math.exp(-reference.excess_cost / self.beta) for reference in references
            ]
            normalizer = sum(unnormalized)
            if not math.isfinite(normalizer) or normalizer <= 0:
                raise ValueError(
                    f"Path utility distribution underflowed for {row.get('example_id')!r}"
                )
            normalized = tuple(
                replace(reference, rank=index + 1, probability=weight / normalizer)
                for index, (reference, weight) in enumerate(
                    zip(references, unnormalized, strict=True)
                )
            )
            self._references[row_index] = normalized
            split = str(row["split"])
            task_id = str(row["task_id"])
            by_split_task[split][task_id].append(row_index)

        self._indices_by_split_task = {
            split: {task_id: tuple(sorted(indices)) for task_id, indices in sorted(tasks.items())}
            for split, tasks in sorted(by_split_task.items())
        }
        if not self._references:
            raise ValueError("No eligible nonterminal reachable search paths were found")

    def eligible_indices(self, split: str | None = None) -> tuple[int, ...]:
        """Return validated state rows, optionally restricted to one split."""

        if split is None:
            return tuple(sorted(self._references))
        return tuple(
            index
            for task_id in sorted(self._indices_by_split_task.get(split, {}))
            for index in self._indices_by_split_task[split][task_id]
        )

    def references(self, row_index: int) -> tuple[SearchPathReference, ...]:
        """Return all utility-weighted references for one eligible state."""

        if row_index not in self._references:
            raise KeyError(f"Row index {row_index} is not an eligible path root")
        return self._references[row_index]

    def sample_state_index(self, split: str, generator: RandomGenerator) -> int:
        """Uniformly sample task, then state, rather than sampling graph rows flat."""

        tasks = self._indices_by_split_task.get(split, {})
        task_ids = tuple(sorted(tasks))
        if not task_ids:
            raise ValueError(f"No eligible path states in split={split!r}")
        task_id = task_ids[_draw_index(len(task_ids), generator)]
        state_indices = tasks[task_id]
        return state_indices[_draw_index(len(state_indices), generator)]

    def sample_reference(
        self,
        row_index: int,
        generator: RandomGenerator,
    ) -> SearchPathReference:
        """Sample a state-local path using its Boltzmann utility distribution."""

        references = self.references(row_index)
        draw = _draw_unit(generator)
        cumulative = 0.0
        for reference in references:
            cumulative += reference.probability
            if draw < cumulative:
                return reference
        # Floating-point accumulation can end infinitesimally below one.
        return references[-1]

    def sample(self, split: str, generator: RandomGenerator) -> SearchPathReference:
        """Hierarchically sample task -> state -> utility-weighted reference path."""

        return self.sample_reference(self.sample_state_index(split, generator), generator)

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        generator: RandomGenerator,
    ) -> tuple[SearchPathReference, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return tuple(self.sample(split, generator) for _ in range(batch_size))

    def report(self, split: str | None = None) -> dict[str, Any]:
        """Return JSON-serializable path coverage, horizon, and utility statistics."""

        indices = self.eligible_indices(split)
        references = [reference for index in indices for reference in self._references[index]]
        tasks = {reference.task_id for reference in references}
        counts_per_state = [len(self._references[index]) for index in indices]
        horizons = [reference.horizon for reference in references]
        excess = [reference.excess_cost for reference in references]
        entropy = []
        for index in indices:
            probabilities = [reference.probability for reference in self._references[index]]
            entropy.append(-sum(value * math.log(value) for value in probabilities if value > 0))
        cost_diverse = sum(
            len({reference.total_cost for reference in self._references[index]}) > 1
            for index in indices
        )
        horizon_histogram = Counter(horizons)
        if split is None:
            excluded_rows = self._exclusion_counts
        else:
            excluded_rows = Counter(
                reason
                for row in self.store.examples
                if row.get("split") == split
                for reason in [
                    (
                        "split_not_loaded"
                        if self.allowed_splits is not None
                        and row.get("split") not in self.allowed_splits
                        else self._exclusion_reason(row)
                    )
                ]
                if reason is not None
            )
        return {
            "split": split or "all",
            "beta": self.beta,
            "eligible_tasks": len(tasks),
            "eligible_states": len(indices),
            "references": len(references),
            "multi_path_states": sum(count > 1 for count in counts_per_state),
            "cost_diverse_states": cost_diverse,
            "max_paths_per_state": max(counts_per_state, default=0),
            "mean_paths_per_state": fmean(counts_per_state) if counts_per_state else None,
            "mean_horizon": fmean(horizons) if horizons else None,
            "max_horizon": max(horizons, default=None),
            "horizon_histogram": {
                str(horizon): count for horizon, count in sorted(horizon_histogram.items())
            },
            "mean_excess_cost": fmean(excess) if excess else None,
            "mean_path_entropy": fmean(entropy) if entropy else None,
            "excluded_rows": dict(sorted(excluded_rows.items())),
        }


__all__ = ["SearchPathReference", "SearchPathStore"]
