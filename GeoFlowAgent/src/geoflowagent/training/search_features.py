"""Tensorize exact-search supervision without changing frozen embeddings.

``SearchFeatureStore`` is the boundary between three immutable artifacts:

* processed state/action rows produced by :mod:`geoflowagent.data.search_supervision`;
* frozen, field-wise embedding arrays produced by
  :mod:`geoflowagent.embeddings.cache`;
* the small trainable goal-conditioned value/geometry model.

    The store deliberately keeps state context, goal, and tool encodings separate.
    State context pools only the model-visible query, current state, history, and
    configured background cards; it explicitly excludes the goal endpoint.  It
    does not pool oracle targets or the private verifier into a model input.
    ``<STOP>`` is represented by the scalar completion target, never by appending a
    synthetic tool prototype.  Labels for unexplored actions stay ``NaN`` and are
    accompanied by false masks; they are not converted to negative examples.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.data.structured import StructuredHasher
from geoflowagent.embeddings.cache import EmbeddingCache
from geoflowagent.utils.io import canonical_json, read_jsonl, sha256_file


def _optional_finite(value: Any, *, label: str) -> float | None:
    """Normalize a numeric target while treating ``None`` as unobserved."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number or null")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite or null, got {value!r}")
    return normalized


def _optional_bool(value: Any, *, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean or null")
    return value


def _mapping_field(row: Mapping[str, Any], name: str, tool_id: str) -> Any:
    value = row.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an action-keyed object")
    return value.get(tool_id)


def _action_rows(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Accept both the canonical list form and a convenient action-keyed form."""

    raw = row.get("action_supervision")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        output: dict[str, Mapping[str, Any]] = {}
        for tool_id, value in raw.items():
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError("action_supervision keys must be non-empty tool IDs")
            if not isinstance(value, Mapping):
                raise TypeError(f"action_supervision[{tool_id!r}] must be an object")
            supplied = value.get("tool_id", value.get("action"))
            if supplied not in {None, tool_id}:
                raise ValueError(
                    f"action_supervision key {tool_id!r} disagrees with tool_id={supplied!r}"
                )
            output[tool_id] = value
        return output
    if not isinstance(raw, list):
        raise TypeError("action_supervision must be a list or action-keyed object")
    output = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise TypeError("action_supervision entries must be objects")
        tool_id = value.get("tool_id", value.get("action"))
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError("action_supervision entries need a non-empty tool_id")
        if tool_id in output:
            raise ValueError(f"Duplicate action supervision for {tool_id!r}")
        output[tool_id] = value
    return output


def _action_value(
    row: Mapping[str, Any],
    action: Mapping[str, Any] | None,
    *,
    action_field: str,
    map_field: str,
    tool_id: str,
) -> Any:
    # An explicit null in the canonical action row is meaningful.  For example,
    # a contract-invalid action has no search regret even if a legacy flat map
    # carries a ranking sentinel.
    if action is not None and action_field in action:
        return action[action_field]
    return _mapping_field(row, map_field, tool_id)


class SearchFeatureStore:
    """Join search examples and frozen embedding arrays by stable identifiers.

    Parameters
    ----------
    processed_dir:
        A checksummed processed directory containing search-supervision examples.
    cache_dir:
        The immutable embedding cache built from exactly that processed manifest.
    structured_dim:
        Width of deterministic structured state/goal/tool features.
    verify_cache:
        Verify every cached array checksum and shape before exposing tensors.

    Notes
    -----
        ``views`` contains complete prototype views plus configured state-only
        genomic views.  A state-only view contributes its real snapshot embedding
        (for example sequence or entity context); its missing goal and tool
        modalities are explicit zero tensors.  ``view_roles`` records every such
        substitution so it cannot be mistaken for observed multimodal evidence.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        cache_dir: str | Path,
        *,
        structured_dim: int = 64,
        verify_cache: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        processed_manifest = verify_processed_dataset(self.processed_dir)
        self.examples = read_jsonl(self.processed_dir / "examples.jsonl")
        self.tools = read_jsonl(self.processed_dir / "tools.jsonl")
        if not self.examples:
            raise ValueError("Search feature store requires at least one example")
        if not self.tools:
            raise ValueError("Search feature store requires at least one real tool")

        self.cache = EmbeddingCache(cache_dir, verify=verify_cache)
        source_manifest = sha256_file(self.processed_dir / "manifest.json")
        if self.cache.manifest.get("source_manifest_sha256") != source_manifest:
            raise ValueError("Embedding cache was built from a different processed manifest")
        if processed_manifest.get("processed_files", {}).get("examples.jsonl") is None:
            raise ValueError("Processed manifest does not bind examples.jsonl")

        self.example_ids = [str(row["example_id"]) for row in self.examples]
        self.tool_ids = [str(row["tool_id"]) for row in self.tools]
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("Processed examples contain duplicate example_id values")
        if len(set(self.tool_ids)) != len(self.tool_ids):
            raise ValueError("Processed tools contain duplicate tool_id values")
        if STOP_TOOL_ID in self.tool_ids:
            raise ValueError("<STOP> must be a completion decision, not a tool prototype")
        if self.example_ids != self.cache.example_ids:
            raise ValueError("Processed examples and embedding cache order/IDs differ")
        if self.tool_ids != self.cache.tool_ids:
            raise ValueError("Processed tools and embedding cache order/IDs differ")

        self.example_index = {example_id: index for index, example_id in enumerate(self.example_ids)}
        self.tool_index = {tool_id: index for index, tool_id in enumerate(self.tool_ids)}
        self.views = tuple(self._select_views())
        if not self.views:
            raise ValueError(
                "At least one usable frozen embedding view is required"
            )

        # Goal is intentionally not pooled into state context: the trainable
        # model receives it as a distinct endpoint.  Snapshot views are retained
        # for transition supervision, where query/history pooling would make the
        # target dependent on a representative search path.
        self.state_views = {
            view: torch.from_numpy(self._state_context_matrix(view)).float()
            for view in self.views
        }
        self.state_snapshot_views = {
            view: torch.from_numpy(
                np.array(
                    self.cache.example_field(view, self._snapshot_field(view)), copy=True
                )
            ).float()
            for view in self.views
        }
        self.goal_views = {
            view: (
                torch.from_numpy(
                    np.array(self.cache.example_field(view, "goal"), copy=True)
                ).float()
                if "goal" in self.cache.manifest["views"][view].get("example_fields", [])
                else torch.zeros(
                    (len(self.examples), int(self.cache.manifest["views"][view]["dim"])),
                    dtype=torch.float32,
                )
            )
            for view in self.views
        }
        self.tool_views = {
            view: (
                torch.from_numpy(np.array(self.cache.tool_matrix(view), copy=True)).float()
                if self.cache.manifest["views"][view].get("tool_fields")
                else torch.zeros(
                    (len(self.tools), int(self.cache.manifest["views"][view]["dim"])),
                    dtype=torch.float32,
                )
            )
            for view in self.views
        }
        self.view_roles = {
            view: self._view_role(view)
            for view in self.views
        }
        self.zero_filled_goal_views = tuple(
            view for view, role in self.view_roles.items() if role["goal_zero_filled"]
        )
        self.zero_filled_tool_views = tuple(
            view for view, role in self.view_roles.items() if role["tool_zero_filled"]
        )

        hasher = StructuredHasher(structured_dim)
        self.structured_state = torch.from_numpy(
            np.stack([hasher.encode_state(row["state"]) for row in self.examples])
        ).float()
        self.structured_goal = torch.from_numpy(
            np.stack([hasher.encode_state(row["goal"]) for row in self.examples])
        ).float()
        self.structured_tools = torch.from_numpy(
            np.stack([hasher.encode_tool(row) for row in self.tools])
        ).float()
        self.structured_dim = int(structured_dim)
        self.feature_ablation = {"zero_views": [], "zero_structured": False}

        self._state_index = self._build_state_index()
        self._state_payload_index = self._build_state_payload_index()
        self._targets = self._tensorize_targets()
        # Training repeatedly gathers wide successor tensors. CUDA runs keep one
        # immutable device copy so indexing happens on the GPU instead of
        # rebuilding and transferring the same frozen sources every batch.
        self._batch_device_cache: dict[str, dict[str, Any]] = {}
        self.split_indices = {
            split: [index for index, row in enumerate(self.examples) if row.get("split") == split]
            for split in ("train", "dev", "test")
        }

    def apply_feature_ablation(self, specification: Mapping[str, Any] | None) -> None:
        """Zero selected input channels without changing architecture or cache files."""

        supplied = dict(specification or {})
        unknown = sorted(set(supplied) - {"zero_views", "zero_structured"})
        if unknown:
            raise ValueError(f"Unknown feature-ablation settings: {unknown}")
        raw_views = supplied.get("zero_views", [])
        if isinstance(raw_views, str):
            raw_views = [raw_views]
        if not isinstance(raw_views, Iterable):
            raise TypeError("feature_ablation.zero_views must be a sequence")
        requested = {str(value) for value in raw_views}
        if "*" in requested:
            requested = set(self.views)
        missing = sorted(requested - set(self.views))
        if missing:
            raise ValueError(f"Cannot ablate unknown frozen views: {missing}")
        normalized = {
            "zero_views": sorted(requested),
            "zero_structured": bool(supplied.get("zero_structured", False)),
        }
        if self.feature_ablation == normalized:
            return
        if self.feature_ablation != {"zero_views": [], "zero_structured": False}:
            raise RuntimeError("A feature store cannot receive two different ablations")
        for view in normalized["zero_views"]:
            self.state_views[view] = torch.zeros_like(self.state_views[view])
            self.state_snapshot_views[view] = torch.zeros_like(
                self.state_snapshot_views[view]
            )
            self.goal_views[view] = torch.zeros_like(self.goal_views[view])
            self.tool_views[view] = torch.zeros_like(self.tool_views[view])
        if normalized["zero_structured"]:
            self.structured_state = torch.zeros_like(self.structured_state)
            self.structured_goal = torch.zeros_like(self.structured_goal)
            self.structured_tools = torch.zeros_like(self.structured_tools)
        self.feature_ablation = normalized
        self._batch_device_cache.clear()

    def _batch_sources(self, device: torch.device) -> dict[str, Any]:
        """Materialize immutable batch sources once per CUDA device."""

        if device.type != "cuda":
            return {
                "state_views": self.state_views,
                "state_snapshot_views": self.state_snapshot_views,
                "goal_views": self.goal_views,
                "tool_views": self.tool_views,
                "structured_state": self.structured_state,
                "structured_goal": self.structured_goal,
                "structured_tools": self.structured_tools,
                "targets": self._targets,
            }
        key = str(device)
        cached = self._batch_device_cache.get(key)
        if cached is None:
            cached = {
                "state_views": {
                    name: value.to(device) for name, value in self.state_views.items()
                },
                "state_snapshot_views": {
                    name: value.to(device)
                    for name, value in self.state_snapshot_views.items()
                },
                "goal_views": {
                    name: value.to(device) for name, value in self.goal_views.items()
                },
                "tool_views": {
                    name: value.to(device) for name, value in self.tool_views.items()
                },
                "structured_state": self.structured_state.to(device),
                "structured_goal": self.structured_goal.to(device),
                "structured_tools": self.structured_tools.to(device),
                "targets": {
                    name: value.to(device) for name, value in self._targets.items()
                },
            }
            self._batch_device_cache[key] = cached
        return cached

    def ablate_runtime_inputs(
        self,
        state_views: Mapping[str, torch.Tensor],
        goal_views: Mapping[str, torch.Tensor],
        structured_state: torch.Tensor,
        structured_goal: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        zero_views = set(self.feature_ablation["zero_views"])
        state = {
            name: torch.zeros_like(value) if name in zero_views else value
            for name, value in state_views.items()
        }
        goal = {
            name: torch.zeros_like(value) if name in zero_views else value
            for name, value in goal_views.items()
        }
        if self.feature_ablation["zero_structured"]:
            structured_state = torch.zeros_like(structured_state)
            structured_goal = torch.zeros_like(structured_goal)
        return state, goal, structured_state, structured_goal

    def _select_views(self) -> list[str]:
        selected: list[str] = []
        for view, metadata in self.cache.manifest["views"].items():
            kind = metadata.get("kind", "prototype")
            if kind not in {"prototype", "state_only"}:
                continue
            fields = set(metadata.get("example_fields", []))
            if kind == "prototype" and {"state", "goal"}.issubset(fields) and metadata.get(
                "tool_fields"
            ):
                selected.append(view)
            elif kind == "state_only" and fields:
                selected.append(view)
        return sorted(selected)

    def _snapshot_field(self, view: str) -> str:
        metadata = self.cache.manifest["views"][view]
        fields = [
            field for field in metadata.get("example_fields", []) if not field.startswith("next_")
        ]
        snapshot_field = metadata.get("snapshot_field")
        if snapshot_field is None:
            if "state" in fields:
                snapshot_field = "state"
            else:
                # Prefer a state/entity modality over a generic query if a
                # legacy state-only config omitted snapshot_field.
                non_query = [field for field in fields if field not in {"query", "goal", "history"}]
                snapshot_field = non_query[0] if non_query else fields[0]
        if snapshot_field not in fields:
            raise ValueError(
                f"View {view!r} snapshot_field={snapshot_field!r} is not a current-state field"
            )
        return str(snapshot_field)

    def _view_role(self, view: str) -> dict[str, Any]:
        metadata = self.cache.manifest["views"][view]
        fields = set(metadata.get("example_fields", []))
        kind = str(metadata.get("kind", "prototype"))
        return {
            "kind": kind,
            "state_semantics": (
                "query+state+history+background excluding goal"
                if kind == "prototype"
                else f"snapshot:{self._snapshot_field(view)}"
            ),
            "snapshot_field": self._snapshot_field(view),
            "goal_zero_filled": "goal" not in fields,
            "tool_zero_filled": not bool(metadata.get("tool_fields")),
        }

    def _state_context_matrix(self, view: str) -> np.ndarray:
        """Pool visible context fields while keeping ``goal`` strictly separate."""

        metadata = self.cache.manifest["views"][view]
        if metadata.get("kind") == "state_only":
            return np.array(
                self.cache.example_field(view, self._snapshot_field(view)), copy=True
            ).astype(np.float32, copy=False)
        fields = [
            field
            for field in metadata.get("example_fields", [])
            if not field.startswith("next_") and field != "goal"
        ]
        if not fields:
            raise ValueError(f"View {view!r} has no model-visible state context fields")
        arrays = [
            np.asarray(self.cache.example_field(view, field), dtype=np.float32)
            for field in fields
        ]
        dim = int(metadata["dim"])
        output = np.zeros((len(self.examples), dim), dtype=np.float32)

        background_relative = f"{view}/background.text.npy"
        cards = (
            np.asarray(self.cache.array(background_relative), dtype=np.float32)
            if background_relative in self.cache.manifest["files"]
            else None
        )
        background_index = {
            str(card_id): index
            for index, card_id in enumerate(
                self.cache.manifest.get("indices", {}).get("background", [])
            )
        }
        for row_index, row in enumerate(self.examples):
            pieces = [
                array[row_index] for array in arrays if np.linalg.norm(array[row_index]) > 0
            ]
            if cards is not None:
                selected_cards = [
                    cards[background_index[str(card_id)]]
                    for card_id in row.get("background_ids", [])
                    if str(card_id) in background_index
                ]
                if selected_cards:
                    # Match the cache/runtime semantics: selected cards form one
                    # contextual piece rather than dominating examples with more
                    # background records.
                    pieces.append(np.stack(selected_cards).mean(axis=0))
            if pieces:
                output[row_index] = np.stack(pieces).mean(axis=0)
        return output

    def _build_state_index(self) -> dict[tuple[str, str], int]:
        output: dict[tuple[str, str], int] = {}
        for index, row in enumerate(self.examples):
            task_id = str(row["task_id"])
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"Example {row['example_id']!r} needs a stable state_id")
            key = (task_id, state_id)
            if key in output:
                raise ValueError(f"Duplicate search state identifier {key!r}")
            output[key] = index
        return output

    def _build_state_payload_index(self) -> dict[tuple[str, str], int]:
        output: dict[tuple[str, str], int] = {}
        for index, row in enumerate(self.examples):
            key = (str(row["task_id"]), canonical_json(row["state"]))
            previous = output.setdefault(key, index)
            if previous != index:
                raise ValueError(
                    f"Task {row['task_id']!r} has duplicate typed states with different IDs"
                )
        return output

    @property
    def view_dims(self) -> dict[str, int]:
        return {view: int(self.state_views[view].shape[-1]) for view in self.views}

    @property
    def all_view_dims(self) -> dict[str, int]:
        """Alias emphasizing that state-only genomic modalities are included."""

        return self.view_dims

    def indices(
        self,
        split: str,
        *,
        include_terminal: bool = True,
        reachable_only: bool = False,
        training_only: bool = False,
    ) -> list[int]:
        """Return row indices while preserving task-level split assignment."""

        indices = list(self.split_indices.get(split, []))
        if not include_terminal:
            indices = [index for index in indices if not bool(self.examples[index].get("terminal"))]
        if reachable_only:
            indices = [
                index for index in indices if self.examples[index].get("goal_reachable") is True
            ]
        if training_only:
            indices = [
                index for index in indices if self.examples[index].get("training_mask", True) is True
            ]
        return indices

    def state_index(self, task_id: str, state: Mapping[str, Any]) -> int | None:
        """Resolve an exactly matching typed Markov state for runtime cache reuse."""

        return self._state_payload_index.get((str(task_id), canonical_json(state)))

    def runtime_features(
        self,
        task_id: str,
        state: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ] | None:
        """Return cached state/goal features for an exact Markov-state hit.

        Histories that merge into the same typed search node share its stable,
        minimum-cost representative context. This is valid only because every
        future-relevant history summary and remaining budget is required to be in
        the typed state before compilation.
        """

        index = self.state_index(task_id, state)
        if index is None:
            return None
        target = torch.device(device)
        return (
            {
                view: matrix[index].unsqueeze(0).to(target)
                for view, matrix in self.state_views.items()
            },
            {
                view: matrix[index].unsqueeze(0).to(target)
                for view, matrix in self.goal_views.items()
            },
            self.structured_state[index].unsqueeze(0).to(target),
            self.structured_goal[index].unsqueeze(0).to(target),
        )

    def _resolve_successor_index(
        self,
        row: Mapping[str, Any],
        action: Mapping[str, Any] | None,
        tool_id: str,
    ) -> int | None:
        successor_id = _action_value(
            row,
            action,
            action_field="successor_id",
            map_field="action_successor_ids",
            tool_id=tool_id,
        )
        task_id = str(row["task_id"])
        if successor_id is not None:
            if not isinstance(successor_id, str) or not successor_id:
                raise ValueError(
                    f"Example {row['example_id']!r} action {tool_id!r} has invalid successor_id"
                )
            resolved = self._state_index.get((task_id, successor_id))
            if resolved is not None:
                return resolved
        successor = action.get("successor") if action is not None else None
        if isinstance(successor, Mapping):
            return self._state_payload_index.get((task_id, canonical_json(successor)))
        return None

    def _tensorize_targets(self) -> dict[str, torch.Tensor]:
        rows = len(self.examples)
        actions = len(self.tool_ids)

        def bool_matrix() -> torch.Tensor:
            return torch.zeros((rows, actions), dtype=torch.bool)

        def float_matrix() -> torch.Tensor:
            return torch.full((rows, actions), torch.nan, dtype=torch.float32)

        candidate_mask = bool_matrix()
        contract_mask = bool_matrix()
        transition_known_mask = bool_matrix()
        transition_observed_target = float_matrix()
        reachability_known_mask = bool_matrix()
        known_action_mask = bool_matrix()
        optimal_action_mask = bool_matrix()
        optimal_action_label_mask = bool_matrix()
        q_star = float_matrix()
        edge_cost = float_matrix()
        regret = float_matrix()
        action_reachability = float_matrix()
        action_executable = float_matrix()
        successor_index = torch.full((rows, actions), -1, dtype=torch.long)
        successor_feature_mask = bool_matrix()
        successor_value = float_matrix()

        value = torch.full((rows,), torch.nan, dtype=torch.float32)
        value_known_mask = torch.zeros((rows,), dtype=torch.bool)
        state_reachability = torch.full((rows,), torch.nan, dtype=torch.float32)
        state_reachability_known_mask = torch.zeros((rows,), dtype=torch.bool)
        completion = torch.full((rows,), torch.nan, dtype=torch.float32)
        training_mask = torch.zeros((rows,), dtype=torch.bool)

        for row_index, row in enumerate(self.examples):
            example_id = str(row["example_id"])
            candidates = row.get("candidate_tools")
            if not isinstance(candidates, list) or not all(
                isinstance(tool_id, str) for tool_id in candidates
            ):
                raise TypeError(f"Example {example_id!r}.candidate_tools must be a list of IDs")
            if STOP_TOOL_ID in candidates:
                raise ValueError(
                    f"Example {example_id!r} stores <STOP> as a candidate tool; use terminal instead"
                )
            unknown_candidates = sorted(set(candidates) - set(self.tool_ids))
            if unknown_candidates:
                raise ValueError(
                    f"Example {example_id!r} references unknown tools: {unknown_candidates}"
                )
            for tool_id in candidates:
                candidate_mask[row_index, self.tool_index[tool_id]] = True

            contract_tools = row.get("contract_valid_tools", [])
            if not isinstance(contract_tools, list) or not all(
                isinstance(tool_id, str) for tool_id in contract_tools
            ):
                raise TypeError(
                    f"Example {example_id!r}.contract_valid_tools must be a list of IDs"
                )
            unknown_contract = sorted(set(contract_tools) - set(self.tool_ids))
            if unknown_contract:
                raise ValueError(
                    f"Example {example_id!r} has unknown contract-valid tools: {unknown_contract}"
                )
            for tool_id in contract_tools:
                contract_mask[row_index, self.tool_index[tool_id]] = True

            actions_by_id = _action_rows(row)
            unexpected = sorted(set(actions_by_id) - set(candidates))
            if unexpected:
                raise ValueError(
                    f"Example {example_id!r} labels non-candidate actions: {unexpected}"
                )
            declared_optimal = row.get("optimal_actions", row.get("valid_next_tools", []))
            if not isinstance(declared_optimal, list):
                raise TypeError(f"Example {example_id!r}.optimal_actions must be a list")
            declared_optimal_set = {tool_id for tool_id in declared_optimal if tool_id != STOP_TOOL_ID}
            unknown_optimal = sorted(declared_optimal_set - set(self.tool_ids))
            if unknown_optimal:
                raise ValueError(
                    f"Example {example_id!r} has unknown optimal tools: {unknown_optimal}"
                )

            for tool_id in candidates:
                tool_index = self.tool_index[tool_id]
                action = actions_by_id.get(tool_id)
                q_known_raw = _action_value(
                    row,
                    action,
                    action_field="label_mask",
                    map_field="action_known_mask",
                    tool_id=tool_id,
                )
                if q_known_raw is None and action is not None:
                    q_known_raw = action.get("q_known_mask", action.get("known"))

                transition_raw = action.get("transition_known") if action is not None else None
                transition_mask_raw = _action_value(
                    row,
                    action,
                    action_field="transition_known_mask",
                    map_field="action_transition_known_mask",
                    tool_id=tool_id,
                )
                reachability_mask_raw = _action_value(
                    row,
                    action,
                    action_field="reachability_known_mask",
                    map_field="action_reachability_known_mask",
                    tool_id=tool_id,
                )

                q_raw = _action_value(
                    row,
                    action,
                    action_field="q_star",
                    map_field="action_q",
                    tool_id=tool_id,
                )
                cost_raw = _action_value(
                    row,
                    action,
                    action_field="cost",
                    map_field="action_cost",
                    tool_id=tool_id,
                )
                regret_raw = _action_value(
                    row,
                    action,
                    action_field="regret",
                    map_field="action_regret",
                    tool_id=tool_id,
                )
                reachable_raw = action.get("reachable") if action is not None else None
                executable_raw = action.get("executable") if action is not None else None
                optimal_raw = action.get("is_optimal") if action is not None else None

                if q_known_raw is None:
                    q_known = q_raw is not None
                elif not isinstance(q_known_raw, bool):
                    raise TypeError(
                        f"Example {example_id!r} action {tool_id!r} Q-known mask must be boolean"
                    )
                else:
                    q_known = q_known_raw
                if not q_known and (q_raw is not None or regret_raw is not None):
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has masked Q/regret labels"
                    )

                if transition_mask_raw is None:
                    transition_labeled = transition_raw is not None or any(
                        item is not None for item in (cost_raw, executable_raw)
                    )
                elif not isinstance(transition_mask_raw, bool):
                    raise TypeError(
                        f"Example {example_id!r} action {tool_id!r} transition mask must be boolean"
                    )
                else:
                    transition_labeled = transition_mask_raw
                if transition_raw is None:
                    transition_observed = bool(
                        q_known
                        or cost_raw is not None
                        or (
                            executable_raw is True
                            and action is not None
                            and action.get("successor") is not None
                        )
                    )
                elif not isinstance(transition_raw, bool):
                    raise TypeError(
                        f"Example {example_id!r} action {tool_id!r}.transition_known must be boolean"
                    )
                else:
                    transition_observed = transition_raw
                if transition_observed and not transition_labeled:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has an unmasked transition"
                    )
                if not transition_observed and cost_raw is not None:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has cost without a transition"
                    )
                if q_known and not transition_observed:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has Q without a transition"
                    )

                if reachability_mask_raw is None:
                    reachability_labeled = reachable_raw is not None
                elif not isinstance(reachability_mask_raw, bool):
                    raise TypeError(
                        f"Example {example_id!r} action {tool_id!r} reachability mask must be boolean"
                    )
                else:
                    reachability_labeled = reachability_mask_raw
                if reachable_raw is not None and not reachability_labeled:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has unmasked reachability"
                    )

                known_action_mask[row_index, tool_index] = q_known
                transition_known_mask[row_index, tool_index] = transition_labeled
                if transition_labeled:
                    transition_observed_target[row_index, tool_index] = float(
                        transition_observed
                    )
                reachability_known_mask[row_index, tool_index] = reachability_labeled

                q_target = _optional_finite(
                    q_raw, label=f"{example_id}.{tool_id}.q_star"
                )
                cost_target = _optional_finite(
                    cost_raw, label=f"{example_id}.{tool_id}.cost"
                )
                regret_target = _optional_finite(
                    regret_raw, label=f"{example_id}.{tool_id}.regret"
                )
                reachable_target = _optional_bool(
                    reachable_raw, label=f"{example_id}.{tool_id}.reachable"
                )
                executable_target = _optional_bool(
                    executable_raw, label=f"{example_id}.{tool_id}.executable"
                )
                optimal_target = _optional_bool(
                    optimal_raw, label=f"{example_id}.{tool_id}.is_optimal"
                )
                if q_known and q_target is None:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} is Q-known but q_star is null"
                    )
                if q_target is not None:
                    q_star[row_index, tool_index] = q_target
                if cost_target is not None:
                    edge_cost[row_index, tool_index] = cost_target
                if regret_target is not None:
                    regret[row_index, tool_index] = regret_target
                if reachability_labeled and reachable_target is None:
                    raise ValueError(
                        f"Example {example_id!r} action {tool_id!r} has a reachability mask but no label"
                    )
                if reachable_target is not None:
                    action_reachability[row_index, tool_index] = float(reachable_target)
                if transition_labeled and executable_target is None:
                    # The transition-observed bit is the conservative fallback
                    # for older rows that lack a distinct executable label.
                    executable_target = transition_observed
                if executable_target is not None:
                    action_executable[row_index, tool_index] = float(executable_target)

                if optimal_target is None and tool_id in declared_optimal_set:
                    optimal_target = True
                if optimal_target is not None:
                    optimal_action_label_mask[row_index, tool_index] = True
                    optimal_action_mask[row_index, tool_index] = optimal_target

                resolved_successor = (
                    self._resolve_successor_index(row, action, tool_id)
                    if transition_observed
                    else None
                )
                if resolved_successor is not None:
                    successor_index[row_index, tool_index] = resolved_successor
                    successor_feature_mask[row_index, tool_index] = True
                    successor_row_value = self.examples[resolved_successor].get(
                        "value_star", self.examples[resolved_successor].get("v_star")
                    )
                    successor_target = _optional_finite(
                        successor_row_value,
                        label=(
                            f"successor {self.examples[resolved_successor]['example_id']}.value_star"
                        ),
                    )
                    if successor_target is not None:
                        successor_value[row_index, tool_index] = successor_target
                if not torch.isfinite(successor_value[row_index, tool_index]):
                    if q_target is not None and cost_target is not None:
                        successor_value[row_index, tool_index] = q_target - cost_target

            value_raw = row.get("value_star", row.get("v_star"))
            raw_value_mask = row.get("value_known_mask", value_raw is not None)
            if not isinstance(raw_value_mask, bool):
                raise TypeError(f"{example_id}.value_known_mask must be boolean")
            if not raw_value_mask and value_raw is not None:
                raise ValueError(f"{example_id} has an unmasked value_star")
            value_target = _optional_finite(value_raw, label=f"{example_id}.value_star")
            if (
                raw_value_mask
                and value_target is None
                and row.get("goal_reachable", row.get("reachable")) is not False
            ):
                raise ValueError(
                    f"{example_id} is reachable/value-known but value_star is null"
                )
            if value_target is not None:
                value[row_index] = value_target
                value_known_mask[row_index] = True
            reachability_raw = row.get("goal_reachable", row.get("reachable"))
            raw_state_reachability_mask = row.get(
                "goal_reachability_known_mask", reachability_raw is not None
            )
            if not isinstance(raw_state_reachability_mask, bool):
                raise TypeError(f"{example_id}.goal_reachability_known_mask must be boolean")
            if reachability_raw is not None and not raw_state_reachability_mask:
                raise ValueError(f"{example_id} has unmasked state reachability")
            if reachability_raw is None and value_target is not None:
                reachability_raw = True
                raw_state_reachability_mask = True
            reachability_target = _optional_bool(
                reachability_raw, label=f"{example_id}.goal_reachable"
            )
            if raw_state_reachability_mask and reachability_target is None:
                raise ValueError(f"{example_id} has a reachability mask but no label")
            if reachability_target is not None:
                state_reachability[row_index] = float(reachability_target)
                state_reachability_known_mask[row_index] = True
            terminal_target = _optional_bool(row.get("terminal"), label=f"{example_id}.terminal")
            if terminal_target is not None:
                completion[row_index] = float(terminal_target)
            raw_training = row.get("training_mask", True)
            if not isinstance(raw_training, bool):
                raise TypeError(f"{example_id}.training_mask must be boolean")
            training_mask[row_index] = raw_training

        completion_bool = torch.isfinite(completion) & (completion > 0.5)
        policy_candidate_mask = candidate_mask & contract_mask & ~completion_bool[:, None]
        return {
            "candidate_mask": candidate_mask,
            "contract_mask": contract_mask,
            "policy_candidate_mask": policy_candidate_mask,
            "transition_known_mask": transition_known_mask,
            "transition_observed_target": transition_observed_target,
            "reachability_known_mask": reachability_known_mask,
            "known_action_mask": known_action_mask,
            "known_candidate_mask": known_action_mask & policy_candidate_mask,
            "optimal_action_mask": optimal_action_mask,
            "optimal_action_label_mask": optimal_action_label_mask,
            "q_star": q_star,
            "q_mask": known_action_mask & torch.isfinite(q_star),
            "edge_cost": edge_cost,
            "edge_cost_mask": transition_known_mask & torch.isfinite(edge_cost),
            "regret": regret,
            "regret_mask": known_action_mask & torch.isfinite(regret),
            "action_reachability_target": action_reachability,
            "action_reachability_mask": reachability_known_mask
            & torch.isfinite(action_reachability),
            "action_executable_target": action_executable,
            "action_executable_mask": transition_known_mask & torch.isfinite(action_executable),
            "successor_index": successor_index,
            "successor_feature_mask": successor_feature_mask,
            "successor_value_target": successor_value,
            "successor_value_mask": torch.isfinite(successor_value),
            "value_target": value,
            "value_mask": value_known_mask,
            "state_reachability_target": state_reachability,
            "state_reachability_mask": state_reachability_known_mask,
            "completion_target": completion,
            "completion_mask": torch.isfinite(completion),
            "training_mask": training_mask,
        }

    @staticmethod
    def _checked_indices(indices: Iterable[int], size: int) -> torch.Tensor:
        values = list(indices)
        if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in values):
            raise TypeError("Batch indices must be integers")
        if any(int(index) < 0 or int(index) >= size for index in values):
            raise IndexError("Batch index is outside the search feature store")
        return torch.as_tensor(values, dtype=torch.long)

    def batch(
        self,
        indices: Iterable[int],
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Return model inputs and dense masked oracle targets for selected rows."""

        index = self._checked_indices(indices, len(self.examples))
        target_device = torch.device(device)
        sources = self._batch_sources(target_device)
        target_tensors = sources["targets"]
        device_index = index.to(target_device)
        selected_successors = target_tensors["successor_index"][device_index]
        safe_successors = selected_successors.clamp_min(0)
        successor_mask = target_tensors["successor_feature_mask"][device_index]

        successor_state_views: dict[str, torch.Tensor] = {}
        for view, matrix in sources["state_snapshot_views"].items():
            gathered = matrix[safe_successors]
            gathered = gathered.masked_fill(~successor_mask.unsqueeze(-1), torch.nan)
            successor_state_views[view] = gathered
        successor_context_views: dict[str, torch.Tensor] = {}
        for view, matrix in sources["state_views"].items():
            gathered = matrix[safe_successors]
            gathered = gathered.masked_fill(~successor_mask.unsqueeze(-1), torch.nan)
            successor_context_views[view] = gathered
        structured_successor = sources["structured_state"][safe_successors]
        structured_successor = structured_successor.masked_fill(
            ~successor_mask.unsqueeze(-1), torch.nan
        )

        batch = {
            "indices": device_index,
            "example_ids": [self.example_ids[int(item)] for item in index],
            "rows": [self.examples[int(item)] for item in index],
            "tool_ids": list(self.tool_ids),
            "state_views": {
                view: matrix[device_index]
                for view, matrix in sources["state_views"].items()
            },
            "state_snapshot_views": {
                view: matrix[device_index]
                for view, matrix in sources["state_snapshot_views"].items()
            },
            "goal_views": {
                view: matrix[device_index]
                for view, matrix in sources["goal_views"].items()
            },
            # Tool prototypes are global [A, D], matching GoalConditionedValueGeometry.
            "tool_views": dict(sources["tool_views"]),
            "structured_state": sources["structured_state"][device_index],
            "structured_goal": sources["structured_goal"][device_index],
            "structured_tools": sources["structured_tools"],
            "successor_state_views": successor_state_views,
            "successor_context_views": successor_context_views,
            "structured_successor_state": structured_successor,
        }
        batch.update(
            {name: tensor[device_index] for name, tensor in target_tensors.items()}
        )
        # Explicit aliases make STOP handling hard to accidentally conflate with
        # the A real-tool columns.
        batch["stop_target"] = batch["completion_target"]
        batch["stop_mask"] = batch["completion_mask"]
        return batch


__all__ = ["SearchFeatureStore"]
