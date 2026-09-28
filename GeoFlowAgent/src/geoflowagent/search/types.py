from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses/tuples into strict JSON-compatible values."""

    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Search artifacts cannot serialize NaN or infinity")
    return value


class JsonSerializable:
    """Small mixin used by persisted oracle artifacts."""

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(self)
        if not isinstance(value, dict):  # pragma: no cover - all users are dataclasses
            raise TypeError("Expected a dataclass object")
        return value

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
            indent=indent,
            allow_nan=False,
        )


def _validate_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


@dataclass(frozen=True)
class SearchLimits(JsonSerializable):
    """Hard bounds for one deterministic search artifact."""

    max_depth: int = 12
    max_states: int = 10_000
    top_k_paths: int = 8
    near_optimal_slack: float = 0.0
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        if self.max_states < 1:
            raise ValueError("max_states must be at least one")
        if self.top_k_paths < 0:
            raise ValueError("top_k_paths must be nonnegative")
        _validate_nonnegative(self.near_optimal_slack, "near_optimal_slack")
        _validate_nonnegative(self.tolerance, "tolerance")


@dataclass(frozen=True)
class EdgeCostConfig(JsonSerializable):
    """Additive nonnegative edge-cost components.

    ``metadata_field`` is read from each tool. A numeric value is added directly.
    For a mapping, only keys present in ``metadata_weights`` are used.
    """

    call_cost: float = 1.0
    failure_cost: float = 1.0
    redundancy_cost: float = 0.25
    metadata_field: str = "search_cost"
    metadata_weights: dict[str, float] = field(default_factory=dict)
    status_costs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonnegative(self.call_cost, "call_cost")
        _validate_nonnegative(self.failure_cost, "failure_cost")
        _validate_nonnegative(self.redundancy_cost, "redundancy_cost")
        if not isinstance(self.metadata_field, str) or not self.metadata_field:
            raise ValueError("metadata_field must be a non-empty string")
        for key, value in self.metadata_weights.items():
            _validate_nonnegative(value, f"metadata_weights[{key!r}]")
        for key, value in self.status_costs.items():
            _validate_nonnegative(value, f"status_costs[{key!r}]")


@dataclass(frozen=True)
class CostBreakdown(JsonSerializable):
    call: float
    failure: float
    redundancy: float
    metadata: float
    status: float
    callback: float | None
    total: float


@dataclass(frozen=True)
class SearchState(JsonSerializable):
    """One canonical bounded-horizon Markov node.

    Depth is part of the canonical key because remaining search horizon changes the
    value of an otherwise identical typed state. ``history`` is only a stable,
    minimum-cost representative prefix; callers that need history-dependent dynamics
    must also carry the required summary in ``context``.
    """

    state_id: str
    canonical_key: str
    state: dict[str, Any]
    context: dict[str, Any]
    depth: int
    goal_satisfied: bool
    parent_state_id: str | None
    parent_edge_id: str | None
    action_history: tuple[str, ...]
    observation_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SearchEdge(JsonSerializable):
    edge_id: str
    source_state_id: str
    target_state_id: str
    tool_id: str
    cost: float
    cost_breakdown: CostBreakdown
    observation: dict[str, Any]
    redundant: bool


@dataclass(frozen=True)
class ActionValue(JsonSerializable):
    """Oracle label for one candidate action at one state.

    ``known_mask`` is deliberately conservative: it is true only when a finite,
    exact Q* target is certified by the explored graph. ``reachable`` has its own
    mask because a fully explored dead end is known unreachable without a numeric Q.
    """

    tool_id: str
    contract_applicable: bool
    executable: bool
    transition_known: bool
    known_mask: bool
    reachability_known: bool
    reachable: bool | None
    q_star: float | None
    best_known_q: float | None
    regret: float | None
    optimal: bool | None
    successor_state_id: str | None
    reason: str


@dataclass(frozen=True)
class StateValue(JsonSerializable):
    state_id: str
    cost_from_start: float | None
    v_star: float | None
    best_known_v: float | None
    value_known: bool
    reachability_known: bool
    reachable: bool | None
    optimal_actions: tuple[str, ...]
    actions: tuple[ActionValue, ...]


@dataclass(frozen=True)
class SearchPath(JsonSerializable):
    rank: int
    state_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    actions: tuple[str, ...]
    cost: float
    excess_cost: float
    optimal_within_known_graph: bool
    certified_optimal: bool


@dataclass(frozen=True)
class SearchDiagnostics(JsonSerializable):
    expanded_states: int
    discovered_states: int
    known_edges: int
    unresolved_actions: int
    hit_depth_limit: bool
    hit_state_limit: bool
    graph_complete: bool


@dataclass(frozen=True)
class SearchResult(JsonSerializable):
    schema_version: str
    initial_state_id: str
    goal: dict[str, Any]
    available_tools: tuple[str, ...]
    limits: SearchLimits
    cost_config: EdgeCostConfig
    states: tuple[SearchState, ...]
    edges: tuple[SearchEdge, ...]
    values: tuple[StateValue, ...]
    paths: tuple[SearchPath, ...]
    diagnostics: SearchDiagnostics

    def state_by_id(self) -> dict[str, SearchState]:
        return {state.state_id: state for state in self.states}

    def value_by_id(self) -> dict[str, StateValue]:
        return {value.state_id: value for value in self.values}

    @property
    def initial_value(self) -> StateValue:
        return self.value_by_id()[self.initial_state_id]
