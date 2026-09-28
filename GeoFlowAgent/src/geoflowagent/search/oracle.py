from __future__ import annotations

import copy
import hashlib
import heapq
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, Transition
from geoflowagent.utils.io import canonical_json

from .types import (
    ActionValue,
    CostBreakdown,
    EdgeCostConfig,
    SearchDiagnostics,
    SearchEdge,
    SearchLimits,
    SearchPath,
    SearchResult,
    SearchState,
    StateValue,
)

SCHEMA_VERSION = "geoflowagent.search-oracle.v1"


@dataclass(frozen=True)
class EdgeCostInput:
    state: dict[str, Any]
    context: dict[str, Any]
    tool_id: str
    tool: dict[str, Any]
    transition: Transition
    next_context: dict[str, Any]
    redundant: bool
    default_cost: float


EdgeCostFn = Callable[[EdgeCostInput], float]
ContextUpdateFn = Callable[
    [dict[str, Any], dict[str, Any], str, Transition], dict[str, Any]
]


@dataclass
class _MutableNode:
    state_id: str
    canonical_key: str
    state: dict[str, Any]
    context: dict[str, Any]
    depth: int
    goal_satisfied: bool


@dataclass(frozen=True)
class _UnresolvedAction:
    tool_id: str
    reason: str
    contract_applicable: bool
    executable: bool


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return result


def _content_id(prefix: str, canonical_key: str) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_key.encode('utf-8')).hexdigest()[:24]}"


class WeightedSearchOracle:
    """Build an exact known transition graph and derive conservative Q*/regret labels.

    Unknown snapshot calls and transitions excluded by a resource bound are retained
    as unknown masks. They are never silently converted into negative examples.
    """

    def __init__(
        self,
        engine: ContractEngine,
        *,
        cost_config: EdgeCostConfig | None = None,
        limits: SearchLimits | None = None,
        edge_cost_fn: EdgeCostFn | None = None,
        context_update_fn: ContextUpdateFn | None = None,
    ) -> None:
        self.engine = engine
        self.cost_config = cost_config or EdgeCostConfig()
        self.limits = limits or SearchLimits()
        self.edge_cost_fn = edge_cost_fn
        self.context_update_fn = context_update_fn

    @staticmethod
    def canonical_markov_key(
        state: dict[str, Any], context: dict[str, Any] | None, depth: int
    ) -> str:
        """Canonical identity for value labels under a bounded horizon."""

        if depth < 0:
            raise ValueError("depth must be nonnegative")
        return canonical_json(
            {"context": copy.deepcopy(context or {}), "depth": depth, "state": state}
        )

    def _next_context(
        self,
        state: dict[str, Any],
        context: dict[str, Any],
        tool_id: str,
        transition: Transition,
    ) -> dict[str, Any]:
        if self.context_update_fn is None:
            return copy.deepcopy(context)
        updated = self.context_update_fn(
            copy.deepcopy(state), copy.deepcopy(context), tool_id, copy.deepcopy(transition)
        )
        if not isinstance(updated, dict):
            raise TypeError("context_update_fn must return a dictionary")
        # Fail early if context is not stable JSON data.
        canonical_json(updated)
        return copy.deepcopy(updated)

    def _edge_cost(
        self,
        state: dict[str, Any],
        context: dict[str, Any],
        tool_id: str,
        transition: Transition,
        next_context: dict[str, Any],
    ) -> CostBreakdown:
        config = self.cost_config
        redundant = canonical_json(state) == canonical_json(transition.state)
        call = float(config.call_cost)
        failure = 0.0 if transition.observation.get("ok") else float(config.failure_cost)
        redundancy = float(config.redundancy_cost) if redundant else 0.0
        status_name = str(transition.observation.get("status", ""))
        status = float(config.status_costs.get(status_name, 0.0))

        tool = self.engine.tools[tool_id]
        raw_metadata = tool.get(config.metadata_field, 0.0)
        metadata = 0.0
        if isinstance(raw_metadata, bool):
            raise TypeError(f"Tool {tool_id!r} metadata cost cannot be boolean")
        if isinstance(raw_metadata, (int, float)):
            metadata = _finite_nonnegative(raw_metadata, f"{tool_id}.metadata")
        elif isinstance(raw_metadata, Mapping):
            for name, weight in config.metadata_weights.items():
                raw_value = raw_metadata.get(name, 0.0)
                value = _finite_nonnegative(raw_value, f"{tool_id}.metadata[{name!r}]")
                metadata += float(weight) * value
        elif raw_metadata is not None:
            raise TypeError(
                f"Tool {tool_id!r}.{config.metadata_field} must be numeric or a mapping"
            )

        default_cost = call + failure + redundancy + metadata + status
        callback_value: float | None = None
        total = default_cost
        if self.edge_cost_fn is not None:
            callback_value = _finite_nonnegative(
                self.edge_cost_fn(
                    EdgeCostInput(
                        state=copy.deepcopy(state),
                        context=copy.deepcopy(context),
                        tool_id=tool_id,
                        tool=copy.deepcopy(tool),
                        transition=copy.deepcopy(transition),
                        next_context=copy.deepcopy(next_context),
                        redundant=redundant,
                        default_cost=default_cost,
                    )
                ),
                "edge_cost_fn result",
            )
            # A callback defines the full task-specific edge cost, rather than being
            # accidentally double-added to the default components.
            total = callback_value
        _finite_nonnegative(total, "edge total")
        return CostBreakdown(call, failure, redundancy, metadata, status, callback_value, total)

    def search(
        self,
        initial_state: dict[str, Any],
        goal: dict[str, Any],
        available_tools: Iterable[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        available = tuple(sorted(set(str(item) for item in available_tools)))
        if STOP_TOOL_ID in available:
            available = tuple(item for item in available if item != STOP_TOOL_ID)
        root_context = copy.deepcopy(context or {})
        root_key = self.canonical_markov_key(initial_state, root_context, 0)
        root_id = _content_id("s", root_key)
        root = _MutableNode(
            root_id,
            root_key,
            copy.deepcopy(initial_state),
            root_context,
            0,
            self.engine.goal_satisfied(initial_state, goal),
        )

        nodes: dict[str, _MutableNode] = {root_id: root}
        key_to_id = {root_key: root_id}
        edges: dict[str, SearchEdge] = {}
        outgoing: dict[str, list[str]] = defaultdict(list)
        incoming: dict[str, list[str]] = defaultdict(list)
        unresolved: dict[str, list[_UnresolvedAction]] = defaultdict(list)
        queue: deque[str] = deque([root_id])
        expanded: set[str] = set()
        hit_depth_limit = False
        hit_state_limit = False

        while queue:
            state_id = queue.popleft()
            node = nodes[state_id]
            if state_id in expanded or node.goal_satisfied:
                continue
            expanded.add(state_id)

            for tool_id in available:
                if tool_id not in self.engine.tools:
                    unresolved[state_id].append(
                        _UnresolvedAction(tool_id, "unknown_tool", False, False)
                    )
                    continue
                applicable = self.engine.applicable(node.state, tool_id)
                if not applicable:
                    continue
                executable = self.engine.executable(node.state, tool_id)
                if not executable:
                    unresolved[state_id].append(
                        _UnresolvedAction(tool_id, "unobserved_snapshot", True, False)
                    )
                    continue
                if node.depth >= self.limits.max_depth:
                    hit_depth_limit = True
                    unresolved[state_id].append(
                        _UnresolvedAction(tool_id, "depth_limit", True, True)
                    )
                    continue

                transition = self.engine.execute(node.state, tool_id)
                next_context = self._next_context(
                    node.state, node.context, tool_id, transition
                )
                target_key = self.canonical_markov_key(
                    transition.state, next_context, node.depth + 1
                )
                target_id = key_to_id.get(target_key)
                if target_id is None:
                    if len(nodes) >= self.limits.max_states:
                        hit_state_limit = True
                        unresolved[state_id].append(
                            _UnresolvedAction(tool_id, "state_limit", True, True)
                        )
                        continue
                    target_id = _content_id("s", target_key)
                    if target_id in nodes and nodes[target_id].canonical_key != target_key:
                        raise RuntimeError("Canonical state ID hash collision")
                    target = _MutableNode(
                        target_id,
                        target_key,
                        copy.deepcopy(transition.state),
                        next_context,
                        node.depth + 1,
                        self.engine.goal_satisfied(transition.state, goal),
                    )
                    nodes[target_id] = target
                    key_to_id[target_key] = target_id
                    queue.append(target_id)

                breakdown = self._edge_cost(
                    node.state, node.context, tool_id, transition, next_context
                )
                edge_key = canonical_json(
                    {"action": tool_id, "source": state_id, "target": target_id}
                )
                edge_id = _content_id("e", edge_key)
                edge = SearchEdge(
                    edge_id=edge_id,
                    source_state_id=state_id,
                    target_state_id=target_id,
                    tool_id=tool_id,
                    cost=breakdown.total,
                    cost_breakdown=breakdown,
                    observation=copy.deepcopy(transition.observation),
                    redundant=canonical_json(node.state) == canonical_json(transition.state),
                )
                if edge_id in edges and edges[edge_id] != edge:
                    raise RuntimeError("Search edge ID hash collision")
                edges[edge_id] = edge
                outgoing[state_id].append(edge_id)
                incoming[target_id].append(edge_id)

        # Stable adjacency makes ties reproducible across Python/hash implementations.
        for edge_ids in outgoing.values():
            edge_ids.sort(key=lambda item: (edges[item].tool_id, item))
        for edge_ids in incoming.values():
            edge_ids.sort(key=lambda item: (edges[item].source_state_id, edges[item].tool_id))

        forward, parent_edge = self._dijkstra_forward(root_id, nodes, edges, outgoing)
        reverse = self._dijkstra_reverse(nodes, edges, incoming)
        uncertain = self._uncertain_ancestors(nodes, edges, incoming, unresolved)

        search_states = self._materialize_states(nodes, edges, forward, parent_edge)
        values = self._make_values(
            nodes, edges, outgoing, unresolved, reverse, forward, uncertain, available
        )
        paths = self._top_paths(root_id, nodes, edges, outgoing, reverse, uncertain)

        graph_complete = not unresolved
        diagnostics = SearchDiagnostics(
            expanded_states=len(expanded),
            discovered_states=len(nodes),
            known_edges=len(edges),
            unresolved_actions=sum(len(items) for items in unresolved.values()),
            hit_depth_limit=hit_depth_limit,
            hit_state_limit=hit_state_limit,
            graph_complete=graph_complete,
        )
        return SearchResult(
            schema_version=SCHEMA_VERSION,
            initial_state_id=root_id,
            goal=copy.deepcopy(goal),
            available_tools=available,
            limits=self.limits,
            cost_config=self.cost_config,
            states=tuple(sorted(search_states, key=lambda item: (item.depth, item.state_id))),
            edges=tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
            values=tuple(
                sorted(values, key=lambda item: (nodes[item.state_id].depth, item.state_id))
            ),
            paths=tuple(paths),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _dijkstra_forward(
        root_id: str,
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        outgoing: Mapping[str, list[str]],
    ) -> tuple[dict[str, float], dict[str, str]]:
        distances = {root_id: 0.0}
        parents: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, root_id)]
        tolerance = 1e-12
        while heap:
            distance, state_id = heapq.heappop(heap)
            if distance > distances.get(state_id, math.inf) + tolerance:
                continue
            for edge_id in outgoing.get(state_id, []):
                edge = edges[edge_id]
                candidate = distance + edge.cost
                current = distances.get(edge.target_state_id, math.inf)
                current_parent = parents.get(edge.target_state_id)
                if candidate < current - tolerance or (
                    abs(candidate - current) <= tolerance
                    and (current_parent is None or edge_id < current_parent)
                ):
                    distances[edge.target_state_id] = candidate
                    parents[edge.target_state_id] = edge_id
                    heapq.heappush(heap, (candidate, edge.target_state_id))
        return distances, parents

    @staticmethod
    def _dijkstra_reverse(
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        incoming: Mapping[str, list[str]],
    ) -> dict[str, float]:
        distances: dict[str, float] = {}
        heap: list[tuple[float, str]] = []
        for state_id, node in nodes.items():
            if node.goal_satisfied:
                distances[state_id] = 0.0
                heapq.heappush(heap, (0.0, state_id))
        tolerance = 1e-12
        while heap:
            distance, state_id = heapq.heappop(heap)
            if distance > distances.get(state_id, math.inf) + tolerance:
                continue
            for edge_id in incoming.get(state_id, []):
                edge = edges[edge_id]
                candidate = distance + edge.cost
                current = distances.get(edge.source_state_id, math.inf)
                if candidate < current - tolerance:
                    distances[edge.source_state_id] = candidate
                    heapq.heappush(heap, (candidate, edge.source_state_id))
        return distances

    @staticmethod
    def _uncertain_ancestors(
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        incoming: Mapping[str, list[str]],
        unresolved: Mapping[str, list[_UnresolvedAction]],
    ) -> set[str]:
        uncertain = set(unresolved)
        queue: deque[str] = deque(sorted(uncertain))
        while queue:
            state_id = queue.popleft()
            for edge_id in incoming.get(state_id, []):
                predecessor = edges[edge_id].source_state_id
                if predecessor not in uncertain:
                    uncertain.add(predecessor)
                    queue.append(predecessor)
        return uncertain & set(nodes)

    @staticmethod
    def _materialize_states(
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        forward: Mapping[str, float],
        parent_edge: Mapping[str, str],
    ) -> list[SearchState]:
        histories: dict[str, tuple[tuple[str, ...], tuple[dict[str, Any], ...]]] = {}

        def history(state_id: str) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
            if state_id in histories:
                return histories[state_id]
            edge_id = parent_edge.get(state_id)
            if edge_id is None:
                result: tuple[tuple[str, ...], tuple[dict[str, Any], ...]] = ((), ())
            else:
                edge = edges[edge_id]
                actions, observations = history(edge.source_state_id)
                result = (
                    actions + (edge.tool_id,),
                    observations + (copy.deepcopy(edge.observation),),
                )
            histories[state_id] = result
            return result

        result: list[SearchState] = []
        for state_id, node in nodes.items():
            actions, observations = history(state_id)
            chosen_edge = parent_edge.get(state_id)
            parent_id = edges[chosen_edge].source_state_id if chosen_edge is not None else None
            result.append(
                SearchState(
                    state_id=state_id,
                    canonical_key=node.canonical_key,
                    state=copy.deepcopy(node.state),
                    context=copy.deepcopy(node.context),
                    depth=node.depth,
                    goal_satisfied=node.goal_satisfied,
                    parent_state_id=parent_id,
                    parent_edge_id=chosen_edge,
                    action_history=actions,
                    observation_history=observations,
                )
            )
        if len(forward) != len(nodes):  # pragma: no cover - all discovered nodes have a parent
            raise RuntimeError("Discovered an unreachable graph node")
        return result

    def _make_values(
        self,
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        outgoing: Mapping[str, list[str]],
        unresolved: Mapping[str, list[_UnresolvedAction]],
        reverse: Mapping[str, float],
        forward: Mapping[str, float],
        uncertain: set[str],
        available: tuple[str, ...],
    ) -> list[StateValue]:
        result: list[StateValue] = []
        tolerance = self.limits.tolerance
        for state_id, node in nodes.items():
            best_known_v = reverse.get(state_id)
            value_known = state_id not in uncertain
            v_star = best_known_v if value_known else None
            if best_known_v is not None:
                reachable: bool | None = True
                reachability_known = True
            elif value_known:
                reachable = False
                reachability_known = True
            else:
                reachable = None
                reachability_known = False

            edge_for_tool = {
                edges[edge_id].tool_id: edges[edge_id]
                for edge_id in outgoing.get(state_id, [])
            }
            unresolved_for_tool = {
                item.tool_id: item for item in unresolved.get(state_id, [])
            }
            action_values: list[ActionValue] = []

            for tool_id in available:
                if node.goal_satisfied:
                    action_values.append(
                        ActionValue(
                            tool_id,
                            False,
                            False,
                            False,
                            False,
                            True,
                            False,
                            None,
                            None,
                            None,
                            False,
                            None,
                            "terminal_state",
                        )
                    )
                    continue
                edge = edge_for_tool.get(tool_id)
                if edge is None:
                    item = unresolved_for_tool.get(tool_id)
                    if item is not None:
                        action_values.append(
                            ActionValue(
                                tool_id,
                                item.contract_applicable,
                                item.executable,
                                False,
                                False,
                                False,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                item.reason,
                            )
                        )
                    else:
                        known_tool = tool_id in self.engine.tools
                        action_values.append(
                            ActionValue(
                                tool_id,
                                False,
                                False,
                                False,
                                False,
                                True,
                                False,
                                None,
                                None,
                                None,
                                False,
                                None,
                                "contract_inapplicable" if known_tool else "unknown_tool",
                            )
                        )
                    continue

                target_id = edge.target_state_id
                target_best = reverse.get(target_id)
                best_known_q = edge.cost + target_best if target_best is not None else None
                q_known = target_id not in uncertain and target_best is not None
                q_star = best_known_q if q_known else None
                if target_best is not None:
                    action_reachable: bool | None = True
                    action_reachability_known = True
                elif target_id not in uncertain:
                    action_reachable = False
                    action_reachability_known = True
                else:
                    action_reachable = None
                    action_reachability_known = False
                regret = None
                optimal: bool | None = None
                if q_star is not None and v_star is not None:
                    regret = max(0.0, q_star - v_star)
                    optimal = regret <= tolerance
                elif action_reachable is False:
                    optimal = False
                action_values.append(
                    ActionValue(
                        tool_id,
                        True,
                        True,
                        True,
                        q_known,
                        action_reachability_known,
                        action_reachable,
                        q_star,
                        best_known_q,
                        regret,
                        optimal,
                        target_id,
                        "known" if q_known else "downstream_value_unknown",
                    )
                )

            if node.goal_satisfied:
                stop_value = ActionValue(
                    STOP_TOOL_ID,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    0.0,
                    0.0,
                    0.0,
                    True,
                    None,
                    "goal_satisfied",
                )
            else:
                stop_value = ActionValue(
                    STOP_TOOL_ID,
                    False,
                    False,
                    True,
                    False,
                    True,
                    False,
                    None,
                    None,
                    None,
                    False,
                    None,
                    "premature_stop",
                )
            action_values.append(stop_value)
            action_values.sort(key=lambda item: item.tool_id)
            optimal_actions = tuple(
                item.tool_id for item in action_values if item.optimal is True
            )
            result.append(
                StateValue(
                    state_id=state_id,
                    cost_from_start=forward.get(state_id),
                    v_star=v_star,
                    best_known_v=best_known_v,
                    value_known=value_known,
                    reachability_known=reachability_known,
                    reachable=reachable,
                    optimal_actions=optimal_actions,
                    actions=tuple(action_values),
                )
            )
        return result

    def _top_paths(
        self,
        root_id: str,
        nodes: Mapping[str, _MutableNode],
        edges: Mapping[str, SearchEdge],
        outgoing: Mapping[str, list[str]],
        reverse: Mapping[str, float],
        uncertain: set[str],
    ) -> list[SearchPath]:
        limit = self.limits.top_k_paths
        best = reverse.get(root_id)
        if limit == 0 or best is None:
            return []
        threshold = best + self.limits.near_optimal_slack
        counter = 0
        # estimated total, actual cost, stable tie counter, state, states, edges, actions
        heap: list[
            tuple[float, float, int, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]
        ] = [(best, 0.0, counter, root_id, (root_id,), (), ())]
        paths: list[SearchPath] = []
        seen_signatures: set[tuple[str, ...]] = set()
        tolerance = self.limits.tolerance
        while heap and len(paths) < limit:
            estimate, cost, _, state_id, state_ids, edge_ids, actions = heapq.heappop(heap)
            if estimate > threshold + tolerance:
                break
            if nodes[state_id].goal_satisfied:
                terminal_actions = actions + (STOP_TOOL_ID,)
                if terminal_actions in seen_signatures:
                    continue
                seen_signatures.add(terminal_actions)
                paths.append(
                    SearchPath(
                        rank=len(paths) + 1,
                        state_ids=state_ids,
                        edge_ids=edge_ids,
                        actions=terminal_actions,
                        cost=cost,
                        excess_cost=max(0.0, cost - best),
                        optimal_within_known_graph=abs(cost - best) <= tolerance,
                        certified_optimal=(
                            root_id not in uncertain and abs(cost - best) <= tolerance
                        ),
                    )
                )
                continue
            for edge_id in outgoing.get(state_id, []):
                edge = edges[edge_id]
                remaining = reverse.get(edge.target_state_id)
                if remaining is None:
                    continue
                next_cost = cost + edge.cost
                next_estimate = next_cost + remaining
                if next_estimate > threshold + tolerance:
                    continue
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        next_estimate,
                        next_cost,
                        counter,
                        edge.target_state_id,
                        state_ids + (edge.target_state_id,),
                        edge_ids + (edge_id,),
                        actions + (edge.tool_id,),
                    ),
                )
        return paths
