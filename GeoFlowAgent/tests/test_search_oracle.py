from __future__ import annotations

import json

import pytest

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine
from geoflowagent.search import EdgeCostConfig, SearchLimits, WeightedSearchOracle


def _symbolic_tool(
    tool_id: str,
    *,
    preconditions: list[dict] | None = None,
    effects: list[dict] | None = None,
    search_cost: float | dict[str, float] | None = None,
) -> dict:
    tool = {
        "tool_id": tool_id,
        "name": tool_id.replace("_", " ").title(),
        "description": f"Deterministic fixture tool {tool_id}.",
        "preconditions": preconditions or [],
        "effects": effects or [],
    }
    if search_cost is not None:
        tool["search_cost"] = search_cost
    return tool


def _action(result, state_id: str, tool_id: str):
    state_value = result.value_by_id()[state_id]
    return next(item for item in state_value.actions if item.tool_id == tool_id)


def test_weighted_reverse_dijkstra_produces_q_regret_and_multiple_paths() -> None:
    tools = [
        _symbolic_tool(
            "collect_a",
            preconditions=[{"field": "a", "op": "missing"}],
            effects=[{"field": "a", "op": "set", "value": True}],
        ),
        _symbolic_tool(
            "collect_b",
            preconditions=[{"field": "b", "op": "missing"}],
            effects=[{"field": "b", "op": "set", "value": True}],
        ),
        _symbolic_tool(
            "detour",
            preconditions=[{"field": "detour", "op": "missing"}],
            effects=[{"field": "detour", "op": "set", "value": True}],
            search_cost=2.0,
        ),
        _symbolic_tool(
            "finish",
            preconditions=[
                {"field": "a", "op": "eq", "value": True},
                {"field": "b", "op": "eq", "value": True},
                {"field": "complete", "op": "missing"},
            ],
            effects=[{"field": "complete", "op": "set", "value": True}],
        ),
    ]
    result = WeightedSearchOracle(
        ContractEngine(tools),
        limits=SearchLimits(max_depth=4, max_states=100, top_k_paths=8),
    ).search({}, {"complete": True}, [tool["tool_id"] for tool in tools])

    root = result.initial_value
    assert root.value_known is True
    assert root.v_star == pytest.approx(3.0)
    assert root.best_known_v == pytest.approx(3.0)
    assert root.optimal_actions == ("collect_a", "collect_b")

    collect_a = _action(result, result.initial_state_id, "collect_a")
    collect_b = _action(result, result.initial_state_id, "collect_b")
    detour = _action(result, result.initial_state_id, "detour")
    stop = _action(result, result.initial_state_id, STOP_TOOL_ID)
    assert collect_a.q_star == pytest.approx(3.0)
    assert collect_b.q_star == pytest.approx(3.0)
    assert collect_a.regret == pytest.approx(0.0)
    assert collect_b.regret == pytest.approx(0.0)
    assert detour.q_star == pytest.approx(6.0)
    assert detour.regret == pytest.approx(3.0)
    assert detour.optimal is False
    assert stop.reason == "premature_stop"
    assert stop.reachability_known is True
    assert stop.reachable is False
    assert stop.known_mask is False

    optimal_paths = [path for path in result.paths if path.optimal_within_known_graph]
    assert {path.actions for path in optimal_paths} == {
        ("collect_a", "collect_b", "finish", STOP_TOOL_ID),
        ("collect_b", "collect_a", "finish", STOP_TOOL_ID),
    }
    assert all(path.certified_optimal for path in optimal_paths)
    assert result.diagnostics.graph_complete is True

    terminal_id = optimal_paths[0].state_ids[-1]
    terminal = result.value_by_id()[terminal_id]
    assert terminal.v_star == 0.0
    assert terminal.optimal_actions == (STOP_TOOL_ID,)
    terminal_stop = _action(result, terminal_id, STOP_TOOL_ID)
    assert terminal_stop.q_star == 0.0
    assert terminal_stop.regret == 0.0


def test_unknown_snapshot_is_masked_not_mislabeled_unreachable() -> None:
    snapshot_tool = {
        "tool_id": "snapshot_lookup",
        "name": "Snapshot lookup",
        "description": "Requires an exact offline response.",
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": ["case_id"],
            "properties": {"case_id": {"type": "string"}},
        },
        "argument_bindings": {"case_id": "case_id"},
        "output_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
        "output_bindings": {"answer": "answer"},
        "preconditions": [{"field": "case_id", "op": "nonempty"}],
        "effects": [],
    }
    fallback = _symbolic_tool(
        "safe_fallback",
        preconditions=[{"field": "done", "op": "missing"}],
        effects=[{"field": "done", "op": "set", "value": True}],
    )
    gated = _symbolic_tool(
        "requires_ready",
        preconditions=[{"field": "ready", "op": "eq", "value": True}],
        effects=[{"field": "done", "op": "set", "value": True}],
    )
    result = WeightedSearchOracle(
        ContractEngine([snapshot_tool, fallback, gated], snapshots=[]),
        limits=SearchLimits(max_depth=2, top_k_paths=2),
    ).search(
        {"case_id": "not-cached"},
        {"done": True},
        ["snapshot_lookup", "safe_fallback", "requires_ready"],
    )

    root = result.initial_value
    # A real path is known, but an unobserved snapshot might be even better. The
    # conservative oracle therefore exposes an upper bound without claiming V*.
    assert root.best_known_v == pytest.approx(1.0)
    assert root.v_star is None
    assert root.value_known is False
    assert root.reachable is True

    unknown = _action(result, result.initial_state_id, "snapshot_lookup")
    assert unknown.contract_applicable is True
    assert unknown.executable is False
    assert unknown.transition_known is False
    assert unknown.known_mask is False
    assert unknown.reachable is None
    assert unknown.reason == "unobserved_snapshot"
    assert all(edge.tool_id != "snapshot_lookup" for edge in result.edges)

    invalid = _action(result, result.initial_state_id, "requires_ready")
    assert invalid.contract_applicable is False
    assert invalid.reachability_known is True
    assert invalid.reachable is False
    assert invalid.reason == "contract_inapplicable"

    fallback_label = _action(result, result.initial_state_id, "safe_fallback")
    assert fallback_label.known_mask is True
    assert fallback_label.q_star == pytest.approx(1.0)
    # Regret remains unknown because the root optimum is not certified.
    assert fallback_label.regret is None
    assert result.paths[0].certified_optimal is False
    assert result.diagnostics.unresolved_actions == 1
    assert result.diagnostics.graph_complete is False


def test_failure_redundancy_metadata_and_callback_costs_are_nonnegative() -> None:
    tool = {
        "tool_id": "empty_lookup",
        "name": "Empty lookup",
        "description": "Pinned empty response.",
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": ["case_id"],
            "properties": {"case_id": {"type": "string"}},
        },
        "argument_bindings": {"case_id": "case_id"},
        "output_schema": {"type": "object"},
        "output_bindings": {"unused": "unused"},
        "preconditions": [{"field": "case_id", "op": "nonempty"}],
        "effects": [],
        "search_cost": {"latency": 2.0},
    }
    snapshot = {
        "snapshot_id": "empty-case",
        "tool_id": "empty_lookup",
        "arguments": {"case_id": "case-1"},
        "output": {},
        "status": "empty",
        "source_revision": "fixture-v1",
    }
    config = EdgeCostConfig(
        call_cost=1.0,
        failure_cost=2.0,
        redundancy_cost=3.0,
        metadata_weights={"latency": 0.5},
        status_costs={"empty": 4.0},
    )
    result = WeightedSearchOracle(
        ContractEngine([tool], [snapshot]),
        cost_config=config,
        limits=SearchLimits(max_depth=1),
    ).search({"case_id": "case-1"}, {"done": True}, ["empty_lookup"])
    edge = result.edges[0]
    assert edge.redundant is True
    assert edge.cost_breakdown.call == 1.0
    assert edge.cost_breakdown.failure == 2.0
    assert edge.cost_breakdown.redundancy == 3.0
    assert edge.cost_breakdown.metadata == 1.0
    assert edge.cost_breakdown.status == 4.0
    assert edge.cost == 11.0

    callback_result = WeightedSearchOracle(
        ContractEngine([tool], [snapshot]),
        cost_config=config,
        limits=SearchLimits(max_depth=1),
        edge_cost_fn=lambda item: item.default_cost / 11.0,
    ).search({"case_id": "case-1"}, {"done": True}, ["empty_lookup"])
    assert callback_result.edges[0].cost == pytest.approx(1.0)
    assert callback_result.edges[0].cost_breakdown.callback == pytest.approx(1.0)

    with pytest.raises(ValueError, match="nonnegative"):
        WeightedSearchOracle(
            ContractEngine([tool], [snapshot]),
            limits=SearchLimits(max_depth=1),
            edge_cost_fn=lambda _: -1.0,
        ).search({"case_id": "case-1"}, {"done": True}, ["empty_lookup"])


def test_context_is_part_of_stable_node_identity_and_history_is_reconstructable() -> None:
    tools = [
        _symbolic_tool(
            "left_route",
            preconditions=[{"field": "done", "op": "missing"}],
            effects=[{"field": "done", "op": "set", "value": True}],
        ),
        _symbolic_tool(
            "right_route",
            preconditions=[{"field": "done", "op": "missing"}],
            effects=[{"field": "done", "op": "set", "value": True}],
        ),
    ]

    def update_context(state, context, tool_id, transition):
        del state, transition
        return {**context, "route": tool_id}

    oracle = WeightedSearchOracle(
        ContractEngine(tools),
        limits=SearchLimits(max_depth=1, top_k_paths=4),
        context_update_fn=update_context,
    )
    first = oracle.search({}, {"done": True}, ["left_route", "right_route"], context={"run": 1})
    second = oracle.search({}, {"done": True}, ["right_route", "left_route"], context={"run": 1})

    assert len(first.states) == 3
    assert [state.state_id for state in first.states] == [
        state.state_id for state in second.states
    ]
    terminal_states = [state for state in first.states if state.goal_satisfied]
    assert {state.context["route"] for state in terminal_states} == {
        "left_route",
        "right_route",
    }
    assert {state.action_history for state in terminal_states} == {
        ("left_route",),
        ("right_route",),
    }
    assert all(len(state.observation_history) == 1 for state in terminal_states)
    assert all(state.parent_state_id == first.initial_state_id for state in terminal_states)
    assert {path.actions for path in first.paths} == {
        ("left_route", STOP_TOOL_ID),
        ("right_route", STOP_TOOL_ID),
    }

    # Persisted oracle output contains no Infinity/NaN and round-trips as JSON.
    payload = json.loads(first.to_json())
    assert payload["initial_state_id"] == first.initial_state_id
    assert payload["states"][0]["state_id"].startswith("s_")


def test_state_limit_turns_omitted_successor_into_unknown_mask() -> None:
    tool = _symbolic_tool(
        "advance",
        preconditions=[{"field": "done", "op": "missing"}],
        effects=[{"field": "done", "op": "set", "value": True}],
    )
    result = WeightedSearchOracle(
        ContractEngine([tool]),
        limits=SearchLimits(max_depth=2, max_states=1),
    ).search({}, {"done": True}, ["advance"])

    action = _action(result, result.initial_state_id, "advance")
    assert action.contract_applicable is True
    assert action.executable is True
    assert action.transition_known is False
    assert action.known_mask is False
    assert action.reachable is None
    assert action.reason == "state_limit"
    assert result.initial_value.reachable is None
    assert result.diagnostics.hit_state_limit is True
    assert not result.edges
