from __future__ import annotations

from typing import Any

import torch

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.evaluation.search_agent import (
    ExactSearchPlanner,
    SearchValuePlanner,
    run_search_episode,
)
from geoflowagent.search import EdgeCostConfig


def _tool(tool_id: str, *, cost: float, field: str = "done") -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "name": tool_id.title(),
        "description": f"Deterministic {tool_id} action.",
        "search_cost": cost,
        "preconditions": [{"field": field, "op": "missing"}],
        "effects": [{"field": field, "op": "set", "value": True}],
    }


class _ConstantCompletionModel(torch.nn.Module):
    def forward(self, *args, candidate_mask: torch.Tensor, **kwargs):
        del args, kwargs
        shape = candidate_mask.shape
        return {
            "completion_logit": torch.full((shape[0],), 20.0),
            "q": torch.zeros(shape),
            "policy_logits": torch.zeros(shape),
            "action_reachability_logit": torch.full(shape, 20.0),
        }


class _ReachabilityModel(torch.nn.Module):
    def __init__(self, *, all_unreachable: bool = False) -> None:
        super().__init__()
        self.all_unreachable = all_unreachable

    def forward(
        self,
        *args,
        candidate_mask: torch.Tensor,
        mask_unreachable: bool,
        reachability_threshold: float,
        **kwargs,
    ):
        del args, kwargs, reachability_threshold
        q = torch.tensor([[0.0, 1.0]], device=candidate_mask.device)
        reachability = torch.tensor(
            [[-20.0, -20.0 if self.all_unreachable else 20.0]],
            device=candidate_mask.device,
        )
        allowed = candidate_mask & (
            (reachability.sigmoid() >= 0.5) if mask_unreachable else True
        )
        return {
            "completion_logit": torch.full((1,), -20.0, device=candidate_mask.device),
            "q": q,
            "masked_q": q.masked_fill(~allowed, torch.inf),
            "policy_logits": (-q + torch.nn.functional.logsigmoid(reachability)).masked_fill(
                ~allowed, -torch.inf
            ),
            "action_reachability_logit": reachability,
        }


class _FakeStore:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools
        self.tool_ids = [row["tool_id"] for row in tools]
        self.tool_index = {tool_id: index for index, tool_id in enumerate(self.tool_ids)}
        self.examples = [{"state": {}}]
        self.tool_views = {"view": torch.ones((len(tools), 2))}
        self.structured_tools = torch.zeros((len(tools), 2))

    def runtime_features(self, task_id, state, device="cpu"):
        del task_id, state
        target = torch.device(device)
        return (
            {"view": torch.ones((1, 2), device=target)},
            {"view": torch.ones((1, 2), device=target)},
            torch.zeros((1, 2), device=target),
            torch.zeros((1, 2), device=target),
        )

    def state_index(self, task_id, state):
        del task_id, state
        return None


def _planner(
    tools: list[dict[str, Any]],
    *,
    policy: str = "learned_policy",
    stop_contract_guard: bool = True,
    retry_limit_per_tool: int = 1,
) -> tuple[SearchValuePlanner, _FakeStore]:
    store = _FakeStore(tools)
    planner = SearchValuePlanner(
        _ConstantCompletionModel(),
        store,  # type: ignore[arg-type]
        object(),  # cache-hit tests never invoke the runtime encoder
        torch.device("cpu"),
        search_depth=4,
        completion_threshold=0.5,
        reachability_threshold=0.5,
        retry_limit_per_tool=retry_limit_per_tool,
        policy=policy,
        stop_contract_guard=stop_contract_guard,
        runtime_embedding_policy="cache_only",
    )
    return planner, store


def test_public_goal_contract_blocks_high_confidence_premature_stop() -> None:
    tools = [_tool("finish", cost=1.0)]
    task = {
        "task_id": "stop-guard",
        "query": "Finish the case.",
        "initial_state": {},
        "goal": {"done": True},
        "available_tools": ["finish"],
    }
    guarded, _ = _planner(tools, stop_contract_guard=True)
    unguarded, _ = _planner(tools, stop_contract_guard=False)

    assert guarded.decide(task, {}, [], []).tool_id == "finish"
    assert unguarded.decide(task, {}, [], []).tool_id == STOP_TOOL_ID
    assert guarded.decide(task, {"done": True}, [], []).tool_id == STOP_TOOL_ID


def test_hard_reachability_guard_and_all_masked_fallback_are_explicit() -> None:
    tools = [_tool("low_q_dead_end", cost=0.0), _tool("reachable", cost=1.0)]
    store = _FakeStore(tools)
    task = {
        "task_id": "reachability-guard",
        "query": "Choose a viable action.",
        "initial_state": {},
        "goal": {"done": True},
        "available_tools": [row["tool_id"] for row in tools],
    }

    def make(model: torch.nn.Module, hard_guard: bool) -> SearchValuePlanner:
        return SearchValuePlanner(
            model,
            store,  # type: ignore[arg-type]
            object(),
            torch.device("cpu"),
            search_depth=4,
            completion_threshold=0.5,
            reachability_threshold=0.5,
            retry_limit_per_tool=1,
            policy="learned_q",
            hard_reachability_guard=hard_guard,
            runtime_embedding_policy="cache_only",
        )

    guarded = make(_ReachabilityModel(), True).decide(task, {}, [], [])
    soft = make(_ReachabilityModel(), False).decide(task, {}, [], [])
    all_masked = make(_ReachabilityModel(all_unreachable=True), True).decide(
        task, {}, [], []
    )

    assert guarded.tool_id == "reachable"
    assert guarded.reachability_guard_fallback is False
    assert soft.tool_id == "low_q_dead_end"
    assert all_masked.tool_id == "low_q_dead_end"
    assert all_masked.reachability_guard_fallback is True


def test_failed_tool_is_masked_and_episode_recovers_via_fallback() -> None:
    tools = [
        _tool("primary", cost=0.0),
        _tool("fallback", cost=1.0),
    ]
    task = {
        "task_id": "retry-recovery",
        "query": "Complete despite a transient primary failure.",
        "initial_state": {},
        "goal": {"done": True},
        "available_tools": ["primary", "fallback"],
    }
    planner, store = _planner(tools, policy="static_cost", retry_limit_per_tool=1)

    result = run_search_episode(
        planner,
        store,  # type: ignore[arg-type]
        task,
        tools,
        [],
        {"done": True},
        [],
        max_steps=4,
        search_depth=4,
        perturbations=[
            {
                "name": "primary-temporary-failure",
                "task_id": task["task_id"],
                "tool_id": "primary",
                "kind": "failure",
                "once": True,
            }
        ],
    )

    assert [row["tool_id"] for row in result["trace"]] == [
        "primary",
        "fallback",
        STOP_TOOL_ID,
    ]
    assert result["trace"][0]["execution_success"] is False
    assert result["success"] is True
    assert result["perturbation_assigned"] is True
    assert result["perturbation_applied"] is True
    assert result["recovered_after_perturbation"] is True
    assert result["invalid_calls"] == 0
    assert result["premature_stops"] == 0


def test_exact_oracle_reuses_certified_clean_search_row() -> None:
    tools = [_tool("finish", cost=1.0)]

    class Store:
        examples = [{"state": {}, "optimal_actions": ["finish"]}]

        @staticmethod
        def state_index(task_id, state):
            assert task_id == "cached-oracle"
            assert state == {}
            return 0

    planner = ExactSearchPlanner(
        tools,
        [],
        {"done": True},
        max_steps=4,
        retry_limit_per_tool=1,
        costs=EdgeCostConfig(),
        store=Store(),  # type: ignore[arg-type]
        search_depth=4,
    )
    decision = planner.decide(
        {
            "task_id": "cached-oracle",
            "initial_state": {},
            "goal": {"done": True},
            "available_tools": ["finish"],
        },
        {},
        [],
        [],
    )

    assert decision.tool_id == "finish"
    assert decision.embedding_source == "cache"
