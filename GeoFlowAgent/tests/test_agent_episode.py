from __future__ import annotations

from typing import Any

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.evaluation.agent import PlanOutput, run_episode


class _EmptyRuntimeStore:
    runtime_context_index: dict[str, int] = {}
    examples: list[dict[str, Any]] = []


class _StopEmittingPlanner:
    """A policy that emits STOP itself; the evaluator never chooses the action."""

    store = _EmptyRuntimeStore()

    def plan(self, *args: object, **kwargs: object) -> PlanOutput:
        return PlanOutput([STOP_TOOL_ID], nfe=1, latency_seconds=0.0)


def _episode(*, goal_satisfied: bool) -> dict[str, Any]:
    task = {
        "task_id": "novel-terminal-context",
        "query": "Finish only when the report is ready.",
        "initial_state": {"report_ready": goal_satisfied},
        "goal": {"report_ready": True},
        "available_tools": ["prepare_report"],
    }
    tools = [
        {
            "tool_id": "prepare_report",
            "name": "Prepare report",
            "description": "Mark the report ready.",
            "preconditions": [{"field": "report_ready", "op": "neq", "value": True}],
            "effects": [{"field": "report_ready", "op": "set", "value": True}],
        }
    ]
    return run_episode(
        _StopEmittingPlanner(),  # type: ignore[arg-type]
        task,
        tools,
        [],
        commit_horizon=1,
        feedback=True,
        max_steps=2,
        perturbations=[],
        counterfactual_policy="observed_only",
    )


def test_observed_only_learned_stop_is_zero_regret_at_novel_goal_state() -> None:
    result = _episode(goal_satisfied=True)

    assert result["success"] is True
    assert result["correct_stop"] is True
    assert result["zero_regret_action_rate"] == 1.0
    assert result["mean_regret"] == 0.0
    assert result["regret_label_coverage"] == 1.0
    assert result["trace"][0]["tool_id"] == STOP_TOOL_ID
    assert result["trace"][0]["valid_set"] == [STOP_TOOL_ID]
    assert result["trace"][0]["regret"] == 0.0
    assert result["trace"][0]["zero_regret"] is True


def test_observed_only_does_not_oracle_validate_premature_stop_on_cache_miss() -> None:
    result = _episode(goal_satisfied=False)

    assert result["success"] is False
    assert result["correct_stop"] is False
    assert result["zero_regret_action_rate"] is None
    assert result["mean_regret"] is None
    assert result["regret_label_coverage"] == 0.0
    assert result["trace"][0]["valid_set"] == []
    assert result["trace"][0]["regret"] is None
    assert result["trace"][0]["zero_regret"] is None
