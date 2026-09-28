from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, set_path
from geoflowagent.utils.io import canonical_json


@dataclass
class StepResult:
    observation: dict[str, Any]
    state: dict[str, Any]
    done: bool
    goal_satisfied: bool
    valid_action: bool
    execution_success: bool
    redundant: bool
    perturbation_applied: bool


class SymbolicGenomicsEnvironment:
    """Deterministic CI environment; replace with snapshot-backed adapters for research."""

    def __init__(
        self,
        task: dict[str, Any],
        tools: list[dict[str, Any]],
        *,
        snapshots: list[dict[str, Any]] | None = None,
        verification_goal: dict[str, Any] | None = None,
        perturbations: list[dict[str, Any]] | None = None,
        max_steps: int = 16,
    ) -> None:
        self.task = task
        self.engine = ContractEngine(tools, snapshots)
        self.verification_goal = copy.deepcopy(verification_goal or task["goal"])
        self.state = copy.deepcopy(task["initial_state"])
        self.history: list[dict[str, Any]] = []
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.perturbations = list(perturbations or [])
        self.applied_perturbations: list[str] = []

    def _matching_perturbation(self, tool_id: str) -> dict[str, Any] | None:
        for item in self.perturbations:
            if item.get("task_id") not in {None, "*", self.task["task_id"]}:
                continue
            if item.get("step") not in {None, self.steps}:
                continue
            if item.get("tool_id") not in {None, "*", tool_id}:
                continue
            name = str(item.get("name", f"perturbation-{len(self.applied_perturbations)}"))
            if item.get("once", True) and name in self.applied_perturbations:
                continue
            return item
        return None

    def step(self, tool_id: str) -> StepResult:
        if self.done:
            raise RuntimeError("Cannot step a completed environment")
        goal_before = self.engine.goal_satisfied(self.state, self.verification_goal)
        if tool_id == STOP_TOOL_ID:
            self.steps += 1
            self.done = True
            observation = {
                "tool_id": STOP_TOOL_ID,
                "ok": goal_before,
                "summary": "goal satisfied" if goal_before else "premature stop",
                "status": "stop",
            }
            self.history.append(observation)
            return StepResult(
                observation,
                copy.deepcopy(self.state),
                True,
                goal_before,
                goal_before,
                goal_before,
                False,
                False,
            )

        valid = tool_id in self.task["available_tools"] and self.engine.applicable(
            self.state, tool_id
        )
        before = canonical_json(self.state)
        self.steps += 1
        # A tool-result perturbation can only trigger after the exact execution
        # contract has admitted the call. Otherwise an invalid attempt would
        # consume a once-only perturbation that was never actually exercised.
        perturbation = self._matching_perturbation(tool_id) if valid else None
        perturbation_applied = perturbation is not None
        arguments = self.engine.resolve_arguments(self.state, tool_id) if valid else {}
        execution_success = False
        if not valid:
            observation = {
                "tool_id": tool_id,
                "ok": False,
                "status": "contract_invalid",
                "summary": "exact precondition failed",
                "arguments": arguments,
                "output": {},
                "changed_fields": [],
            }
        elif perturbation and perturbation.get("kind") in {"failure", "empty_result"}:
            observation = {
                "tool_id": tool_id,
                "ok": False,
                "status": perturbation["kind"],
                "summary": str(perturbation.get("message", perturbation["kind"])),
                "arguments": arguments,
                "output": {},
                "changed_fields": [],
            }
        else:
            try:
                transition = self.engine.execute(self.state, tool_id)
            except LookupError:
                observation = {
                    "tool_id": tool_id,
                    "ok": False,
                    "status": "unobserved",
                    "summary": "no exact offline snapshot exists for these arguments",
                    "arguments": arguments,
                    "output": {},
                    "changed_fields": [],
                }
            else:
                self.state = transition.state
                observation = transition.observation
                execution_success = bool(observation.get("ok"))
                if perturbation:
                    for field, value in perturbation.get("set_state", {}).items():
                        set_path(self.state, field, value)
                    observation["status"] = str(perturbation.get("kind", "state_override"))
                    observation["summary"] = str(
                        perturbation.get("message", observation["summary"])
                    )
        if perturbation:
            self.applied_perturbations.append(
                str(perturbation.get("name", f"perturbation-{self.steps}"))
            )
        redundant = before == canonical_json(self.state) and observation.get("status") == "success"
        goal_after = self.engine.goal_satisfied(self.state, self.verification_goal)
        # Reaching the goal does not silently end the episode: the planner must emit STOP.
        self.done = self.steps >= self.max_steps
        self.history.append(observation)
        return StepResult(
            copy.deepcopy(observation),
            copy.deepcopy(self.state),
            self.done,
            goal_after,
            valid,
            execution_success,
            redundant,
            perturbation_applied,
        )
