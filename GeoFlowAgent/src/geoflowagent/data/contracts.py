from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from geoflowagent.constants import INVALID_REGRET, STOP_TOOL_ID
from geoflowagent.data.schema import validate_snapshot
from geoflowagent.utils.io import canonical_json

MISSING = object()
_NO_DEFAULT = object()

# Compiled once per distinct schema.
#
# ``Draft202012Validator(schema)`` rebuilds the reference resolver on every
# construction and ``check_schema`` re-validates the schema against the
# metaschema.  The contract engine validates arguments once per candidate action
# per state, so a single oracle task rebuilt the same handful of validators tens
# of thousands of times: a cProfile run of six ACMG tasks spent 105 of 125
# seconds inside jsonschema, most of it in URL parsing for $ref resolution.
#
# A validator holds no state about the instance it checks, so reusing one is
# exactly equivalent to building a fresh one.  Verified by comparing
# ``dataset_content_sha256`` of a processed package built with and without the
# cache: identical.
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _validator_for(schema: dict[str, Any]) -> Draft202012Validator:
    key = canonical_json(schema)
    validator = _VALIDATOR_CACHE.get(key)
    if validator is None:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        _VALIDATOR_CACHE[key] = validator
    return validator


def get_path(value: dict[str, Any], dotted_path: str, default: Any = _NO_DEFAULT) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            if default is _NO_DEFAULT:
                raise KeyError(dotted_path)
            return default
        current = current[part]
    return current


def set_path(value: dict[str, Any], dotted_path: str, item: Any) -> None:
    parts = dotted_path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(item)


def delete_path(value: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def condition_holds(state: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = str(condition["field"])
    op = str(condition["op"])
    actual = get_path(state, field, None)
    expected = condition.get("value")
    if op == "eq":
        return actual == expected
    if op == "neq":
        return actual != expected
    if op == "exists":
        return get_path(state, field, MISSING) is not MISSING
    if op == "missing":
        return get_path(state, field, MISSING) is MISSING
    if op == "nonempty":
        return actual is not None and hasattr(actual, "__len__") and len(actual) > 0
    if op == "in":
        return actual in (expected or [])
    if op == "not_in":
        return actual not in (expected or [])
    if op == "contains":
        return actual is not None and expected in actual
    if op == "gte":
        return actual is not None and actual >= expected
    if op == "lte":
        return actual is not None and actual <= expected
    raise ValueError(f"Unsupported condition op={op!r}")


def goal_conditions(goal: dict[str, Any]) -> list[dict[str, Any]]:
    if "conditions" in goal:
        conditions = goal["conditions"]
        if not isinstance(conditions, list):
            raise TypeError("goal.conditions must be a list")
        return conditions
    return [{"field": key, "op": "eq", "value": value} for key, value in goal.items()]


def combine_verification_goal(
    public_goal: dict[str, Any], private_verifier: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Combine public intent with hidden evaluator predicates without mutating either."""

    conditions = copy.deepcopy(goal_conditions(public_goal))
    if private_verifier is not None:
        conditions.extend(copy.deepcopy(goal_conditions(private_verifier)))
    return {"conditions": conditions}


@dataclass(frozen=True)
class Transition:
    state: dict[str, Any]
    observation: dict[str, Any]
    changed_fields: tuple[str, ...]


class ContractEngine:
    """Deterministic typed-state executor used for preprocessing and smoke evaluation.

    This engine is intentionally strict. Dense embeddings may suggest candidates, but
    genome build, accession/version, coordinate system, and schema compatibility remain
    exact predicates here.
    """

    def __init__(
        self,
        tools: Iterable[dict[str, Any]],
        snapshots: Iterable[dict[str, Any]] | None = None,
    ):
        self.tools = {str(tool["tool_id"]): tool for tool in tools}
        self.snapshots: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots or []:
            validate_snapshot(snapshot)
            tool_id = str(snapshot["tool_id"])
            if tool_id not in self.tools:
                raise ValueError(
                    f"Snapshot {snapshot.get('snapshot_id')!r} references unknown tool {tool_id!r}"
                )
            tool = self.tools[tool_id]
            if tool.get("execution_mode", "symbolic") != "snapshot":
                raise ValueError(
                    f"Snapshot {snapshot.get('snapshot_id')!r} targets non-snapshot tool {tool_id!r}"
                )
            arguments = snapshot["arguments"]
            self._validate_instance(
                arguments,
                tool.get("input_schema", {}),
                label=f"Snapshot {snapshot.get('snapshot_id')!r} arguments",
            )
            if snapshot.get("status") == "success":
                self._validate_instance(
                    snapshot["output"],
                    tool.get("output_schema", {}),
                    label=f"Snapshot {snapshot.get('snapshot_id')!r} output",
                )
                for output_path in tool.get("output_bindings", {}):
                    try:
                        get_path(snapshot["output"], str(output_path))
                    except KeyError:
                        raise ValueError(
                            f"Snapshot {snapshot.get('snapshot_id')!r} is missing bound output "
                            f"path {output_path!r}"
                        ) from None
            key = self._snapshot_key(tool_id, arguments)
            if key in self.snapshots:
                other = self.snapshots[key].get("snapshot_id")
                raise ValueError(
                    "Duplicate snapshot for exact tool+arguments key: "
                    f"{other!r} and {snapshot.get('snapshot_id')!r}"
                )
            self.snapshots[key] = copy.deepcopy(snapshot)

    @staticmethod
    def _snapshot_key(tool_id: str, arguments: dict[str, Any]) -> str:
        return canonical_json({"tool_id": tool_id, "arguments": arguments})

    @staticmethod
    def _validate_instance(instance: Any, schema: dict[str, Any], *, label: str) -> None:
        errors = sorted(
            _validator_for(schema).iter_errors(instance),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
            raise ValueError(f"{label} fails schema at {location}: {errors[0].message}")

    def resolve_arguments(self, state: dict[str, Any], tool_id: str) -> dict[str, Any]:
        """Bind a selected tool's arguments from typed state and validate them."""

        tool = self.tools[tool_id]
        schema = tool.get("input_schema", {})
        if schema.get("type", "object") != "object":
            raise ValueError(f"Tool {tool_id!r} input_schema must describe an object")
        properties = schema.get("properties", {})
        required = list(schema.get("required", []))
        bindings = tool.get("argument_bindings", {})
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise TypeError(f"Malformed input_schema for tool {tool_id!r}")
        if not isinstance(bindings, dict):
            raise TypeError(f"Tool {tool_id!r}.argument_bindings must be an object")
        arguments: dict[str, Any] = {}
        for name in sorted(set(required) | set(properties) | set(bindings)):
            path = str(bindings.get(name, name))
            try:
                value = get_path(state, path)
            except KeyError:
                if name in required:
                    raise ValueError(
                        f"Tool {tool_id!r} requires argument {name!r} from state path {path!r}"
                    ) from None
                continue
            arguments[name] = copy.deepcopy(value)
        self._validate_instance(arguments, schema, label=f"Tool {tool_id!r} arguments")
        return arguments

    def applicable(self, state: dict[str, Any], tool_id: str) -> bool:
        if tool_id == STOP_TOOL_ID:
            return False
        tool = self.tools[tool_id]
        if not all(condition_holds(state, condition) for condition in tool["preconditions"]):
            return False
        try:
            self.resolve_arguments(state, tool_id)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def applicable_tools(self, state: dict[str, Any], available_tools: Iterable[str]) -> list[str]:
        return [tool_id for tool_id in available_tools if self.applicable(state, tool_id)]

    def executable(self, state: dict[str, Any], tool_id: str) -> bool:
        """Whether the exact offline executor has an outcome for a contract-valid action."""

        if not self.applicable(state, tool_id):
            return False
        tool = self.tools[tool_id]
        if tool.get("execution_mode", "symbolic") != "snapshot":
            return True
        arguments = self.resolve_arguments(state, tool_id)
        return self._snapshot_key(tool_id, arguments) in self.snapshots

    def goal_satisfied(self, state: dict[str, Any], goal: dict[str, Any]) -> bool:
        return all(condition_holds(state, condition) for condition in goal_conditions(goal))

    def execute(self, state: dict[str, Any], tool_id: str) -> Transition:
        if tool_id not in self.tools:
            raise KeyError(f"Unknown tool {tool_id!r}")
        if not self.applicable(state, tool_id):
            raise ValueError(f"Tool {tool_id!r} is not applicable to state={canonical_json(state)}")
        arguments = self.resolve_arguments(state, tool_id)
        next_state = copy.deepcopy(state)
        changed: list[str] = []
        tool = self.tools[tool_id]
        output: dict[str, Any] | None = None
        status = "success"
        summary = tool.get("observation", f"{tool_id} completed")
        if tool.get("execution_mode", "symbolic") == "snapshot":
            key = self._snapshot_key(tool_id, arguments)
            if key not in self.snapshots:
                raise LookupError(
                    f"No exact snapshot for tool={tool_id!r}, arguments={canonical_json(arguments)}"
                )
            snapshot = self.snapshots[key]
            status = str(snapshot["status"])
            output = copy.deepcopy(snapshot["output"])
            summary = snapshot.get("summary", f"{tool_id} snapshot returned {status}")
            if status == "success":
                self._validate_instance(
                    output,
                    tool.get("output_schema", {}),
                    label=f"Tool {tool_id!r} output",
                )
                # Direction is deliberately fixed: output dotted path -> state dotted path.
                for output_path, state_path in tool.get("output_bindings", {}).items():
                    set_path(next_state, str(state_path), get_path(output, str(output_path)))
                    changed.append(str(state_path))

        # Failed/empty snapshots are observations, not successful state transitions.
        effects = tool["effects"] if status == "success" else []
        for effect in effects:
            op = effect["op"]
            field = effect["field"]
            if op == "set":
                set_path(next_state, field, effect.get("value"))
            elif op == "copy":
                set_path(next_state, field, get_path(next_state, str(effect["from"])))
            elif op == "delete":
                delete_path(next_state, field)
            elif op == "append":
                values = list(get_path(next_state, field, []))
                values.append(copy.deepcopy(effect.get("value")))
                set_path(next_state, field, values)
            elif op == "increment":
                set_path(next_state, field, get_path(next_state, field, 0) + effect.get("value", 1))
            else:
                raise ValueError(f"Unsupported effect op={op!r}")
            changed.append(field)
        observation = {
            "tool_id": tool_id,
            "ok": status == "success",
            "status": status,
            "arguments": arguments,
            "changed_fields": sorted(set(changed)),
            "summary": summary,
        }
        if output is not None:
            observation["output"] = output
        return Transition(next_state, observation, tuple(sorted(set(changed))))

    def shortest_distance(
        self,
        state: dict[str, Any],
        goal: dict[str, Any],
        available_tools: list[str],
        max_depth: int = 12,
    ) -> int | None:
        if self.goal_satisfied(state, goal):
            return 0
        queue: deque[tuple[dict[str, Any], int]] = deque([(copy.deepcopy(state), 0)])
        seen = {canonical_json(state)}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for tool_id in self.applicable_tools(current, available_tools):
                if not self.executable(current, tool_id):
                    continue
                successor = self.execute(current, tool_id).state
                key = canonical_json(successor)
                if key in seen:
                    continue
                if self.goal_satisfied(successor, goal):
                    return depth + 1
                seen.add(key)
                queue.append((successor, depth + 1))
        return None

    def action_analysis(
        self,
        state: dict[str, Any],
        goal: dict[str, Any],
        available_tools: list[str],
        max_depth: int = 12,
    ) -> tuple[list[str], dict[str, float]]:
        """Return the set of shortest-path actions and normalized action regret."""
        if self.goal_satisfied(state, goal):
            regrets = {tool_id: INVALID_REGRET for tool_id in available_tools}
            regrets[STOP_TOOL_ID] = 0.0
            return [STOP_TOOL_ID], regrets

        base_distance = self.shortest_distance(state, goal, available_tools, max_depth=max_depth)
        regrets: dict[str, float] = {STOP_TOOL_ID: INVALID_REGRET}
        optimal: list[str] = []
        for tool_id in available_tools:
            if not self.applicable(state, tool_id) or not self.executable(state, tool_id):
                regrets[tool_id] = INVALID_REGRET
                continue
            successor = self.execute(state, tool_id).state
            next_distance = self.shortest_distance(
                successor, goal, available_tools, max_depth=max_depth
            )
            if base_distance is None or next_distance is None:
                regrets[tool_id] = INVALID_REGRET
                continue
            raw_regret = max(0, 1 + next_distance - base_distance)
            regrets[tool_id] = min(1.0, raw_regret / max(1, max_depth))
            if raw_regret == 0:
                optimal.append(tool_id)
        return sorted(optimal), regrets
