from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

REQUIRED_TOOL_FIELDS = {
    "tool_id": str,
    "name": str,
    "description": str,
    "preconditions": list,
    "effects": list,
}
REQUIRED_TASK_FIELDS = {
    "task_id": str,
    "query": str,
    "initial_state": dict,
    "goal": dict,
    "available_tools": list,
}
REQUIRED_TRAJECTORY_FIELDS = {"task_id": str, "tool_ids": list}
REQUIRED_BACKGROUND_FIELDS = {"card_id": str, "text": str}
REQUIRED_PAIR_FIELDS = {
    "pair_id": str,
    "relation": str,
    "changed_fields": list,
    "split": str,
}
REQUIRED_VERIFIER_FIELDS = {"task_id": str, "private_verifier": dict}
REQUIRED_SNAPSHOT_FIELDS = {
    "snapshot_id": str,
    "tool_id": str,
    "arguments": dict,
    "output": dict,
    "status": str,
    "source_revision": str,
}
SNAPSHOT_STATUSES = {"success", "empty", "domain_error", "transport_error", "parse_error"}
CONDITION_OPS = {
    "eq",
    "neq",
    "exists",
    "missing",
    "nonempty",
    "in",
    "not_in",
    "contains",
    "gte",
    "lte",
}
EFFECT_OPS = {"set", "copy", "delete", "append", "increment"}
MODEL_FORBIDDEN_KEYS = {
    "expected_answer",
    "private_verifier",
    "verification_goal",
    "gold_tool_ids",
    "gold_next_tool",
    "gold_suffix_tool_ids",
    "stop_target",
}


def _validate_fields(row: dict[str, Any], fields: dict[str, type], kind: str) -> None:
    for field, expected_type in fields.items():
        if field not in row:
            raise ValueError(f"{kind} is missing required field {field!r}: {row}")
        if not isinstance(row[field], expected_type):
            raise TypeError(
                f"{kind}.{field} must be {expected_type.__name__}, got {type(row[field]).__name__}"
            )


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in MODEL_FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_tool(row: dict[str, Any]) -> None:
    _validate_fields(row, REQUIRED_TOOL_FIELDS, "tool")
    if not row["tool_id"].strip():
        raise ValueError("tool_id cannot be empty")
    if row["tool_id"] in {"<STOP>", "<PAD>"}:
        raise ValueError(f"{row['tool_id']} is a planner sentinel, not a callable tool")
    if not isinstance(row.get("input_schema", {}), dict):
        raise TypeError(f"Tool {row['tool_id']}.input_schema must be an object")
    Draft202012Validator.check_schema(row.get("input_schema", {}))
    if not isinstance(row.get("output_schema", {}), dict):
        raise TypeError(f"Tool {row['tool_id']}.output_schema must be an object")
    Draft202012Validator.check_schema(row.get("output_schema", {}))
    bindings = row.get("argument_bindings", {})
    if not isinstance(bindings, dict) or not all(
        isinstance(name, str) and name and isinstance(path, str) and path
        for name, path in bindings.items()
    ):
        raise TypeError(f"Tool {row['tool_id']}.argument_bindings must map names to paths")
    execution_mode = row.get("execution_mode", "symbolic")
    if execution_mode not in {"symbolic", "snapshot"}:
        raise ValueError(f"Tool {row['tool_id']}.execution_mode must be 'symbolic' or 'snapshot'")
    output_bindings = row.get("output_bindings", {})
    if not isinstance(output_bindings, dict) or not all(
        isinstance(output_path, str) and output_path and isinstance(state_path, str) and state_path
        for output_path, state_path in output_bindings.items()
    ):
        raise TypeError(
            f"Tool {row['tool_id']}.output_bindings must map output paths to state paths"
        )
    if execution_mode == "snapshot" and not output_bindings:
        raise ValueError(f"Snapshot tool {row['tool_id']} needs at least one output_binding")
    for condition in row["preconditions"]:
        if not isinstance(condition, dict) or "field" not in condition or "op" not in condition:
            raise ValueError(f"Malformed precondition in {row['tool_id']}: {condition}")
        if condition["op"] not in CONDITION_OPS:
            raise ValueError(f"Unsupported precondition op in {row['tool_id']}: {condition}")
    for effect in row["effects"]:
        if not isinstance(effect, dict) or "field" not in effect or "op" not in effect:
            raise ValueError(f"Malformed effect in {row['tool_id']}: {effect}")
        if effect["op"] not in EFFECT_OPS:
            raise ValueError(f"Unsupported effect op in {row['tool_id']}: {effect}")
        if effect["op"] == "copy" and not isinstance(effect.get("from"), str):
            raise ValueError(f"Copy effect needs a source path in {row['tool_id']}: {effect}")


def validate_task(row: dict[str, Any]) -> None:
    _validate_fields(row, REQUIRED_TASK_FIELDS, "task")
    if not row["task_id"].strip() or not row["query"].strip():
        raise ValueError("task_id and query cannot be empty")
    if not row["available_tools"]:
        raise ValueError(f"Task {row['task_id']} has no available tools")
    if not row["goal"]:
        raise ValueError(f"Task {row['task_id']} has an empty goal")
    _validate_goal_like(row["goal"], f"task {row['task_id']}.goal")
    if row.get("split") not in {None, "train", "dev", "test"}:
        raise ValueError(f"Task {row['task_id']} has invalid split={row.get('split')!r}")
    if "split_group" in row and (
        not isinstance(row["split_group"], str) or not row["split_group"].strip()
    ):
        raise ValueError(f"Task {row['task_id']}.split_group must be a non-empty string")
    if not isinstance(row.get("background_ids", []), list):
        raise TypeError(f"Task {row['task_id']}.background_ids must be a list")
    if not isinstance(row.get("provenance", {}), dict):
        raise TypeError(f"Task {row['task_id']}.provenance must be an object")
    if not all(
        isinstance(tool_id, str) and tool_id and tool_id not in {"<STOP>", "<PAD>"}
        for tool_id in row["available_tools"]
    ):
        raise ValueError(f"Task {row['task_id']} contains an invalid available tool ID")
    leaked = _forbidden_paths(row)
    if leaked:
        raise ValueError(
            f"Task {row['task_id']} contains evaluation-only/model-leaking fields: {sorted(leaked)}"
        )


def validate_trajectory(row: dict[str, Any]) -> None:
    _validate_fields(row, REQUIRED_TRAJECTORY_FIELDS, "trajectory")
    if not all(isinstance(tool_id, str) and tool_id for tool_id in row["tool_ids"]):
        raise ValueError(f"Trajectory {row['task_id']} contains an invalid tool ID")


def validate_background(row: dict[str, Any]) -> None:
    _validate_fields(row, REQUIRED_BACKGROUND_FIELDS, "background")
    if not row["card_id"].strip() or not row["text"].strip():
        raise ValueError("background.card_id and background.text cannot be empty")


def validate_minimal_pair(row: dict[str, Any]) -> None:
    _validate_fields(row, REQUIRED_PAIR_FIELDS, "minimal_pair")
    if row["relation"] not in {"invariant", "functional_sensitive"}:
        raise ValueError(f"Unknown minimal-pair relation: {row['relation']!r}")
    if not row["pair_id"].strip():
        raise ValueError("minimal_pair.pair_id cannot be empty")
    if row["split"] not in {"train", "dev", "test"}:
        raise ValueError(f"Minimal pair {row['pair_id']} has invalid split={row.get('split')!r}")
    if "source_task_id" in row and (
        not isinstance(row["source_task_id"], str) or not row["source_task_id"].strip()
    ):
        raise ValueError(f"Minimal pair {row['pair_id']}.source_task_id must be a non-empty string")
    if not row["changed_fields"] or not all(
        isinstance(field, str) and field for field in row["changed_fields"]
    ):
        raise ValueError(f"Minimal pair {row['pair_id']} needs non-empty changed_fields")
    if not any(key in row for key in ("left", "left_text")) or not any(
        key in row for key in ("right", "right_text")
    ):
        raise ValueError(f"Minimal pair {row['pair_id']} needs left and right values")


def _validate_goal_like(goal: dict[str, Any], kind: str) -> None:
    if not goal:
        raise ValueError(f"{kind} cannot be empty")
    if "conditions" not in goal:
        return
    conditions = goal["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"{kind}.conditions must be a non-empty list")
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("field"), str):
            raise ValueError(f"Malformed condition in {kind}: {condition}")
        if condition.get("op") not in CONDITION_OPS:
            raise ValueError(f"Unsupported condition op in {kind}: {condition}")


def validate_verifier(row: dict[str, Any]) -> None:
    """Validate an evaluation-only goal fragment kept outside model-facing tasks."""

    _validate_fields(row, REQUIRED_VERIFIER_FIELDS, "verifier")
    if not row["task_id"].strip():
        raise ValueError("verifier.task_id cannot be empty")
    _validate_goal_like(row["private_verifier"], f"verifier {row['task_id']}")


def validate_snapshot(row: dict[str, Any]) -> None:
    """Validate the envelope of a deterministic, normalized offline tool result."""

    _validate_fields(row, REQUIRED_SNAPSHOT_FIELDS, "snapshot")
    if not row["snapshot_id"].strip() or not row["tool_id"].strip():
        raise ValueError("snapshot_id and snapshot.tool_id cannot be empty")
    if row["status"] not in SNAPSHOT_STATUSES:
        raise ValueError(f"Snapshot {row['snapshot_id']} has unsupported status={row['status']!r}")
    if not row["source_revision"].strip():
        raise ValueError(f"Snapshot {row['snapshot_id']}.source_revision cannot be empty")
    if "summary" in row and not isinstance(row["summary"], str):
        raise TypeError(f"Snapshot {row['snapshot_id']}.summary must be a string")
    if not isinstance(row.get("provenance", {}), dict):
        raise TypeError(f"Snapshot {row['snapshot_id']}.provenance must be an object")


def unique_by_id(rows: list[dict[str, Any]], field: str, kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in output:
            raise ValueError(f"Duplicate {kind} {field}={key!r}")
        output[key] = row
    return output
