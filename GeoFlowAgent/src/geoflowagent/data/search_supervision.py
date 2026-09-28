"""Compile weighted-search output into leakage-safe GeoFlowAgent examples.

This module deliberately does *not* invent trajectories.  A deterministic weighted
search implementation is expected to return states, action costs, Q values, and
top-k paths.  The compiler validates those labels against the exact contract/
snapshot executor, keeps unobserved counterfactuals unknown, and emits rows that
the existing embedding cache can consume.

The public API is intentionally small:

``compile_search_supervision``
    Compile one task and its oracle result into cache-compatible examples.
``task_balanced_split`` / ``validate_preassigned_splits``
    Assign or validate task/group-level splits (never prefix-level splits).
``write_search_dataset`` / ``read_search_dataset``
    Persist an immutable, checksummed processed dataset without demonstrations.

Oracle input format
-------------------

The oracle result must contain a ``states`` list (a mapping keyed by state id is
also accepted).  Each state has ``state``, ``v_star``, ``reachable``, ``actions``,
and ``top_k_paths``.  Actions may be a list or mapping and use this shape::

    {
      "tool_id": "normalize_variant",
      "known": true,
      "executable": true,
      "successor": {...},
      "successor_id": "state-2",
      "cost": 1.0,
      "q_star": 3.0,
      "regret": 0.0,
      "is_optimal": true,
      "reachable": true
    }

Missing applicable actions and explicit ``known=false`` actions are serialized
with every outcome/utility label set to null and ``label_mask=false``.  They are
not silently converted into negative examples.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from geoflowagent.constants import INVALID_REGRET, SCHEMA_VERSION, STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.data.schema import (
    MODEL_FORBIDDEN_KEYS,
    unique_by_id,
    validate_background,
    validate_snapshot,
    validate_task,
    validate_tool,
    validate_verifier,
)
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

SEARCH_SUPERVISION_VERSION = "geoflowagent.search-supervision.v1"
SEARCH_EXAMPLES_FILE = "examples.jsonl"

SPLITS = ("train", "dev", "test")
GROUP_KINDS = ("case", "entity", "template", "workflow", "source", "release")
DEFAULT_AUDIT_GROUP_KINDS = ("entity", "template", "workflow", "source", "release")

_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "entity": ("entity_group", "entity_id", "variant_id", "gene_id"),
    "template": ("template_group", "template_id"),
    "workflow": (
        "workflow_group",
        "workflow_id",
        "workflow_family",
        "topology_group",
        "category",
    ),
    "source": ("source_group", "source_id", "source"),
    "release": ("release_group", "release_id", "release", "source_revision"),
}

_SEARCH_CONTEXT_FIELD = "__search_context__"


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).strip().lower()
            if normalized in MODEL_FORBIDDEN_KEYS or normalized in {
                "expected",
                "answer_key",
                "private_goal",
                "verifier_answer",
            }:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def model_input_view(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only fields permitted to enter a frozen encoder/planner input.

    Supervision fields intentionally live beside these fields for compatibility
    with the existing cache, but callers should pass this whitelisted view to any
    new serializer instead of serializing a complete training row.
    """

    return {
        "query": copy.deepcopy(example["query"]),
        "goal": copy.deepcopy(example["goal"]),
        "state": copy.deepcopy(example["state"]),
        "history": copy.deepcopy(example.get("history", [])),
        "candidate_tools": copy.deepcopy(example["candidate_tools"]),
        "background_ids": copy.deepcopy(example.get("background_ids", [])),
    }


def validate_model_input(example_or_input: Mapping[str, Any]) -> None:
    """Fail closed if evaluation-only labels occur in model-facing content."""

    if {"query", "goal", "state", "candidate_tools"}.issubset(example_or_input):
        visible = model_input_view(example_or_input)
    else:
        visible = copy.deepcopy(dict(example_or_input))
    leaked = _forbidden_paths(visible)
    if leaked:
        raise ValueError(f"Model input contains evaluation-only fields: {sorted(leaked)}")


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return result


def _optional_cost(value: Any, *, label: str) -> float | None:
    return None if value is None else _finite_nonnegative(value, label=label)


def _normalize_group_values(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized: set[str] = set()
    for item in values:
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise TypeError(f"{label} group values must be non-empty strings or integers")
        text = str(item).strip()
        if not text:
            raise ValueError(f"{label} group values cannot be empty")
        normalized.add(text)
    return sorted(normalized)


def extract_group_ids(task: Mapping[str, Any]) -> dict[str, list[str]]:
    """Extract explicit, auditable group IDs without guessing from biomedical text."""

    explicit = task.get("group_ids", task.get("groups", {}))
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, Mapping):
        raise TypeError(f"Task {task.get('task_id')!r}.group_ids must be an object")
    unknown = sorted(set(explicit) - set(GROUP_KINDS))
    if unknown:
        raise ValueError(f"Task {task.get('task_id')!r} has unknown group kinds: {unknown}")

    provenance = task.get("provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    result: dict[str, list[str]] = {
        "case": _normalize_group_values(
            explicit.get("case", task.get("split_group", task["task_id"])), label="case"
        )
    }
    for kind in GROUP_KINDS[1:]:
        value = explicit.get(kind)
        if value is None:
            for alias in _GROUP_ALIASES[kind]:
                if alias in task:
                    value = task[alias]
                    break
                if alias in provenance:
                    value = provenance[alias]
                    break
        values = _normalize_group_values(value, label=kind)
        if values:
            result[kind] = values
    return result


def _state_id(state_row: Mapping[str, Any], task_id: str) -> str:
    supplied = state_row.get("state_id")
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied.strip():
            raise ValueError(f"Task {task_id} has an empty/non-string state_id")
        return supplied
    payload = {
        "task_id": task_id,
        "state": state_row.get("state"),
        "history": state_row.get("history", []),
        "budget": state_row.get("budget"),
        "source_availability": state_row.get("source_availability"),
    }
    return "state:" + sha256_text(canonical_json(payload))[:24]


def _normalize_oracle_states(oracle_output: Mapping[str, Any], task_id: str) -> list[dict[str, Any]]:
    raw_states = oracle_output.get("states")
    if isinstance(raw_states, Mapping):
        states = []
        for state_id, value in raw_states.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Oracle state {state_id!r} for {task_id} must be an object")
            row = copy.deepcopy(dict(value))
            if "state_id" in row and row["state_id"] != str(state_id):
                raise ValueError(f"Oracle state mapping key disagrees with state_id={row['state_id']!r}")
            row["state_id"] = str(state_id)
            states.append(row)
    elif isinstance(raw_states, list):
        if not all(isinstance(row, Mapping) for row in raw_states):
            raise TypeError(f"Oracle states for {task_id} must contain only objects")
        states = [copy.deepcopy(dict(row)) for row in raw_states]
    else:
        raise TypeError(f"Oracle output for {task_id} needs a states list or mapping")
    if not states:
        raise ValueError(f"Oracle output for {task_id} has no states")

    seen: set[str] = set()
    for row in states:
        row["state_id"] = _state_id(row, task_id)
        if row["state_id"] in seen:
            raise ValueError(f"Duplicate oracle state_id={row['state_id']!r} for task {task_id}")
        seen.add(row["state_id"])
        if not isinstance(row.get("state"), dict):
            raise TypeError(f"Oracle state {row['state_id']!r}.state must be an object")
        if not isinstance(row.get("history", []), list):
            raise TypeError(f"Oracle state {row['state_id']!r}.history must be a list")
        leaked = _forbidden_paths({"state": row["state"], "history": row.get("history", [])})
        if leaked:
            raise ValueError(
                f"Oracle state {row['state_id']!r} exposes evaluation-only fields: {sorted(leaked)}"
            )
    return sorted(states, key=lambda row: row["state_id"])


def _adapt_weighted_search_output(oracle_output: Mapping[str, Any]) -> dict[str, Any]:
    """Join ``SearchResult.to_dict()`` states/values/edges into compiler rows.

    The search package intentionally stores graph topology separately from value
    labels.  Keeping the adapter here avoids coupling persistence code to Python
    dataclasses and also makes JSON round-trips an explicit integration boundary.
    """

    if "values" not in oracle_output or "edges" not in oracle_output:
        return copy.deepcopy(dict(oracle_output))
    raw_states = oracle_output.get("states")
    raw_values = oracle_output.get("values")
    raw_edges = oracle_output.get("edges")
    if not isinstance(raw_states, list) or not isinstance(raw_values, list) or not isinstance(
        raw_edges, list
    ):
        raise TypeError("SearchResult states, values, and edges must be lists")
    states_by_id = {str(row["state_id"]): row for row in raw_states}
    values_by_id = {str(row["state_id"]): row for row in raw_values}
    if len(states_by_id) != len(raw_states) or len(values_by_id) != len(raw_values):
        raise ValueError("SearchResult contains duplicate state/value IDs")
    if set(states_by_id) != set(values_by_id):
        raise ValueError("SearchResult states and values do not cover the same IDs")
    edge_by_source_tool: dict[tuple[str, str], Mapping[str, Any]] = {}
    for edge in raw_edges:
        key = (str(edge["source_state_id"]), str(edge["tool_id"]))
        if key in edge_by_source_tool:
            raise ValueError(f"SearchResult has duplicate deterministic edge {key}")
        edge_by_source_tool[key] = edge

    maximum_depth = int(oracle_output.get("limits", {}).get("max_depth", 0))

    def visible_state(state_id: str) -> dict[str, Any]:
        source = states_by_id[state_id]
        state = copy.deepcopy(source["state"])
        context = copy.deepcopy(source.get("context", {}))
        depth = int(source.get("depth", 0))
        # Depth/remaining budget and search context can alter future values.  They
        # therefore belong in the model-visible Markov state, not only provenance.
        if context or maximum_depth:
            state[_SEARCH_CONTEXT_FIELD] = {
                "context": context,
                "depth": depth,
                "remaining_depth": max(0, maximum_depth - depth),
            }
        return state

    # Produce at least one certified optimal suffix for every value-known state.
    # The SearchResult's explicit top-k paths are rooted only at initial_state_id.
    path_limit = max(1, int(oracle_output.get("limits", {}).get("top_k_paths", 1)))
    path_memo: dict[str, list[dict[str, Any]]] = {}

    def optimal_suffixes(state_id: str, visiting: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
        if state_id in path_memo:
            return copy.deepcopy(path_memo[state_id])
        if state_id in visiting:
            return []
        state = states_by_id[state_id]
        if bool(state.get("goal_satisfied")):
            result = [{"tool_ids": [], "total_cost": 0.0, "state_ids": [state_id]}]
            path_memo[state_id] = result
            return copy.deepcopy(result)
        value = values_by_id[state_id]
        result: list[dict[str, Any]] = []
        for action in sorted(value.get("actions", []), key=lambda row: str(row["tool_id"])):
            tool_id = str(action["tool_id"])
            if tool_id == STOP_TOOL_ID or action.get("optimal") is not True:
                continue
            successor_id = action.get("successor_state_id")
            edge = edge_by_source_tool.get((state_id, tool_id))
            if not isinstance(successor_id, str) or edge is None:
                continue
            for suffix in optimal_suffixes(successor_id, visiting | {state_id}):
                result.append(
                    {
                        "tool_ids": [tool_id, *suffix["tool_ids"]],
                        "total_cost": float(edge["cost"]) + float(suffix["total_cost"]),
                        "state_ids": [state_id, *suffix["state_ids"]],
                    }
                )
                if len(result) >= path_limit:
                    break
            if len(result) >= path_limit:
                break
        result.sort(key=lambda path: (path["total_cost"], path["tool_ids"]))
        path_memo[state_id] = result[:path_limit]
        return copy.deepcopy(path_memo[state_id])

    explicit_root_paths: list[dict[str, Any]] = []
    for path in oracle_output.get("paths", []):
        actions = list(path.get("actions", []))
        if actions and actions[-1] == STOP_TOOL_ID:
            actions.pop()
        explicit_root_paths.append(
            {
                "tool_ids": actions,
                "total_cost": path["cost"],
                "state_ids": list(path.get("state_ids", [])),
            }
        )
    explicit_root_paths.sort(key=lambda path: (path["total_cost"], path["tool_ids"]))

    joined: list[dict[str, Any]] = []
    for state_id, search_state in states_by_id.items():
        value = values_by_id[state_id]
        actions: list[dict[str, Any]] = []
        for action in value.get("actions", []):
            tool_id = str(action["tool_id"])
            if tool_id == STOP_TOOL_ID or action.get("contract_applicable") is not True:
                continue
            transition_known = bool(action.get("transition_known", False))
            known_mask = bool(action.get("known_mask", False))
            if not transition_known:
                actions.append({"tool_id": tool_id, "known": False})
                continue
            edge = edge_by_source_tool.get((state_id, tool_id))
            successor_id = action.get("successor_state_id")
            if edge is None or not isinstance(successor_id, str):
                actions.append({"tool_id": tool_id, "known": False})
                continue
            # Full Q labels are conservative.  A known transition whose downstream
            # optimum is uncertain remains useful for transition learning, but is
            # not a Q/regret target (q_known=false).
            actions.append(
                {
                    "tool_id": tool_id,
                    "known": transition_known,
                    "q_known": known_mask,
                    "transition_known": True,
                    "reachability_known": bool(action.get("reachability_known", False)),
                    "executable": bool(action.get("executable", True)),
                    "successor": visible_state(successor_id),
                    "successor_id": successor_id,
                    "cost": edge["cost"],
                    "q_star": action.get("q_star") if known_mask else None,
                    "regret": action.get("regret") if known_mask else None,
                    "is_optimal": (
                        action.get("optimal")
                        if known_mask
                        or (
                            bool(action.get("reachability_known", False))
                            and action.get("reachable") is False
                        )
                        else None
                    ),
                    "reachable": (
                        action.get("reachable")
                        if bool(action.get("reachability_known", False))
                        else None
                    ),
                }
            )
        value_known = bool(value.get("value_known", False))
        reachability_known = bool(value.get("reachability_known", False))
        paths = (
            explicit_root_paths
            if state_id == oracle_output.get("initial_state_id") and explicit_root_paths
            else optimal_suffixes(state_id)
        )
        joined.append(
            {
                "state_id": state_id,
                "state": visible_state(state_id),
                "history": copy.deepcopy(search_state.get("observation_history", [])),
                "prefix_tool_ids": list(search_state.get("action_history", [])),
                "depth": int(search_state.get("depth", 0)),
                "terminal": bool(search_state.get("goal_satisfied", False)),
                "v_star": value.get("v_star") if value_known else None,
                "best_known_v": value.get("best_known_v"),
                "value_known": value_known,
                "reachability_known": reachability_known,
                "reachable": value.get("reachable") if reachability_known else None,
                "actions": actions,
                "optimal_actions": [
                    tool_id
                    for tool_id in value.get("optimal_actions", [])
                    if tool_id != STOP_TOOL_ID
                ],
                "top_k_paths": paths,
            }
        )
    output = copy.deepcopy(dict(oracle_output))
    output["states"] = joined
    output.pop("values", None)
    output.pop("edges", None)
    output.pop("paths", None)
    output.setdefault("algorithm", "weighted_dijkstra")
    output.setdefault("cost_spec", copy.deepcopy(oracle_output.get("cost_config", {})))
    output.setdefault("oracle_version", oracle_output.get("schema_version"))
    return output


def _normalize_actions(raw_actions: Any, *, state_id: str) -> dict[str, dict[str, Any]]:
    if raw_actions is None:
        return {}
    if isinstance(raw_actions, Mapping):
        rows: list[dict[str, Any]] = []
        for tool_id, value in raw_actions.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Action {tool_id!r} at {state_id} must be an object")
            row = copy.deepcopy(dict(value))
            supplied = row.get("tool_id", row.get("action"))
            if supplied is not None and supplied != str(tool_id):
                raise ValueError(f"Action mapping key {tool_id!r} disagrees with {supplied!r}")
            row["tool_id"] = str(tool_id)
            rows.append(row)
    elif isinstance(raw_actions, list):
        if not all(isinstance(row, Mapping) for row in raw_actions):
            raise TypeError(f"Actions at {state_id} must contain only objects")
        rows = [copy.deepcopy(dict(row)) for row in raw_actions]
    else:
        raise TypeError(f"Actions at {state_id} must be a list or mapping")

    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        tool_id = row.get("tool_id", row.get("action"))
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError(f"Action at {state_id} needs a non-empty tool_id")
        if tool_id in output:
            raise ValueError(f"Duplicate action {tool_id!r} at {state_id}")
        row["tool_id"] = tool_id
        output[tool_id] = row
    return output


def _unknown_action(tool_id: str, *, contract_applicable: bool) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "known": False,
        "label_mask": False,
        "q_known_mask": False,
        "transition_known": False,
        "transition_known_mask": False,
        "reachability_known_mask": False,
        "contract_applicable": contract_applicable,
        "executable": None,
        "successor": None,
        "successor_id": None,
        "cost": None,
        "q_star": None,
        "regret": None,
        "is_optimal": None,
        "reachable": None,
        "outcome_status": "unobserved",
    }


def _contract_invalid_action(tool_id: str) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "known": False,
        "label_mask": False,
        "q_known_mask": False,
        "transition_known": False,
        "transition_known_mask": True,
        "reachability_known_mask": True,
        "contract_applicable": False,
        "executable": False,
        "successor": None,
        "successor_id": None,
        "cost": None,
        "q_star": None,
        "regret": None,
        "is_optimal": False,
        "reachable": False,
        "outcome_status": "contract_invalid",
    }


def _normalize_top_paths(
    raw_paths: Any,
    *,
    state_id: str,
    available_tools: set[str],
) -> list[dict[str, Any]]:
    if raw_paths is None:
        return []
    if not isinstance(raw_paths, list):
        raise TypeError(f"top_k_paths at {state_id} must be a list")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw in enumerate(raw_paths):
        if isinstance(raw, list):
            raw = {"tool_ids": raw}
        if not isinstance(raw, Mapping):
            raise TypeError(f"top_k_paths[{index}] at {state_id} must be an object or list")
        tool_ids = raw.get("tool_ids", raw.get("actions"))
        if not isinstance(tool_ids, list) or not all(
            isinstance(tool_id, str) and tool_id for tool_id in tool_ids
        ):
            raise ValueError(f"top_k_paths[{index}] at {state_id} needs valid tool_ids")
        unknown = sorted(set(tool_ids) - available_tools)
        if unknown:
            raise ValueError(f"top_k_paths[{index}] at {state_id} uses unknown tools: {unknown}")
        key = tuple(tool_ids)
        if key in seen:
            raise ValueError(f"Duplicate top-k action path at {state_id}: {list(key)}")
        seen.add(key)
        total_cost = raw.get("total_cost", raw.get("cost"))
        if total_cost is None:
            raise ValueError(f"top_k_paths[{index}] at {state_id} is missing total_cost")
        path: dict[str, Any] = {
            "tool_ids": list(tool_ids),
            "total_cost": _finite_nonnegative(
                total_cost, label=f"top_k_paths[{index}].total_cost at {state_id}"
            ),
        }
        if "weight" in raw:
            path["weight"] = _finite_nonnegative(
                raw["weight"], label=f"top_k_paths[{index}].weight at {state_id}"
            )
        if "probability" in raw:
            probability = _finite_nonnegative(
                raw["probability"], label=f"top_k_paths[{index}].probability at {state_id}"
            )
            if probability > 1:
                raise ValueError(f"top_k_paths[{index}].probability exceeds one at {state_id}")
            path["probability"] = probability
        if "state_ids" in raw:
            state_ids = raw["state_ids"]
            if not isinstance(state_ids, list) or not all(
                isinstance(value, str) and value for value in state_ids
            ):
                raise ValueError(f"top_k_paths[{index}].state_ids at {state_id} is invalid")
            path["state_ids"] = list(state_ids)
        output.append(path)
    return sorted(output, key=lambda path: (path["total_cost"], path["tool_ids"]))


def _successor_id(
    action: Mapping[str, Any],
    successor: Mapping[str, Any],
    state_id_by_payload: Mapping[str, str],
) -> str:
    supplied = action.get("successor_id", action.get("successor_state_id"))
    payload_key = canonical_json(successor)
    inferred = state_id_by_payload.get(payload_key)
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied:
            raise ValueError("successor_id must be a non-empty string")
        if inferred is not None and supplied != inferred:
            raise ValueError(
                f"successor_id={supplied!r} disagrees with oracle state payload id={inferred!r}"
            )
        return supplied
    return inferred or ("state:" + sha256_text(payload_key)[:24])


def _executor_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop search-only Markov metadata before checking an executor transition."""

    output = copy.deepcopy(dict(value))
    output.pop(_SEARCH_CONTEXT_FIELD, None)
    return output


def _compile_action(
    *,
    raw: Mapping[str, Any] | None,
    tool_id: str,
    state: dict[str, Any],
    state_id: str,
    state_v: float | None,
    state_reachable: bool,
    engine: ContractEngine,
    state_values: Mapping[str, tuple[bool, float | None]],
    state_id_by_payload: Mapping[str, str],
    optimal_epsilon: float,
    consistency_tolerance: float,
) -> dict[str, Any]:
    contract_applicable = engine.applicable(state, tool_id)
    if raw is None:
        return (
            _unknown_action(tool_id, contract_applicable=True)
            if contract_applicable
            else _contract_invalid_action(tool_id)
        )

    transition_known = raw.get("transition_known", raw.get("known", True))
    if not isinstance(transition_known, bool):
        raise TypeError(f"Action {tool_id!r} at {state_id}.transition_known must be boolean")
    q_known = raw.get("q_known", raw.get("known", raw.get("label_mask", True)))
    if not isinstance(q_known, bool):
        raise TypeError(f"Action {tool_id!r} at {state_id}.known must be boolean")
    if q_known and not transition_known:
        raise ValueError(f"Action {tool_id!r} at {state_id} has Q label without a transition")
    if not transition_known:
        non_null = [
            key
            for key in (
                "executable",
                "successor",
                "successor_state",
                "successor_id",
                "successor_state_id",
                "cost",
                "edge_cost",
                "q_star",
                "regret",
                "is_optimal",
                "reachable",
            )
            if raw.get(key) is not None
        ]
        if non_null:
            raise ValueError(
                f"Unknown action {tool_id!r} at {state_id} carries labels: {sorted(non_null)}"
            )
        return _unknown_action(tool_id, contract_applicable=contract_applicable)

    if not contract_applicable:
        executable = raw.get("executable", False)
        if executable is not False:
            raise ValueError(
                f"Oracle marks contract-invalid action {tool_id!r} executable at {state_id}"
            )
        forbidden = [
            key
            for key in ("successor", "successor_state", "cost", "edge_cost", "q_star", "regret")
            if raw.get(key) is not None
        ]
        if forbidden:
            raise ValueError(
                f"Contract-invalid action {tool_id!r} at {state_id} has transition labels: {forbidden}"
            )
        return _contract_invalid_action(tool_id)

    if not engine.executable(state, tool_id):
        raise ValueError(
            f"Action {tool_id!r} at {state_id} is claimed known but has no exact snapshot outcome"
        )
    if raw.get("executable", True) is not True:
        raise ValueError(f"Known executable action {tool_id!r} at {state_id} is contradictory")

    transition = engine.execute(_executor_state(state), tool_id)
    supplied_successor = raw.get("successor", raw.get("successor_state"))
    successor = transition.state if supplied_successor is None else supplied_successor
    if not isinstance(successor, dict):
        raise TypeError(f"Action {tool_id!r} at {state_id}.successor must be an object")
    if canonical_json(_executor_state(successor)) != canonical_json(transition.state):
        raise ValueError(
            f"Oracle successor for action {tool_id!r} at {state_id} disagrees with exact executor"
        )
    leaked = _forbidden_paths(successor)
    if leaked:
        raise ValueError(
            f"Successor for action {tool_id!r} at {state_id} has evaluation-only fields: {leaked}"
        )
    successor_id = _successor_id(raw, successor, state_id_by_payload)
    cost = raw.get("cost", raw.get("edge_cost"))
    if cost is None:
        raise ValueError(f"Known action {tool_id!r} at {state_id} is missing weighted edge cost")
    cost_value = _finite_nonnegative(cost, label=f"Action {tool_id!r} at {state_id}.cost")
    reachability_known = raw.get("reachability_known", raw.get("reachable") is not None)
    if not isinstance(reachability_known, bool):
        raise TypeError(f"Action {tool_id!r} at {state_id}.reachability_known must be boolean")
    reachable = raw.get("reachable")
    if reachability_known and not isinstance(reachable, bool):
        raise TypeError(f"Action {tool_id!r} at {state_id}.reachable must be boolean")
    if not reachability_known and reachable is not None:
        raise ValueError(f"Action {tool_id!r} at {state_id} has unmasked reachability")
    q_star = _optional_cost(raw.get("q_star"), label=f"Action {tool_id!r} at {state_id}.q_star")
    if q_known and q_star is None:
        raise ValueError(f"Reachable action {tool_id!r} at {state_id} needs q_star")
    if not q_known and q_star is not None:
        raise ValueError(f"Q-unknown action {tool_id!r} at {state_id} must use q_star=null")
    if q_known and reachable is not True:
        raise ValueError(f"Q-known action {tool_id!r} at {state_id} must be known reachable")

    if q_known and successor_id in state_values:
        successor_reachable, successor_v = state_values[successor_id]
        if not successor_reachable or successor_v is None:
            raise ValueError(
                f"Action {tool_id!r} at {state_id} is reachable but successor is not"
            )
        expected_q = cost_value + successor_v
        if not math.isclose(float(q_star), expected_q, abs_tol=consistency_tolerance, rel_tol=0):
            raise ValueError(
                f"Bellman inconsistency for {tool_id!r} at {state_id}: "
                f"q_star={q_star}, cost+V(successor)={expected_q}"
            )

    regret = _optional_cost(raw.get("regret"), label=f"Action {tool_id!r} at {state_id}.regret")
    if q_known and state_reachable and state_v is not None:
        expected_regret = max(0.0, float(q_star) - state_v)
        if regret is None:
            regret = expected_regret
        elif not math.isclose(regret, expected_regret, abs_tol=consistency_tolerance, rel_tol=0):
            raise ValueError(
                f"Regret inconsistency for {tool_id!r} at {state_id}: "
                f"regret={regret}, q_star-v_star={expected_regret}"
            )
    elif regret is not None:
        raise ValueError(
            f"Action {tool_id!r} at {state_id} cannot have regret without reachable state/action"
        )
    derived_optimal: bool | None
    if q_known and state_reachable and state_v is not None:
        derived_optimal = bool(regret is not None and regret <= optimal_epsilon)
    elif reachability_known and reachable is False:
        derived_optimal = False
    else:
        derived_optimal = None
    supplied_optimal = raw.get("is_optimal", derived_optimal)
    if supplied_optimal is not None and not isinstance(supplied_optimal, bool):
        raise TypeError(f"Action {tool_id!r} at {state_id}.is_optimal must be boolean or null")
    if supplied_optimal != derived_optimal:
        raise ValueError(
            f"Action {tool_id!r} at {state_id}.is_optimal disagrees with regret={regret}"
        )
    return {
        "tool_id": tool_id,
        "known": q_known,
        "label_mask": q_known,
        "q_known_mask": q_known,
        "transition_known": True,
        "transition_known_mask": True,
        "reachability_known_mask": reachability_known,
        "contract_applicable": True,
        "executable": True,
        "successor": copy.deepcopy(successor),
        "successor_id": successor_id,
        "cost": cost_value,
        "q_star": q_star,
        "regret": regret,
        "is_optimal": derived_optimal,
        "reachable": reachable,
        "outcome_status": str(transition.observation.get("status", "success")),
    }


def _record_sha256(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "record_sha256"}
    return sha256_text(canonical_json(payload))


def _with_record_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    output["record_sha256"] = _record_sha256(output)
    return output


def compile_search_supervision(
    task: dict[str, Any],
    tools: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]] | None,
    oracle_output: Mapping[str, Any],
    *,
    split: str | None = None,
    optimal_epsilon: float = 1e-8,
    consistency_tolerance: float = 1e-6,
    include_unreachable: bool = False,
    include_uncertain: bool = False,
) -> list[dict[str, Any]]:
    """Compile one task's weighted-search labels into cache-compatible examples.

    ``include_unreachable`` defaults to false because the legacy Flow loader treats
    an empty suffix as a STOP target.  Unreachable graph states can be retained for
    a dedicated reachability head, but such rows carry ``training_mask=false`` and
    must not be consumed by the legacy Flow loss.
    """

    validate_task(task)
    if not isinstance(oracle_output, Mapping):
        raise TypeError("oracle_output must be an object")
    oracle_output = _adapt_weighted_search_output(oracle_output)
    if oracle_output.get("task_id", task["task_id"]) != task["task_id"]:
        raise ValueError("Oracle task_id disagrees with task.task_id")
    for tool in tools:
        validate_tool(tool)
    for snapshot in snapshots or []:
        validate_snapshot(snapshot)
    tool_map = unique_by_id(list(tools), "tool_id", "tool")
    unique_by_id(list(snapshots or []), "snapshot_id", "snapshot")
    unknown_snapshot_tools = sorted(
        {str(snapshot["tool_id"]) for snapshot in snapshots or []} - set(tool_map)
    )
    if unknown_snapshot_tools:
        raise ValueError(f"Snapshots reference unknown tools: {unknown_snapshot_tools}")
    available = [str(tool_id) for tool_id in task["available_tools"]]
    missing = sorted(set(available) - set(tool_map))
    if missing:
        raise ValueError(f"Task {task['task_id']} references unknown tools: {missing}")
    selected_split = split if split is not None else task.get("split")
    if selected_split not in {*SPLITS, None}:
        raise ValueError(f"Invalid split={selected_split!r}")
    if split is not None and task.get("split") not in {None, split}:
        raise ValueError(
            f"Requested split={split!r} contradicts task split={task.get('split')!r}"
        )

    relevant_snapshots = [
        snapshot for snapshot in snapshots or [] if str(snapshot["tool_id"]) in set(available)
    ]
    engine = ContractEngine([tool_map[tool_id] for tool_id in available], relevant_snapshots)
    states = _normalize_oracle_states(oracle_output, str(task["task_id"]))
    state_id_by_payload: dict[str, str] = {}
    state_values: dict[str, tuple[bool, float | None]] = {}
    state_label_status: dict[str, tuple[bool, bool, bool | None, float | None]] = {}
    for row in states:
        payload = canonical_json(row["state"])
        previous = state_id_by_payload.setdefault(payload, row["state_id"])
        if previous != row["state_id"]:
            raise ValueError(
                "Two oracle nodes have identical typed state but different IDs. Include all "
                "future-relevant history/budget/source status inside the typed state."
            )
        value_known = row.get("value_known", row.get("v_star") is not None)
        if not isinstance(value_known, bool):
            raise TypeError(f"Oracle state {row['state_id']}.value_known must be boolean")
        reachability_known = row.get(
            "reachability_known", row.get("reachable") is not None
        )
        if not isinstance(reachability_known, bool):
            raise TypeError(
                f"Oracle state {row['state_id']}.reachability_known must be boolean"
            )
        reachable = row.get("reachable")
        if reachability_known and not isinstance(reachable, bool):
            raise TypeError(f"Oracle state {row['state_id']}.reachable must be boolean")
        if not reachability_known and reachable is not None:
            raise ValueError(f"Oracle state {row['state_id']} has unmasked reachability")
        value = _optional_cost(row.get("v_star"), label=f"State {row['state_id']}.v_star")
        if value_known and reachable is True and value is None:
            raise ValueError(f"Reachable state {row['state_id']} needs v_star")
        if value_known and reachability_known is not True:
            raise ValueError(
                f"Value-known state {row['state_id']} needs known reachability"
            )
        if not value_known and value is not None:
            raise ValueError(f"Value-unknown state {row['state_id']} must use v_star=null")
        if reachable is False and value is not None:
            raise ValueError(f"Unreachable state {row['state_id']} must use v_star=null")
        best_known = _optional_cost(
            row.get("best_known_v"), label=f"State {row['state_id']}.best_known_v"
        )
        state_values[row["state_id"]] = (value_known and reachable is True, value)
        state_label_status[row["state_id"]] = (
            value_known,
            reachability_known,
            reachable,
            best_known,
        )

    groups = extract_group_ids(task)
    available_set = set(available)
    oracle_hash = sha256_text(canonical_json(oracle_output))
    tools_hash = sha256_text(
        canonical_json([tool_map[tool_id] for tool_id in sorted(available)])
    )
    snapshots_hash = sha256_text(
        canonical_json(sorted(relevant_snapshots, key=lambda row: row["snapshot_id"]))
    )
    safe_search_metadata = {
        key: copy.deepcopy(oracle_output[key])
        for key in (
            "algorithm",
            "cost_spec",
            "oracle_version",
            "source_revision",
            "diagnostics",
            "limits",
        )
        if key in oracle_output
    }
    if _forbidden_paths(safe_search_metadata):
        raise ValueError("Search metadata contains evaluation-only fields")

    examples: list[dict[str, Any]] = []
    for state_index, state_row in enumerate(states):
        state_id = str(state_row["state_id"])
        state = copy.deepcopy(state_row["state"])
        history = copy.deepcopy(state_row.get("history", []))
        state_value_known, state_reachability_known, state_reachable_raw, best_known_v = (
            state_label_status[state_id]
        )
        state_reachable = state_reachable_raw is True
        state_v = state_values[state_id][1]
        terminal = state_row.get("terminal", engine.goal_satisfied(state, task["goal"]))
        if not isinstance(terminal, bool):
            raise TypeError(f"State {state_id}.terminal must be boolean")
        if terminal and (
            not state_value_known
            or not state_reachable
            or state_v is None
            or state_v > optimal_epsilon
        ):
            raise ValueError(f"Terminal state {state_id} must be reachable with v_star=0")
        if not state_value_known and not include_uncertain:
            continue
        if state_reachability_known and not state_reachable and not include_unreachable:
            continue

        raw_actions = _normalize_actions(state_row.get("actions"), state_id=state_id)
        extra_actions = sorted(set(raw_actions) - available_set)
        if extra_actions:
            raise ValueError(f"Oracle state {state_id} labels unavailable tools: {extra_actions}")
        actions = [
            _compile_action(
                raw=raw_actions.get(tool_id),
                tool_id=tool_id,
                state=state,
                state_id=state_id,
                state_v=state_v,
                state_reachable=state_reachable,
                engine=engine,
                state_values=state_values,
                state_id_by_payload=state_id_by_payload,
                optimal_epsilon=optimal_epsilon,
                consistency_tolerance=consistency_tolerance,
            )
            for tool_id in available
        ]
        derived_optimal = sorted(
            action["tool_id"] for action in actions if action["is_optimal"] is True
        )
        supplied_optimal = state_row.get("optimal_actions")
        if supplied_optimal is not None:
            if not isinstance(supplied_optimal, list) or not all(
                isinstance(tool_id, str) for tool_id in supplied_optimal
            ):
                raise TypeError(f"State {state_id}.optimal_actions must be a list of strings")
            if sorted(set(supplied_optimal)) != derived_optimal:
                raise ValueError(
                    f"State {state_id}.optimal_actions={sorted(set(supplied_optimal))} "
                    f"disagrees with action regrets={derived_optimal}"
                )
        optimal_actions = derived_optimal
        top_paths = _normalize_top_paths(
            state_row.get("top_k_paths", []),
            state_id=state_id,
            available_tools=available_set,
        )

        if terminal:
            if top_paths and (top_paths[0]["tool_ids"] or top_paths[0]["total_cost"] > optimal_epsilon):
                raise ValueError(f"Terminal state {state_id} may only have an empty zero-cost path")
            gold_next_tool: str | None = STOP_TOOL_ID
            gold_suffix: list[str] = []
            next_state = copy.deepcopy(state)
            valid_next_tools = [STOP_TOOL_ID]
            training_mask = True
        elif state_value_known and state_reachable:
            if not optimal_actions:
                raise ValueError(f"Reachable nonterminal state {state_id} has no optimal action")
            if not top_paths:
                raise ValueError(
                    f"Reachable nonterminal state {state_id} needs at least one complete top-k path"
                )
            best_path = top_paths[0]
            if not best_path["tool_ids"]:
                raise ValueError(f"Nonterminal state {state_id} has an empty best path")
            if best_path["tool_ids"][0] not in optimal_actions:
                raise ValueError(
                    f"Best path at {state_id} starts with non-optimal action "
                    f"{best_path['tool_ids'][0]!r}"
                )
            if state_v is not None and not math.isclose(
                best_path["total_cost"], state_v, abs_tol=consistency_tolerance, rel_tol=0
            ):
                raise ValueError(
                    f"Best path cost at {state_id} is {best_path['total_cost']}, v_star={state_v}"
                )
            gold_next_tool = best_path["tool_ids"][0]
            gold_suffix = list(best_path["tool_ids"])
            selected_action = next(
                action for action in actions if action["tool_id"] == gold_next_tool
            )
            if not selected_action["known"] or selected_action["successor"] is None:
                raise ValueError(f"Best action {gold_next_tool!r} at {state_id} has no known successor")
            next_state = copy.deepcopy(selected_action["successor"])
            valid_next_tools = optimal_actions
            training_mask = True
        else:
            gold_next_tool = None
            gold_suffix = []
            next_state = copy.deepcopy(state)
            valid_next_tools = []
            training_mask = False

        contract_valid = engine.applicable_tools(state, available)
        action_regret = {
            action["tool_id"]: (
                action["regret"]
                if action["known"] and action["regret"] is not None
                else (INVALID_REGRET if action["outcome_status"] == "contract_invalid" else None)
            )
            for action in actions
        }
        action_regret_mask = {
            action["tool_id"]: bool(
                (action["known"] and action["regret"] is not None)
                or action["outcome_status"] == "contract_invalid"
            )
            for action in actions
        }
        phase = state_row.get("phase", state_row.get("depth", state_index) + 1)
        if not isinstance(phase, int) or isinstance(phase, bool) or phase < 0:
            raise ValueError(f"State {state_id}.phase/depth must yield a non-negative integer")
        example: dict[str, Any] = {
            "example_id": f"{task['task_id']}:search:{state_id}",
            "task_id": task["task_id"],
            "state_id": state_id,
            "split": selected_split,
            "phase": phase,
            "terminal": terminal,
            "training_mask": training_mask,
            "query": task["query"],
            "category": task.get("category", "genomics"),
            "goal": copy.deepcopy(task["goal"]),
            "state": state,
            "next_state": next_state,
            "history": history,
            "prefix_tool_ids": copy.deepcopy(state_row.get("prefix_tool_ids", [])),
            "gold_suffix_tool_ids": gold_suffix,
            "gold_next_tool": gold_next_tool,
            "valid_next_tools": valid_next_tools,
            "candidate_tools": list(available),
            "contract_valid_tools": contract_valid,
            "background_ids": copy.deepcopy(task.get("background_ids", [])),
            "value_star": state_v,
            "best_known_value": best_known_v,
            "value_known_mask": state_value_known,
            "goal_reachable": state_reachable_raw,
            "goal_reachability_known_mask": state_reachability_known,
            "optimal_actions": optimal_actions,
            "top_k_paths": top_paths,
            "action_supervision": actions,
            "action_q": {action["tool_id"]: action["q_star"] for action in actions},
            "action_cost": {action["tool_id"]: action["cost"] for action in actions},
            "action_successor_ids": {
                action["tool_id"]: action["successor_id"] for action in actions
            },
            "action_known_mask": {
                action["tool_id"]: action["label_mask"] for action in actions
            },
            "action_transition_known_mask": {
                action["tool_id"]: action["transition_known_mask"] for action in actions
            },
            "action_reachability_known_mask": {
                action["tool_id"]: action["reachability_known_mask"] for action in actions
            },
            "action_regret": action_regret,
            "action_regret_mask": action_regret_mask,
            "action_outcome_status": {
                action["tool_id"]: action["outcome_status"] for action in actions
            },
            "group_ids": copy.deepcopy(groups),
            "provenance": {
                **copy.deepcopy(task.get("provenance", {})),
                "search_supervision": {
                    "version": SEARCH_SUPERVISION_VERSION,
                    "task_sha256": sha256_text(canonical_json(task)),
                    "tools_sha256": tools_hash,
                    "snapshots_sha256": snapshots_hash,
                    "oracle_sha256": oracle_hash,
                    "metadata": safe_search_metadata,
                },
            },
        }
        if not isinstance(example["prefix_tool_ids"], list) or not all(
            tool_id in available_set for tool_id in example["prefix_tool_ids"]
        ):
            raise ValueError(f"State {state_id}.prefix_tool_ids contains unavailable tools")
        validate_model_input(example)
        examples.append(_with_record_hash(example))
    if not examples:
        raise ValueError(f"No compilable states remain for task {task['task_id']}")
    return sorted(examples, key=lambda row: row["example_id"])


def validate_search_example(row: Mapping[str, Any]) -> None:
    required = {
        "example_id",
        "task_id",
        "state_id",
        "query",
        "goal",
        "state",
        "next_state",
        "history",
        "candidate_tools",
        "contract_valid_tools",
        "valid_next_tools",
        "gold_next_tool",
        "gold_suffix_tool_ids",
        "action_supervision",
        "action_q",
        "action_cost",
        "action_successor_ids",
        "action_known_mask",
        "group_ids",
        "provenance",
        "record_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Search example is missing fields: {missing}")
    if row.get("split") not in {*SPLITS, None}:
        raise ValueError(f"Search example {row['example_id']} has invalid split={row.get('split')!r}")
    actions = row["action_supervision"]
    if not isinstance(actions, list):
        raise TypeError("action_supervision must be a list")
    by_tool: dict[str, Mapping[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping) or not isinstance(action.get("tool_id"), str):
            raise TypeError("Each action_supervision row needs a string tool_id")
        tool_id = str(action["tool_id"])
        if tool_id in by_tool:
            raise ValueError(f"Duplicate action_supervision tool_id={tool_id!r}")
        by_tool[tool_id] = action
        if action.get("known") is False:
            if action.get("label_mask") is not False:
                raise ValueError(f"Q-unknown action {tool_id!r} must have label_mask=false")
            for field in ("q_star", "regret"):
                if action.get(field) is not None:
                    raise ValueError(f"Q-unknown action {tool_id!r} must have {field}=null")
        if (
            action.get("transition_known") is False
            and action.get("outcome_status") == "unobserved"
        ):
            for field in (
                "executable",
                "successor",
                "successor_id",
                "cost",
                "is_optimal",
                "reachable",
            ):
                if action.get(field) is not None:
                    raise ValueError(
                        f"Transition-unknown action {tool_id!r} must have {field}=null"
                    )
    if list(by_tool) != list(row["candidate_tools"]):
        raise ValueError("action_supervision order/set must equal candidate_tools")
    for map_name, action_field in (
        ("action_q", "q_star"),
        ("action_cost", "cost"),
        ("action_successor_ids", "successor_id"),
        ("action_known_mask", "label_mask"),
    ):
        mapping = row[map_name]
        if not isinstance(mapping, Mapping) or set(mapping) != set(by_tool):
            raise ValueError(f"{map_name} must cover every candidate action exactly once")
        if any(mapping[tool_id] != by_tool[tool_id].get(action_field) for tool_id in by_tool):
            raise ValueError(f"{map_name} disagrees with action_supervision.{action_field}")
    validate_model_input(row)
    if row["record_sha256"] != _record_sha256(row):
        raise ValueError(f"Search example {row['example_id']} checksum mismatch")


def _task_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[str, ...]]]:
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        raw_groups = row.get("group_ids", {"case": [task_id]})
        if not isinstance(raw_groups, Mapping):
            raise TypeError(f"Task {task_id}.group_ids must be an object")
        groups = {
            str(kind): tuple(_normalize_group_values(value, label=str(kind)))
            for kind, value in raw_groups.items()
        }
        unknown = sorted(set(groups) - set(GROUP_KINDS))
        if unknown:
            raise ValueError(f"Task {task_id} has unknown group kinds: {unknown}")
        groups.setdefault("case", (task_id,))
        previous = output.setdefault(task_id, groups)
        if previous != groups:
            raise ValueError(f"Task {task_id} has inconsistent group IDs across state rows")
    return output


def audit_group_leakage(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_kinds: Sequence[str] = DEFAULT_AUDIT_GROUP_KINDS,
) -> dict[str, Any]:
    """Report task and requested group IDs that occur in more than one split."""

    unknown_kinds = sorted(set(group_kinds) - set(GROUP_KINDS))
    if unknown_kinds:
        raise ValueError(f"Unknown group kinds requested for audit: {unknown_kinds}")
    _task_groups(rows)  # also validates per-task metadata consistency
    task_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, dict[str, set[str]]] = {
        kind: defaultdict(set) for kind in group_kinds
    }
    missing_split_tasks: set[str] = set()
    for row in rows:
        task_id = str(row["task_id"])
        split = row.get("split")
        if split not in SPLITS:
            missing_split_tasks.add(task_id)
            continue
        task_splits[task_id].add(str(split))
        for kind in group_kinds:
            for group_id in row.get("group_ids", {}).get(kind, []):
                group_splits[kind][str(group_id)].add(str(split))
    task_conflicts = {
        task_id: sorted(splits) for task_id, splits in task_splits.items() if len(splits) > 1
    }
    overlaps = {
        kind: {
            group_id: sorted(splits)
            for group_id, splits in sorted(values.items())
            if len(splits) > 1
        }
        for kind, values in group_splits.items()
    }
    overlaps = {kind: values for kind, values in overlaps.items() if values}
    return {
        "ok": not task_conflicts and not overlaps and not missing_split_tasks,
        "audited_group_kinds": list(group_kinds),
        "missing_split_tasks": sorted(missing_split_tasks),
        "task_split_conflicts": task_conflicts,
        "group_split_overlaps": overlaps,
    }


def validate_preassigned_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_kinds: Sequence[str] = DEFAULT_AUDIT_GROUP_KINDS,
) -> dict[str, Any]:
    audit = audit_group_leakage(rows, group_kinds=group_kinds)
    if not audit["ok"]:
        raise ValueError(f"Preassigned split leakage/invalidity: {canonical_json(audit)}")
    return audit


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def task_balanced_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 17,
    ratios: Mapping[str, float] | None = None,
    group_kinds: Sequence[str] = ("case",),
) -> list[dict[str, Any]]:
    """Assign deterministic splits by task count while keeping groups intact.

    Prefix/state row counts never affect the target proportions.  Add ``entity``,
    ``template``, ``workflow``, ``source``, or ``release`` to ``group_kinds`` for a
    corresponding holdout protocol.  Connected tasks are assigned as one component.
    """

    if not rows:
        raise ValueError("Cannot split an empty search dataset")
    ratios = dict(ratios or {"train": 0.7, "dev": 0.15, "test": 0.15})
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must define exactly {SPLITS}")
    if any(isinstance(value, bool) or value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative")
    total_ratio = float(sum(ratios.values()))
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    ratios = {split: float(ratios[split]) / total_ratio for split in SPLITS}
    unknown_kinds = sorted(set(group_kinds) - set(GROUP_KINDS))
    if unknown_kinds:
        raise ValueError(f"Unknown group kinds requested for split: {unknown_kinds}")

    groups_by_task = _task_groups(rows)
    task_ids = sorted(groups_by_task)
    union_find = _UnionFind(task_ids)
    for kind in group_kinds:
        tasks_by_value: dict[str, list[str]] = defaultdict(list)
        for task_id, groups in groups_by_task.items():
            for value in groups.get(kind, ()):
                tasks_by_value[value].append(task_id)
        for related in tasks_by_value.values():
            for task_id in related[1:]:
                union_find.union(related[0], task_id)
    components: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        components[union_find.find(task_id)].append(task_id)
    ordered_components = sorted(
        (sorted(component) for component in components.values()),
        key=lambda component: (
            -len(component),
            sha256_text(f"{seed}:{canonical_json(component)}"),
        ),
    )
    target = {split: ratios[split] * len(task_ids) for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    split_by_task: dict[str, str] = {}
    for component in ordered_components:
        eligible = [split for split in SPLITS if ratios[split] > 0]
        chosen = min(
            eligible,
            key=lambda split: (
                (counts[split] + len(component) - target[split]) / max(target[split], 1.0),
                counts[split] / max(target[split], 1.0),
                sha256_text(f"{seed}:{component[0]}:{split}"),
            ),
        )
        for task_id in component:
            split_by_task[task_id] = chosen
        counts[chosen] += len(component)

    output: list[dict[str, Any]] = []
    for row in rows:
        updated = copy.deepcopy(dict(row))
        updated["split"] = split_by_task[str(row["task_id"])]
        output.append(_with_record_hash(updated))
    output.sort(key=lambda row: row["example_id"])
    validate_preassigned_splits(output, group_kinds=group_kinds)
    return output


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    return sha256_text(text)


def write_search_dataset(
    output_dir: str | Path,
    tasks: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    examples: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]] | None = None,
    *,
    background: Sequence[dict[str, Any]] | None = None,
    private_verifiers: Sequence[dict[str, Any]] | None = None,
    seed: int = 17,
    split_group_kinds: Sequence[str] = ("case",),
    oracle_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a deterministic processed directory consumable by the current cache."""

    if not examples:
        raise ValueError("Cannot write a search dataset without examples")
    for task in tasks:
        validate_task(task)
    for tool in tools:
        validate_tool(tool)
    for snapshot in snapshots or []:
        validate_snapshot(snapshot)
    for card in background or []:
        validate_background(card)
    for verifier in private_verifiers or []:
        validate_verifier(verifier)
    task_map = unique_by_id(list(tasks), "task_id", "task")
    tool_map = unique_by_id(list(tools), "tool_id", "tool")
    unique_by_id(list(snapshots or []), "snapshot_id", "snapshot")
    unique_by_id(list(background or []), "card_id", "background card")
    verifier_map = unique_by_id(
        list(private_verifiers or []), "task_id", "private verifier"
    )
    unknown_verifier_tasks = sorted(set(verifier_map) - set(task_map))
    if unknown_verifier_tasks:
        raise ValueError(
            f"Private verifiers reference unknown tasks: {unknown_verifier_tasks}"
        )
    sorted_examples = sorted((copy.deepcopy(row) for row in examples), key=lambda row: row["example_id"])
    if len({row["example_id"] for row in sorted_examples}) != len(sorted_examples):
        raise ValueError("Duplicate example_id in search dataset")
    for row in sorted_examples:
        validate_search_example(row)
        if row["task_id"] not in task_map:
            raise ValueError(f"Example references unknown task {row['task_id']!r}")
        unknown_tools = sorted(set(row["candidate_tools"]) - set(tool_map))
        if unknown_tools:
            raise ValueError(f"Example {row['example_id']} references unknown tools: {unknown_tools}")
    split_audit = validate_preassigned_splits(
        sorted_examples, group_kinds=split_group_kinds
    )

    split_by_task: dict[str, str] = {}
    for row in sorted_examples:
        split_by_task.setdefault(str(row["task_id"]), str(row["split"]))
    missing_tasks = sorted(set(task_map) - set(split_by_task))
    if missing_tasks:
        raise ValueError(f"Tasks have no compiled examples: {missing_tasks}")
    processed_tasks: list[dict[str, Any]] = []
    for task_id in sorted(task_map):
        task = copy.deepcopy(task_map[task_id])
        assigned = split_by_task[task_id]
        if task.get("split") not in {None, assigned}:
            raise ValueError(
                f"Task {task_id} split={task.get('split')!r} disagrees with examples={assigned!r}"
            )
        task["split"] = assigned
        processed_tasks.append(task)
    processed_tools = [copy.deepcopy(tool_map[tool_id]) for tool_id in sorted(tool_map)]
    processed_snapshots = sorted(
        (copy.deepcopy(row) for row in snapshots or []), key=lambda row: row["snapshot_id"]
    )
    processed_background = sorted(
        (copy.deepcopy(row) for row in background or []), key=lambda row: row["card_id"]
    )
    processed_verifiers = [copy.deepcopy(verifier_map[task_id]) for task_id in sorted(verifier_map)]
    metadata = copy.deepcopy(dict(oracle_metadata or {}))
    leaked_metadata = _forbidden_paths(metadata)
    if leaked_metadata:
        raise ValueError(f"oracle_metadata exposes evaluation-only fields: {leaked_metadata}")

    source_hashes = {
        "tasks": _canonical_rows_sha256(processed_tasks),
        "tools": _canonical_rows_sha256(processed_tools),
        "examples": _canonical_rows_sha256(sorted_examples),
        "snapshots": _canonical_rows_sha256(processed_snapshots),
        "background": _canonical_rows_sha256(processed_background),
        "private_verifiers": _canonical_rows_sha256(processed_verifiers),
        "oracle_metadata": sha256_text(canonical_json(metadata)),
    }
    dataset_content_sha256 = sha256_text(
        canonical_json(
            {
                "version": SEARCH_SUPERVISION_VERSION,
                "seed": seed,
                "split_group_kinds": list(split_group_kinds),
                "source_hashes": source_hashes,
            }
        )
    )
    output_dir = Path(output_dir)
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        existing = read_json(existing_manifest)
        if existing.get("dataset_content_sha256") != dataset_content_sha256:
            raise RuntimeError(
                "Refusing to overwrite an immutable search dataset with different content"
            )
        verify_processed_dataset(output_dir)
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to write into non-empty directory without manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_rows: dict[str, Sequence[dict[str, Any]]] = {
        "tools.jsonl": processed_tools,
        "tasks.jsonl": processed_tasks,
        SEARCH_EXAMPLES_FILE: sorted_examples,
    }
    if processed_snapshots:
        files_to_rows["snapshots.jsonl"] = processed_snapshots
    if processed_background:
        files_to_rows["background.jsonl"] = processed_background
    if processed_verifiers:
        files_to_rows["verifiers.private.jsonl"] = processed_verifiers
    for name, rows in files_to_rows.items():
        write_jsonl(output_dir / name, rows)

    split_task_ids = {
        split: sorted(task_id for task_id, value in split_by_task.items() if value == split)
        for split in SPLITS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "weighted_search_supervision",
        "search_supervision_version": SEARCH_SUPERVISION_VERSION,
        "dataset_content_sha256": dataset_content_sha256,
        "seed": seed,
        "split_protocol": {
            "unit": "task",
            "protected_group_kinds": list(split_group_kinds),
            "audit": split_audit,
        },
        "counts": {
            "tools": len(processed_tools),
            "tasks": len(processed_tasks),
            "examples": len(sorted_examples),
            "snapshots": len(processed_snapshots),
            "background": len(processed_background),
            "private_verifiers": len(processed_verifiers),
            "terminal_examples": sum(bool(row.get("terminal")) for row in sorted_examples),
            "reachable_examples": sum(bool(row.get("goal_reachable")) for row in sorted_examples),
            "known_action_labels": sum(
                sum(bool(value) for value in row["action_known_mask"].values())
                for row in sorted_examples
            ),
            "unknown_action_labels": sum(
                sum(not bool(value) for value in row["action_known_mask"].values())
                for row in sorted_examples
            ),
            "split_tasks": {split: len(task_ids) for split, task_ids in split_task_ids.items()},
            "split_examples": {
                split: sum(row["split"] == split for row in sorted_examples) for split in SPLITS
            },
        },
        "task_ids_by_split": split_task_ids,
        "task_groups_by_split": {
            split: sorted(
                {
                    group
                    for row in sorted_examples
                    if row["split"] == split
                    for group in row.get("group_ids", {}).get("case", [row["task_id"]])
                }
            )
            for split in SPLITS
        },
        "source_objects": source_hashes,
        "oracle_metadata": metadata,
        "processed_files": {
            name: sha256_file(output_dir / name) for name in sorted(files_to_rows)
        },
        "invariants": {
            "task_level_split": True,
            "declared_split_groups_kept_disjoint": True,
            "trajectory_file_required": False,
            "unknown_counterfactual_is_null": True,
            "unknown_counterfactual_loss_mask": False,
            "future_observation_in_context": False,
            "private_verifier_in_model_input": False,
            "private_verifier_stored_separately": bool(processed_verifiers),
            "weighted_q_bellman_checked_when_successor_known": True,
        },
    }
    write_json(existing_manifest, manifest)
    verify_processed_dataset(output_dir)
    return manifest


def read_search_dataset(root: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Read a persisted search dataset and verify checksums/row invariants."""

    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("search_supervision_version") != SEARCH_SUPERVISION_VERSION:
        raise ValueError(
            "Unsupported search supervision version: "
            f"{manifest.get('search_supervision_version')!r}"
        )
    if verify:
        verify_processed_dataset(root)
    examples = read_jsonl(root / SEARCH_EXAMPLES_FILE)
    if verify:
        for row in examples:
            validate_search_example(row)
        validate_preassigned_splits(
            examples,
            group_kinds=manifest.get("split_protocol", {}).get(
                "protected_group_kinds", ("case",)
            ),
        )
        actual_content = {
            "tasks": _canonical_rows_sha256(read_jsonl(root / "tasks.jsonl")),
            "tools": _canonical_rows_sha256(read_jsonl(root / "tools.jsonl")),
            "examples": _canonical_rows_sha256(examples),
            "snapshots": _canonical_rows_sha256(
                read_jsonl(root / "snapshots.jsonl")
                if (root / "snapshots.jsonl").exists()
                else []
            ),
            "background": _canonical_rows_sha256(
                read_jsonl(root / "background.jsonl")
                if (root / "background.jsonl").exists()
                else []
            ),
            "private_verifiers": _canonical_rows_sha256(
                read_jsonl(root / "verifiers.private.jsonl")
                if (root / "verifiers.private.jsonl").exists()
                else []
            ),
            "oracle_metadata": sha256_text(canonical_json(manifest.get("oracle_metadata", {}))),
        }
        if actual_content != manifest.get("source_objects"):
            raise ValueError("Search dataset canonical object hashes do not match manifest")
    return {
        "manifest": manifest,
        "tools": read_jsonl(root / "tools.jsonl"),
        "tasks": read_jsonl(root / "tasks.jsonl"),
        "examples": examples,
        "snapshots": (
            read_jsonl(root / "snapshots.jsonl") if (root / "snapshots.jsonl").exists() else []
        ),
        "background": (
            read_jsonl(root / "background.jsonl") if (root / "background.jsonl").exists() else []
        ),
        "private_verifiers": (
            read_jsonl(root / "verifiers.private.jsonl")
            if (root / "verifiers.private.jsonl").exists()
            else []
        ),
    }
