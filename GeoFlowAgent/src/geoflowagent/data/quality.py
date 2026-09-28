"""Fail-closed quality audit for raw and search-supervised datasets.

The audit is deliberately separate from model training.  It answers whether a
corpus is large/diverse enough for a declared protocol, whether group-held-out
splits are actually disjoint, and whether weighted-search labels are informative.
It does *not* turn missing counterfactual outcomes into negative labels.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from geoflowagent.data.contracts import ContractEngine
from geoflowagent.data.schema import (
    MODEL_FORBIDDEN_KEYS,
    validate_background,
    validate_minimal_pair,
    validate_snapshot,
    validate_task,
    validate_tool,
    validate_trajectory,
    validate_verifier,
)
from geoflowagent.utils.io import canonical_json, read_json, read_jsonl, sha256_file, sha256_text

QUALITY_REPORT_VERSION = "geoflowagent.dataset-quality.v1"
SPLITS = ("train", "dev", "test")

# Keep the quality gate aligned with the typed group IDs emitted by
# ``search_supervision.extract_group_ids``.  Configs should prefer the canonical
# names (case/entity/template/workflow/source/release), while the aliases retain
# compatibility with raw task provenance and older configs.
_CANONICAL_SPLIT_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "case": ("split_group",),
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
    "release": (
        "release_group",
        "release_id",
        "release",
        "source_revision",
    ),
}
_SPLIT_GROUP_KIND_BY_KEY = {
    alias: kind
    for kind, aliases in _CANONICAL_SPLIT_GROUP_ALIASES.items()
    for alias in aliases
}


@dataclass(frozen=True)
class DatasetQualityThresholds:
    """Acceptance criteria for :func:`audit_dataset_quality`.

    Defaults enforce integrity but remain permissive about scale.  A real study
    should explicitly raise the count/diversity and search-label thresholds in a
    checked-in experiment config.  Fractions are in ``[0, 1]``.
    """

    min_tasks: int = 1
    min_tasks_per_split: dict[str, int] = field(default_factory=dict)
    min_workflow_families: int = 0
    min_entities: int = 0
    min_sequence_contexts: int = 0
    min_templates: int = 0
    min_topologies: int = 0
    min_sources: int = 0
    min_source_releases: int = 0
    min_verifier_coverage: float = 0.0
    split_group_keys: tuple[str, ...] = ("entity_group",)
    require_complete_split_group_metadata: bool = False
    test_only_sources: tuple[str, ...] = ()
    require_assigned_splits: bool = True
    fail_on_duplicate_content: bool = True
    fail_on_private_label_leakage: bool = True
    require_snapshot_source_revisions: bool = True
    require_search_supervision: bool = False
    min_known_applicable_action_fraction: float = 0.0
    max_unknown_applicable_action_fraction: float = 1.0
    min_branch_state_fraction: float = 0.0
    min_branch_label_coverage: float = 0.0
    min_nontrivial_regret_fraction: float = 0.0
    min_multi_path_state_fraction: float = 0.0
    max_oracle_contract_collapse_fraction: float = 1.0
    min_search_complete_fraction: float = 0.0
    max_incomplete_or_unknown_state_fraction: float = 1.0
    allow_synthetic: bool = True

    def __post_init__(self) -> None:
        count_fields = (
            "min_tasks",
            "min_workflow_families",
            "min_entities",
            "min_sequence_contexts",
            "min_templates",
            "min_topologies",
            "min_sources",
            "min_source_releases",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if set(self.min_tasks_per_split) - set(SPLITS):
            raise ValueError(f"min_tasks_per_split keys must be among {SPLITS}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.min_tasks_per_split.values()
        ):
            raise ValueError("min_tasks_per_split values must be non-negative integers")
        fraction_fields = (
            "min_verifier_coverage",
            "min_known_applicable_action_fraction",
            "max_unknown_applicable_action_fraction",
            "min_branch_state_fraction",
            "min_branch_label_coverage",
            "min_nontrivial_regret_fraction",
            "min_multi_path_state_fraction",
            "max_oracle_contract_collapse_fraction",
            "min_search_complete_fraction",
            "max_incomplete_or_unknown_state_fraction",
        )
        for name in fraction_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be a finite fraction in [0, 1]")
        if not all(isinstance(key, str) and key for key in self.split_group_keys):
            raise ValueError("split_group_keys must contain non-empty strings")
        if not isinstance(self.require_complete_split_group_metadata, bool):
            raise TypeError("require_complete_split_group_metadata must be boolean")
        if not all(isinstance(source, str) and source.strip() for source in self.test_only_sources):
            raise ValueError("test_only_sources must contain non-empty strings")


class DatasetQualityError(ValueError):
    """Raised when a completed quality report does not pass its thresholds."""


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _json_safe_thresholds(thresholds: DatasetQualityThresholds) -> dict[str, Any]:
    value = asdict(thresholds)
    value["split_group_keys"] = list(thresholds.split_group_keys)
    value["test_only_sources"] = list(thresholds.test_only_sources)
    return value


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "message": message}
    if details:
        row["details"] = details
    return row


def _read_optional_jsonl(
    root: Path,
    name: str,
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = root / name
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except Exception as exc:  # audit reports malformed inputs instead of crashing
        failures.append(_issue("invalid_jsonl", f"Could not read {path}: {exc}", file=name))
        return []


def _validate_rows(
    rows: Sequence[dict[str, Any]],
    validator: Any,
    kind: str,
    failures: list[dict[str, Any]],
) -> None:
    for index, row in enumerate(rows):
        try:
            validator(row)
        except Exception as exc:
            failures.append(
                _issue(
                    "schema_invalid",
                    f"Invalid {kind} row {index}: {exc}",
                    kind=kind,
                    row_index=index,
                )
            )


def _duplicates(rows: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if field_name in row:
            positions[str(row[field_name])].append(index)
    return {key: values for key, values in sorted(positions.items()) if len(values) > 1}


def _nested_values(row: Mapping[str, Any], key: str) -> list[Any]:
    containers: list[Mapping[str, Any]] = [row]
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        containers.append(provenance)
        metadata = provenance.get("metadata")
        if isinstance(metadata, Mapping):
            containers.append(metadata)
    values: list[Any] = []
    for container in containers:
        if key in container:
            value = container[key]
            if isinstance(value, (list, tuple, set)):
                values.extend(value)
            else:
                values.append(value)
    return values


def _text_values(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[str]:
    output: set[str] = set()
    for row in rows:
        for key in keys:
            for value in _nested_values(row, key):
                if isinstance(value, Mapping):
                    output.add(canonical_json(value))
                elif value is not None and not isinstance(value, bool):
                    text = str(value).strip()
                    if text:
                        output.add(text)
    return sorted(output)


def _task_split_map(
    tasks: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    observed: dict[str, set[str]] = defaultdict(set)
    all_task_ids = {str(row.get("task_id")) for row in tasks if row.get("task_id") is not None}
    for row in tasks:
        if row.get("task_id") is not None and row.get("split") in SPLITS:
            observed[str(row["task_id"])].add(str(row["split"]))
    for row in examples:
        if row.get("task_id") is not None:
            all_task_ids.add(str(row["task_id"]))
            if row.get("split") in SPLITS:
                observed[str(row["task_id"])].add(str(row["split"]))
    conflicts = {
        task_id: sorted(splits) for task_id, splits in sorted(observed.items()) if len(splits) > 1
    }
    resolved = {
        task_id: next(iter(splits)) for task_id, splits in observed.items() if len(splits) == 1
    }
    missing = sorted(all_task_ids - set(resolved) - set(conflicts))
    return resolved, conflicts, missing


def _test_only_source_violations(
    tasks: Sequence[Mapping[str, Any]],
    split_by_task: Mapping[str, str],
    protected_sources: Sequence[str],
) -> list[dict[str, str]]:
    """Find protected external-benchmark rows outside the test split.

    Dataset adapters do not all use the same provenance key, so the guard checks
    the small set of established aliases. Matching is exact after trimming and
    case-folding; a substring match would make similarly named internal sources
    fail unexpectedly.
    """

    protected = {str(value).strip().casefold() for value in protected_sources}
    if not protected:
        return []
    violations: list[dict[str, str]] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        sources = {
            str(value).strip()
            for key in ("dataset_id", "source", "source_id", "dataset")
            for value in _nested_values(task, key)
            if value is not None and str(value).strip()
        }
        matched = sorted(value for value in sources if value.casefold() in protected)
        split = split_by_task.get(task_id)
        if matched and split != "test":
            violations.extend(
                {"task_id": task_id, "source": source, "split": split or "unassigned"}
                for source in matched
            )
    return violations


def _group_value(task: Mapping[str, Any], key: str) -> list[str]:
    """Resolve one configured group using the search pipeline's canonical mapping.

    Explicit ``group_ids`` take precedence, including an explicitly empty value.
    Raw tasks are then resolved from their top-level/provenance aliases.  ``case``
    deliberately requires ``group_ids.case``, ``case_group``, or ``split_group``:
    silently substituting a unique task ID would make a strict case-family split
    audit pass without protecting related cases from leakage.
    """

    canonical_key = _SPLIT_GROUP_KIND_BY_KEY.get(key, key)
    group_ids = task.get("group_ids")
    if isinstance(group_ids, Mapping):
        value = group_ids.get(canonical_key)
        if value is None and key != canonical_key:
            value = group_ids.get(key)
        if value is not None:
            values = value if isinstance(value, (list, tuple, set)) else [value]
            return sorted(
                {
                    str(item).strip()
                    for item in values
                    if item is not None and str(item).strip()
                }
            )

    aliases = _CANONICAL_SPLIT_GROUP_ALIASES.get(canonical_key, (key,))
    for alias in aliases:
        values = sorted(
            {
                str(value).strip()
                for value in _nested_values(task, alias)
                if value is not None and str(value).strip()
            }
        )
        if values:
            return values
    return []


def _split_leakage(
    tasks: Sequence[Mapping[str, Any]],
    split_by_task: Mapping[str, str],
    group_keys: Sequence[str],
    *,
    require_complete_metadata: bool = False,
) -> dict[str, Any]:
    overlaps: dict[str, dict[str, list[str]]] = {}
    missing: dict[str, list[str]] = {}
    for key in group_keys:
        split_sets: dict[str, set[str]] = defaultdict(set)
        missing_ids: list[str] = []
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            values = _group_value(task, key)
            if not values:
                missing_ids.append(task_id)
                continue
            split = split_by_task.get(task_id)
            if split is not None:
                for value in values:
                    split_sets[value].add(split)
        found = {
            value: sorted(splits)
            for value, splits in sorted(split_sets.items())
            if len(splits) > 1
        }
        if found:
            overlaps[key] = found
        if missing_ids:
            missing[key] = sorted(missing_ids)
    return {
        "audited_group_keys": list(group_keys),
        "require_complete_metadata": require_complete_metadata,
        "overlaps": overlaps,
        "missing_group_ids": missing,
        "ok": not overlaps and (not require_complete_metadata or not missing),
    }


def _content_duplicate_groups(
    rows: Sequence[Mapping[str, Any]],
    split_by_task: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        payload = {
            key: copy.deepcopy(row.get(key))
            for key in ("query", "initial_state", "goal", "available_tools", "background_ids")
        }
        digest = sha256_text(canonical_json(payload))
        by_hash[digest].append(str(row.get("task_id", "")))
    output: list[dict[str, Any]] = []
    for digest, task_ids in sorted(by_hash.items()):
        distinct_ids = sorted(set(task_ids))
        splits = sorted({split_by_task[item] for item in distinct_ids if item in split_by_task})
        if len(distinct_ids) > 1 and len(splits) > 1:
            output.append({"sha256": digest, "task_ids": distinct_ids, "splits": splits})
    return output


def _forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    aliases = set(MODEL_FORBIDDEN_KEYS) | {
        "answer_key",
        "expected",
        "private_goal",
        "verifier_answer",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in aliases:
                paths.append(path)
            paths.extend(_forbidden_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, f"{prefix}[{index}]"))
    return paths


def _private_literals(verifier: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()

    def visit_label_value(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit_label_value(child)
        elif isinstance(value, list):
            for child in value:
                visit_label_value(child)
        elif isinstance(value, str) and len(value.strip()) >= 4:
            output.add(value.strip().casefold())

    private = verifier.get("private_verifier", {})
    if not isinstance(private, Mapping):
        return output
    conditions = private.get("conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, Mapping) and "value" in condition:
                # Scan hidden target values, never the structural words in
                # ``field``/``op`` (e.g. "gene_symbol" or "contains").
                visit_label_value(condition["value"])
    else:
        # Compact goal form: every mapping value is a target.
        for value in private.values():
            visit_label_value(value)
    return output


def _leakage_audit(
    tasks: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    verifiers: Sequence[Mapping[str, Any]],
    forbidden_literals: Iterable[str],
) -> dict[str, Any]:
    structural: list[dict[str, Any]] = []
    for row in tasks:
        visible = {
            key: row.get(key)
            for key in ("query", "initial_state", "goal", "available_tools", "background_ids")
        }
        paths = _forbidden_key_paths(visible)
        if paths:
            structural.append({"id": str(row.get("task_id")), "paths": sorted(paths)})
    for row in examples:
        visible = {
            key: row.get(key)
            for key in ("query", "goal", "state", "history", "candidate_tools", "background_ids")
        }
        paths = _forbidden_key_paths(visible)
        if paths:
            structural.append({"id": str(row.get("example_id")), "paths": sorted(paths)})

    verifier_by_task = {
        str(row.get("task_id")): _private_literals(row) for row in verifiers
    }
    explicit = {str(value).strip().casefold() for value in forbidden_literals if str(value).strip()}
    literal_hits: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        # Private truth appearing in the *initial* request is leakage.  The same
        # value in a later state may be a legitimate observed tool result.
        initial_visible = {
            key: task.get(key) for key in ("query", "initial_state", "goal")
        }
        haystack = canonical_json(initial_visible).casefold()
        matched_private = sorted(value for value in verifier_by_task.get(task_id, set()) if value in haystack)
        matched_explicit = sorted(value for value in explicit if value in haystack)
        if matched_private or matched_explicit:
            literal_hits.append(
                {
                    "task_id": task_id,
                    "private_literals": matched_private,
                    "forbidden_literals": matched_explicit,
                }
            )
    return {
        "structural_key_hits": structural,
        "initial_private_literal_hits": literal_hits,
        "ok": not structural and not literal_hits,
        "note": "Post-tool state values are not literal-scanned because legitimate observations may equal private truth.",
    }


def _as_actions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("action_supervision")
    if isinstance(raw, list):
        return [dict(action) for action in raw if isinstance(action, Mapping)]
    if isinstance(raw, Mapping):
        output: list[dict[str, Any]] = []
        for tool_id, value in raw.items():
            if isinstance(value, Mapping):
                action = dict(value)
                action.setdefault("tool_id", str(tool_id))
                output.append(action)
        return output

    candidates = row.get("candidate_tools", [])
    if not isinstance(candidates, list):
        return []
    masks = row.get("action_known_mask", row.get("action_regret_mask", {}))
    q_values = row.get("action_q", {})
    regrets = row.get("action_regret", {})
    statuses = row.get("action_outcome_status", {})
    contract_valid = set(row.get("contract_valid_tools", []))
    optimal = set(row.get("optimal_actions", row.get("valid_next_tools", [])))
    actions: list[dict[str, Any]] = []
    for tool_id in candidates:
        status = statuses.get(tool_id) if isinstance(statuses, Mapping) else None
        applicable = tool_id in contract_valid if "contract_valid_tools" in row else status != "contract_invalid"
        known = bool(masks.get(tool_id, False)) if isinstance(masks, Mapping) else False
        actions.append(
            {
                "tool_id": tool_id,
                "known": known,
                "label_mask": known,
                "contract_applicable": applicable,
                "q_star": q_values.get(tool_id) if isinstance(q_values, Mapping) else None,
                "regret": regrets.get(tool_id) if isinstance(regrets, Mapping) else None,
                "is_optimal": tool_id in optimal if known else None,
                "reachable": None,
                "reachability_known": False,
                "outcome_status": status,
            }
        )
    return actions


def _complete_flag(row: Mapping[str, Any]) -> bool | None:
    candidates: list[Any] = [
        row.get("graph_complete"),
        row.get("search_complete"),
    ]
    for key in ("search_diagnostics", "oracle_diagnostics", "diagnostics"):
        value = row.get(key)
        if isinstance(value, Mapping):
            candidates.append(value.get("graph_complete"))
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        search = provenance.get("search_supervision")
        if isinstance(search, Mapping):
            metadata = search.get("metadata")
            if isinstance(metadata, Mapping):
                diagnostics = metadata.get("diagnostics")
                if isinstance(diagnostics, Mapping):
                    candidates.append(diagnostics.get("graph_complete"))
    for value in candidates:
        if isinstance(value, bool):
            return value
    return None


def summarize_search_supervision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize search labels while preserving unknown/negative separation."""

    search_rows = [
        row
        for row in rows
        if any(
            key in row
            for key in ("action_supervision", "action_known_mask", "action_regret_mask")
        )
    ]
    total_actions = 0
    applicable_actions = 0
    known_applicable_actions = 0
    unknown_applicable_actions = 0
    known_unreachable_applicable_actions = 0
    nonterminal_states = 0
    eligible_branch_states = 0
    label_covered_branch_states = 0
    regret_evaluable_states = 0
    nontrivial_regret_states = 0
    path_labeled_states = 0
    multi_path_states = 0
    total_path_references = 0
    collapse_evaluable_states = 0
    oracle_contract_collapse_states = 0
    label_complete_states = 0
    metadata_complete = 0
    metadata_incomplete = 0
    metadata_unknown = 0

    for row in search_rows:
        actions = _as_actions(row)
        total_actions += len(actions)
        terminal = bool(row.get("terminal"))
        # Once STOP is correct, real tools are outside the policy decision and
        # search intentionally does not expand them. Do not call those branches
        # "unknown" in label-coverage statistics.
        applicable = (
            []
            if terminal
            else [action for action in actions if action.get("contract_applicable") is True]
        )
        # "Known action" here means the exact executor/search established at
        # least transition or reachability status. A certified dead end has no
        # finite Q* target, but it is known unreachable—not an unknown action.
        known_applicable = [
            action
            for action in applicable
            if any(
                action.get(key) is True
                for key in (
                    "label_mask",
                    "known",
                    "transition_known_mask",
                    "transition_known",
                    "reachability_known_mask",
                    "reachability_known",
                )
            )
        ]
        known_action_ids = {id(action) for action in known_applicable}
        unknown_applicable = [
            action for action in applicable if id(action) not in known_action_ids
        ]
        applicable_actions += len(applicable)
        known_applicable_actions += len(known_applicable)
        unknown_applicable_actions += len(unknown_applicable)
        for action in known_applicable:
            reachability_known = action.get(
                "reachability_known_mask", action.get("reachability_known")
            )
            known_unreachable = action.get("reachable") is False and (
                reachability_known is True
                or (reachability_known is None and action.get("known") is True)
            )
            if known_unreachable:
                known_unreachable_applicable_actions += 1
        if not unknown_applicable:
            label_complete_states += 1

        complete = _complete_flag(row)
        if complete is True:
            metadata_complete += 1
        elif complete is False:
            metadata_incomplete += 1
        else:
            metadata_unknown += 1

        if terminal:
            continue
        nonterminal_states += 1
        top_paths = row.get("top_k_paths", [])
        if isinstance(top_paths, list) and top_paths:
            path_labeled_states += 1
            total_path_references += len(top_paths)
            if len(top_paths) >= 2:
                multi_path_states += 1
        if len(applicable) >= 2:
            eligible_branch_states += 1
            if len(known_applicable) >= 2:
                label_covered_branch_states += 1

        numeric_regrets = [
            float(action["regret"])
            for action in known_applicable
            if isinstance(action.get("regret"), (int, float))
            and not isinstance(action.get("regret"), bool)
            and math.isfinite(float(action["regret"]))
        ]
        if len(numeric_regrets) >= 2:
            regret_evaluable_states += 1
            if max(numeric_regrets) - min(numeric_regrets) > 1e-9:
                nontrivial_regret_states += 1

        if applicable and not unknown_applicable:
            supplied_optimal = {
                str(action.get("tool_id"))
                for action in known_applicable
                if action.get("is_optimal") is True
            }
            if supplied_optimal:
                collapse_evaluable_states += 1
                if supplied_optimal == {
                    str(action.get("tool_id")) for action in applicable
                }:
                    oracle_contract_collapse_states += 1

    known_fraction = _ratio(known_applicable_actions, applicable_actions)
    unknown_fraction = _ratio(unknown_applicable_actions, applicable_actions)
    branch_fraction = _ratio(eligible_branch_states, nonterminal_states)
    branch_label_coverage = _ratio(label_covered_branch_states, eligible_branch_states)
    regret_fraction = _ratio(nontrivial_regret_states, regret_evaluable_states)
    multi_path_fraction = _ratio(multi_path_states, path_labeled_states)
    collapse_fraction = _ratio(oracle_contract_collapse_states, collapse_evaluable_states)
    metadata_denominator = metadata_complete + metadata_incomplete
    complete_fraction = _ratio(metadata_complete, metadata_denominator)
    incomplete_or_unknown_fraction = _ratio(
        metadata_incomplete + metadata_unknown, len(search_rows)
    )
    return {
        "present": bool(search_rows),
        "states": len(search_rows),
        "nonterminal_states": nonterminal_states,
        "total_actions": total_actions,
        "applicable_actions": applicable_actions,
        "known_applicable_actions": known_applicable_actions,
        "unknown_applicable_actions": unknown_applicable_actions,
        "known_unreachable_applicable_actions": known_unreachable_applicable_actions,
        "known_applicable_action_fraction": known_fraction,
        "unknown_applicable_action_fraction": unknown_fraction,
        "eligible_branch_states": eligible_branch_states,
        "label_covered_branch_states": label_covered_branch_states,
        "branch_state_fraction": branch_fraction,
        "branch_label_coverage": branch_label_coverage,
        "regret_evaluable_states": regret_evaluable_states,
        "nontrivial_regret_states": nontrivial_regret_states,
        "nontrivial_regret_fraction": regret_fraction,
        "path_labeled_states": path_labeled_states,
        "multi_path_states": multi_path_states,
        "total_path_references": total_path_references,
        "multi_path_state_fraction": multi_path_fraction,
        "collapse_evaluable_states": collapse_evaluable_states,
        "oracle_contract_collapse_states": oracle_contract_collapse_states,
        "oracle_contract_collapse_fraction": collapse_fraction,
        "label_complete_states": label_complete_states,
        "label_complete_state_fraction": _ratio(label_complete_states, len(search_rows)),
        "search_complete_metadata": {
            "complete": metadata_complete,
            "incomplete": metadata_incomplete,
            "unknown": metadata_unknown,
            "complete_fraction_among_declared": complete_fraction,
            "incomplete_or_unknown_state_fraction": incomplete_or_unknown_fraction,
        },
        "unknown_is_not_negative": True,
    }


def _load_forbidden_literals(
    denylist_path: str | Path | None,
    supplied: Iterable[str] | None,
    failures: list[dict[str, Any]],
) -> list[str]:
    values = list(supplied or [])
    if denylist_path is None:
        return sorted({str(value) for value in values if str(value)})
    try:
        raw = read_json(denylist_path)
        if isinstance(raw, list):
            values.extend(raw)
        elif isinstance(raw, Mapping):
            entries = raw.get("forbidden_literals", [])
            if not isinstance(entries, list):
                raise TypeError("forbidden_literals must be a list")
            values.extend(entries)
        else:
            raise TypeError("denylist must be a list or object")
    except Exception as exc:
        failures.append(_issue("invalid_denylist", f"Could not read denylist: {exc}"))
    return sorted({str(value) for value in values if str(value)})


def audit_dataset_quality(
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
    *,
    thresholds: DatasetQualityThresholds | None = None,
    forbidden_literals: Iterable[str] | None = None,
    denylist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit raw contracts and optional processed/search-supervision examples.

    The function always returns a JSON-safe report, including malformed-input
    failures.  Use :func:`require_dataset_quality` to turn a failed report into an
    exception before embedding, training, or evaluation.
    """

    thresholds = thresholds or DatasetQualityThresholds()
    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir) if processed_dir is not None else None
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not raw_root.exists():
        failures.append(_issue("raw_dir_missing", f"Raw directory does not exist: {raw_root}"))
    raw_names = {
        "tools": "tools.jsonl",
        "tasks": "tasks.jsonl",
        "trajectories": "trajectories.jsonl",
        "minimal_pairs": "minimal_pairs.jsonl",
        "background": "background.jsonl",
        "verifiers": "verifiers.private.jsonl",
        "snapshots": "snapshots.jsonl",
    }
    raw_rows = {
        key: _read_optional_jsonl(raw_root, name, failures)
        for key, name in raw_names.items()
    }
    for required_key in ("tools", "tasks"):
        if not (raw_root / raw_names[required_key]).exists():
            failures.append(
                _issue("required_file_missing", f"Missing {raw_names[required_key]}")
            )
    validators = {
        "tools": validate_tool,
        "tasks": validate_task,
        "trajectories": validate_trajectory,
        "minimal_pairs": validate_minimal_pair,
        "background": validate_background,
        "verifiers": validate_verifier,
        "snapshots": validate_snapshot,
    }
    for kind, validator in validators.items():
        _validate_rows(raw_rows[kind], validator, kind, failures)

    id_fields = {
        "tools": "tool_id",
        "tasks": "task_id",
        "trajectories": "task_id",
        "minimal_pairs": "pair_id",
        "background": "card_id",
        "verifiers": "task_id",
        "snapshots": "snapshot_id",
    }
    duplicate_ids: dict[str, dict[str, list[int]]] = {}
    for kind, field_name in id_fields.items():
        found = _duplicates(raw_rows[kind], field_name)
        if found:
            duplicate_ids[kind] = found
            failures.append(
                _issue("duplicate_ids", f"Duplicate {kind} IDs found", kind=kind, values=found)
            )

    raw_tool_ids = {
        str(row.get("tool_id")) for row in raw_rows["tools"] if row.get("tool_id") is not None
    }
    raw_task_ids = {
        str(row.get("task_id")) for row in raw_rows["tasks"] if row.get("task_id") is not None
    }
    raw_background_ids = {
        str(row.get("card_id"))
        for row in raw_rows["background"]
        if row.get("card_id") is not None
    }
    reference_errors: list[dict[str, Any]] = []
    for task in raw_rows["tasks"]:
        task_id = str(task.get("task_id", ""))
        available = task.get("available_tools", [])
        if isinstance(available, list):
            unknown = sorted(str(value) for value in available if str(value) not in raw_tool_ids)
            if unknown:
                reference_errors.append(
                    {"kind": "task_tools", "task_id": task_id, "unknown": unknown}
                )
        background = task.get("background_ids", [])
        if isinstance(background, list):
            unknown = sorted(
                str(value) for value in background if str(value) not in raw_background_ids
            )
            if unknown:
                reference_errors.append(
                    {"kind": "task_background", "task_id": task_id, "unknown": unknown}
                )
    trajectory_task_ids: set[str] = set()
    task_by_id = {
        str(row["task_id"]): row for row in raw_rows["tasks"] if "task_id" in row
    }
    for trajectory in raw_rows["trajectories"]:
        task_id = str(trajectory.get("task_id", ""))
        trajectory_task_ids.add(task_id)
        if task_id not in raw_task_ids:
            reference_errors.append(
                {"kind": "trajectory_task", "task_id": task_id, "unknown": [task_id]}
            )
            continue
        steps = trajectory.get("tool_ids", [])
        if isinstance(steps, list):
            available = set(task_by_id[task_id].get("available_tools", []))
            unknown = sorted(str(value) for value in steps if value not in available)
            if unknown:
                reference_errors.append(
                    {"kind": "trajectory_tools", "task_id": task_id, "unknown": unknown}
                )
    if raw_rows["trajectories"]:
        missing_trajectories = sorted(raw_task_ids - trajectory_task_ids)
        if missing_trajectories:
            reference_errors.append(
                {
                    "kind": "missing_trajectories",
                    "task_ids": missing_trajectories,
                }
            )
    unknown_snapshot_tools = sorted(
        {
            str(row.get("tool_id"))
            for row in raw_rows["snapshots"]
            if row.get("tool_id") is not None
        }
        - raw_tool_ids
    )
    if unknown_snapshot_tools:
        reference_errors.append(
            {"kind": "snapshot_tools", "unknown": unknown_snapshot_tools}
        )
    if reference_errors:
        failures.append(
            _issue(
                "reference_integrity",
                "Raw rows contain missing or inconsistent references",
                errors=reference_errors,
            )
        )
    if raw_rows["tools"] and not duplicate_ids.get("tools") and not duplicate_ids.get("snapshots"):
        try:
            ContractEngine(raw_rows["tools"], raw_rows["snapshots"])
        except Exception as exc:
            failures.append(
                _issue(
                    "contract_snapshot_integrity",
                    f"Tools/snapshots do not form an exact executable contract: {exc}",
                )
            )

    processed_rows: dict[str, list[dict[str, Any]]] = {
        "tasks": [],
        "tools": [],
        "examples": [],
        "snapshots": [],
        "verifiers": [],
    }
    if processed_root is not None:
        if not processed_root.exists():
            failures.append(
                _issue("processed_dir_missing", f"Processed directory does not exist: {processed_root}")
            )
        else:
            for key, name in (
                ("tasks", "tasks.jsonl"),
                ("tools", "tools.jsonl"),
                ("examples", "examples.jsonl"),
                ("snapshots", "snapshots.jsonl"),
                ("verifiers", "verifiers.private.jsonl"),
            ):
                processed_rows[key] = _read_optional_jsonl(processed_root, name, failures)
            manifest_path = processed_root / "manifest.json"
            if not manifest_path.exists():
                failures.append(_issue("processed_manifest_missing", "Processed manifest is missing"))
            else:
                try:
                    value = read_json(manifest_path)
                    if not isinstance(value, dict):
                        raise TypeError("manifest must be an object")
                    files = value.get("processed_files")
                    if not isinstance(files, Mapping) or not files:
                        failures.append(
                            _issue("processed_manifest_unbound", "Manifest has no processed_files")
                        )
                    else:
                        for name, expected in files.items():
                            path = processed_root / str(name)
                            if not path.exists() or sha256_file(path) != expected:
                                failures.append(
                                    _issue(
                                        "processed_checksum_mismatch",
                                        f"Processed checksum mismatch: {name}",
                                    )
                                )
                except Exception as exc:
                    failures.append(
                        _issue("processed_manifest_invalid", f"Could not validate manifest: {exc}")
                    )
            _validate_rows(processed_rows["tasks"], validate_task, "processed task", failures)
            _validate_rows(processed_rows["tools"], validate_tool, "processed tool", failures)
            _validate_rows(
                processed_rows["snapshots"], validate_snapshot, "processed snapshot", failures
            )
            for index, row in enumerate(processed_rows["examples"]):
                try:
                    if "action_supervision" in row:
                        from geoflowagent.data.search_supervision import validate_search_example

                        validate_search_example(row)
                    else:
                        schema_path = Path(__file__).resolve().parents[3] / "data" / "schemas" / "processed_example.schema.json"
                        if schema_path.exists():
                            schema = read_json(schema_path)
                            errors = sorted(
                                Draft202012Validator(schema).iter_errors(row),
                                key=lambda error: list(error.absolute_path),
                            )
                            if errors:
                                raise ValueError(errors[0].message)
                except Exception as exc:
                    failures.append(
                        _issue(
                            "schema_invalid",
                            f"Invalid processed example row {index}: {exc}",
                            kind="processed example",
                            row_index=index,
                        )
                    )

            processed_example_duplicates = _duplicates(
                processed_rows["examples"], "example_id"
            )
            if processed_example_duplicates:
                duplicate_ids["processed_examples"] = processed_example_duplicates
                failures.append(
                    _issue(
                        "duplicate_ids",
                        "Duplicate processed example IDs found",
                        kind="processed_examples",
                        values=processed_example_duplicates,
                    )
                )
            processed_task_ids = {
                str(row.get("task_id"))
                for row in processed_rows["tasks"]
                if row.get("task_id") is not None
            }
            processed_tool_ids = {
                str(row.get("tool_id"))
                for row in processed_rows["tools"]
                if row.get("tool_id") is not None
            }
            processed_reference_errors: list[dict[str, Any]] = []
            for row in processed_rows["examples"]:
                example_id = str(row.get("example_id", ""))
                if str(row.get("task_id", "")) not in processed_task_ids:
                    processed_reference_errors.append(
                        {"example_id": example_id, "kind": "task"}
                    )
                candidates = row.get("candidate_tools", [])
                if isinstance(candidates, list):
                    unknown = sorted(
                        str(value) for value in candidates if str(value) not in processed_tool_ids
                    )
                    if unknown:
                        processed_reference_errors.append(
                            {
                                "example_id": example_id,
                                "kind": "candidate_tools",
                                "unknown": unknown,
                            }
                        )
            if processed_reference_errors:
                failures.append(
                    _issue(
                        "processed_reference_integrity",
                        "Processed examples reference missing tasks or tools",
                        errors=processed_reference_errors,
                    )
                )

    tasks = processed_rows["tasks"] or raw_rows["tasks"]
    examples = processed_rows["examples"]
    verifiers = raw_rows["verifiers"] or processed_rows["verifiers"]
    snapshots = processed_rows["snapshots"] or raw_rows["snapshots"]
    task_ids = {str(row.get("task_id")) for row in tasks if row.get("task_id") is not None}
    verifier_ids = {
        str(row.get("task_id")) for row in verifiers if row.get("task_id") is not None
    }
    unknown_verifiers = sorted(verifier_ids - task_ids)
    if unknown_verifiers:
        failures.append(
            _issue(
                "unknown_verifier_tasks",
                "Private verifiers reference unknown tasks",
                task_ids=unknown_verifiers,
            )
        )
    verifier_coverage = _ratio(len(task_ids & verifier_ids), len(task_ids)) or 0.0

    split_by_task, split_conflicts, missing_splits = _task_split_map(tasks, examples)
    if split_conflicts:
        failures.append(
            _issue("task_split_conflict", "A task appears in multiple splits", values=split_conflicts)
        )
    if missing_splits and thresholds.require_assigned_splits:
        failures.append(
            _issue("unassigned_tasks", "Tasks are missing split assignments", task_ids=missing_splits)
        )
    split_counts = {
        split: sum(value == split for value in split_by_task.values()) for split in SPLITS
    }
    test_only_source_violations = _test_only_source_violations(
        tasks, split_by_task, thresholds.test_only_sources
    )
    if test_only_source_violations:
        failures.append(
            _issue(
                "test_only_source_leakage",
                "A protected external-test source occurs outside the test split",
                rows=test_only_source_violations,
            )
        )
    # Split-group protection is an inventory property, not a model-input
    # property.  A sealed dev run may deliberately materialize only train/dev
    # processed tasks; auditing that subset would silently skip malformed or
    # leaking raw test tasks.  Always inspect the complete raw task inventory
    # when it is available, while allowing processed-only callers as a fallback.
    split_group_inventory = raw_rows["tasks"] or tasks
    inventory_split_by_task, _, _ = _task_split_map(split_group_inventory, examples)
    split_audit = _split_leakage(
        split_group_inventory,
        inventory_split_by_task,
        thresholds.split_group_keys,
        require_complete_metadata=thresholds.require_complete_split_group_metadata,
    )
    split_audit["inventory_source"] = (
        "raw_tasks" if raw_rows["tasks"] else "processed_tasks"
    )
    split_audit["inventory_task_count"] = len(
        {
            str(row.get("task_id"))
            for row in split_group_inventory
            if row.get("task_id") is not None
        }
    )
    if split_audit["overlaps"]:
        failures.append(
            _issue(
                "split_group_leakage",
                "Protected group IDs occur in multiple splits",
                overlaps=split_audit["overlaps"],
            )
        )
    if split_audit["missing_group_ids"]:
        missing_group_issue = _issue(
            "split_group_metadata_missing",
            "Some tasks lack configured split-group metadata",
            missing=split_audit["missing_group_ids"],
        )
        if thresholds.require_complete_split_group_metadata:
            failures.append(missing_group_issue)
        else:
            warnings.append(missing_group_issue)

    duplicate_content = _content_duplicate_groups(tasks, split_by_task)
    if duplicate_content and thresholds.fail_on_duplicate_content:
        failures.append(
            _issue(
                "duplicate_content_across_splits",
                "Identical public task content occurs across splits",
                groups=duplicate_content,
            )
        )

    supplied_forbidden = _load_forbidden_literals(
        denylist_path, forbidden_literals, failures
    )
    leakage = _leakage_audit(tasks, examples, verifiers, supplied_forbidden)
    if not leakage["ok"] and thresholds.fail_on_private_label_leakage:
        failures.append(
            _issue(
                "private_label_leakage",
                "Evaluation-only keys or hidden literals occur in model-visible input",
                structural_key_hits=leakage["structural_key_hits"],
                initial_private_literal_hits=leakage["initial_private_literal_hits"],
            )
        )

    missing_snapshot_revisions = sorted(
        str(row.get("snapshot_id", index))
        for index, row in enumerate(snapshots)
        if not isinstance(row.get("source_revision"), str) or not row["source_revision"].strip()
    )
    if missing_snapshot_revisions and thresholds.require_snapshot_source_revisions:
        failures.append(
            _issue(
                "snapshot_revision_missing",
                "Snapshots must bind a non-empty source revision",
                snapshot_ids=missing_snapshot_revisions,
            )
        )

    diversity_values = {
        "workflow_families": _text_values(tasks, ("workflow_family", "category")),
        "entities": _text_values(tasks, ("entity_group", "entity_id", "variant_id")),
        "sequence_contexts": sorted(
            {
                str(task.get("initial_state", {}).get("sequence_context", "")).strip()
                for task in tasks
                if isinstance(task.get("initial_state"), Mapping)
                and str(task["initial_state"].get("sequence_context", "")).strip()
            }
        ),
        "templates": _text_values(tasks, ("template_group", "template_id")),
        "topologies": _text_values(tasks, ("topology_group", "workflow_group")),
        "sources": _text_values(tasks, ("source", "source_id", "dataset")),
        "source_releases": sorted(
            set(_text_values(tasks, ("source_revision", "release", "release_id")))
            | {
                str(row["source_revision"])
                for row in snapshots
                if isinstance(row.get("source_revision"), str) and row["source_revision"].strip()
            }
        ),
    }
    diversity = {
        key: {"count": len(values), "values": values} for key, values in diversity_values.items()
    }
    count_checks = {
        "workflow_families": thresholds.min_workflow_families,
        "entities": thresholds.min_entities,
        "sequence_contexts": thresholds.min_sequence_contexts,
        "templates": thresholds.min_templates,
        "topologies": thresholds.min_topologies,
        "sources": thresholds.min_sources,
        "source_releases": thresholds.min_source_releases,
    }
    if len(task_ids) < thresholds.min_tasks:
        failures.append(
            _issue(
                "too_few_tasks",
                f"Found {len(task_ids)} tasks; require at least {thresholds.min_tasks}",
            )
        )
    for split, required in thresholds.min_tasks_per_split.items():
        if split_counts[split] < required:
            failures.append(
                _issue(
                    "too_few_split_tasks",
                    f"Split {split} has {split_counts[split]} tasks; require at least {required}",
                    split=split,
                )
            )
    for key, required in count_checks.items():
        if diversity[key]["count"] < required:
            failures.append(
                _issue(
                    "insufficient_diversity",
                    f"{key} has {diversity[key]['count']} values; require at least {required}",
                    dimension=key,
                )
            )
    if verifier_coverage < thresholds.min_verifier_coverage:
        failures.append(
            _issue(
                "insufficient_verifier_coverage",
                f"Verifier coverage is {verifier_coverage:.4f}; require {thresholds.min_verifier_coverage:.4f}",
            )
        )

    search = summarize_search_supervision(examples)
    if thresholds.require_search_supervision and not search["present"]:
        failures.append(_issue("search_supervision_missing", "No search-label examples found"))
    if search["present"]:
        metric_checks = (
            (
                "known_applicable_action_fraction",
                thresholds.min_known_applicable_action_fraction,
                "minimum",
            ),
            (
                "unknown_applicable_action_fraction",
                thresholds.max_unknown_applicable_action_fraction,
                "maximum",
            ),
            ("branch_state_fraction", thresholds.min_branch_state_fraction, "minimum"),
            ("branch_label_coverage", thresholds.min_branch_label_coverage, "minimum"),
            (
                "nontrivial_regret_fraction",
                thresholds.min_nontrivial_regret_fraction,
                "minimum",
            ),
            (
                "multi_path_state_fraction",
                thresholds.min_multi_path_state_fraction,
                "minimum",
            ),
            (
                "oracle_contract_collapse_fraction",
                thresholds.max_oracle_contract_collapse_fraction,
                "maximum",
            ),
        )
        for name, threshold, direction in metric_checks:
            value = search[name]
            if value is None:
                if threshold > 0 if direction == "minimum" else threshold < 1:
                    failures.append(
                        _issue(
                            "search_metric_unevaluable",
                            f"Search metric {name} is unavailable for a non-permissive threshold",
                            metric=name,
                        )
                    )
            elif (direction == "minimum" and value < threshold) or (
                direction == "maximum" and value > threshold
            ):
                failures.append(
                    _issue(
                        "search_metric_threshold",
                        f"Search metric {name}={value:.4f} violates {direction} {threshold:.4f}",
                        metric=name,
                    )
                )
        declared_fraction = search["search_complete_metadata"][
            "complete_fraction_among_declared"
        ]
        unknown_state_fraction = search["search_complete_metadata"][
            "incomplete_or_unknown_state_fraction"
        ]
        if declared_fraction is None:
            if thresholds.min_search_complete_fraction > 0:
                failures.append(
                    _issue(
                        "search_completeness_missing",
                        "Search completeness metadata is required by the configured threshold",
                    )
                )
            else:
                warnings.append(
                    _issue(
                        "search_completeness_unknown",
                        "No explicit graph-completeness metadata was found",
                    )
                )
        elif declared_fraction < thresholds.min_search_complete_fraction:
            failures.append(
                _issue(
                    "search_completeness_threshold",
                    f"Declared complete fraction {declared_fraction:.4f} is below threshold",
                )
            )
        if (
            unknown_state_fraction is not None
            and unknown_state_fraction > thresholds.max_incomplete_or_unknown_state_fraction
        ):
            failures.append(
                _issue(
                    "search_unknown_state_threshold",
                    f"Incomplete/unknown state fraction {unknown_state_fraction:.4f} exceeds threshold",
                )
            )

    synthetic_flags: list[str] = []
    for task in tasks:
        if task.get("initial_state", {}).get("is_synthetic") is True:
            synthetic_flags.append(str(task.get("task_id")))
        elif any(value is True for value in _nested_values(task, "synthetic")):
            synthetic_flags.append(str(task.get("task_id")))
    manifest_evidence = None
    fixture_manifest = raw_root / "fixture_manifest.json"
    if fixture_manifest.exists():
        try:
            fixture_data = read_json(fixture_manifest)
            if isinstance(fixture_data, Mapping):
                manifest_evidence = fixture_data.get("research_evidence")
        except Exception as exc:
            failures.append(_issue("fixture_manifest_invalid", str(exc)))
    synthetic_only = bool(tasks) and len(set(synthetic_flags)) == len(task_ids)
    pipeline_validation_only = synthetic_only or manifest_evidence is False
    if synthetic_flags:
        warnings.append(
            _issue(
                "synthetic_pipeline_validation_only",
                "Synthetic/generated rows validate code paths but are not empirical research evidence",
                task_count=len(set(synthetic_flags)),
            )
        )
    if synthetic_flags and not thresholds.allow_synthetic:
        failures.append(
            _issue(
                "synthetic_data_disallowed",
                "Configured quality protocol requires non-synthetic evidence",
            )
        )

    raw_hashes = {
        name: sha256_file(raw_root / name)
        for name in raw_names.values()
        if (raw_root / name).is_file()
    }
    processed_hashes: dict[str, str] = {}
    if processed_root is not None and processed_root.exists():
        processed_hashes = {
            path.name: sha256_file(path)
            for path in sorted(processed_root.glob("*.json*"))
            if path.is_file()
        }
    report: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_VERSION,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "thresholds": _json_safe_thresholds(thresholds),
        "counts": {
            "tasks": len(task_ids),
            "tools": len(raw_rows["tools"]),
            "trajectories": len(raw_rows["trajectories"]),
            "private_verifiers": len(verifier_ids),
            "snapshots": len(snapshots),
            "processed_examples": len(examples),
            "split_tasks": split_counts,
        },
        "diversity": diversity,
        "verifier": {
            "covered_tasks": len(task_ids & verifier_ids),
            "coverage": verifier_coverage,
            "unknown_task_ids": unknown_verifiers,
        },
        "split_integrity": {
            "task_split_conflicts": split_conflicts,
            "unassigned_tasks": missing_splits,
            "test_only_sources": list(thresholds.test_only_sources),
            "test_only_source_violations": test_only_source_violations,
            "groups": split_audit,
            "duplicate_content_across_splits": duplicate_content,
        },
        "leakage": leakage,
        "snapshot_integrity": {
            "missing_source_revision_ids": missing_snapshot_revisions,
            "source_revisions": diversity_values["source_releases"],
        },
        "search_supervision": search,
        "evidence_status": {
            "synthetic_task_ids": sorted(set(synthetic_flags)),
            "synthetic_only": synthetic_only,
            "pipeline_validation_only": pipeline_validation_only,
            "research_evidence": False if pipeline_validation_only else manifest_evidence,
        },
        "duplicates": duplicate_ids,
        "hashes": {
            "raw_files": raw_hashes,
            "processed_files": processed_hashes,
        },
    }
    report["hashes"]["audit_input_sha256"] = sha256_text(
        canonical_json(
            {
                "raw_files": raw_hashes,
                "processed_files": processed_hashes,
                "thresholds": report["thresholds"],
                "forbidden_literals_sha256": sha256_text(canonical_json(supplied_forbidden)),
            }
        )
    )
    # Prove JSON safety, including the absence of NaN/Infinity.
    canonical_json(report)
    return report


def require_dataset_quality(report: Mapping[str, Any]) -> None:
    """Raise :class:`DatasetQualityError` unless ``report`` passed."""

    if report.get("passed") is True:
        return
    failures = report.get("failures", [])
    codes = sorted(
        str(item.get("code")) for item in failures if isinstance(item, Mapping)
    )
    raise DatasetQualityError(
        "Dataset quality gate failed"
        + (f" ({', '.join(codes)})" if codes else "")
    )


def audit_and_require_dataset_quality(
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
    *,
    thresholds: DatasetQualityThresholds | None = None,
    forbidden_literals: Iterable[str] | None = None,
    denylist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the audit, raise on failure, and return the passing report."""

    report = audit_dataset_quality(
        raw_dir,
        processed_dir,
        thresholds=thresholds,
        forbidden_literals=forbidden_literals,
        denylist_path=denylist_path,
    )
    require_dataset_quality(report)
    return report


__all__ = [
    "DatasetQualityError",
    "DatasetQualityThresholds",
    "QUALITY_REPORT_VERSION",
    "audit_and_require_dataset_quality",
    "audit_dataset_quality",
    "require_dataset_quality",
    "summarize_search_supervision",
]
