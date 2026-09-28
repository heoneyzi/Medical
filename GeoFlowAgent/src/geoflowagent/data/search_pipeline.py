"""End-to-end construction of exact weighted-search supervision.

The functions in this module operate only on local, versioned task/tool/snapshot
packages.  They never call a live biomedical API while labels are being built.
Consequently a dataset can be regenerated exactly and an unobserved action remains
unknown instead of becoming an accidental negative label.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.schema import (
    unique_by_id,
    validate_background,
    validate_snapshot,
    validate_task,
    validate_tool,
    validate_verifier,
)
from geoflowagent.data.search_supervision import (
    SPLITS,
    compile_search_supervision,
    write_search_dataset,
)
from geoflowagent.search import EdgeCostConfig, SearchLimits, WeightedSearchOracle
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_text,
)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _search_task_shard(
    task: Mapping[str, Any],
    *,
    tools: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]],
    private_verifier: Mapping[str, Any],
    limits: SearchLimits,
    costs: EdgeCostConfig,
    include_unreachable: bool,
    require_complete_graph: bool,
    require_reachable_root: bool,
    search_fingerprint: str,
) -> dict[str, Any]:
    """Build one independently resumable oracle shard.

    This deliberately contains no shared mutable state, so callers may execute
    separate tasks in worker processes without changing any labels or ordering
    in the final compiled dataset.
    """

    task_id = str(task["task_id"])
    verification_goal = combine_verification_goal(task["goal"], private_verifier["private_verifier"])
    result = WeightedSearchOracle(
        ContractEngine(tools, snapshots), cost_config=costs, limits=limits
    ).search(task["initial_state"], verification_goal, task["available_tools"])
    diagnostics = result.diagnostics.to_dict()
    diagnostics["task_id"] = task_id
    if require_complete_graph and not result.diagnostics.graph_complete:
        raise RuntimeError(
            f"Search graph for {task_id} is incomplete: {diagnostics}. Add exact snapshots/stateful "
            "retry context or raise resource bounds; do not turn unresolved actions into negatives."
        )
    if require_reachable_root and result.initial_value.reachable is not True:
        raise RuntimeError(f"No verified path reaches the private goal for {task_id}")
    oracle_output = result.to_dict()
    oracle_output["task_id"] = task_id
    oracle_output["source_revision"] = task.get("provenance", {}).get("source_revision")
    return {
        "search_fingerprint": search_fingerprint,
        "task_id": task_id,
        "diagnostics": diagnostics,
        "examples": compile_search_supervision(
            task, tools, snapshots, oracle_output, include_unreachable=include_unreachable
        ),
    }


def _optional_jsonl(root: Path, name: str) -> list[dict[str, Any]]:
    path = root / name
    return read_jsonl(path) if path.exists() else []


def _deduplicate_exact_snapshots(
    snapshots: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate exact calls and reject contradictory source snapshots.

    Multiple cases often query the same gene/release.  The executor keys a call by
    ``tool_id + resolved arguments`` rather than patient/case ID, so identical
    observations are safely shared.  Conflicting observations for that key are a
    source-data error and fail closed.
    """

    by_call: dict[str, dict[str, Any]] = {}
    for row in sorted(snapshots, key=lambda item: str(item["snapshot_id"])):
        validate_snapshot(row)
        key = canonical_json({"tool_id": row["tool_id"], "arguments": row["arguments"]})
        previous = by_call.get(key)
        if previous is None:
            by_call[key] = copy.deepcopy(row)
            continue
        comparable = ("status", "output", "source_revision")
        if any(canonical_json(previous[field]) != canonical_json(row[field]) for field in comparable):
            raise ValueError(
                "Conflicting exact snapshots for tool+arguments: "
                f"{previous['snapshot_id']!r} vs {row['snapshot_id']!r}"
            )
    return sorted(by_call.values(), key=lambda row: row["snapshot_id"])


def _task_release_tokens(task: Mapping[str, Any]) -> set[str]:
    """Return explicit release identifiers that can scope an offline snapshot."""

    values: set[str] = set()
    provenance = task.get("provenance", {})
    initial_state = task.get("initial_state", {})
    for container in (provenance, initial_state):
        if not isinstance(container, Mapping):
            continue
        for key in ("source_revision", "database_release", "release", "release_id"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    return values


def _snapshot_splits(
    snapshot: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Resolve snapshot ownership without inspecting a held-out task's search graph.

    A multi-split corpus must make snapshot ownership auditable.  Preferred forms
    are ``provenance.task_id``, ``provenance.task_ids``, or ``provenance.split``.
    A release identifier may also scope a snapshot when that release occurs in
    exactly one split.  Truly shared public snapshots must opt in with
    ``provenance.shared_across_splits=true``; ambiguous rows fail closed.
    """

    provenance = snapshot.get("provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    all_splits = {str(task["split"]) for task in tasks_by_id.values()}
    if provenance.get("shared_across_splits") is True:
        return all_splits

    task_ids: list[str] = []
    direct_task_id = provenance.get("task_id", snapshot.get("task_id"))
    if isinstance(direct_task_id, str) and direct_task_id:
        task_ids.append(direct_task_id)
    raw_task_ids = provenance.get("task_ids", snapshot.get("task_ids", []))
    if isinstance(raw_task_ids, list):
        task_ids.extend(str(value) for value in raw_task_ids)
    if task_ids:
        unknown = sorted(set(task_ids) - set(tasks_by_id))
        if unknown:
            raise ValueError(
                f"Snapshot {snapshot['snapshot_id']} references unknown task IDs: {unknown}"
            )
        return {str(tasks_by_id[task_id]["split"]) for task_id in task_ids}

    declared = provenance.get("split", snapshot.get("split"))
    if isinstance(declared, str):
        declared_values = [declared]
    elif isinstance(declared, list):
        declared_values = declared
    else:
        declared_values = []
    if declared_values:
        normalized = {str(value) for value in declared_values}
        invalid = sorted(normalized - set(SPLITS))
        if invalid:
            raise ValueError(
                f"Snapshot {snapshot['snapshot_id']} has invalid split scope: {invalid}"
            )
        return normalized

    revision = snapshot.get("source_revision")
    release_splits = {
        str(task["split"])
        for task in tasks_by_id.values()
        if isinstance(revision, str) and revision in _task_release_tokens(task)
    }
    if len(release_splits) == 1:
        return release_splits
    if len(all_splits) == 1:
        return all_splits
    raise ValueError(
        f"Snapshot {snapshot['snapshot_id']} has ambiguous split ownership. Add "
        "provenance.task_id/task_ids/split or shared_across_splits=true."
    )


def _active_rows(
    tasks: Sequence[dict[str, Any]],
    verifiers: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    background: Sequence[dict[str, Any]],
    *,
    active_splits: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build the physical split boundary used by search, cache, and quality."""

    selected_splits = tuple(dict.fromkeys(str(value) for value in active_splits))
    if not selected_splits or set(selected_splits) - set(SPLITS):
        raise ValueError(f"active_splits must be a non-empty subset of {SPLITS}")
    missing_split = sorted(str(row["task_id"]) for row in tasks if row.get("split") not in SPLITS)
    if missing_split:
        raise ValueError(
            "Split-aware search requires every raw task to have a preassigned split; "
            f"missing/invalid={missing_split}"
        )
    tasks_by_id = {str(row["task_id"]): row for row in tasks}
    selected_tasks = [
        copy.deepcopy(row) for row in tasks if str(row["split"]) in selected_splits
    ]
    if not selected_tasks:
        raise ValueError(f"No tasks belong to active_splits={list(selected_splits)}")
    selected_task_ids = {str(row["task_id"]) for row in selected_tasks}
    selected_tool_ids = {
        str(tool_id) for row in selected_tasks for tool_id in row["available_tools"]
    }
    selected_background_ids = {
        str(card_id) for row in selected_tasks for card_id in row.get("background_ids", [])
    }
    selected_verifiers = [
        copy.deepcopy(row) for row in verifiers if str(row["task_id"]) in selected_task_ids
    ]
    selected_snapshots = [
        copy.deepcopy(row)
        for row in snapshots
        if str(row["tool_id"]) in selected_tool_ids
        and bool(_snapshot_splits(row, tasks_by_id) & set(selected_splits))
    ]
    selected_tools = [
        copy.deepcopy(row) for row in tools if str(row["tool_id"]) in selected_tool_ids
    ]
    selected_background = [
        copy.deepcopy(row)
        for row in background
        if str(row["card_id"]) in selected_background_ids
    ]
    return (
        selected_tasks,
        selected_verifiers,
        _deduplicate_exact_snapshots(selected_snapshots),
        selected_tools,
        selected_background,
    )


def prepare_search_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    seed: int = 17,
    max_depth: int = 16,
    max_states: int = 20_000,
    top_k_paths: int = 8,
    near_optimal_slack: float = 0.0,
    call_cost: float = 1.0,
    failure_cost: float = 1.0,
    redundancy_cost: float = 0.25,
    status_costs: Mapping[str, float] | None = None,
    include_unreachable: bool = True,
    require_complete_graph: bool = True,
    require_reachable_root: bool = True,
    split_group_kinds: Sequence[str] = ("case", "entity", "template", "source", "release"),
    active_splits: Sequence[str] = ("train", "dev"),
    progress_dir: str | Path | None = None,
    progress_backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Search every task and write a cache-compatible supervised state dataset.

    Private verifiers are used only to decide whether a searched state is truly
    terminal.  They are persisted in ``verifiers.private.jsonl`` for evaluation,
    never copied into task text, typed state, history, background, or embeddings.
    """

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    required = ("tools.jsonl", "tasks.jsonl", "verifiers.private.jsonl")
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Search corpus is missing required files: {missing}")
    tools = read_jsonl(raw_dir / "tools.jsonl")
    tasks = read_jsonl(raw_dir / "tasks.jsonl")
    verifiers = read_jsonl(raw_dir / "verifiers.private.jsonl")
    snapshots = _optional_jsonl(raw_dir, "snapshots.jsonl")
    background = _optional_jsonl(raw_dir, "background.jsonl")
    if not tasks:
        raise ValueError("Search corpus has no tasks")
    for row in tools:
        validate_tool(row)
    for row in tasks:
        validate_task(row)
    for row in verifiers:
        validate_verifier(row)
    for row in background:
        validate_background(row)
    tool_map = unique_by_id(tools, "tool_id", "tool")
    task_map = unique_by_id(tasks, "task_id", "task")
    verifier_map = unique_by_id(verifiers, "task_id", "verifier")
    missing_verifiers = sorted(set(task_map) - set(verifier_map))
    extra_verifiers = sorted(set(verifier_map) - set(task_map))
    if missing_verifiers or extra_verifiers:
        raise ValueError(
            "Private verifier coverage must be exactly one per task; "
            f"missing={missing_verifiers}, extra={extra_verifiers}"
        )
    raw_split_counts = {
        split: sum(row.get("split") == split for row in tasks) for split in SPLITS
    }
    (
        tasks,
        verifiers,
        snapshots,
        tools,
        background,
    ) = _active_rows(
        tasks,
        verifiers,
        snapshots,
        tools,
        background,
        active_splits=active_splits,
    )
    tool_map = unique_by_id(tools, "tool_id", "active tool")
    verifier_map = unique_by_id(verifiers, "task_id", "active verifier")
    engine = ContractEngine(tools, snapshots)
    limits = SearchLimits(
        max_depth=max_depth,
        max_states=max_states,
        top_k_paths=top_k_paths,
        near_optimal_slack=near_optimal_slack,
    )
    costs = EdgeCostConfig(
        call_cost=call_cost,
        failure_cost=failure_cost,
        redundancy_cost=redundancy_cost,
        status_costs=dict(status_costs or {}),
    )
    active_source_hashes = {
        "tools": sha256_text(canonical_json(tools)),
        "tasks": sha256_text(canonical_json(tasks)),
        "verifiers": sha256_text(canonical_json(verifiers)),
        "snapshots": sha256_text(canonical_json(snapshots)),
        "background": sha256_text(canonical_json(background)),
    }
    search_fingerprint = sha256_text(
        canonical_json(
            {
                "active_source_hashes": active_source_hashes,
                "active_splits": list(dict.fromkeys(str(value) for value in active_splits)),
                "limits": limits.to_dict(),
                "cost_config": costs.to_dict(),
                "include_unreachable": include_unreachable,
                "require_complete_graph": require_complete_graph,
                "require_reachable_root": require_reachable_root,
            }
        )
    )
    progress_root = (
        Path(progress_dir)
        if progress_dir is not None
        else processed_dir.parent / f".{processed_dir.name}.oracle_progress"
    )
    progress_root.mkdir(parents=True, exist_ok=True)
    progress_manifest_path = progress_root / "manifest.json"
    if progress_manifest_path.exists():
        progress_manifest = read_json(progress_manifest_path)
        if progress_manifest.get("search_fingerprint") != search_fingerprint:
            raise RuntimeError(
                "Search progress belongs to different sources or search settings; "
                "choose a new progress_dir instead of mixing oracle labels."
            )
    else:
        progress_manifest = {
            "status": "running",
            "search_fingerprint": search_fingerprint,
            "processed_dir": str(processed_dir),
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "completed_task_ids": [],
        }
        _atomic_write_json(progress_manifest_path, progress_manifest)
    progress_backup = Path(progress_backup_path) if progress_backup_path is not None else None

    def persist_progress(completed_task_ids: Sequence[str], *, status: str = "running") -> None:
        payload = {
            "status": status,
            "search_fingerprint": search_fingerprint,
            "processed_dir": str(processed_dir),
            "progress_dir": str(progress_root),
            "total_tasks": len(tasks),
            "completed_tasks": len(completed_task_ids),
            "completed_task_ids_sha256": sha256_text(canonical_json(completed_task_ids)),
            "last_completed_task_id": completed_task_ids[-1] if completed_task_ids else None,
        }
        _atomic_write_json(progress_manifest_path, payload)
        if progress_backup is not None:
            _atomic_write_json(progress_backup, payload)

    examples: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    completed_task_ids: list[str] = []
    for task in sorted(tasks, key=lambda row: row["task_id"]):
        unknown_tools = sorted(set(task["available_tools"]) - set(tool_map))
        if unknown_tools:
            raise ValueError(f"Task {task['task_id']} references unknown tools: {unknown_tools}")
        task_id = str(task["task_id"])
        shard_path = progress_root / f"{sha256_text(task_id)}.json"
        if shard_path.exists():
            shard = read_json(shard_path)
            if (
                shard.get("search_fingerprint") != search_fingerprint
                or shard.get("task_id") != task_id
            ):
                raise RuntimeError(f"Invalid oracle progress shard for task {task_id}")
            task_diagnostics = dict(shard["diagnostics"])
            task_examples = list(shard["examples"])
        else:
            verification_goal = combine_verification_goal(
                task["goal"], verifier_map[task_id]["private_verifier"]
            )
            result = WeightedSearchOracle(engine, cost_config=costs, limits=limits).search(
                task["initial_state"], verification_goal, task["available_tools"]
            )
            task_diagnostics = result.diagnostics.to_dict()
            task_diagnostics["task_id"] = task_id
            if require_complete_graph and not result.diagnostics.graph_complete:
                raise RuntimeError(
                    f"Search graph for {task_id} is incomplete: {task_diagnostics}. "
                    "Add exact snapshots/stateful retry context or raise resource bounds; do not "
                    "turn unresolved actions into negatives."
                )
            if require_reachable_root and result.initial_value.reachable is not True:
                raise RuntimeError(f"No verified path reaches the private goal for {task_id}")
            oracle_output = result.to_dict()
            oracle_output["task_id"] = task_id
            oracle_output["source_revision"] = task.get("provenance", {}).get(
                "source_revision"
            )
            task_examples = compile_search_supervision(
                task,
                tools,
                snapshots,
                oracle_output,
                include_unreachable=include_unreachable,
            )
            _atomic_write_json(
                shard_path,
                {
                    "search_fingerprint": search_fingerprint,
                    "task_id": task_id,
                    "diagnostics": task_diagnostics,
                    "examples": task_examples,
                },
            )
        diagnostics.append(task_diagnostics)
        examples.extend(task_examples)
        completed_task_ids.append(task_id)
        persist_progress(completed_task_ids)

    aggregate = {
        "algorithm": "bounded_weighted_graph_search_and_reverse_dijkstra",
        "limits": limits.to_dict(),
        "cost_config": costs.to_dict(),
        "tasks": len(tasks),
        "active_splits": list(dict.fromkeys(str(value) for value in active_splits)),
        "raw_inventory": {
            "tasks": len(task_map),
            "split_tasks": raw_split_counts,
            "tasks_file_sha256": sha256_text(canonical_json(sorted(task_map))),
        },
        "all_graphs_complete": all(row["graph_complete"] for row in diagnostics),
        "total_discovered_states": sum(int(row["discovered_states"]) for row in diagnostics),
        "total_known_edges": sum(int(row["known_edges"]) for row in diagnostics),
        "total_unresolved_actions": sum(int(row["unresolved_actions"]) for row in diagnostics),
        # These hashes bind only active rows.  A sealed test record must not
        # influence the development dataset/cache identity.
        "active_source_hashes": active_source_hashes,
        "search_fingerprint": search_fingerprint,
    }
    manifest = write_search_dataset(
        processed_dir,
        tasks,
        tools,
        examples,
        snapshots,
        background=background,
        private_verifiers=verifiers,
        seed=seed,
        split_group_kinds=split_group_kinds,
        oracle_metadata=aggregate,
    )
    persist_progress(completed_task_ids, status="complete")
    return manifest
