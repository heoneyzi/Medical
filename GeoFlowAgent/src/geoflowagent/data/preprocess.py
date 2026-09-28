from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from geoflowagent.constants import INVALID_REGRET, SCHEMA_VERSION, STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.schema import (
    unique_by_id,
    validate_background,
    validate_minimal_pair,
    validate_snapshot,
    validate_task,
    validate_tool,
    validate_trajectory,
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


def verify_processed_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Verify schema version and every generated file bound by the manifest."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported processed schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    files = manifest.get("processed_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Processed manifest does not bind generated file checksums")
    required = {"tools.jsonl", "tasks.jsonl", "examples.jsonl"}
    missing_required = sorted(required - set(files))
    if missing_required:
        raise ValueError(f"Processed manifest is missing required files: {missing_required}")
    known_optional = {
        "minimal_pairs.jsonl",
        "background.jsonl",
        "verifiers.private.jsonl",
        "snapshots.jsonl",
    }
    unbound = sorted(
        name for name in known_optional if (output_dir / name).exists() and name not in files
    )
    if unbound:
        raise ValueError(f"Processed directory contains unbound generated files: {unbound}")
    for relative, expected in files.items():
        path = output_dir / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Processed dataset checksum mismatch: {path}")
    return manifest


def _hashed_split(group_id: str, seed: int) -> str:
    bucket = int(sha256_text(f"{seed}:{group_id}")[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def _assign_task_splits(tasks: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Keep related cases together and reject contradictory explicit assignments."""

    explicit_by_group: dict[str, str] = {}
    for task in tasks:
        group_id = str(task.get("split_group") or task["task_id"])
        requested = task.get("split")
        if requested not in {"train", "dev", "test", None}:
            raise ValueError(f"Invalid split {requested!r} for task {task['task_id']}")
        if requested is None:
            continue
        previous = explicit_by_group.setdefault(group_id, str(requested))
        if previous != requested:
            raise ValueError(
                f"Split group {group_id!r} crosses splits: {previous!r} vs {requested!r}"
            )
    split_by_group: dict[str, str] = dict(explicit_by_group)
    assigned: dict[str, str] = {}
    for task in tasks:
        group_id = str(task.get("split_group") or task["task_id"])
        split_by_group.setdefault(group_id, _hashed_split(group_id, seed))
        assigned[str(task["task_id"])] = split_by_group[group_id]
    return assigned


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 17,
    max_search_depth: int = 12,
    require_optimal_demo: bool = True,
    counterfactual_policy: str = "symbolic",
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    tool_path = raw_dir / "tools.jsonl"
    task_path = raw_dir / "tasks.jsonl"
    trajectory_path = raw_dir / "trajectories.jsonl"
    pair_path = raw_dir / "minimal_pairs.jsonl"
    background_path = raw_dir / "background.jsonl"
    verifier_path = raw_dir / "verifiers.private.jsonl"
    snapshot_path = raw_dir / "snapshots.jsonl"
    if counterfactual_policy not in {"symbolic", "observed_only"}:
        raise ValueError("counterfactual_policy must be 'symbolic' or 'observed_only'")
    if counterfactual_policy == "observed_only" and require_optimal_demo:
        raise ValueError(
            "require_optimal_demo cannot be verified under observed_only; "
            "disable it or provide trusted symbolic/snapshot counterfactuals"
        )

    tools = read_jsonl(tool_path)
    tasks = read_jsonl(task_path)
    trajectories = read_jsonl(trajectory_path)
    pairs = read_jsonl(pair_path) if pair_path.exists() else []
    background = read_jsonl(background_path) if background_path.exists() else []
    verifiers = read_jsonl(verifier_path) if verifier_path.exists() else []
    snapshots = read_jsonl(snapshot_path) if snapshot_path.exists() else []
    for row in tools:
        validate_tool(row)
    for row in tasks:
        validate_task(row)
    for row in trajectories:
        validate_trajectory(row)
    for row in pairs:
        validate_minimal_pair(row)
    for row in background:
        validate_background(row)
    for row in verifiers:
        validate_verifier(row)
    for row in snapshots:
        validate_snapshot(row)
    tool_map = unique_by_id(tools, "tool_id", "tool")
    task_map = unique_by_id(tasks, "task_id", "task")
    trajectory_map = unique_by_id(trajectories, "task_id", "trajectory")
    pair_map = unique_by_id(pairs, "pair_id", "minimal pair")
    background_map = unique_by_id(background, "card_id", "background card")
    verifier_map = unique_by_id(verifiers, "task_id", "private verifier")
    unique_by_id(snapshots, "snapshot_id", "snapshot")
    if set(task_map) != set(trajectory_map):
        missing = sorted(set(task_map) - set(trajectory_map))
        extra = sorted(set(trajectory_map) - set(task_map))
        raise ValueError(f"Task/trajectory mismatch: missing={missing}, extra={extra}")
    unknown_verifiers = sorted(set(verifier_map) - set(task_map))
    if unknown_verifiers:
        raise ValueError(f"Private verifiers reference unknown tasks: {unknown_verifiers}")

    task_splits = _assign_task_splits(tasks, seed)
    for pair in pairs:
        source_task_id = pair.get("source_task_id")
        if source_task_id is None:
            continue
        if source_task_id not in task_splits:
            raise ValueError(
                f"Minimal pair {pair['pair_id']} references unknown source task {source_task_id!r}"
            )
        if pair["split"] != task_splits[source_task_id]:
            raise ValueError(
                f"Minimal pair {pair['pair_id']} crosses its source task split: "
                f"pair={pair['split']}, task={task_splits[source_task_id]}"
            )

    engine = ContractEngine(tools, snapshots)
    examples: list[dict[str, Any]] = []
    split_task_ids: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for task_id in sorted(task_map):
        task = task_map[task_id]
        trajectory = trajectory_map[task_id]
        available = [str(item) for item in task["available_tools"]]
        unknown = sorted(set(available) - set(tool_map))
        if unknown:
            raise ValueError(f"Task {task_id} references unknown available tools: {unknown}")
        unknown_steps = sorted(set(trajectory["tool_ids"]) - set(available))
        if unknown_steps:
            raise ValueError(f"Trajectory {task_id} uses unavailable tools: {unknown_steps}")
        unknown_background = sorted(set(task.get("background_ids", [])) - set(background_map))
        if unknown_background:
            raise ValueError(
                f"Task {task_id} references unknown background cards: {unknown_background}"
            )
        split = task_splits[task_id]
        split_task_ids[split].append(task_id)
        state = copy.deepcopy(task["initial_state"])
        history: list[dict[str, Any]] = []
        prefix: list[str] = []
        tool_ids = [str(item) for item in trajectory["tool_ids"]]
        private_row = verifier_map.get(task_id)
        verification_goal = combine_verification_goal(
            task["goal"], private_row["private_verifier"] if private_row else None
        )

        for step_index, gold_tool_id in enumerate(tool_ids):
            if counterfactual_policy == "symbolic":
                valid_next, regrets = engine.action_analysis(
                    state, verification_goal, available, max_depth=max_search_depth
                )
            else:
                # Observed-only data must not execute or infer the outcome of a non-gold edge.
                # `valid_next_tools` is behavioral supervision here; only the demonstrated
                # edge receives a known utility label.
                valid_next = [gold_tool_id]
                regrets = {STOP_TOOL_ID: INVALID_REGRET, gold_tool_id: 0.0}
            outcome_status: dict[str, str] = {STOP_TOOL_ID: "contract_invalid"}
            regret_mask: dict[str, bool] = {STOP_TOOL_ID: True}
            for tool_id in available:
                if not engine.applicable(state, tool_id):
                    outcome_status[tool_id] = "contract_invalid"
                    regret_mask[tool_id] = True
                    regrets[tool_id] = INVALID_REGRET
                elif tool_id == gold_tool_id:
                    outcome_status[tool_id] = "observed"
                    regret_mask[tool_id] = True
                elif counterfactual_policy == "symbolic":
                    if engine.executable(state, tool_id):
                        outcome_status[tool_id] = "simulated"
                        regret_mask[tool_id] = True
                    else:
                        outcome_status[tool_id] = "unobserved"
                        regret_mask[tool_id] = False
                        regrets[tool_id] = None
                else:
                    outcome_status[tool_id] = "unobserved"
                    regret_mask[tool_id] = False
                    regrets[tool_id] = None
            if not valid_next and require_optimal_demo:
                raise ValueError(
                    f"Could not verify an optimal demonstration at {task_id}[{step_index}]: "
                    f"no shortest-path action was found within max_search_depth={max_search_depth}"
                )
            if not valid_next:
                valid_next = [gold_tool_id]
                regrets[gold_tool_id] = 0.0
            if gold_tool_id not in valid_next and require_optimal_demo:
                raise ValueError(
                    f"Non-optimal demonstration at {task_id}[{step_index}]: "
                    f"gold={gold_tool_id}, shortest-path actions={valid_next}"
                )
            if not engine.applicable(state, gold_tool_id):
                raise ValueError(
                    f"Inapplicable gold action at {task_id}[{step_index}]: {gold_tool_id}"
                )
            transition = engine.execute(state, gold_tool_id)
            example = {
                "example_id": f"{task_id}:h{step_index + 1}",
                "task_id": task_id,
                "split": split,
                "phase": step_index + 1,
                "terminal": False,
                "query": task["query"],
                "category": task.get("category", "genomics"),
                "goal": task["goal"],
                "state": copy.deepcopy(state),
                "next_state": copy.deepcopy(transition.state),
                "history": copy.deepcopy(history),
                "prefix_tool_ids": list(prefix),
                "gold_suffix_tool_ids": tool_ids[step_index:],
                "gold_next_tool": gold_tool_id,
                "valid_next_tools": valid_next,
                "action_regret": regrets,
                "action_regret_mask": regret_mask,
                "action_outcome_status": outcome_status,
                "candidate_tools": available,
                "contract_valid_tools": engine.applicable_tools(state, available),
                "background_ids": task.get("background_ids", []),
                "provenance": task.get("provenance", {}),
            }
            example["record_sha256"] = sha256_text(canonical_json(example))
            examples.append(example)
            state = transition.state
            history.append(transition.observation)
            prefix.append(gold_tool_id)

        if not engine.goal_satisfied(state, verification_goal):
            raise ValueError(f"Trajectory for {task_id} does not satisfy goal; final_state={state}")

        # Explicit h=m+1 row. Without it, the model never sees STOP in slot zero.
        if counterfactual_policy == "symbolic":
            _, terminal_regret = engine.action_analysis(
                state, verification_goal, available, max_depth=max_search_depth
            )
        else:
            terminal_regret = {STOP_TOOL_ID: 0.0}
        terminal_contract_tools = engine.applicable_tools(state, available)
        terminal_outcome_status: dict[str, str] = {STOP_TOOL_ID: "observed"}
        terminal_regret_mask: dict[str, bool] = {STOP_TOOL_ID: True}
        for tool_id in available:
            if tool_id not in terminal_contract_tools:
                terminal_outcome_status[tool_id] = "contract_invalid"
                terminal_regret_mask[tool_id] = True
                terminal_regret[tool_id] = INVALID_REGRET
            elif counterfactual_policy == "symbolic":
                terminal_outcome_status[tool_id] = "simulated"
                terminal_regret_mask[tool_id] = True
            else:
                terminal_outcome_status[tool_id] = "unobserved"
                terminal_regret_mask[tool_id] = False
                terminal_regret[tool_id] = None
        terminal = {
            "example_id": f"{task_id}:h{len(tool_ids) + 1}",
            "task_id": task_id,
            "split": split,
            "phase": len(tool_ids) + 1,
            "terminal": True,
            "query": task["query"],
            "category": task.get("category", "genomics"),
            "goal": task["goal"],
            "state": copy.deepcopy(state),
            "next_state": copy.deepcopy(state),
            "history": copy.deepcopy(history),
            "prefix_tool_ids": list(prefix),
            "gold_suffix_tool_ids": [],
            "gold_next_tool": STOP_TOOL_ID,
            "valid_next_tools": [STOP_TOOL_ID],
            "action_regret": terminal_regret,
            "action_regret_mask": terminal_regret_mask,
            "action_outcome_status": terminal_outcome_status,
            "candidate_tools": available,
            "contract_valid_tools": terminal_contract_tools,
            "background_ids": task.get("background_ids", []),
            "provenance": task.get("provenance", {}),
        }
        terminal["record_sha256"] = sha256_text(canonical_json(terminal))
        examples.append(terminal)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "tools.jsonl", tools)
    write_jsonl(output_dir / "tasks.jsonl", tasks)
    write_jsonl(output_dir / "examples.jsonl", examples)
    processed_paths = [
        output_dir / "tools.jsonl",
        output_dir / "tasks.jsonl",
        output_dir / "examples.jsonl",
    ]
    if pair_path.exists():
        write_jsonl(output_dir / "minimal_pairs.jsonl", list(pair_map.values()))
        processed_paths.append(output_dir / "minimal_pairs.jsonl")
    elif (output_dir / "minimal_pairs.jsonl").exists():
        (output_dir / "minimal_pairs.jsonl").unlink()
    if background_path.exists():
        write_jsonl(output_dir / "background.jsonl", list(background_map.values()))
        processed_paths.append(output_dir / "background.jsonl")
    elif (output_dir / "background.jsonl").exists():
        (output_dir / "background.jsonl").unlink()
    if verifier_path.exists():
        write_jsonl(output_dir / "verifiers.private.jsonl", list(verifier_map.values()))
        processed_paths.append(output_dir / "verifiers.private.jsonl")
    elif (output_dir / "verifiers.private.jsonl").exists():
        (output_dir / "verifiers.private.jsonl").unlink()
    if snapshot_path.exists():
        write_jsonl(output_dir / "snapshots.jsonl", snapshots)
        processed_paths.append(output_dir / "snapshots.jsonl")
    elif (output_dir / "snapshots.jsonl").exists():
        (output_dir / "snapshots.jsonl").unlink()

    input_files = [tool_path, task_path, trajectory_path]
    if pair_path.exists():
        input_files.append(pair_path)
    if background_path.exists():
        input_files.append(background_path)
    if verifier_path.exists():
        input_files.append(verifier_path)
    if snapshot_path.exists():
        input_files.append(snapshot_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "max_search_depth": max_search_depth,
        "require_optimal_demo": require_optimal_demo,
        "counterfactual_policy": counterfactual_policy,
        "counts": {
            "tools": len(tools),
            "tasks": len(tasks),
            "examples": len(examples),
            "terminal_examples": sum(bool(row["terminal"]) for row in examples),
            "private_verifiers": len(verifiers),
            "snapshots": len(snapshots),
            "split_tasks": {key: len(value) for key, value in split_task_ids.items()},
            "split_examples": {
                split: sum(row["split"] == split for row in examples)
                for split in ("train", "dev", "test")
            },
        },
        "task_ids_by_split": split_task_ids,
        "task_groups_by_split": {
            split: sorted(
                {str(task_map[task_id].get("split_group") or task_id) for task_id in task_ids}
            )
            for split, task_ids in split_task_ids.items()
        },
        "source_files": {path.name: sha256_file(path) for path in input_files},
        "processed_files": {path.name: sha256_file(path) for path in processed_paths},
        "invariants": {
            "task_level_split": True,
            "declared_split_groups_kept_disjoint": True,
            "terminal_row_per_task": True,
            "terminal_target": STOP_TOOL_ID,
            "future_observation_in_context": False,
            "private_verifier_in_model_input": False,
            "verification_goal_used_for_labels": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build typed prefix/suffix GeoFlowAgent data")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-search-depth", type=int, default=12)
    parser.add_argument("--allow-nonoptimal-demo", action="store_true")
    parser.add_argument(
        "--counterfactual-policy",
        choices=["symbolic", "observed_only"],
        default="symbolic",
        help="Whether deterministic contracts may label unexecuted candidate actions.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(
        args.raw_dir,
        args.output_dir,
        seed=args.seed,
        max_search_depth=args.max_search_depth,
        require_optimal_demo=not args.allow_nonoptimal_demo,
        counterfactual_policy=args.counterfactual_policy,
    )
    print(canonical_json(manifest["counts"]))


if __name__ == "__main__":
    main()
