"""Wiring: artefacts on disk -> the estimators -> findings on disk.

Kept apart from the experiment modules so that ``probes`` and ``ladder`` stay
testable without a corpus, and this file stays readable as plumbing.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from geoflowagent.geoacmg import ladder, probes
from geoflowagent.geoacmg.claims import Finding, Role
from geoflowagent.geoacmg.inference import paired_contrast
from geoflowagent.utils.io import read_jsonl

SEARCH_CONTEXT = "__search_context__"


# ------------------------------------------------------------------ RQ1 probe


def _state_depth(state: dict[str, Any]) -> int:
    context = state.get(SEARCH_CONTEXT) or {}
    return int(context.get("depth", 0))


def load_probe_table(processed_dir: str | Path) -> dict[str, Any]:
    """Pull the supervised targets a linear probe is asked to recover.

    ``value_star`` is the oracle's remaining cost from this state; ``optimal``
    is the action it would take.  Rows without a known value are dropped rather
    than imputed -- an unknown is not a zero.
    """

    rows = read_jsonl(Path(processed_dir) / "examples.jsonl")
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("value_known_mask"):
            continue
        optimal = row.get("optimal_actions") or []
        if not optimal:
            continue
        kept.append(
            {
                "example_id": row["example_id"],
                "task_id": row["task_id"],
                "gene": (row.get("group_ids", {}).get("entity") or [row["task_id"]])[0]
                if isinstance(row.get("group_ids", {}).get("entity"), list)
                else row.get("group_ids", {}).get("entity", row["task_id"]),
                "value_star": float(row["value_star"]),
                "optimal": sorted(optimal)[0],
                "depth": _state_depth(row.get("state", {})),
                "split": row.get("split", "train"),
            }
        )
    return {
        "rows": kept,
        "counts": {
            "examples_read": len(rows),
            "with_known_value_and_optimal_action": len(kept),
            "tasks": len({r["task_id"] for r in kept}),
            "genes": len({r["gene"] for r in kept}),
        },
    }


def probe_findings(
    processed_dir: str | Path,
    cache_dir: str | Path | None,
    *,
    views: Sequence[str] = (),
    splits: Sequence[str] = ("train", "dev"),
    seed: int = 17,
    permutations: int = 200,
) -> tuple[list[Finding], dict[str, Any]]:
    """Run RQ1 over whichever views the embedding cache actually contains."""

    table = load_probe_table(processed_dir)
    rows = [row for row in table["rows"] if row["split"] in splits]
    if not rows:
        raise ValueError(
            "no probe rows in the requested splits; run the oracle stage first"
        )
    report: dict[str, Any] = {"counts": table["counts"], "used_rows": len(rows), "views": {}}
    findings: list[Finding] = []
    if cache_dir is None:
        report["skipped"] = "no embedding cache; run the embed stage to measure RQ1"
        return findings, report

    from geoflowagent.embeddings.cache import EmbeddingCache

    cache = EmbeddingCache(Path(cache_dir), verify=False)
    index = cache.example_index
    available = sorted(cache.manifest.get("views", {}))
    chosen = [view for view in (views or available) if view in available]
    usable = [row for row in rows if row["example_id"] in index]
    report["rows_with_embeddings"] = len(usable)
    if not usable:
        report["skipped"] = "the cache holds no vector for any probe row"
        return findings, report

    positions = [index[row["example_id"]] for row in usable]
    targets = [row["value_star"] for row in usable]
    actions = [row["optimal"] for row in usable]
    genes = [row["gene"] for row in usable]
    depths = [row["depth"] for row in usable]
    tasks = [row["task_id"] for row in usable]

    for view in chosen:
        relative = f"{view}/examples.state.npy"
        if relative not in cache.manifest.get("files", {}):
            report["views"][view] = "no state encoding"
            continue
        vectors = np.asarray(cache.array(relative), dtype=np.float64)[positions]
        findings.extend(
            probes.information_findings(
                vectors,
                remaining_cost=targets,
                optimal_action=actions,
                genes=genes,
                view=view,
                seed=seed,
                permutations=permutations,
            )
        )
        goal_relative = f"{view}/examples.goal.npy"
        if goal_relative in cache.manifest.get("files", {}):
            goals = np.asarray(cache.array(goal_relative), dtype=np.float64)[positions]
            unit_state = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
            unit_goal = goals / (np.linalg.norm(goals, axis=1, keepdims=True) + 1e-12)
            distance = 1.0 - np.sum(unit_state * unit_goal, axis=1)
            findings.extend(
                probes.alignment_findings(
                    distance.tolist(),
                    remaining_cost=targets,
                    depths=depths,
                    tasks=tasks,
                    genes=genes,
                    view=view,
                )
            )
        report["views"][view] = {
            "rows": len(usable),
            "hubness": probes.hubness(vectors[: min(len(vectors), 1500)]),
        }
    return findings, report


# ----------------------------------------------------------------- RQ2 ladder


def _oracle_scores(labels: dict[str, dict[str, float]]):
    def scorer(state, candidates, context):
        key = state.get("record_id", "")
        table = labels.get(key, {})
        return {tool: table.get(tool, 1e9) for tool in candidates}

    return scorer


def ladder_findings(
    package_dir: str | Path,
    processed_dir: str | Path,
    *,
    commits: Sequence[int] = (1, 2, 4, 8),
    seed: int = 17,
    max_tasks: int | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """Run the rungs that need no trained model, and report the rest as pending.

    The hindsight rung (the oracle's own ordering) and the random rung bracket
    every learned policy, so they are worth having before any training: a learned
    policy that does not sit between them is broken rather than interesting.
    """

    from geoflowagent.data.contracts import ContractEngine

    package = Path(package_dir)
    tasks = read_jsonl(package / "tasks.jsonl")
    tools = read_jsonl(package / "tools.jsonl")
    snapshots = read_jsonl(package / "snapshots.jsonl")
    verifiers = {
        row["task_id"]: row["private_verifier"]
        for row in read_jsonl(package / "verifiers.private.jsonl")
    }
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in snapshots:
        by_task[row["provenance"]["task_id"]].append(row)
    if max_tasks:
        tasks = tasks[:max_tasks]

    results: dict[str, list[ladder.Episode]] = collections.defaultdict(list)
    per_commit: dict[int, list[ladder.Episode]] = collections.defaultdict(list)

    for task in tasks:
        task_id = task["task_id"]
        engine = ContractEngine(tools, by_task[task_id])
        verifier = verifier_of = verifiers[task_id]
        required = list(task["provenance"]["required_families"])
        target = next(
            c["value"] for c in verifier_of["conditions"] if c["field"] == "report.label"
        )
        report_tool = "report_" + target.lower().replace(" ", "_")
        gene = task["group_ids"].get("entity", task_id)

        def candidates_fn(state, _task=task, _engine=engine):
            return _engine.applicable_tools(dict(state), _task["available_tools"])

        def execute_fn(state, tool, _engine=engine):
            transition = _engine.execute(dict(state), tool)
            payload = (transition.observation.get("output") or {}).get("payload") or {}
            points = int(payload.get("points", 0))
            cost = float(
                next(t for t in tools if t["tool_id"] == tool).get("search_cost", 1.0)
            ) + 1.0
            return transition.state, cost, points

        def solved_fn(state, _engine=engine, _v=verifier):
            return _engine.goal_satisfied(dict(state), _v)

        def terminal_fn(state):
            return bool((state.get("report") or {}).get("ready"))

        hindsight_plan = [f"probe_{family}" for family in required] + [report_tool]

        def hindsight_sampler(state, candidates, context, _plan=hindsight_plan):
            return [tool for tool in _plan if tool in candidates]

        rungs = {
            "hindsight_whole_plan": ladder.WholePlan(hindsight_sampler, name="hindsight"),
            "random": ladder.RandomPolicy(seed=seed),
        }
        for name, planner in rungs.items():
            results[name].append(
                ladder.run_episode(
                    planner=planner, task_id=task_id, gene=gene,
                    initial_state=task["initial_state"],
                    candidates_fn=candidates_fn, execute_fn=execute_fn,
                    solved_fn=solved_fn, terminal_fn=terminal_fn, commit=1,
                )
            )
        for commit in commits:
            per_commit[commit].append(
                ladder.run_episode(
                    planner=ladder.WholePlan(hindsight_sampler, name="hindsight"),
                    task_id=task_id, gene=gene, initial_state=task["initial_state"],
                    candidates_fn=candidates_fn, execute_fn=execute_fn,
                    solved_fn=solved_fn, terminal_fn=terminal_fn, commit=commit,
                )
            )
        results["blind_replan"].append(
            ladder.run_episode(
                planner=ladder.WholePlan(hindsight_sampler, name="hindsight"),
                task_id=task_id, gene=gene, initial_state=task["initial_state"],
                candidates_fn=candidates_fn, execute_fn=execute_fn,
                solved_fn=solved_fn, terminal_fn=terminal_fn, commit=1, observe=False,
            )
        )

    findings: list[Finding] = []
    reference = results["hindsight_whole_plan"]
    for name, episodes in sorted(results.items()):
        if name == "hindsight_whole_plan" or not episodes:
            continue
        findings.append(
            paired_contrast(
                claim_id="C3",
                name=f"ladder_hindsight_minus_{name}",
                left=[float(e.solved) for e in reference],
                right=[float(e.solved) for e in episodes],
                clusters=[e.gene for e in reference],
                unit="gene", role=Role.EXPLORATORY,
                detail={"rung": name, **ladder.summarise(episodes)},
            )
        )
    report = {
        "rungs": {name: ladder.summarise(eps) for name, eps in sorted(results.items())},
        "commit_curve": ladder.commit_curve(per_commit),
        "pending": [
            "greedy / beam over the learned distance head (needs train_value)",
            "flow whole-plan and raw-L2 ablation (needs train_flow)",
        ],
        "note": (
            "hindsight and random bracket every learned policy. A trained rung that "
            "falls outside the bracket is a bug, not a finding"
        ),
    }
    return findings, report


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
