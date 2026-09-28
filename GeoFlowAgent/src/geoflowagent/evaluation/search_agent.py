"""Closed-loop evaluation for the search-distilled value geometry."""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.embeddings.cache import EmbeddingCache, verify_evaluation_cache_superset
from geoflowagent.embeddings.runtime import RuntimeFeatureEncoder
from geoflowagent.evaluation.environment import SymbolicGenomicsEnvironment
from geoflowagent.search import EdgeCostConfig, SearchLimits, WeightedSearchOracle
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import load_value_geometry_checkpoint
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)

SEARCH_CONTEXT_FIELD = "__search_context__"
FAILURE_STATUSES = {
    "empty",
    "empty_result",
    "failure",
    "domain_error",
    "transport_error",
    "parse_error",
    "unobserved",
}


def _perturbation_assigned(
    task: Mapping[str, Any], perturbations: Sequence[Mapping[str, Any]]
) -> bool:
    """Use a policy-independent denominator for intervention robustness."""

    available = set(task["available_tools"])
    return any(
        item.get("task_id") in {None, "*", task["task_id"]}
        and (item.get("tool_id") in {None, "*"} or item.get("tool_id") in available)
        for item in perturbations
    )


@dataclass(frozen=True)
class SearchDecision:
    tool_id: str
    latency_seconds: float
    embedding_source: str
    completion_probability: float | None = None
    action_reachability_probability: float | None = None
    reachability_guard_fallback: bool = False


def _markov_state(
    state: Mapping[str, Any], *, depth: int, maximum_depth: int, include_context: bool
) -> dict[str, Any]:
    output = copy.deepcopy(dict(state))
    if include_context:
        output[SEARCH_CONTEXT_FIELD] = {
            "context": {},
            "depth": int(depth),
            "remaining_depth": max(0, int(maximum_depth) - int(depth)),
        }
    return output


def _failure_counts(history: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(row.get("tool_id"))
        for row in history
        if row.get("ok") is False or str(row.get("status")) in FAILURE_STATUSES
    )


class SearchValuePlanner:
    """One-step receding-horizon policy with exact contracts and retry budgets."""

    def __init__(
        self,
        model: torch.nn.Module,
        store: SearchFeatureStore,
        runtime: RuntimeFeatureEncoder,
        device: torch.device,
        *,
        search_depth: int,
        completion_threshold: float,
        reachability_threshold: float,
        retry_limit_per_tool: int,
        policy: str = "learned_policy",
        stop_contract_guard: bool = True,
        hard_reachability_guard: bool = True,
        runtime_embedding_policy: str = "cache_or_encode",
        seed: int = 17,
    ) -> None:
        if policy not in {"learned_policy", "learned_q", "random", "static_cost"}:
            raise ValueError(f"Unsupported search planner policy={policy!r}")
        if runtime_embedding_policy not in {"cache_or_encode", "cache_only", "encode"}:
            raise ValueError("runtime_embedding_policy must be cache_or_encode, cache_only, or encode")
        if retry_limit_per_tool < 0:
            raise ValueError("retry_limit_per_tool must be non-negative")
        self.model = model
        self.store = store
        self.runtime = runtime
        self.device = device
        self.search_depth = int(search_depth)
        self.completion_threshold = float(completion_threshold)
        self.reachability_threshold = float(reachability_threshold)
        self.retry_limit_per_tool = int(retry_limit_per_tool)
        self.policy = policy
        self.stop_contract_guard = bool(stop_contract_guard)
        self.hard_reachability_guard = bool(hard_reachability_guard)
        self.runtime_embedding_policy = runtime_embedding_policy
        self.rng = random.Random(seed)
        self.contracts = ContractEngine(store.tools)

    def _candidates(
        self,
        task: Mapping[str, Any],
        state: dict[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        failures = _failure_counts(history)
        return [
            tool_id
            for tool_id in task["available_tools"]
            if tool_id in self.store.tool_index
            and self.contracts.applicable(state, tool_id)
            and (
                self.retry_limit_per_tool == 0
                or failures[tool_id] < self.retry_limit_per_tool
            )
        ]

    @torch.inference_mode()
    def decide(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        background: list[dict[str, Any]],
    ) -> SearchDecision:
        start = time.perf_counter()
        candidates = self._candidates(task, state, history)
        if self.policy == "random":
            tool_id = self.rng.choice(candidates) if candidates else STOP_TOOL_ID
            return SearchDecision(tool_id, time.perf_counter() - start, "none")
        if self.policy == "static_cost":
            tool_id = (
                min(
                    candidates,
                    key=lambda item: (
                        float(self.contracts.tools[item].get("search_cost", 0.0)),
                        item,
                    ),
                )
                if candidates
                else STOP_TOOL_ID
            )
            return SearchDecision(tool_id, time.perf_counter() - start, "none")

        visible_state = _markov_state(
            state,
            depth=len(history),
            maximum_depth=self.search_depth,
            include_context=any(SEARCH_CONTEXT_FIELD in row["state"] for row in self.store.examples[:1]),
        )
        cached = (
            None
            if self.runtime_embedding_policy == "encode"
            else self.store.runtime_features(task["task_id"], visible_state, self.device)
        )
        if cached is None:
            if self.runtime_embedding_policy == "cache_only":
                raise RuntimeError(
                    f"No cached frozen features for task={task['task_id']} depth={len(history)}"
                )
            state_views, goal_views, structured_state, structured_goal = self.runtime.encode_search(
                task, visible_state, history, background
            )
            state_views, goal_views, structured_state, structured_goal = (
                self.store.ablate_runtime_inputs(
                    state_views,
                    goal_views,
                    structured_state,
                    structured_goal,
                )
            )
            state_views = {name: value.to(self.device) for name, value in state_views.items()}
            goal_views = {name: value.to(self.device) for name, value in goal_views.items()}
            structured_state = structured_state.to(self.device)
            structured_goal = structured_goal.to(self.device)
            embedding_source = "encoder"
        else:
            state_views, goal_views, structured_state, structured_goal = cached
            embedding_source = "cache"
        candidate_mask = torch.zeros(
            (1, len(self.store.tool_ids)), dtype=torch.bool, device=self.device
        )
        for tool_id in candidates:
            candidate_mask[0, self.store.tool_index[tool_id]] = True
        output = self.model(
            state_views,
            goal_views,
            {name: value.to(self.device) for name, value in self.store.tool_views.items()},
            structured_state,
            structured_goal,
            self.store.structured_tools.to(self.device),
            candidate_mask=candidate_mask,
            mask_unreachable=self.hard_reachability_guard,
            reachability_threshold=self.reachability_threshold,
        )
        completion_probability = float(output["completion_logit"].sigmoid().item())
        learned_stop_allowed = (
            not self.stop_contract_guard
            or self.contracts.goal_satisfied(state, task["goal"])
        )
        if (
            completion_probability >= self.completion_threshold and learned_stop_allowed
        ) or not candidates:
            return SearchDecision(
                STOP_TOOL_ID,
                time.perf_counter() - start,
                embedding_source,
                completion_probability,
            )
        if self.policy == "learned_q":
            score = -output["masked_q"]
        else:
            score = output["policy_logits"]
        # A learned reachability head may be over-conservative under shift.  Hard
        # screening is therefore allowed to remove predicted dead ends, but it
        # may not force an invalid early STOP when every contract-valid action is
        # below threshold.  In that corner case, fall back to the corresponding
        # soft score and let the exact contract/retry guards retain authority.
        reachability_fallback = not bool(
            torch.isfinite(score.masked_fill(~candidate_mask, -torch.inf)).any().item()
        )
        if reachability_fallback:
            if self.policy == "learned_q":
                score = -output["q"]
            else:
                score = -output["q"] + torch.nn.functional.logsigmoid(
                    output["action_reachability_logit"]
                )
        selected = int(score.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1).item())
        probability = float(output["action_reachability_logit"][0, selected].sigmoid().item())
        return SearchDecision(
            self.store.tool_ids[selected],
            time.perf_counter() - start,
            embedding_source,
            completion_probability,
            probability,
            reachability_fallback,
        )


class ExactSearchPlanner:
    """Private-goal upper bound used only as an evaluation control."""

    def __init__(
        self,
        tools: Sequence[dict[str, Any]],
        snapshots: Sequence[dict[str, Any]],
        private_goal: dict[str, Any],
        *,
        max_steps: int,
        retry_limit_per_tool: int,
        costs: EdgeCostConfig,
        store: SearchFeatureStore | None = None,
        search_depth: int | None = None,
    ) -> None:
        self.engine = ContractEngine(tools, snapshots)
        self.private_goal = private_goal
        self.max_steps = max_steps
        self.retry_limit_per_tool = retry_limit_per_tool
        self.costs = costs
        self.store = store
        self.search_depth = int(search_depth if search_depth is not None else max_steps)

    def decide(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        background: list[dict[str, Any]],
    ) -> SearchDecision:
        del background
        start = time.perf_counter()
        if self.engine.goal_satisfied(state, self.private_goal):
            return SearchDecision(STOP_TOOL_ID, time.perf_counter() - start, "none")
        failures = _failure_counts(history)
        # Clean states already have certified labels in the immutable search
        # package. Reusing them gives exactly the same private-verifier oracle
        # decision without recomputing an exponential search at every step.
        # Once a runtime-only failure occurs, fall back to a fresh search with
        # the failed tool removed by the retry budget.
        if self.store is not None and not failures:
            row = _row_for_state(
                self.store,
                task["task_id"],
                state,
                depth=len(history),
                search_depth=self.search_depth,
            )
            if row is not None:
                optimal = list(row.get("optimal_actions", []))
                if optimal:
                    return SearchDecision(
                        str(optimal[0]), time.perf_counter() - start, "cache"
                    )
        available = [
            tool_id
            for tool_id in task["available_tools"]
            if self.retry_limit_per_tool == 0
            or failures[tool_id] < self.retry_limit_per_tool
        ]
        remaining = max(0, self.max_steps - len(history))
        result = WeightedSearchOracle(
            self.engine,
            cost_config=self.costs,
            limits=SearchLimits(
                max_depth=remaining,
                max_states=20_000,
                top_k_paths=1,
            ),
        ).search(state, self.private_goal, available)
        action = (
            result.initial_value.optimal_actions[0]
            if result.initial_value.optimal_actions
            else STOP_TOOL_ID
        )
        return SearchDecision(action, time.perf_counter() - start, "none")


def _row_for_state(
    store: SearchFeatureStore,
    task_id: str,
    state: dict[str, Any],
    *,
    depth: int,
    search_depth: int,
) -> dict[str, Any] | None:
    visible = _markov_state(
        state,
        depth=depth,
        maximum_depth=search_depth,
        include_context=any(SEARCH_CONTEXT_FIELD in row["state"] for row in store.examples[:1]),
    )
    index = store.state_index(task_id, visible)
    return store.examples[index] if index is not None else None


def run_search_episode(
    planner: SearchValuePlanner | ExactSearchPlanner,
    store: SearchFeatureStore,
    task: dict[str, Any],
    tools: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]],
    private_goal: dict[str, Any],
    background: list[dict[str, Any]],
    *,
    max_steps: int,
    search_depth: int,
    perturbations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    environment = SymbolicGenomicsEnvironment(
        task,
        list(tools),
        snapshots=list(snapshots),
        verification_goal=private_goal,
        perturbations=list(perturbations),
        max_steps=max_steps,
    )
    trace: list[dict[str, Any]] = []
    latency = 0.0
    cache_calls = 0
    runtime_calls = 0
    known_regrets: list[float] = []
    optimal_hits: list[float] = []
    invalid_calls = 0
    premature_stops = 0
    perturbation_assigned = _perturbation_assigned(task, perturbations)
    while not environment.done and environment.steps < max_steps:
        label_row = _row_for_state(
            store,
            task["task_id"],
            environment.state,
            depth=len(environment.history),
            search_depth=search_depth,
        )
        decision = planner.decide(
            task, environment.state, environment.history, background
        )
        latency += decision.latency_seconds
        cache_calls += int(decision.embedding_source == "cache")
        runtime_calls += int(decision.embedding_source == "encoder")
        regret: float | None = None
        optimal: bool | None = None
        if label_row is not None and decision.tool_id != STOP_TOOL_ID:
            action = next(
                (
                    row
                    for row in label_row.get("action_supervision", [])
                    if row["tool_id"] == decision.tool_id
                ),
                None,
            )
            if action is not None and action.get("regret") is not None:
                regret = float(action["regret"])
                optimal = bool(action.get("is_optimal"))
                known_regrets.append(regret)
                optimal_hits.append(float(optimal))
        step = environment.step(decision.tool_id)
        premature_stop = decision.tool_id == STOP_TOOL_ID and not step.valid_action
        premature_stops += int(premature_stop)
        invalid_calls += int(decision.tool_id != STOP_TOOL_ID and not step.valid_action)
        trace.append(
            {
                "step": environment.steps,
                "tool_id": decision.tool_id,
                "completion_probability": decision.completion_probability,
                "action_reachability_probability": decision.action_reachability_probability,
                "reachability_guard_fallback": decision.reachability_guard_fallback,
                "regret": regret,
                "optimal_action": optimal,
                "contract_valid": step.valid_action,
                "execution_success": step.execution_success,
                "observation": step.observation,
                "goal_satisfied": step.goal_satisfied,
                "embedding_source": decision.embedding_source,
            }
        )
        if decision.tool_id == STOP_TOOL_ID:
            break
    goal_reached = environment.engine.goal_satisfied(environment.state, private_goal)
    stopped = bool(trace) and trace[-1]["tool_id"] == STOP_TOOL_ID
    correct_stop = stopped and bool(trace[-1]["observation"].get("ok"))
    success = goal_reached and correct_stop
    return {
        "task_id": task["task_id"],
        "success": success,
        "goal_reached": goal_reached,
        "correct_stop": correct_stop,
        "steps": environment.steps,
        "tool_calls": sum(row["tool_id"] != STOP_TOOL_ID for row in trace),
        "invalid_calls": invalid_calls,
        "premature_stops": premature_stops,
        "reachability_guard_fallbacks": sum(
            bool(row["reachability_guard_fallback"]) for row in trace
        ),
        "mean_known_regret": float(np.mean(known_regrets)) if known_regrets else None,
        "optimal_action_rate": float(np.mean(optimal_hits)) if optimal_hits else None,
        "regret_label_coverage": len(known_regrets)
        / max(1, sum(row["tool_id"] != STOP_TOOL_ID for row in trace)),
        "planner_latency_seconds": latency,
        "cache_embedding_calls": cache_calls,
        "runtime_embedding_calls": runtime_calls,
        "cache_hit_rate": cache_calls / max(1, cache_calls + runtime_calls),
        "perturbation_assigned": perturbation_assigned,
        "perturbation_applied": bool(environment.applied_perturbations),
        "recovered_after_perturbation": bool(environment.applied_perturbations) and success,
        "final_state": environment.state,
        "trace": trace,
    }


def _bootstrap(rows: Sequence[dict[str, Any]], field: str, *, seed: int, reps: int) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not len(values):
        return {"mean": None, "low": None, "high": None, "tasks": 0}
    point = float(values.mean())
    if len(values) == 1 or reps <= 0:
        return {"mean": point, "low": point, "high": point, "tasks": len(values)}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return {
        "mean": point,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "tasks": len(values),
    }


def _summarize(rows: Sequence[dict[str, Any]], *, seed: int, reps: int) -> dict[str, Any]:
    perturbed = [row for row in rows if row["perturbation_applied"]]
    assigned = [row for row in rows if row["perturbation_assigned"]]
    regrets = [float(row["mean_known_regret"]) for row in rows if row["mean_known_regret"] is not None]
    optimal = [float(row["optimal_action_rate"]) for row in rows if row["optimal_action_rate"] is not None]
    return {
        "tasks": len(rows),
        "success": _bootstrap(rows, "success", seed=seed, reps=reps),
        "goal_reached_rate": float(np.mean([row["goal_reached"] for row in rows])),
        "correct_stop_rate": float(np.mean([row["correct_stop"] for row in rows])),
        "mean_tool_calls": float(np.mean([row["tool_calls"] for row in rows])),
        "mean_invalid_calls": float(np.mean([row["invalid_calls"] for row in rows])),
        "mean_premature_stops": float(np.mean([row["premature_stops"] for row in rows])),
        "mean_reachability_guard_fallbacks": float(
            np.mean([row["reachability_guard_fallbacks"] for row in rows])
        ),
        "mean_known_regret": float(np.mean(regrets)) if regrets else None,
        "mean_optimal_action_rate": float(np.mean(optimal)) if optimal else None,
        "mean_regret_label_coverage": float(
            np.mean([row["regret_label_coverage"] for row in rows])
        ),
        "mean_planner_latency_seconds": float(
            np.mean([row["planner_latency_seconds"] for row in rows])
        ),
        "mean_cache_hit_rate": float(np.mean([row["cache_hit_rate"] for row in rows])),
        "assigned_perturbation_tasks": len(assigned),
        "assigned_task_success_rate": (
            float(np.mean([row["success"] for row in assigned])) if assigned else None
        ),
        "perturbation_triggered_tasks": len(perturbed),
        "triggered_recovery_rate": (
            float(np.mean([row["recovered_after_perturbation"] for row in perturbed]))
            if perturbed
            else None
        ),
    }


def evaluate_search_agent(
    processed_dir: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    split: str = "dev",
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError("Search-agent evaluation uses dev or explicitly unsealed test")
    config = read_yaml(config_path)
    eval_config = config.get("search_agent_evaluation", {})
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    model, checkpoint = load_value_geometry_checkpoint(checkpoint_path, device)
    store = SearchFeatureStore(
        processed_dir,
        cache_dir,
        structured_dim=int(checkpoint["model_config"]["structured_dim"]),
    )
    store.apply_feature_ablation(checkpoint.get("feature_ablation"))
    if checkpoint["views"] != list(store.views):
        raise ValueError("Value checkpoint view order differs from the evaluation cache")
    registry_superset = None
    if checkpoint["tool_ids"] != store.tool_ids:
        training_cache_dir = eval_config.get("training_cache_dir")
        if training_cache_dir is None:
            raise ValueError("Value checkpoint tool order differs from the evaluation cache")
        if checkpoint["tool_ids"] != EmbeddingCache(training_cache_dir).tool_ids:
            raise ValueError("Configured training cache does not match checkpoint tool order")
        registry_superset = verify_evaluation_cache_superset(training_cache_dir, store.cache)
    same_training_cache = checkpoint["cache_content_sha256"] == store.cache.content_sha256
    if not same_training_cache:
        expected_fingerprint = checkpoint.get("cache_evaluation_fingerprint_sha256")
        if expected_fingerprint is None:
            raise ValueError(
                "Legacy value checkpoint may only use its exact training cache"
            )
        if (
            expected_fingerprint != store.cache.evaluation_fingerprint_sha256
            and registry_superset is None
        ):
            raise ValueError(
                "Value checkpoint and evaluation cache use different frozen coordinates "
                "or tool prototypes"
            )
        if checkpoint.get("cache_config_sha256") != store.cache.manifest.get(
            "config_sha256"
        ):
            raise ValueError("Value checkpoint and evaluation cache use different configs")
    embedding_config = read_yaml(config_path).get("embedding", {})
    if sha256_text(canonical_json(embedding_config)) != store.cache.manifest["config_sha256"]:
        raise ValueError("Runtime embedding settings differ from the immutable cache")
    runtime = RuntimeFeatureEncoder(
        config_path,
        list(store.views),
        structured_dim=store.structured_dim,
        expected_view_metadata=store.cache.manifest["views"],
    )
    processed_dir = Path(processed_dir)
    tasks = [row for row in read_jsonl(processed_dir / "tasks.jsonl") if row["split"] == split]
    if not tasks:
        raise ValueError(f"No tasks in evaluation split={split!r}")
    tools = read_jsonl(processed_dir / "tools.jsonl")
    snapshots = read_jsonl(processed_dir / "snapshots.jsonl")
    verifier_rows = read_jsonl(processed_dir / "verifiers.private.jsonl")
    verifiers = {row["task_id"]: row["private_verifier"] for row in verifier_rows}
    background_rows = (
        read_jsonl(processed_dir / "background.jsonl")
        if (processed_dir / "background.jsonl").exists()
        else []
    )
    background_map = {row["card_id"]: row for row in background_rows}
    manifest = read_json(processed_dir / "manifest.json")
    search_depth = int(manifest.get("oracle_metadata", {}).get("limits", {}).get("max_depth", 16))
    max_steps = int(eval_config.get("max_steps", search_depth))
    retry_limit = int(eval_config.get("retry_limit_per_tool", 1))
    runtime_policy = str(eval_config.get("runtime_embedding_policy", "cache_or_encode"))
    costs_config = manifest.get("oracle_metadata", {}).get("cost_config", {})
    costs = EdgeCostConfig(**costs_config) if costs_config else EdgeCostConfig()
    scenarios = {
        "clean": [],
        "perturbed": list(eval_config.get("perturbations", [])),
    }
    conditions = (
        ("exact_search_oracle", "exact_search_oracle", True, True),
        ("learned_policy", "learned_policy", True, True),
        ("learned_policy_soft_reachability", "learned_policy", True, False),
        ("learned_policy_no_stop_guard", "learned_policy", False, True),
        ("learned_q", "learned_q", True, True),
        ("static_cost", "static_cost", True, True),
        ("random", "random", True, True),
    )
    details: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for scenario_name, perturbations in scenarios.items():
        for task_index, task in enumerate(sorted(tasks, key=lambda row: row["task_id"])):
            private_goal = combine_verification_goal(
                task["goal"], verifiers[task["task_id"]]
            )
            background = [
                background_map[card_id]
                for card_id in task.get("background_ids", [])
                if card_id in background_map
            ]
            for condition, planner_policy, stop_guard, hard_reachability in conditions:
                if condition == "exact_search_oracle":
                    planner: SearchValuePlanner | ExactSearchPlanner = ExactSearchPlanner(
                        tools,
                        snapshots,
                        private_goal,
                        max_steps=max_steps,
                        retry_limit_per_tool=retry_limit,
                        costs=costs,
                        store=store,
                        search_depth=search_depth,
                    )
                else:
                    planner = SearchValuePlanner(
                        model,
                        store,
                        runtime,
                        device,
                        search_depth=search_depth,
                        completion_threshold=float(checkpoint["completion_threshold"]),
                        reachability_threshold=float(checkpoint["reachability_threshold"]),
                        retry_limit_per_tool=retry_limit,
                        policy=planner_policy,
                        stop_contract_guard=stop_guard,
                        hard_reachability_guard=hard_reachability,
                        runtime_embedding_policy=runtime_policy,
                        seed=seed + task_index,
                    )
                row = run_search_episode(
                    planner,
                    store,
                    task,
                    tools,
                    snapshots,
                    private_goal,
                    background,
                    max_steps=max_steps,
                    search_depth=search_depth,
                    perturbations=perturbations,
                )
                detail = {"scenario": scenario_name, "condition": condition, **row}
                details.append(detail)
                grouped[(scenario_name, condition)].append(row)
    bootstrap_reps = int(eval_config.get("bootstrap_reps", 1000))
    summaries = [
        {
            "scenario": scenario,
            "condition": condition,
            **_summarize(rows, seed=seed, reps=bootstrap_reps),
        }
        for (scenario, condition), rows in sorted(grouped.items())
    ]
    for summary in summaries:
        scenario = str(summary["scenario"])
        condition = str(summary["condition"])
        oracle = {
            row["task_id"]: row
            for row in grouped.get((scenario, "exact_search_oracle"), [])
        }
        paired = [
            float(row["tool_calls"] - oracle[row["task_id"]]["tool_calls"])
            for row in grouped[(scenario, condition)]
            if row["task_id"] in oracle
        ]
        summary["mean_excess_tool_calls_vs_exact"] = (
            float(np.mean(paired)) if paired else None
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{split}_details.jsonl"
    write_jsonl(details_path, details)
    result = {
        "evaluation_split": split,
        "test_unsealed": split == "test",
        "seed": seed,
        "conditions": summaries,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cache_content_sha256": store.cache.content_sha256,
        "checkpoint_training_cache_sha256": checkpoint["cache_content_sha256"],
        "evaluation_cache_matches_training_cache": same_training_cache,
        "registry_superset_verification": registry_superset,
        "cache_evaluation_fingerprint_sha256": (
            store.cache.evaluation_fingerprint_sha256
        ),
        "details_sha256": sha256_file(details_path),
        "evaluation_config": eval_config,
        "run_provenance": collect_run_provenance(config_path, device=device),
        "notes": {
            "exact_search_oracle": "Private-verifier upper bound; never deployable model input.",
            "retry_guard": (
                "Failed/empty tools are masked after the configured retry limit; this is an "
                "exact safety controller, not a learned genomics judgment."
            ),
            "reachability_guard": (
                "learned_policy uses the dev-calibrated action-reachability threshold; the "
                "soft-reachability condition is its ablation. If every valid action is screened, "
                "the planner falls back to the soft ranking instead of forcing an invalid STOP."
            ),
            "success": "Private goal reached and STOP emitted correctly.",
            "paired_efficiency": (
                "Excess tool calls are paired by task against the exact-search condition "
                "under the same clean or perturbed scenario."
            ),
            "test_policy": "Test is reported only when split=test is explicitly requested.",
        },
    }
    write_json(output_dir / f"{split}_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate search-distilled agent closed loop")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_search_agent(
        args.processed_dir,
        args.cache_dir,
        args.checkpoint,
        args.output_dir,
        args.config,
        split=args.split,
    )
    print(result["conditions"])


if __name__ == "__main__":
    main()
