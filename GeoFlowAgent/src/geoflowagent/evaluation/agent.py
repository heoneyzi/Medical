from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, combine_verification_goal
from geoflowagent.data.serialize import runtime_context_key
from geoflowagent.embeddings.runtime import RuntimeFeatureEncoder
from geoflowagent.evaluation.environment import SymbolicGenomicsEnvironment
from geoflowagent.training.features import FeatureStore
from geoflowagent.training.flow import (
    load_flow_checkpoint,
    masked_tool_set_context,
)
from geoflowagent.training.metric import load_metric_checkpoint
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
from geoflowagent.utils.reporting import markdown_table, write_markdown
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)


@dataclass
class PlanOutput:
    tool_ids: list[str]
    nfe: int
    latency_seconds: float
    embedding_source: str = "none"


class LearnedPlanner:
    def __init__(
        self,
        geometry: torch.nn.Module,
        store: FeatureStore,
        runtime: RuntimeFeatureEncoder,
        device: torch.device,
        *,
        flow: torch.nn.Module | None = None,
        nfe: int = 16,
        stop_threshold: float = 0.5,
        seed: int = 17,
        runtime_embedding_policy: str = "cache_or_encode",
        apply_contract_mask: bool = True,
        selection_policy: str = "learned",
    ) -> None:
        self.geometry = geometry
        self.store = store
        self.runtime = runtime
        self.device = device
        self.flow = flow
        self.nfe = nfe
        self.stop_threshold = stop_threshold
        self.seed = seed
        self.apply_contract_mask = bool(apply_contract_mask)
        if selection_policy not in {"learned", "random"}:
            raise ValueError("selection_policy must be 'learned' or 'random'")
        self.selection_policy = selection_policy
        if runtime_embedding_policy not in {"cache_or_encode", "cache_only", "encode"}:
            raise ValueError(
                "runtime_embedding_policy must be cache_or_encode, cache_only, or encode"
            )
        self.runtime_embedding_policy = runtime_embedding_policy
        self.calls = 0
        self.engine = ContractEngine(store.tools)

    @torch.inference_mode()
    def plan(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        background_cards: list[dict[str, Any]],
    ) -> PlanOutput:
        start = time.perf_counter()
        self.calls += 1
        if self.flow is None and self.engine.goal_satisfied(state, task["goal"]):
            return PlanOutput([STOP_TOOL_ID], 0, time.perf_counter() - start)
        if self.selection_policy == "random":
            candidates = [
                tool_id
                for tool_id in task["available_tools"]
                if tool_id in self.store.tool_index
                and (not self.apply_contract_mask or self.engine.applicable(state, tool_id))
            ]
            if not candidates:
                return PlanOutput([STOP_TOOL_ID], 0, time.perf_counter() - start)
            choice = random.Random(self.seed + self.calls).choice(candidates)
            return PlanOutput([choice], 0, time.perf_counter() - start)
        cached = (
            None
            if self.runtime_embedding_policy == "encode"
            else self.store.runtime_features(task, state, history, self.device)
        )
        if cached is None:
            if self.runtime_embedding_policy == "cache_only":
                raise RuntimeError(
                    f"No frozen embedding cached for runtime state of task {task['task_id']}"
                )
            all_views, structured = self.runtime.encode(task, state, history, background_cards)
            all_views = {name: value.to(self.device) for name, value in all_views.items()}
            structured = structured.to(self.device)
            embedding_source = "encoder"
        else:
            all_views, structured = cached
            embedding_source = "cache"
        views = {name: all_views[name] for name in self.store.views}
        aux_views = {name: all_views[name] for name in self.store.aux_views}
        tool_views = {name: value.to(self.device) for name, value in self.store.tool_views.items()}
        structured_tools = self.store.structured_tools.to(self.device)
        candidate = torch.zeros((1, len(self.store.tool_ids)), dtype=torch.bool, device=self.device)
        contract = torch.zeros_like(candidate)
        for tool_id in task["available_tools"]:
            if tool_id in self.store.tool_index:
                candidate[0, self.store.tool_index[tool_id]] = True
                if self.engine.applicable(state, tool_id):
                    contract[0, self.store.tool_index[tool_id]] = True
        if self.flow is None and self.apply_contract_mask and not contract.any():
            return PlanOutput([STOP_TOOL_ID], 0, time.perf_counter() - start, embedding_source)
        if self.flow is None:
            result = self.geometry(
                views,
                tool_views,
                structured,
                structured_tools,
                aux_views=aux_views,
                candidate_mask=candidate,
                contract_mask=contract,
                apply_contract_mask=self.apply_contract_mask,
            )
            tool_index = int(result["energy"].argmin(dim=1).item())
            return PlanOutput(
                [self.store.tool_ids[tool_index]],
                0,
                time.perf_counter() - start,
                embedding_source,
            )

        prototypes = self.geometry.tool_prototypes(tool_views)
        context = self.geometry.encode_context(views, structured, aux_views)
        set_context = masked_tool_set_context(prototypes, candidate)
        # Generate on CPU for reproducible support across CPU/CUDA/MPS; MPS does
        # not consistently expose a device-local torch.Generator.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.calls)
        noise = torch.randn(
            1,
            self.flow.max_plan_length,
            self.flow.dim,
            generator=generator,
            device="cpu",
        ).to(self.device)
        plan, _ = self.flow.solve(
            context, set_context, nfe=self.nfe, noise=noise, return_transport=False
        )
        decoded = self.flow.decode(plan, prototypes, candidate, stop_threshold=self.stop_threshold)[
            0
        ]
        if not decoded:
            return PlanOutput(
                [STOP_TOOL_ID], self.nfe, time.perf_counter() - start, embedding_source
            )
        if self.apply_contract_mask and not contract.any():
            return PlanOutput(
                [STOP_TOOL_ID], self.nfe, time.perf_counter() - start, embedding_source
            )
        # Exact contract applies to the action that will actually be executed now.
        first = decoded[0]
        if self.apply_contract_mask and not bool(contract[0, first]):
            anchor = F.normalize(plan[:, 0], dim=-1)
            score = 1.0 - anchor @ F.normalize(prototypes, dim=-1).T
            score = score.masked_fill(~contract, torch.finfo(score.dtype).max / 100)
            decoded[0] = int(score.argmin(dim=1).item())
        tool_ids = [self.store.tool_ids[index] for index in decoded]
        if len(decoded) < self.flow.max_plan_length:
            tool_ids.append(STOP_TOOL_ID)
        return PlanOutput(tool_ids, self.nfe, time.perf_counter() - start, embedding_source)


def _background_for_task(
    task: dict[str, Any], card_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [card_map[card_id] for card_id in task.get("background_ids", []) if card_id in card_map]


def _perturbation_assigned(task: dict[str, Any], perturbations: list[dict[str, Any]]) -> bool:
    """Whether the protocol assigned at least one perturbation to this task.

    This is determined before running a policy. It therefore gives an unbiased
    denominator even when a policy never selects the tool that would trigger the
    perturbation.
    """

    available = set(task["available_tools"])
    for item in perturbations:
        if item.get("task_id") not in {None, "*", task["task_id"]}:
            continue
        if item.get("tool_id") in {None, "*"} or item.get("tool_id") in available:
            return True
    return False


def _validate_checkpoint_chain(
    metric_checkpoint: str | Path,
    geometry_checkpoint: dict[str, Any],
    flow_meta: dict[str, Any],
    store: FeatureStore,
) -> None:
    """Fail closed unless cache, geometry, and flow describe one coordinate system."""

    if sha256_file(metric_checkpoint) != flow_meta.get("metric_checkpoint_sha256"):
        raise ValueError("Flow checkpoint was trained with a different metric checkpoint")
    for name, metadata in (("Metric", geometry_checkpoint), ("Flow", flow_meta)):
        if metadata.get("tool_ids") != store.tool_ids:
            raise ValueError(f"{name} checkpoint tool IDs differ from current dataset")
        if metadata.get("cache_config_sha256") != store.cache.manifest["config_sha256"]:
            raise ValueError(f"{name} checkpoint was trained on a different embedding config")
        if metadata.get("cache_content_sha256") != store.cache.content_sha256:
            raise ValueError(f"{name} checkpoint was trained on different embedding cache contents")
        if metadata.get("source_manifest_sha256") != store.cache.manifest["source_manifest_sha256"]:
            raise ValueError(f"{name} checkpoint was trained on a different processed dataset")
        if metadata.get("feature_spec_version") != store.cache.manifest["feature_spec_version"]:
            raise ValueError(f"{name} checkpoint uses different feature semantics")


def run_episode(
    planner: LearnedPlanner,
    task: dict[str, Any],
    tools: list[dict[str, Any]],
    background_cards: list[dict[str, Any]],
    *,
    snapshots: list[dict[str, Any]] | None = None,
    verification_goal: dict[str, Any] | None = None,
    commit_horizon: int,
    feedback: bool,
    max_steps: int,
    perturbations: list[dict[str, Any]],
    counterfactual_policy: str = "symbolic",
    planner_call_budget: int | None = None,
) -> dict[str, Any]:
    environment = SymbolicGenomicsEnvironment(
        task,
        tools,
        snapshots=snapshots,
        verification_goal=verification_goal,
        perturbations=perturbations,
        max_steps=max_steps,
    )
    planner_calls = 0
    total_nfe = 0
    total_latency = 0.0
    cache_embedding_calls = 0
    runtime_embedding_miss_calls = 0
    zero_regret_actions = 0
    known_regret_actions = 0
    contract_valid_actions = 0
    execution_success_actions = 0
    invalid_actions = 0
    redundant_calls = 0
    regrets = []
    contract_candidate_counts: list[int] = []
    trace = []
    perturbation_assigned = _perturbation_assigned(task, perturbations)
    hidden_state = copy.deepcopy(environment.state)
    hidden_history: list[dict[str, Any]] = []
    while (
        not environment.done
        and environment.steps < max_steps
        and (planner_call_budget is None or planner_calls < planner_call_budget)
    ):
        visible_state = environment.state if feedback else hidden_state
        visible_history = environment.history if feedback else hidden_history
        plan = planner.plan(task, visible_state, visible_history, background_cards)
        planner_calls += 1
        total_nfe += plan.nfe
        total_latency += plan.latency_seconds
        cache_embedding_calls += int(plan.embedding_source == "cache")
        runtime_embedding_miss_calls += int(plan.embedding_source == "encoder")
        provisional = plan.tool_ids or [STOP_TOOL_ID]
        for tool_id in provisional[:commit_horizon]:
            contract_candidates = environment.engine.applicable_tools(
                environment.state, task["available_tools"]
            )
            contract_candidate_counts.append(len(contract_candidates))
            if counterfactual_policy == "observed_only":
                context_index = planner.store.runtime_context_index.get(
                    runtime_context_key(task, environment.state, environment.history)
                )
                if context_index is None:
                    # A cache miss makes demonstrated counterfactual outcomes
                    # unknown, but terminality itself is still an exact contract
                    # fact.  The planner must emit STOP; the evaluator does not
                    # inject it.  Once emitted at a verified goal state, however,
                    # STOP is the sole valid zero-regret action even for a novel
                    # runtime context.
                    terminal = environment.engine.goal_satisfied(
                        environment.state, environment.verification_goal
                    )
                    valid_set = [STOP_TOOL_ID] if terminal else []
                    regret = 0.0 if tool_id == STOP_TOOL_ID and terminal else None
                else:
                    label_row = planner.store.examples[context_index]
                    valid_set = list(label_row["valid_next_tools"])
                    value = label_row.get("action_regret", {}).get(tool_id)
                    known = bool(label_row.get("action_regret_mask", {}).get(tool_id, False))
                    regret = float(value) if known and value is not None else None
            else:
                valid_set, regret_map = environment.engine.action_analysis(
                    environment.state, environment.verification_goal, task["available_tools"]
                )
                regret = regret_map.get(tool_id)
                if (
                    tool_id != STOP_TOOL_ID
                    and environment.engine.applicable(environment.state, tool_id)
                    and not environment.engine.executable(environment.state, tool_id)
                ):
                    regret = None
            regret_known = regret is not None
            if regret_known:
                regrets.append(float(regret))
                known_regret_actions += 1
            result = environment.step(tool_id)
            zero_regret_actions += int(regret_known and tool_id in valid_set)
            contract_valid_actions += int(result.valid_action)
            execution_success_actions += int(result.execution_success)
            invalid_actions += int(not result.valid_action)
            redundant_calls += int(result.redundant and tool_id != STOP_TOOL_ID)
            trace.append(
                {
                    "step": environment.steps,
                    "tool_id": tool_id,
                    "contract_valid_candidates": contract_candidates,
                    "contract_valid_candidate_count": len(contract_candidates),
                    "valid_set": valid_set,
                    "regret": regret,
                    "regret_known": regret_known,
                    "contract_valid": result.valid_action,
                    "execution_success": result.execution_success,
                    "zero_regret": tool_id in valid_set if regret_known else None,
                    "observation": result.observation,
                    "goal_satisfied": result.goal_satisfied,
                }
            )
            if environment.done:
                break
        if provisional == [STOP_TOOL_ID]:
            break
    # Compute-matched controls may terminate early; spend the remaining planner budget
    # without executing additional actions so total calls/NFE stay comparable.
    while planner_call_budget is not None and planner_calls < planner_call_budget:
        visible_state = environment.state if feedback else hidden_state
        visible_history = environment.history if feedback else hidden_history
        plan = planner.plan(task, visible_state, visible_history, background_cards)
        planner_calls += 1
        total_nfe += plan.nfe
        total_latency += plan.latency_seconds
        cache_embedding_calls += int(plan.embedding_source == "cache")
        runtime_embedding_miss_calls += int(plan.embedding_source == "encoder")
    goal_reached = environment.engine.goal_satisfied(
        environment.state, environment.verification_goal
    )
    stopped = bool(environment.history) and environment.history[-1].get("tool_id") == STOP_TOOL_ID
    correct_stop = stopped and bool(environment.history[-1].get("ok"))
    success = goal_reached and correct_stop
    return {
        "task_id": task["task_id"],
        "success": success,
        "goal_reached": goal_reached,
        "correct_stop": correct_stop,
        "steps": environment.steps,
        "planner_calls": planner_calls,
        "total_nfe": total_nfe,
        "planner_latency_seconds": total_latency,
        "cache_embedding_calls": cache_embedding_calls,
        "runtime_embedding_miss_calls": runtime_embedding_miss_calls,
        "embedding_cache_hit_rate": cache_embedding_calls
        / max(1, cache_embedding_calls + runtime_embedding_miss_calls),
        "contract_valid_action_rate": contract_valid_actions / max(1, len(trace)),
        "execution_success_rate": execution_success_actions / max(1, len(trace)),
        "zero_regret_action_rate": (
            zero_regret_actions / known_regret_actions if known_regret_actions else None
        ),
        "regret_label_coverage": known_regret_actions / max(1, len(trace)),
        "mean_contract_valid_candidates": (
            float(np.mean(contract_candidate_counts)) if contract_candidate_counts else 0.0
        ),
        "invalid_calls": invalid_actions,
        "redundant_calls": redundant_calls,
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "perturbation_assigned": perturbation_assigned,
        "perturbation_applied": bool(environment.applied_perturbations),
        "recovered_after_perturbation": bool(environment.applied_perturbations) and success,
        "final_state": environment.state,
        "trace": trace,
    }


def evaluate_agent(
    processed_dir: str | Path,
    cache_dir: str | Path,
    embedding_config: str | Path,
    metric_checkpoint: str | Path,
    flow_checkpoint: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    split: str = "dev",
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError("Agent evaluation split must be 'dev' or explicitly unsealed 'test'")
    config = read_yaml(config_path)
    eval_config = config.get("agent_evaluation", {})
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    run_provenance = collect_run_provenance(config_path, device=device)
    geometry, geometry_checkpoint = load_metric_checkpoint(metric_checkpoint, device)
    flow, flow_meta = load_flow_checkpoint(flow_checkpoint, device)
    store = FeatureStore(
        processed_dir,
        cache_dir,
        structured_dim=int(geometry_checkpoint["model_config"]["structured_dim"]),
    )
    embedding_settings = read_yaml(embedding_config)
    embedding_settings = embedding_settings.get("embedding", embedding_settings)
    if sha256_text(canonical_json(embedding_settings)) != store.cache.manifest["config_sha256"]:
        raise ValueError("Runtime embedding config differs from the immutable training cache")
    _validate_checkpoint_chain(metric_checkpoint, geometry_checkpoint, flow_meta, store)
    runtime = RuntimeFeatureEncoder(
        embedding_config,
        store.views + store.aux_views,
        structured_dim=store.structured_dim,
        expected_view_metadata=store.cache.manifest["views"],
    )
    runtime_embedding_policy = str(eval_config.get("runtime_embedding_policy", "cache_or_encode"))
    prewarm_runtime_encoders = bool(eval_config.get("prewarm_runtime_encoders", False))
    if prewarm_runtime_encoders and runtime_embedding_policy != "cache_only":
        runtime.warmup()
    tasks = read_jsonl(Path(processed_dir) / "tasks.jsonl")
    evaluation_task_ids = {row["task_id"] for row in store.examples if row["split"] == split}
    tasks = sorted(
        (row for row in tasks if row["task_id"] in evaluation_task_ids),
        key=lambda row: row["task_id"],
    )
    if not tasks:
        raise ValueError(f"No tasks in requested agent evaluation split {split!r}")
    background_path = Path(processed_dir) / "background.jsonl"
    background = read_jsonl(background_path) if background_path.exists() else []
    card_map = {row["card_id"]: row for row in background}
    verifier_path = Path(processed_dir) / "verifiers.private.jsonl"
    verifier_rows = read_jsonl(verifier_path) if verifier_path.exists() else []
    verifier_map = {row["task_id"]: row["private_verifier"] for row in verifier_rows}
    snapshot_path = Path(processed_dir) / "snapshots.jsonl"
    snapshots = read_jsonl(snapshot_path) if snapshot_path.exists() else []
    processed_manifest = read_json(Path(processed_dir) / "manifest.json")
    counterfactual_policy = str(processed_manifest.get("counterfactual_policy", "symbolic"))
    perturbations = list(eval_config.get("perturbations", []))
    max_steps = int(eval_config.get("max_steps", 16))
    nfe = int(eval_config.get("nfe", flow_meta.get("nfe", 16)))
    stop_threshold = float(eval_config.get("stop_threshold", flow_meta.get("stop_threshold", 0.5)))
    conditions = [
        {
            "name": "random_contract_oracle_stop",
            "use_flow": False,
            "selection_policy": "random",
            "contract_mask": True,
            "stop_controller": "public_goal_oracle",
            "commit": 1,
            "feedback": True,
        },
        {
            "name": "metric_no_contract_mask_oracle_stop",
            "use_flow": False,
            "selection_policy": "learned",
            "contract_mask": False,
            "stop_controller": "public_goal_oracle",
            "commit": 1,
            "feedback": True,
        },
        {
            "name": "metric_closed_loop",
            "use_flow": False,
            "selection_policy": "learned",
            "contract_mask": True,
            "stop_controller": "public_goal_oracle",
            "commit": 1,
            "feedback": True,
        },
        {
            "name": "flow_open_loop",
            "use_flow": True,
            "selection_policy": "learned",
            "contract_mask": True,
            "stop_controller": "learned",
            "commit": max_steps,
            "feedback": True,
        },
        {
            "name": "flow_closed_loop_commit1",
            "use_flow": True,
            "selection_policy": "learned",
            "contract_mask": True,
            "stop_controller": "learned",
            "commit": 1,
            "feedback": True,
        },
        {
            "name": "flow_closed_loop_commit1_no_contract_mask",
            "use_flow": True,
            "selection_policy": "learned",
            "contract_mask": False,
            "stop_controller": "learned",
            "commit": 1,
            "feedback": True,
        },
        {
            "name": "flow_closed_loop_commit2",
            "use_flow": True,
            "selection_policy": "learned",
            "contract_mask": True,
            "stop_controller": "learned",
            "commit": 2,
            "feedback": True,
        },
        {
            "name": "flow_compute_matched_blind_replan",
            "use_flow": True,
            "selection_policy": "learned",
            "contract_mask": True,
            "stop_controller": "learned",
            "commit": 1,
            "feedback": False,
        },
    ]
    details = []
    summary_rows = []
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        commit1_budget = {
            row["task_id"]: int(row["planner_calls"])
            for row in rows_by_condition.get("flow_closed_loop_commit1", [])
        }
        rows = []
        for task_index, task in enumerate(tasks):
            # Reset the planner per task. Every flow condition receives the same
            # noise sequence for call k, so feedback/commit horizon—not RNG drift
            # from the previous task—is the controlled difference.
            planner = LearnedPlanner(
                geometry,
                store,
                runtime,
                device,
                flow=flow if condition["use_flow"] else None,
                nfe=nfe,
                stop_threshold=stop_threshold,
                seed=seed + task_index * 10_000,
                runtime_embedding_policy=runtime_embedding_policy,
                apply_contract_mask=bool(condition["contract_mask"]),
                selection_policy=str(condition["selection_policy"]),
            )
            if condition["name"] == "flow_open_loop":
                planner_call_budget = 1
            elif condition["name"] == "flow_compute_matched_blind_replan":
                planner_call_budget = commit1_budget.get(task["task_id"])
            else:
                planner_call_budget = None
            rows.append(
                run_episode(
                    planner,
                    task,
                    store.tools,
                    _background_for_task(task, card_map),
                    snapshots=snapshots,
                    verification_goal=combine_verification_goal(
                        task["goal"], verifier_map.get(task["task_id"])
                    ),
                    commit_horizon=int(condition["commit"]),
                    feedback=bool(condition["feedback"]),
                    max_steps=max_steps,
                    perturbations=perturbations,
                    counterfactual_policy=counterfactual_policy,
                    planner_call_budget=planner_call_budget,
                )
            )
        rows_by_condition[condition["name"]] = rows
        for row in rows:
            details.append({"condition": condition["name"], **row})
        perturbed = [row for row in rows if row["perturbation_applied"]]
        assigned = [row for row in rows if row["perturbation_assigned"]]
        known_zero_regret_rates = [
            float(row["zero_regret_action_rate"])
            for row in rows
            if row["zero_regret_action_rate"] is not None
        ]
        known_mean_regrets = [
            float(row["mean_regret"]) for row in rows if row["mean_regret"] is not None
        ]
        summary_rows.append(
            {
                "condition": condition["name"],
                "stop_controller": condition["stop_controller"],
                "contract_mask": bool(condition["contract_mask"]),
                "tasks": len(rows),
                "task_success": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
                "goal_reached": float(np.mean([row["goal_reached"] for row in rows]))
                if rows
                else 0.0,
                "correct_stop_rate": float(np.mean([row["correct_stop"] for row in rows]))
                if rows
                else 0.0,
                "contract_valid_action_rate": float(
                    np.mean([row["contract_valid_action_rate"] for row in rows])
                )
                if rows
                else 0.0,
                "execution_success_rate": float(
                    np.mean([row["execution_success_rate"] for row in rows])
                )
                if rows
                else 0.0,
                "zero_regret_action_rate": (
                    float(np.mean(known_zero_regret_rates)) if known_zero_regret_rates else None
                ),
                "mean_regret_label_coverage": float(
                    np.mean([row["regret_label_coverage"] for row in rows])
                )
                if rows
                else 0.0,
                "assigned_perturbation_tasks": len(assigned),
                "triggered_perturbation_tasks": len(perturbed),
                "perturbation_trigger_rate": (len(perturbed) / len(assigned) if assigned else 0.0),
                "assigned_task_success_rate": (
                    float(np.mean([row["success"] for row in assigned])) if assigned else 0.0
                ),
                "triggered_recovery_rate": float(
                    np.mean([row["recovered_after_perturbation"] for row in perturbed])
                )
                if perturbed
                else 0.0,
                "mean_regret": float(np.mean(known_mean_regrets)) if known_mean_regrets else None,
                "mean_planner_calls": float(np.mean([row["planner_calls"] for row in rows]))
                if rows
                else 0.0,
                "mean_total_nfe": float(np.mean([row["total_nfe"] for row in rows]))
                if rows
                else 0.0,
                "mean_latency_seconds": float(
                    np.mean([row["planner_latency_seconds"] for row in rows])
                )
                if rows
                else 0.0,
                "mean_embedding_cache_hit_rate": float(
                    np.mean([row["embedding_cache_hit_rate"] for row in rows])
                )
                if rows
                else 0.0,
                "mean_runtime_embedding_miss_calls": float(
                    np.mean([row["runtime_embedding_miss_calls"] for row in rows])
                )
                if rows
                else 0.0,
                "mean_redundant_calls": float(np.mean([row["redundant_calls"] for row in rows]))
                if rows
                else 0.0,
                "mean_contract_valid_candidates": float(
                    np.mean([row["mean_contract_valid_candidates"] for row in rows])
                )
                if rows
                else 0.0,
            }
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "agent_details.jsonl"
    write_jsonl(details_path, details)
    result = {
        "evaluation_split": split,
        "seed": seed,
        "experiment_config_sha256": sha256_file(config_path),
        "evaluation_config_sha256": sha256_text(canonical_json(eval_config)),
        "evaluation_config": eval_config,
        "run_provenance": run_provenance,
        "cache_content_sha256": store.cache.content_sha256,
        "metric_checkpoint_sha256": sha256_file(metric_checkpoint),
        "flow_checkpoint_sha256": sha256_file(flow_checkpoint),
        "agent_details_sha256": sha256_file(details_path),
        "prewarmed_runtime_encoders": prewarm_runtime_encoders,
        "conditions": summary_rows,
        "notes": {
            "flow_open_loop": "exactly one planner call followed by committing the provisional suffix",
            "blind_replan": "same repeated planning compute but no new observation/state is exposed",
            "success": "task_success requires both typed-goal satisfaction and a correct STOP",
            "oracle_stop_baselines": (
                "random/metric baselines have no learned STOP head and stop when the public "
                "completion predicate is true; compare their action quality, not correct_stop, "
                "directly against flow"
            ),
            "contract_ablation": (
                "no-contract-mask conditions expose embedding/planner errors; masked conditions "
                "measure the deployable exact-contract pipeline"
            ),
            "perturbation_denominators": (
                "assigned_task_success_rate uses tasks fixed before policy execution; "
                "triggered_recovery_rate is conditional on the target actually being triggered"
            ),
            "runtime_embeddings": (
                f"policy={runtime_embedding_policy}; known prefixes use immutable cache entries"
            ),
            "latency": (
                "planner encoding plus solve only; symbolic tool latency is excluded. Lazy model "
                "loading is included unless prewarm_runtime_encoders=true"
            ),
        },
    }
    write_json(output_dir / "agent_summary.json", result)
    write_markdown(
        output_dir / "agent_summary.md",
        f"GeoFlowAgent closed-loop evaluation ({split})",
        [
            ("Results", markdown_table(summary_rows)),
            (
                "Interpretation",
                "Compare commit-1 with both open-loop and compute-matched blind replanning. "
                "Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.",
            ),
        ],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run receding-horizon GeoFlowAgent evaluation")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--embedding-config", required=True)
    parser.add_argument("--metric-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Defaults to dev. Passing test is an explicit test-unsealing action.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_agent(
        args.processed_dir,
        args.cache_dir,
        args.embedding_config,
        args.metric_checkpoint,
        args.flow_checkpoint,
        args.output_dir,
        args.config,
        split=args.split,
    )
    print(result["conditions"])


if __name__ == "__main__":
    main()
