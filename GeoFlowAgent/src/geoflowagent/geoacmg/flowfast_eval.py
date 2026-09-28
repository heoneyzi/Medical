"""``evaluate_search_state_flow`` 의 배치 대체본.

원본(``training/state_flow.py``)은 ``for row_index in indices:`` 루프 안에서 루트마다
모델을 수십 번 부른다.  이 모듈은 **같은 지표를 같은 난수로** 내되 모델 호출을
루트 축으로 묶는다.  원본 함수는 건드리지 않는다 — 나란히 돌려 비교할 수 있어야
교체가 정당해지기 때문이다.

세 단계로 나뉜다.

1. ``flowfast.batch_sample_plans`` — 전 루트의 계획을 한 번에 적분 (실측 63.8x)
2. 개루프 디코드 — 계획별 순차 디코드.  현재는 원본 ``_decode_open_loop`` 를 그대로
   쓴다.  여기를 배치화할지는 **재본 뒤에** 정한다 (지어내지 않는다).
3. ``flowfast.lockstep_rollouts`` — 롤아웃 3종을 보조 맞춰 전진

blind arm 의 예산
-----------------
원본은 blind 롤아웃에 **행마다 다른** ``planner_call_budget`` (그 행의 observed
롤아웃이 쓴 호출 수)을 준다.  ``lockstep_rollouts`` 는 배치 전체에 하나의 예산만
받으므로, 예산이 같은 행끼리 묶어 그룹마다 한 번씩 부른다.  예산은 1..max_steps
범위라 그룹 수는 최대 ``max_steps`` 개다 — 584회 순차가 최대 24회 배치가 된다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Any

import torch

from geoflowagent.geoacmg import flowfast
from geoflowagent.training import state_flow as SF


def _row_generator(seed: int, row_index: int) -> torch.Generator:
    """``state_flow.py`` 의 행별 시드 규칙과 같아야 한다 (row_seed)."""

    return torch.Generator(device="cpu").manual_seed(int(seed) + 10_000 * int(row_index))


def _store_hooks(store: Any):
    """``lockstep_rollouts`` 가 요구하는 세 콜백을 실제 store 에 잇는다."""

    def candidate_fn(index: int):
        return SF._candidate_transition_indices(store, int(index))  # noqa: SLF001

    def terminal_fn(index: int) -> bool:
        return bool(store.examples[int(index)].get("terminal"))

    def edge_cost_fn(current: int, tool_index: int):
        return store._targets["edge_cost"][int(current), int(tool_index)]  # noqa: SLF001

    return candidate_fn, terminal_fn, edge_cost_fn


def batched_rollouts_with_per_row_budget(
    model: Any,
    features: Any,
    row_indices: Sequence[int],
    device: Any,
    *,
    budgets: dict[int, int],
    nfe: int,
    stop_threshold: float,
    max_steps: int,
    seed: int,
    feedback: bool,
    guard_premature_stop: bool = False,
    label: str = "blind",
) -> dict[int, dict[str, Any]]:
    """예산이 같은 행끼리 묶어 ``lockstep_rollouts`` 를 그룹마다 부른다."""

    candidate_fn, terminal_fn, edge_cost_fn = _store_hooks(features.store)
    by_budget: dict[int, list[int]] = defaultdict(list)
    for row in row_indices:
        by_budget[int(budgets[row])].append(int(row))

    out: dict[int, dict[str, Any]] = {}
    for budget in sorted(by_budget):
        group = by_budget[budget]
        out.update(
            flowfast.lockstep_rollouts(
                model, features, group, device,
                nfe=nfe, stop_threshold=stop_threshold, max_steps=max_steps,
                generator_for=lambda row: _row_generator(seed, row),
                candidate_fn=candidate_fn, terminal_fn=terminal_fn, edge_cost_fn=edge_cost_fn,
                guard_premature_stop=guard_premature_stop, feedback=feedback,
                planner_call_budget=budget,
                label=f"{label}[budget={budget}]",
            )
        )
    return out


@torch.inference_mode()
def evaluate_batched(
    model: Any,
    store: Any,
    path_store: Any,
    split: str,
    device: Any,
    *,
    samples_per_state: int = 4,
    nfe: int = 12,
    stop_threshold: float = 0.5,
    seed: int = 17,
    root_only: bool = False,
    max_replan_steps: int | None = None,
    max_batch_rows: int | None = None,
) -> dict[str, Any]:
    """원본 ``evaluate_search_state_flow`` 와 같은 지표 dict 를 배치로 낸다."""

    if samples_per_state <= 0:
        raise ValueError("samples_per_state must be positive")
    if nfe <= 0:
        raise ValueError("nfe must be positive")

    indices = list(path_store.eligible_indices(split))
    if root_only:
        indices = [i for i in indices if not store.examples[i].get("prefix_tool_ids")]
    if not indices:
        return {"split": split, "eligible_states": 0, "eligible_tasks": 0}
    if split == "test" and path_store.allowed_splits != frozenset({"test"}):
        raise ValueError("Test evaluation requires a test-only SearchPathStore")

    features = SF.StateFlowFeatureSpace(store)
    model.eval()
    max_replan_steps = max_replan_steps or (model.max_plan_length * 2)
    candidate_fn, terminal_fn, edge_cost_fn = _store_hooks(store)

    # --- 1. 계획 표집: 전 루트 한 번에 --------------------------------------
    plans_by_row = flowfast.batch_sample_plans(
        model, features, indices, device,
        samples=samples_per_state, nfe=nfe,
        generators={r: _row_generator(seed, r) for r in indices},
        max_batch_rows=max_batch_rows,
    )

    # --- 2. 개루프 디코드 (계획별 순차) --------------------------------------
    acc: dict[str, list[float]] = defaultdict(list)
    task_goal_hits: defaultdict[str, list[float]] = defaultdict(list)
    task_replan_hits: defaultdict[str, list[float]] = defaultdict(list)
    task_blind_hits: defaultdict[str, list[float]] = defaultdict(list)
    row_goal_mean: dict[int, float] = {}
    progress = flowfast.parallel.Progress(total=len(indices), label="decode_open_loop")
    for row_index in indices:
        row = store.examples[row_index]
        task_id = str(row["task_id"])
        value_star = float(row["value_star"])
        optimal = store._targets["optimal_action_mask"][row_index]  # noqa: SLF001
        references = path_store.references(row_index)
        reference_tools = {r.tool_indices for r in references}
        observed: set[tuple[int, ...]] = set()
        row_hits: list[float] = []
        for plan in plans_by_row[row_index]:
            decoded = SF._decode_open_loop(  # noqa: SLF001
                model, plan, row_index, features, stop_threshold=stop_threshold
            )
            tools = decoded["tool_indices"]
            observed.add(tools)
            acc["first"].append(float(bool(tools) and bool(optimal[tools[0]])))
            goal = float(decoded["goal_reached"])
            acc["goal"].append(goal)
            row_hits.append(goal)
            task_goal_hits[task_id].append(goal)
            acc["valid"].append(float(decoded["valid_chain"]))
            acc["stop"].append(float(decoded["stop_valid"]))
            acc["premature"].append(float(decoded["premature_stop"]))
            in_reference = tools in reference_tools
            acc["reference"].append(float(in_reference))
            acc["novel"].append(float(decoded["goal_reached"] and decoded["valid_chain"] and not in_reference))
            if decoded["goal_reached"]:
                excess = max(0.0, float(decoded["total_cost"]) - value_star)
                acc["excess"].append(excess)
                acc["optimal_cost"].append(float(excess <= 1e-6))
            acc["edit"].append(
                min(
                    SF._edit_distance(tools, r.tool_indices)  # noqa: SLF001
                    / max(1, len(tools), len(r.tool_indices))
                    for r in references
                )
            )
        acc["coverage"].append(len(observed & reference_tools) / len(reference_tools))
        acc["unique"].append(float(len(observed)))
        row_goal_mean[row_index] = fmean(row_hits)
        progress.tick()

    # --- 3. 롤아웃 3종 -------------------------------------------------------
    common = dict(
        nfe=nfe, stop_threshold=stop_threshold, max_steps=max_replan_steps,
        generator_for=lambda row: _row_generator(seed, row),
        candidate_fn=candidate_fn, terminal_fn=terminal_fn, edge_cost_fn=edge_cost_fn,
    )
    observed_rollouts = flowfast.lockstep_rollouts(
        model, features, indices, device, label="replan", **common)
    guarded_rollouts = flowfast.lockstep_rollouts(
        model, features, indices, device, guard_premature_stop=True, label="guarded", **common)
    blind_rollouts = batched_rollouts_with_per_row_budget(
        model, features, indices, device,
        budgets={r: int(observed_rollouts[r]["planner_calls"]) for r in indices},
        nfe=nfe, stop_threshold=stop_threshold, max_steps=max_replan_steps,
        seed=seed, feedback=False, label="blind",
    )

    replan_failures: defaultdict[str, int] = defaultdict(int)
    guarded_failures: defaultdict[str, int] = defaultdict(int)
    blind_failures: defaultdict[str, int] = defaultdict(int)
    root_open_loop: list[float] = []
    root_replan: list[float] = []
    root_blind: list[float] = []

    for row_index in indices:
        row = store.examples[row_index]
        task_id = str(row["task_id"])
        value_star = float(row["value_star"])
        roll = observed_rollouts[row_index]
        guard = guarded_rollouts[row_index]
        blind = blind_rollouts[row_index]

        acc["replan_success"].append(float(roll["success"]))
        acc["replan_calls"].append(float(roll["planner_calls"]))
        task_replan_hits[task_id].append(float(roll["success"]))
        if roll["success"]:
            acc["replan_excess"].append(max(0.0, float(roll["total_cost"]) - value_star))
        elif roll["failure"] is not None:
            replan_failures[str(roll["failure"])] += 1

        acc["guarded_success"].append(float(guard["success"]))
        acc["guarded_calls"].append(float(guard["planner_calls"]))
        if guard["success"]:
            acc["guarded_excess"].append(max(0.0, float(guard["total_cost"]) - value_star))
        elif guard["failure"] is not None:
            guarded_failures[str(guard["failure"])] += 1

        acc["blind_success"].append(float(blind["success"]))
        acc["blind_calls"].append(float(blind["planner_calls"]))
        acc["matched_calls"].append(float(blind["planner_calls"] == roll["planner_calls"]))
        task_blind_hits[task_id].append(float(blind["success"]))
        if blind["success"]:
            acc["blind_excess"].append(max(0.0, float(blind["total_cost"]) - value_star))
        elif blind["failure"] is not None:
            blind_failures[str(blind["failure"])] += 1

        if not row.get("prefix_tool_ids"):
            root_open_loop.append(row_goal_mean[row_index])
            root_replan.append(float(roll["success"]))
            root_blind.append(float(blind["success"]))

    m = SF._mean  # noqa: SLF001
    task_macro = m(fmean(v) for v in task_goal_hits.values())
    return {
        "split": split,
        "evaluation_scope": "task_roots" if root_only else "all_eligible_states",
        "eligible_states": len(indices),
        "eligible_tasks": len(task_goal_hits),
        "samples_per_state": samples_per_state,
        "sample_count": len(acc["goal"]),
        "nfe": nfe,
        "stop_threshold": stop_threshold,
        "open_loop": {
            "first_action_optimal_set_accuracy": m(acc["first"]),
            "valid_transition_chain_rate": m(acc["valid"]),
            "goal_completion_rate": m(acc["goal"]),
            "task_macro_goal_completion_rate": task_macro,
            "valid_stop_after_goal_rate": m(acc["stop"]),
            "premature_stop_rate": m(acc["premature"]),
            "serialized_top_k_path_rate": m(acc["reference"]),
            "novel_valid_goal_path_rate": m(acc["novel"]),
            "optimal_cost_given_goal_rate": m(acc["optimal_cost"]),
            "mean_excess_cost_given_goal": m(acc["excess"]),
            "closest_reference_normalized_edit_distance": m(acc["edit"]),
            "mean_reference_coverage_at_k": m(acc["coverage"]),
            "mean_unique_paths_at_k": m(acc["unique"]),
        },
        "receding_horizon": {
            "rollouts": len(acc["replan_success"]),
            "goal_completion_rate": m(acc["replan_success"]),
            "mean_excess_cost_given_goal": m(acc["replan_excess"]),
            "mean_planner_calls": m(acc["replan_calls"]),
            "mean_total_nfe": nfe * m(acc["replan_calls"]),
            "failure_counts": dict(sorted(replan_failures.items())),
        },
        "receding_horizon_verifier_stop_guard": {
            "rollouts": len(acc["guarded_success"]),
            "goal_completion_rate": m(acc["guarded_success"]),
            "mean_excess_cost_given_goal": m(acc["guarded_excess"]),
            "mean_planner_calls": m(acc["guarded_calls"]),
            "mean_total_nfe": nfe * m(acc["guarded_calls"]),
            "failure_counts": dict(sorted(guarded_failures.items())),
        },
        "compute_matched_blind_replanning": {
            "rollouts": len(acc["blind_success"]),
            "goal_completion_rate": m(acc["blind_success"]),
            "mean_excess_cost_given_goal": m(acc["blind_excess"]),
            "mean_planner_calls": m(acc["blind_calls"]),
            "mean_total_nfe": nfe * m(acc["blind_calls"]),
            "per_state_planner_call_match_rate": m(acc["matched_calls"]),
            "failure_counts": dict(sorted(blind_failures.items())),
        },
        "feedback_effects": {
            "replanning_minus_one_shot_task_macro": SF._paired_task_effect(  # noqa: SLF001
                task_replan_hits, task_goal_hits, seed=seed + 700_001),
            "replanning_minus_blind_task_macro": SF._paired_task_effect(  # noqa: SLF001
                task_replan_hits, task_blind_hits, seed=seed + 700_002),
            "root_start": {
                "tasks": len(root_replan),
                "one_shot_goal_completion_rate": m(root_open_loop),
                "replanning_goal_completion_rate": m(root_replan),
                "blind_replanning_goal_completion_rate": m(root_blind),
                "replanning_minus_one_shot": SF._bootstrap_effect(  # noqa: SLF001
                    [a - b for a, b in zip(root_replan, root_open_loop, strict=True)],
                    seed=seed + 700_003),
                "replanning_minus_blind": SF._bootstrap_effect(  # noqa: SLF001
                    [a - b for a, b in zip(root_replan, root_blind, strict=True)],
                    seed=seed + 700_004),
            },
        },
        "test_unsealed": split == "test",
        "implementation": "geoflowagent.geoacmg.flowfast_eval.evaluate_batched",
    }
