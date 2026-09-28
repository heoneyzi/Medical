"""flow 평가의 CPU 병목 제거 — 배치화된 원시연산.

병목의 실체 (``training/state_flow.py`` 확인):

``evaluate_search_state_flow`` 는 ``for row_index in indices:`` 파이썬 루프다.
루트 하나마다

* ``_sample_plans`` — ``condition.expand(samples, -1)`` 로 **한 행의 4샘플만** 배치
* ``model.stop_probability(plan.unsqueeze(0))`` — 계획 하나씩
* ``_replanning_rollout`` **3종**, 각각 최대 ``max_plan_length * 2`` 스텝 순차

루트당 모델 호출 약 73회이고 각 호출 안에 Euler ``nfe`` 스텝이 들어간다.
dev root-only 584 루트면 4만 회가 넘는 단일행 호출을 파이썬이 몬다.  GPU 는 놀고
CPU 한 코어가 병목이 되는 이유가 이것이다 (저널 §21: 128코어에서 1.2코어).

중요한 사실: ``StateFlowFeatureSpace.condition`` 은 이미
``row_indices: Sequence[int]`` 를 받고 ``SearchStateFlow.solve`` 도 배치를 받는다.
**모델은 처음부터 배치를 할 수 있었고 호출부가 안 했을 뿐이다.**

이 모듈은 원본 함수를 고치지 않는다.  같은 일을 배치로 하는 원시연산을 따로 주고,
``equivalence`` 로 원본과 출력이 같은지 먼저 증명하게 한다.  증명 전에는 바꾸지 않는다.

난수 보존
---------
원본은 행마다 자기 ``torch.Generator`` 로 ``torch.randn(samples, L, D)`` 를 뽑는다.
배치화해도 **행별 생성기에서 같은 모양으로 같은 순서로** 뽑아 쌓으면 난수열이
바이트 단위로 같다.  배치는 잡음 생성이 아니라 적분만 합친다.  따라서 동등성은
부동소수점 누적 순서 차이만 남고, 그건 ``equivalence`` 가 허용오차와 **결정 일치**
(선택된 도구 인덱스가 같은지)로 따로 검정한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:  # torch 는 분석 전용 환경에는 없을 수 있다
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from geoflowagent.geoacmg import parallel

# ------------------------------------------------------------------ 계획 표집


def row_noise(
    model: Any, samples: int, generator: Any, *, device: Any
) -> Any:
    """원본과 **동일한 순서·모양**으로 한 행의 잡음을 뽑는다.

    원본 ``_sample_plans`` 와 같아야 하므로 CPU 에서 뽑아 옮긴다.
    """

    return torch.randn(
        samples,
        model.max_plan_length,
        model.dim,
        generator=generator,
        device="cpu",
    ).to(device)


def batch_sample_plans(
    model: Any,
    features: Any,
    row_indices: Sequence[int],
    device: Any,
    *,
    samples: int,
    nfe: int,
    generators: Mapping[int, Any],
    max_batch_rows: int | None = None,
) -> dict[int, Any]:
    """모든 루트의 계획을 **한 번에** 적분한다.

    원본은 루트마다 ``solve`` 를 불러 배치 크기가 ``samples``(=4)였다.
    여기서는 ``R`` 개 루트를 쌓아 배치 ``R * samples`` 로 ``nfe`` 스텝을 돈다.
    커널 디스패치가 ``R * nfe`` 회에서 ``nfe`` 회로 줄어든다.

    반환: ``row_index -> (samples, L, D)``  — 원본 ``_sample_plans`` 와 같은 모양.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch 가 필요하다")
    rows = list(row_indices)
    if not rows:
        return {}
    chunk = max_batch_rows or len(rows)
    out: dict[int, Any] = {}
    progress = parallel.Progress(total=len(rows), label="batch_sample_plans")
    for group in parallel.batched(rows, chunk):
        condition = features.condition(model, list(group), device)      # (G, C)
        expanded = condition.repeat_interleave(samples, dim=0)          # (G*S, C)
        noise = torch.cat(
            [row_noise(model, samples, generators[row], device=device) for row in group],
            dim=0,
        )                                                                # (G*S, L, D)
        with torch.inference_mode():
            plans, _ = model.solve(expanded, nfe=nfe, noise=noise)
        plans = plans.view(len(group), samples, model.max_plan_length, model.dim)
        for position, row in enumerate(group):
            out[row] = plans[position]
        progress.tick(len(group))
    return out


def batch_stop_probabilities(model: Any, plans_by_row: Mapping[int, Any]) -> dict[int, Any]:
    """모든 계획의 정지 확률을 한 번에 계산한다.

    원본은 계획마다 ``model.stop_probability(plan.unsqueeze(0))`` 를 불렀다.
    순수 함수라 배치화에 아무 제약이 없는데도 하나씩 돌고 있었다.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch 가 필요하다")
    rows = list(plans_by_row)
    if not rows:
        return {}
    stacked = torch.cat([plans_by_row[row] for row in rows], dim=0)      # (sum S, L, D)
    with torch.inference_mode():
        probabilities = model.stop_probability(stacked)                  # (sum S, L)
    out: dict[int, Any] = {}
    cursor = 0
    for row in rows:
        count = plans_by_row[row].shape[0]
        out[row] = probabilities[cursor : cursor + count]
        cursor += count
    return out


# ------------------------------------------------------------- 보조 맞춘 롤아웃


@dataclass
class RolloutState:
    """루트 하나의 롤아웃 진행 상태.

    루트 간에는 서로 독립이므로 **보조를 맞춰** 한 스텝씩 함께 전진시킬 수 있다.
    같은 스텝에서 살아 있는 롤아웃들의 계획 표집을 하나의 배치로 묶는 것이
    이 방식의 전부이고, 그것이 전체 시간의 대부분을 차지하던 부분이다.
    """

    row_index: int
    current: int
    generator: Any
    visited: set[int] = field(default_factory=set)
    tools: list[int] = field(default_factory=list)
    cost: float = 0.0
    planner_calls: int = 0
    failure: str | None = None
    done: bool = False
    steps_taken: int = 0

    def result(self, terminal_fn: Callable[[int], bool]) -> dict[str, Any]:
        # 원본과 같은 형태: success 는 최종 상태가 terminal 인지로 정한다.
        return {
            "success": bool(terminal_fn(self.current)),
            "tool_indices": tuple(self.tools),
            "total_cost": self.cost,
            "failure": self.failure,
            "planner_calls": self.planner_calls,
        }


def lockstep_rollouts(
    model: Any,
    features: Any,
    row_indices: Sequence[int],
    device: Any,
    *,
    nfe: int,
    stop_threshold: float,
    max_steps: int,
    generator_for: Callable[[int], Any],
    candidate_fn: Callable[[int], tuple[Any, Any]],
    terminal_fn: Callable[[int], bool],
    edge_cost_fn: Callable[[int, int], Any],
    guard_premature_stop: bool = False,
    feedback: bool = True,
    planner_call_budget: int | None = None,
    label: str = "rollouts",
) -> dict[int, dict[str, Any]]:
    """모든 루트의 재계획 롤아웃을 보조를 맞춰 동시에 전진시킨다.

    원본 ``_replanning_rollout`` 은 루트마다 최대 ``max_steps`` 번 순차로
    ``_sample_plan`` 을 불렀고, 그것을 롤아웃 3종에 대해 반복했다.  여기서는
    스텝마다 **살아 있는 모든 롤아웃**의 계획을 한 배치로 표집한다.  모델 호출이
    ``R * max_steps`` 회에서 ``max_steps`` 회로 줄어든다.

    원본의 세부를 그대로 보존한다.

    * **사이클 검출** — 이미 방문한 상태로 돌아가면 ``cycle`` 로 끝낸다.
    * **``max_steps``** — 루프가 break 없이 끝났을 때만 붙는 실패다.
    * **예산 패딩** — ``planner_call_budget`` 이 남으면 행동 없이 계획만 더 뽑는다.
      난수를 실제로 소비하므로, 빼먹으면 난수열이 어긋난다.

    난수: 롤아웃마다 자기 생성기를 들고 자기 스텝에서만 뽑는다.  끝난 롤아웃은
    원본과 마찬가지로 더 뽑지 않으므로 난수열이 일치한다.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch 가 필요하다")
    import torch.nn.functional as F

    states = [
        RolloutState(row_index=row, current=int(row), generator=generator_for(row), visited={int(row)})
        for row in row_indices
    ]
    progress = parallel.Progress(total=len(states), label=label)
    retired = 0

    def retire(state: RolloutState, failure: str | None) -> None:
        nonlocal retired
        state.done = True
        if failure is not None:
            state.failure = failure
        retired += 1
        progress.tick()

    for _step in range(max_steps):
        running = [s for s in states if not s.done]
        if not running:
            break
        still: list[RolloutState] = []
        for state in running:
            state.steps_taken += 1
            if terminal_fn(state.current):
                retire(state, None)
            elif planner_call_budget is not None and state.planner_calls >= planner_call_budget:
                retire(state, "planner_call_budget")
            else:
                still.append(state)
        if not still:
            continue

        visible = [s.current if feedback else s.row_index for s in still]
        condition = features.condition(model, visible, device)
        noise = torch.cat(
            [row_noise(model, 1, s.generator, device=device) for s in still], dim=0
        )
        with torch.inference_mode():
            plans, _ = model.solve(condition, nfe=nfe, noise=noise)      # (K, L, D)
            stop = model.stop_probability(plans[:, :1])                  # (K, 1)
        tensors = features.device_tensors(device)

        for position, state in enumerate(still):
            state.planner_calls += 1
            if not guard_premature_stop and float(stop[position, 0].item()) >= stop_threshold:
                retire(state, "premature_stop")
                continue
            look_at = state.current if feedback else state.row_index
            candidates, successors = candidate_fn(look_at)
            if not len(candidates):
                retire(state, "no_decodable_transition")
                continue
            anchors = model.transition_anchor(
                tensors["snapshot"][successors.to(device)],
                tensors["tool"][candidates.to(device)],
            )
            chosen = int(
                F.cosine_similarity(plans[position, 0].unsqueeze(0), anchors, dim=-1)
                .argmax()
                .item()
            )
            tool_index = int(candidates[chosen].item())
            actual_candidates, actual_successors = candidate_fn(state.current)
            match = torch.where(actual_candidates == tool_index)[0]
            if len(match) != 1:
                retire(state, "invalid_action" if feedback else "blind_invalid_action")
                continue
            cost = edge_cost_fn(state.current, tool_index)
            finite = (
                bool(torch.isfinite(cost).item())
                if torch.is_tensor(cost)
                else (cost == cost and abs(float(cost)) != float("inf"))
            )
            if not finite:
                retire(state, "missing_transition_cost")
                continue
            state.tools.append(tool_index)
            state.cost += float(cost)
            state.current = int(actual_successors[int(match[0].item())].item())
            if state.current in state.visited:
                retire(state, "cycle")
                continue
            state.visited.add(state.current)

    # 루프를 break 없이 다 쓴 롤아웃만 max_steps 실패다 (원본의 for...else).
    for state in states:
        if not state.done:
            state.failure = "max_steps"
            state.done = True

    # 예산 패딩 — 행동 없이 계획만 더 뽑는다. 난수를 실제로 소비한다.
    if planner_call_budget is not None:
        while True:
            pending = [s for s in states if s.planner_calls < planner_call_budget]
            if not pending:
                break
            visible = [s.current if feedback else s.row_index for s in pending]
            condition = features.condition(model, visible, device)
            noise = torch.cat(
                [row_noise(model, 1, s.generator, device=device) for s in pending], dim=0
            )
            with torch.inference_mode():
                model.solve(condition, nfe=nfe, noise=noise)
            for state in pending:
                state.planner_calls += 1

    return {s.row_index: s.result(terminal_fn) for s in states}


# --------------------------------------------------------------------- 동등성


@dataclass(frozen=True)
class EquivalenceReport:
    checked: int
    decision_matches: int
    max_absolute_difference: float
    mismatches: tuple[dict[str, Any], ...]

    @property
    def decisions_identical(self) -> bool:
        return self.decision_matches == self.checked

    def to_payload(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "decision_matches": self.decision_matches,
            "decisions_identical": self.decisions_identical,
            "max_absolute_difference": self.max_absolute_difference,
            "mismatches": list(self.mismatches[:20]),
            "rule": (
                "배치화는 적분만 합치고 난수는 행별 생성기에서 그대로 뽑는다. "
                "따라서 결정(선택된 도구 인덱스)은 반드시 일치해야 한다. "
                "부동소수점 값 차이는 배치 축약 순서 때문에 허용하되 크기를 보고한다."
            ),
        }


def compare_plans(
    reference: Mapping[int, Any], candidate: Mapping[int, Any], *, atol: float = 1e-4
) -> EquivalenceReport:
    """원본 계획 텐서와 배치본을 비교한다.

    **결정 일치가 통과 조건이고, 값 차이는 보고 대상이다.**  배치 축약 순서가
    바뀌면 마지막 자리는 달라질 수 있지만 argmax 가 바뀌면 안 된다.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch 가 필요하다")
    rows = sorted(set(reference) & set(candidate))
    worst = 0.0
    matches = 0
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        left = reference[row].detach().float().cpu()
        right = candidate[row].detach().float().cpu()
        if left.shape != right.shape:
            mismatches.append({"row": row, "reason": f"shape {tuple(left.shape)} vs {tuple(right.shape)}"})
            continue
        difference = float((left - right).abs().max().item())
        worst = max(worst, difference)
        same_decision = bool(
            torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))
        )
        if same_decision:
            matches += 1
        else:
            mismatches.append({"row": row, "reason": "argmax differs", "max_abs": difference})
        if difference > atol:
            mismatches.append({"row": row, "reason": "value drift", "max_abs": difference})
    return EquivalenceReport(len(rows), matches, worst, tuple(mismatches))


def projected_speedup(
    n_roots: int, *, samples: int, nfe: int, rollout_kinds: int, max_steps: int
) -> dict[str, Any]:
    """호출 횟수로 본 이득. 시간 예측이 아니라 **디스패치 수 산술**이다.

    실제 배속은 하드웨어와 배치 크기가 정한다 — 여기서 시간을 지어내지 않는다.
    """

    per_root = 1 + rollout_kinds * max_steps
    original = n_roots * per_root
    batched_calls = 1 + rollout_kinds * max_steps
    return {
        "n_roots": n_roots,
        "model_calls_original": original,
        "model_calls_batched": batched_calls,
        "call_reduction_factor": round(original / max(batched_calls, 1), 1),
        "euler_steps_per_call": nfe,
        "note": (
            "호출 수 비교다. 배치 하나가 더 큰 만큼 GPU 시간은 늘지만, "
            "지난 실행에서 병목은 GPU 가 아니라 파이썬 디스패치였다 (1.2코어)."
        ),
    }
