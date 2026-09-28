"""flowfast 동등성 검정 — 원본과 같은 답을 내는지 먼저 증명한다.

배치화는 성능 작업이지만, 답이 달라지면 성능은 의미가 없다.  그래서 교체 전에
원본 ``_sample_plans`` / ``_replanning_rollout`` 과 **같은 시드에서 같은 출력**을
내는지 검정한다.  이 테스트가 통과하기 전에는 평가 경로를 바꾸지 않는다.

torch 가 없는 환경(분석 전용)에서는 건너뛴다.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geoflowagent.geoacmg import flowfast  # noqa: E402
from geoflowagent.models.search_state_flow import SearchStateFlow  # noqa: E402
from geoflowagent.training import state_flow as SF  # noqa: E402

SEED = 17
NFE = 12
LAYERS = 4
N_STATES = 96
N_TOOLS = 5


def _model():
    torch.manual_seed(0)
    return SearchStateFlow(
        snapshot_input_dim=16,
        context_input_dim=16,
        goal_input_dim=16,
        tool_input_dim=16,
        dim=8,
        max_plan_length=6,
        hidden_dim=32,
        layers=2,
        heads=2,
    ).eval()


def _generator(row: int):
    """원본 evaluate_search_state_flow 와 같은 시드 규칙."""

    return torch.Generator(device="cpu").manual_seed(SEED + 10_000 * row)


@pytest.fixture(scope="module")
def world():
    model = _model()
    rng = torch.Generator().manual_seed(7)
    layer = torch.arange(N_STATES) % LAYERS
    successor = torch.full((N_STATES, N_TOOLS), -1, dtype=torch.long)
    candidate = torch.zeros(N_STATES, N_TOOLS, dtype=torch.bool)
    cost = torch.full((N_STATES, N_TOOLS), float("inf"))
    for state in range(N_STATES):
        if layer[state] == LAYERS - 1:
            continue
        for tool in range(N_TOOLS):
            if torch.rand(1, generator=rng).item() < 0.6:
                nxt = int(torch.randint(0, N_STATES, (1,), generator=rng).item())
                nxt = min(nxt - (nxt % LAYERS) + int(layer[state]) + 1, N_STATES - 1)
                successor[state, tool] = nxt
                candidate[state, tool] = True
                cost[state, tool] = float(torch.rand(1, generator=rng).item()) + 0.1

    class Store:
        examples = [{"terminal": bool(layer[i] == LAYERS - 1)} for i in range(N_STATES)]
        _targets = {
            "policy_candidate_mask": candidate,
            "transition_known_mask": torch.ones_like(candidate),
            "successor_feature_mask": torch.ones_like(candidate),
            "successor_index": successor,
            "edge_cost": cost,
        }

    store = Store()
    context = torch.randn(N_STATES, 16)
    goal = torch.randn(N_STATES, 16)
    snapshot = torch.randn(N_STATES, 16)
    tools = torch.randn(N_TOOLS, 16)

    class Features:
        def __init__(self) -> None:
            self.store = store

        def condition(self, model, row_indices, device):
            index = torch.as_tensor(list(row_indices), dtype=torch.long)
            return model.encode_condition(context[index], goal[index])

        def device_tensors(self, device):
            return {"snapshot": snapshot, "tool": tools}

    roots = [i for i in range(N_STATES) if layer[i] == 0]
    return model, Features(), store, roots, torch.device("cpu")


def test_batched_sampling_makes_identical_decisions(world):
    """배치 표집이 원본과 같은 결정을 내는지.

    난수는 행별 생성기에서 그대로 뽑으므로 결정은 반드시 같아야 한다.
    값은 배치 축약 순서 때문에 마지막 자리가 다를 수 있고, 그건 보고 대상이다.
    """

    model, features, _store, roots, device = world
    samples = 4

    with torch.inference_mode():
        original = {
            row: SF._sample_plans(
                model, features, row, device, samples=samples, nfe=NFE, generator=_generator(row)
            )
            for row in roots
        }
    batched = flowfast.batch_sample_plans(
        model,
        features,
        roots,
        device,
        samples=samples,
        nfe=NFE,
        generators={row: _generator(row) for row in roots},
    )

    report = flowfast.compare_plans(original, batched)
    assert report.decisions_identical, report.to_payload()
    assert report.max_absolute_difference < 1e-4


@pytest.mark.parametrize(
    ("feedback", "guard", "budget", "label"),
    [
        (True, False, None, "replan"),
        (True, True, None, "guarded"),
        (False, False, 6, "blind_with_budget"),
    ],
)
def test_lockstep_rollouts_match_original(world, feedback, guard, budget, label):
    """롤아웃 3종 전부가 원본과 정확히 같은 결과를 내는지.

    사이클 검출, max_steps 의미, 예산 패딩(난수를 더 소비한다)까지 보존되어야 한다.
    하나라도 빠지면 난수열이 어긋나 결과가 갈린다.
    """

    model, features, store, roots, device = world

    original = {
        row: SF._replanning_rollout(
            model,
            features,
            row,
            device,
            nfe=NFE,
            stop_threshold=0.5,
            max_steps=12,
            generator=_generator(row),
            guard_premature_stop=guard,
            feedback=feedback,
            planner_call_budget=budget,
        )
        for row in roots
    }
    fast = flowfast.lockstep_rollouts(
        model,
        features,
        roots,
        device,
        nfe=NFE,
        stop_threshold=0.5,
        max_steps=12,
        generator_for=_generator,
        candidate_fn=lambda index: SF._candidate_transition_indices(store, index),
        terminal_fn=lambda index: bool(store.examples[index].get("terminal")),
        edge_cost_fn=lambda source, tool: store._targets["edge_cost"][source, tool],
        guard_premature_stop=guard,
        feedback=feedback,
        planner_call_budget=budget,
        label=label,
    )

    for row in roots:
        for key in ("success", "tool_indices", "failure", "planner_calls"):
            assert original[row][key] == fast[row][key], (row, key, original[row], fast[row])
        assert original[row]["total_cost"] == pytest.approx(fast[row]["total_cost"], abs=1e-6)


def test_projected_speedup_is_call_arithmetic():
    payload = flowfast.projected_speedup(584, samples=4, nfe=12, rollout_kinds=3, max_steps=24)
    assert payload["model_calls_original"] == 584 * (1 + 3 * 24)
    assert payload["model_calls_batched"] == 1 + 3 * 24
    assert payload["call_reduction_factor"] == pytest.approx(584.0)
