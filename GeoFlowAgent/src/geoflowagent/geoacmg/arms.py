"""단계 7 — 빠진 arm.

등록한 PRIMARY 4개 중 3개가 arm 부재로 산출되지 않았다 (P1 raw-L2 planning arm 없음,
P2 from-scratch encoder arm 없음, P3 tool cost 가 placeholder).  나온 것은 B0
0.6898 [0.6323, 0.7391] 하나였다.

* **P1** — 동결 공간의 raw L2 로 계획하는 arm.  학습된 에너지 없이 같은 예산·
  같은 commit length 로 돌린다.
* **P2** — ``winnability.assess`` 가 비퇴화 축을 찾지 못하면 만들지 않는다.
* **P3** — placeholder 비용으로 cost-matched 결론을 내지 않는다.  비용 모델을
  **여러 개** 두고, 결론이 전부에서 유지되는지를 결과로 보고한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from geoflowagent.geoacmg import cited, estimators, ladder
from geoflowagent.geoacmg.cited import Cited, Provenance

# ------------------------------------------------------------------ P1: raw L2


def raw_metric_scorer(
    embeddings: Mapping[str, np.ndarray],
    goal_key: str,
    *,
    metric: str = "l2",
):
    """동결 임베딩 위의 raw 거리로 행동을 고르는 scorer.

    학습된 trunk 도, 학습된 에너지도 쓰지 않는다.  E2 에서 raw 기준선 3종이
    전부 우연과 구별되지 않았으므로(cosine 0.5055, sq-L2 0.5078, dot 0.4966),
    이 arm 이 낮게 나오는 것 자체는 예상된 결과다.  중요한 것은 **얼마나** 낮은지와
    그것이 층별로 어떻게 달라지는지다.
    """

    if metric not in {"l2", "cosine", "dot"}:
        raise ValueError(f"알 수 없는 raw metric: {metric}")

    def score(
        state: Mapping[str, Any], actions: Sequence[str], world: Mapping[str, Any]
    ) -> Mapping[str, float]:
        goal = np.asarray(embeddings[goal_key], dtype=np.float64)
        out: dict[str, float] = {}
        for action in actions:
            key = f"{state.get('id', '')}|{action}"
            vector = embeddings.get(key)
            if vector is None:
                out[action] = float("-inf")
                continue
            vector = np.asarray(vector, dtype=np.float64)
            if metric == "l2":
                energy = float(((vector - goal) ** 2).sum())
            elif metric == "cosine":
                a = vector / max(np.linalg.norm(vector), 1e-12)
                b = goal / max(np.linalg.norm(goal), 1e-12)
                energy = float(1.0 - (a * b).sum())
            else:
                energy = float(-(vector * goal).sum())
            out[action] = -energy  # scorer 는 클수록 좋다
        return out

    return score


def raw_metric_plan_source(
    embeddings: Mapping[str, np.ndarray], goal_key: str, *, metric: str = "l2"
) -> ladder.PlanSource:
    return ladder.Greedy(
        scorer=raw_metric_scorer(embeddings, goal_key, metric=metric),
        name=f"raw_{metric}",
    )


# -------------------------------------------------------------- P3: cost models


@dataclass(frozen=True)
class CostModel:
    """도구 비용을 재는 한 가지 방법.

    ``per_tool`` 의 모든 값은 ``Provenance.MEASURED`` 여야 한다.  placeholder 로
    cost-matched 결론을 내는 것을 타입 수준에서 막는다.
    """

    key: str
    unit: str
    per_tool: Mapping[str, Cited[float]]

    def __post_init__(self) -> None:
        for tool, constant in self.per_tool.items():
            cited.assert_decision_safe(constant, f"cost model {self.key!r}, tool {tool!r}")
            if constant.provenance is not Provenance.MEASURED:
                raise ValueError(
                    f"cost model {self.key!r} 의 {tool!r} 비용이 측정값이 아니다 "
                    f"({constant.provenance.value}). placeholder 비용으로 cost-matched "
                    "결론을 낼 수 없다."
                )

    def total(self, actions: Sequence[str]) -> float:
        return float(sum(float(self.per_tool[a].value) for a in actions if a in self.per_tool))


def measured_cost_model(
    key: str, unit: str, measurements: Mapping[str, Sequence[float]], *, source: str
) -> CostModel:
    """실제 측정치에서 비용 모델을 만든다.

    ``measurements`` 는 도구마다 반복 측정한 원시 값이다.  중앙값을 쓰고,
    표본 크기를 ``sample`` 에 남겨 감사 가능하게 한다.
    """

    per_tool: dict[str, Cited[float]] = {}
    for tool, samples in measurements.items():
        values = [float(x) for x in samples]
        if not values:
            raise ValueError(f"{tool} 의 측정 표본이 비어 있다")
        per_tool[tool] = cited.measured(
            float(np.median(values)),
            source=f"{source}:{tool}",
            sample=f"n={len(values)}, median of observed {unit}",
        )
    return CostModel(key=key, unit=unit, per_tool=per_tool)


def conclusion_stability(
    episodes_by_arm: Mapping[str, Sequence[ladder.Episode]],
    cost_models: Sequence[CostModel],
    *,
    reference_arm: str,
    resamples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """비용 모델마다 결론을 다시 계산하고, 전부에서 유지되는지 본다.

    단일 비용 모델은 placeholder 를 피해도 자의성을 못 피한다.  wall-clock 이냐
    호출 수냐 금액이냐에 따라 뒤집히는 결론이라면, 그 사실 자체가 결과다.
    """

    rows: list[dict[str, Any]] = []
    for model in cost_models:
        reference = episodes_by_arm[reference_arm]
        reference_cost = {e.task_id: model.total(e.actions) for e in reference}
        reference_solved = {e.task_id: float(e.solved) for e in reference}
        for arm, episodes in episodes_by_arm.items():
            if arm == reference_arm:
                continue
            shared = [e for e in episodes if e.task_id in reference_cost]
            if not shared:
                continue
            clusters = [e.gene for e in shared]
            cost_difference = [
                model.total(e.actions) - reference_cost[e.task_id] for e in shared
            ]
            solve_difference = [
                float(e.solved) - reference_solved[e.task_id] for e in shared
            ]
            cost_interval = estimators.cluster_bootstrap(
                cost_difference, clusters, resamples=resamples, seed=seed
            )
            solve_interval = estimators.cluster_bootstrap(
                solve_difference, clusters, resamples=resamples, seed=seed
            )
            rows.append(
                {
                    "cost_model": model.key,
                    "unit": model.unit,
                    "arm": arm,
                    "reference": reference_arm,
                    "cost_delta": {
                        "estimate": cost_interval.estimate,
                        "ci": [cost_interval.low, cost_interval.high],
                    },
                    "solve_delta": {
                        "estimate": solve_interval.estimate,
                        "ci": [solve_interval.low, solve_interval.high],
                    },
                    "n_units": cost_interval.units,
                }
            )

    directions: dict[str, set[str]] = {}
    for row in rows:
        low, high = row["solve_delta"]["ci"]
        direction = "up" if low > 0 else "down" if high < 0 else "unresolved"
        directions.setdefault(row["arm"], set()).add(direction)

    return {
        "rows": rows,
        "stable_arms": sorted(a for a, d in directions.items() if len(d) == 1 and "unresolved" not in d),
        "unstable_arms": sorted(a for a, d in directions.items() if len(d) > 1),
        "reading": (
            "결론이 비용 모델에 따라 뒤집히면 단일 모델로 보고하지 않는다. "
            "뒤집힌다는 사실 자체를 결과로 싣는다."
        ),
    }
