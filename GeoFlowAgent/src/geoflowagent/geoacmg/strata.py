"""단계 3 — horizon 축.

풀링된 숫자 하나로는 해석이 안 되는 상태다.  action-level ordering 은 이미
0.9254 이고 남은 여유가 0.0746, 기하 효과는 0.0113 이다.  이 상태에서 P1 을
평균으로 내면 UNRESOLVED 가 나와도 "기하가 계획에 도움이 안 된다"인지
"도울 여지가 0.0746 뿐이었다"인지 구분할 수 없다.  그건 보고 가능한 null 이
아니라 측정 실패다.

해법은 과제를 새로 만드는 것이 아니라 **평균 내기를 그만두고 축으로 놓는 것**이다.
그 축은 이미 데이터 안에 있다 (root V*, horizon length, task 길이 쏠림).

층 경계는 **고르지 않는다**.  상수를 고르면 그 상수를 방어해야 하고, 그건
"0.01 넘으면 된다" 와 같은 종류의 논리가 된다.  대신:

* 1차 보고는 경계가 없는 **연속 기울기** (``estimators.cluster_slope``).
* 층별 표는 표시용이며, 경계는 dev 경험분위수로 생성하고 파일에 기록한다.
* 층마다 효과와 **여유(headroom)** 를 나란히 낸다 — 포화가 어디서 무는지 보이게.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from geoflowagent.geoacmg import estimators, inference
from geoflowagent.geoacmg.claims import Finding, Role


@dataclass(frozen=True)
class Stratum:
    index: int
    low: float
    high: float
    n: int
    effect: float
    ci_low: float
    ci_high: float
    absolute: float
    headroom: float


@dataclass(frozen=True)
class HorizonReport:
    moderator_name: str
    slope: Finding
    boundaries: tuple[float, ...]
    boundary_rule: str
    strata: tuple[Stratum, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "moderator": self.moderator_name,
            "primary": "continuous slope (no binning)",
            "slope": {
                "estimate": self.slope.estimate,
                "ci": [self.slope.ci_low, self.slope.ci_high],
                "n_units": self.slope.n_units,
                "method": self.slope.method,
            },
            "boundary_rule": self.boundary_rule,
            "boundaries": list(self.boundaries),
            "strata": [
                {
                    "index": s.index,
                    "range": [s.low, s.high],
                    "n": s.n,
                    "effect": s.effect,
                    "ci": [s.ci_low, s.ci_high],
                    "absolute": s.absolute,
                    "headroom": s.headroom,
                }
                for s in self.strata
            ],
            "note": (
                "층은 표시용이다. 판정은 연속 기울기로 한다. "
                "headroom = 1 - 해당 층의 절대 성능 — 포화가 어디서 무는지 보이게 함께 싣는다."
            ),
        }


def quantile_boundaries(moderator: Sequence[float], *, n_strata: int = 4) -> tuple[float, ...]:
    """dev 경험분위수로 경계를 만든다. 사람이 고른 상수가 아니다."""

    values = np.asarray(list(moderator), dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_strata + 1)
    edges = np.quantile(values, quantiles)
    # 동률로 경계가 겹치면 층을 줄인다 — 빈 층을 만들어 표를 늘리지 않는다.
    unique = np.unique(edges)
    return tuple(float(x) for x in unique)


def stratified_report(
    *,
    claim_id: str,
    name: str,
    moderator: Sequence[float],
    treatment: Sequence[float],
    control: Sequence[float],
    clusters: Sequence[Any],
    moderator_name: str = "root_v_star",
    n_strata: int = 4,
    role: Role = Role.EXPLORATORY,
    seed: int = 17,
) -> HorizonReport:
    """연속 기울기(1차) + 층별 표(표시용)를 함께 낸다."""

    mod = np.asarray(list(moderator), dtype=np.float64)
    treat = np.asarray(list(treatment), dtype=np.float64)
    ctrl = np.asarray(list(control), dtype=np.float64)
    labels = np.asarray(list(clusters))
    if not (mod.size == treat.size == ctrl.size == labels.size):
        raise ValueError("moderator/treatment/control/clusters 길이가 다르다")

    benefit = treat - ctrl

    slope = inference.moderator_slope(
        claim_id=claim_id,
        name=f"{name}:slope",
        moderator=mod.tolist(),
        benefit=benefit.tolist(),
        clusters=labels.tolist(),
        moderator_name=moderator_name,
        role=role,
        seed=seed,
    )

    boundaries = quantile_boundaries(mod, n_strata=n_strata)
    strata: list[Stratum] = []
    for index in range(len(boundaries) - 1):
        low, high = boundaries[index], boundaries[index + 1]
        last = index == len(boundaries) - 2
        mask = (mod >= low) & ((mod <= high) if last else (mod < high))
        if not mask.any():
            continue
        interval = estimators.cluster_bootstrap(
            benefit[mask].tolist(), labels[mask].tolist(), seed=seed
        )
        absolute = float(treat[mask].mean())
        strata.append(
            Stratum(
                index=index,
                low=float(low),
                high=float(high),
                n=int(mask.sum()),
                effect=float(interval.estimate),
                ci_low=float(interval.low),
                ci_high=float(interval.high),
                absolute=absolute,
                headroom=float(1.0 - absolute),
            )
        )

    return HorizonReport(
        moderator_name=moderator_name,
        slope=slope,
        boundaries=boundaries,
        boundary_rule=f"dev empirical quantiles, n_strata={n_strata}",
        strata=tuple(strata),
    )


def saturation_profile(
    absolute_by_level: Mapping[str, float],
    effect_by_level: Mapping[str, float],
) -> dict[str, Any]:
    """어느 출력 수준이 포화돼 있는지 한 표로.

    풀링 기준값(저널 §12): state-level range 0.2396 / 여유 0.4130,
    action-level 0.9254 / 여유 0.0746 / 기하 효과 0.0113, policy range 0.0188.
    """

    rows = []
    for level, absolute in absolute_by_level.items():
        headroom = 1.0 - float(absolute)
        effect = float(effect_by_level.get(level, float("nan")))
        rows.append(
            {
                "level": level,
                "absolute": float(absolute),
                "headroom": headroom,
                "geometry_effect": effect,
                "effect_over_headroom": (effect / headroom) if headroom > 1e-12 else float("nan"),
            }
        )
    rows.sort(key=lambda r: r["headroom"])
    return {
        "rows": rows,
        "reading": (
            "headroom 이 작은 수준에서는 기하 차이가 정책으로 번역될 물리적 여지가 없다. "
            "effect_over_headroom 은 남은 여지 중 실제로 쓴 몫이다 — "
            "작은 절대 효과가 '작은 효과'인지 '꽉 찬 천장'인지 가른다."
        ),
    }
