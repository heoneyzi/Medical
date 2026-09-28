"""단계 5 — P2 가 애초에 이길 수 있는 비교인지 판정.

C4 는 사전등록한 조절변수가 **구조적으로 검정 불가**였다.  모든 엣지가
prefix-increasing 이라 비가역성 ρ 가 1.0 상수였고, task 그래프가 전부 동일해서
(192 노드, diameter 7, sd 0.0000) δ 도 변하지 않았다.  GPU 를 태우기 전에
종이에서 끝났어야 할 일이었다.

P2("동결 인코더 vs 처음부터 학습한 인코더")가 같은 결함을 갖고 있을 수 있다.
동결 MedCPT 는 109M, 학습되는 trunk 는 335,942 파라미터, 말뭉치는 ClinGen
13,278 행에서 나온 task 2,921 개(oracle 예제 448,704)다.

* 파라미터를 맞추면 → 109M 급을 2,921 task 로 학습, 자동 패배
* 데이터를 맞추면 → 같은 말, 자동 패배
* 연산을 맞추면 → 여기만 살아날 여지가 있다

중요한 것: **"몇 파라미터당 몇 표본이면 퇴화"라는 상수를 쓰지 않는다.**  그런 상수는
정확히 거부되는 종류의 논리다.  대신 퇴화 여부를 **측정**하는 학습곡선 프로브를
정의하고, 그 곡선이 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geoflowagent.geoacmg import estimators


@dataclass(frozen=True)
class Scale:
    """비교에 들어가는 실제 수치. 지어내지 않고 측정해 채운다."""

    frozen_encoder_params: int
    trunk_params: int
    n_tasks: int
    n_examples: int
    n_gene_clusters: int
    effective_clusters: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "frozen_encoder_params": self.frozen_encoder_params,
            "trunk_params": self.trunk_params,
            "n_tasks": self.n_tasks,
            "n_examples": self.n_examples,
            "n_gene_clusters": self.n_gene_clusters,
            "effective_clusters": self.effective_clusters,
            "params_per_effective_cluster": (
                self.frozen_encoder_params / self.effective_clusters
                if self.effective_clusters
                else float("inf")
            ),
        }


@dataclass(frozen=True)
class MatchingAxis:
    key: str
    description: str
    from_scratch_gets: str
    degenerate_reason: str | None
    empirical_test: str | None

    @property
    def survivable(self) -> bool:
        return self.degenerate_reason is None


def enumerate_axes(scale: Scale) -> tuple[MatchingAxis, ...]:
    """매칭 축마다 비퇴화 승리 가능성이 있는지 적는다."""

    return (
        MatchingAxis(
            key="parameters",
            description="두 arm 의 학습 파라미터 수를 맞춘다",
            from_scratch_gets=f"{scale.frozen_encoder_params:,} 파라미터",
            degenerate_reason=(
                f"유효 클러스터 {scale.effective_clusters:.1f} 개 위에서 "
                f"{scale.frozen_encoder_params:,} 파라미터를 학습한다. "
                "동결 arm 은 말뭉치 밖에서 이미 학습된 표현을 들고 오므로, "
                "이 축의 비교는 '사전학습이 도움이 되는가'가 아니라 "
                "'적은 데이터로 큰 모델을 처음부터 학습할 수 있는가'를 묻게 된다."
            ),
            empirical_test=None,
        ),
        MatchingAxis(
            key="data",
            description="두 arm 이 보는 데이터를 맞춘다",
            from_scratch_gets=f"task {scale.n_tasks:,} / 예제 {scale.n_examples:,}",
            degenerate_reason=(
                "동결 arm 의 표현은 말뭉치 밖에서 왔다. 데이터를 맞추는 순간 "
                "'사전학습 말뭉치를 못 쓰게 한다'는 뜻이 되어, 사전학습의 가치를 "
                "정의상 0 으로 만든 뒤 사전학습을 평가하게 된다."
            ),
            empirical_test=None,
        ),
        MatchingAxis(
            key="compute",
            description="총 학습 연산(FLOPs·wall-clock)을 맞추고 구조는 자유롭게 둔다",
            from_scratch_gets="동일 연산 예산 안에서 고를 수 있는 아무 크기의 인코더",
            degenerate_reason=None,
            empirical_test=(
                "learning_curve_probe: 데이터 비율을 올려가며 from-scratch arm 을 학습해 "
                "곡선이 전 구간에서 측정된 우연 수준에 붙어 있는지 본다. "
                "붙어 있으면 퇴화(어떤 예산으로도 배우지 못함), "
                "상승 중이면 비퇴화(예산을 늘리면 경쟁이 성립)."
            ),
        ),
        MatchingAxis(
            key="wall_clock",
            description="벽시계 시간을 맞춘다",
            from_scratch_gets="동일 시간",
            degenerate_reason=(
                "하드웨어·구현 효율이 결론을 좌우한다. 연산 축과 같은 것을 재면서 "
                "재현 불가능한 교란만 더한다."
            ),
            empirical_test=None,
        ),
    )


@dataclass
class LearningCurve:
    fractions: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    chance: float = float("nan")
    clusters: list[Any] = field(default_factory=list)

    def rising(self, *, resamples: int = 2000, seed: int = 17) -> dict[str, Any]:
        """곡선이 상승 중인지 기울기와 구간으로 답한다.

        임계값으로 "충분히 배웠다"를 선언하지 않는다.  기울기 구간이 0 을
        배제하면 상승 중이고, 포함하면 데이터를 늘려도 움직이지 않는다는 뜻이다.
        """

        if len(self.fractions) < 3:
            raise ValueError("학습곡선 판정에는 최소 3개 비율이 필요하다")
        clusters = self.clusters or list(range(len(self.fractions)))
        interval = estimators.cluster_slope(
            self.fractions, self.scores, clusters, resamples=resamples, seed=seed
        )
        top = float(np.asarray(self.scores, dtype=np.float64).max())
        return {
            "slope": interval.estimate,
            "ci": [interval.low, interval.high],
            "best_score": top,
            "measured_chance": self.chance,
            "above_chance_margin": top - self.chance,
            "reading": (
                "기울기 구간이 0 을 배제하고 최고점이 측정된 우연을 넘으면 비퇴화. "
                "전 구간이 우연에 붙어 있으면 이 축에서도 퇴화 — C4 와 같이 선언한다."
            ),
        }


def assess(scale: Scale, curve: LearningCurve | None = None) -> dict[str, Any]:
    axes = enumerate_axes(scale)
    survivable = [a for a in axes if a.survivable]
    payload: dict[str, Any] = {
        "question": "P2(동결 vs 처음부터 학습)가 비퇴화 비교를 가질 수 있는가",
        "scale": scale.to_payload(),
        "axes": [
            {
                "axis": a.key,
                "description": a.description,
                "from_scratch_gets": a.from_scratch_gets,
                "degenerate_reason": a.degenerate_reason,
                "empirical_test": a.empirical_test,
                "survivable": a.survivable,
            }
            for a in axes
        ],
        "survivable_axes": [a.key for a in survivable],
    }
    if curve is not None:
        payload["learning_curve"] = curve.rising()
    if not survivable:
        payload["declaration"] = declaration(scale, reason="어떤 매칭 축도 비퇴화 비교를 주지 않는다")
    return payload


def declaration(scale: Scale, *, reason: str) -> dict[str, Any]:
    """C4 와 같은 형식의 '구조적 검정 불가' 선언.

    형식을 맞추는 이유: 한 번은 정직하게 접었고 그것이 논문에 실렸다.
    두 번째도 같은 형식으로 접어야 독자가 같은 기준이 적용됐음을 안다.
    """

    return {
        "claim": "P2",
        "status": "structurally untestable in this benchmark",
        "reason": reason,
        "precedent": (
            "C4 와 동일한 처리. C4 는 엣지 1,458,964 개가 전부 prefix-increasing 이라 "
            "ρ = 1.0 상수였고, task 그래프가 전부 동일(192 노드, diameter 7, sd 0.0000)했다."
        ),
        "scale": scale.to_payload(),
        "consequence": "arm 을 만들지 않는다. GPU 를 태우기 전에 종이에서 끝낸다.",
        "what_would_change_it": (
            "말뭉치가 유효 클러스터 기준으로 크게 늘거나, 사전학습 말뭉치를 포함한 "
            "연산 예산 매칭이 정의되면 다시 검정 가능해진다."
        ),
    }
