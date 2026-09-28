"""단계 4 — cross-model 통제를 표준으로.

C5 는 within-model 상관이 +0.33~+0.61 로 지지처럼 보였다가, cross-model 통제에서
A=B +0.3282 ≈ A≠B +0.3289 로 무너졌다.  **과제 난이도 인공물**이었다.

한 번 무너진 것이 중요한 게 아니라, **이 데이터셋이 그런 인공물을 생산한다는 것이
확인됐다**는 게 중요하다.  그러면 남아 있는 상관 주장도 같은 방식으로 죽을 수 있다.
그래서 통제를 사람의 기억이 아니라 타입에 건다: 상관 근거를 쓰는 Finding 이
통제값을 달고 있지 않으면 리포트가 거부된다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from geoflowagent.geoacmg import estimators
from geoflowagent.geoacmg.claims import Finding

CONTROL_KEY = "crossmodel_control"

#: 통제가 반드시 필요한 근거 종류.  상관/기울기 계열은 난이도 인공물에 취약하다.
CORRELATIONAL_METHODS = ("slope", "correlation", "moderator", "alignment", "linkage")


class MissingControl(RuntimeError):
    """상관 주장인데 cross-model 통제값이 붙어 있지 않다."""


@dataclass(frozen=True)
class ControlReport:
    statistic_name: str
    matched: float
    matched_ci: tuple[float, float]
    mismatched: float
    mismatched_ci: tuple[float, float]
    difference: float
    difference_ci: tuple[float, float]
    n_units: int

    @property
    def survives(self) -> str:
        """판정을 임계값으로 내리지 않는다. 구간 위치를 문장으로 돌려준다."""

        low, high = self.difference_ci
        if low > 0:
            return "matched > mismatched, interval excludes zero"
        if high < 0:
            return "mismatched > matched, interval excludes zero"
        return "interval includes zero — indistinguishable from a task-difficulty artifact"

    def to_payload(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic_name,
            "matched": {"estimate": self.matched, "ci": list(self.matched_ci)},
            "mismatched": {"estimate": self.mismatched, "ci": list(self.mismatched_ci)},
            "difference": {"estimate": self.difference, "ci": list(self.difference_ci)},
            "n_units": self.n_units,
            "reading": self.survives,
            "precedent": (
                "C5: within-model +0.33~+0.61 로 지지처럼 보였으나 "
                "A=B +0.3282 vs A≠B +0.3289 로 통제에서 무너짐 (저널 §6)"
            ),
        }


def crossmodel_control(
    *,
    statistic_name: str,
    matched: Sequence[float],
    mismatched: Sequence[float],
    clusters: Sequence[Any],
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> ControlReport:
    """일치쌍(A=B)과 불일치쌍(A≠B)에서 같은 통계를 재고 차이를 구간으로 낸다.

    같은 단위(보통 유전자 클러스터)에서 짝지어 들어와야 한다.  일치쌍만 높고
    불일치쌍도 같이 높다면, 잰 것은 모델 사이의 관계가 아니라 과제 난이도다.
    """

    if not (len(matched) == len(mismatched) == len(clusters)):
        raise ValueError("matched/mismatched/clusters 길이가 다르다")
    if not matched:
        raise ValueError("빈 표본으로 통제를 계산할 수 없다")

    matched_interval = estimators.cluster_bootstrap(
        [float(x) for x in matched], list(clusters), resamples=resamples, alpha=alpha, seed=seed
    )
    mismatched_interval = estimators.cluster_bootstrap(
        [float(x) for x in mismatched], list(clusters), resamples=resamples, alpha=alpha, seed=seed
    )
    differences = [float(a) - float(b) for a, b in zip(matched, mismatched, strict=True)]
    difference_interval = estimators.cluster_bootstrap(
        differences, list(clusters), resamples=resamples, alpha=alpha, seed=seed
    )
    return ControlReport(
        statistic_name=statistic_name,
        matched=float(matched_interval.estimate),
        matched_ci=(float(matched_interval.low), float(matched_interval.high)),
        mismatched=float(mismatched_interval.estimate),
        mismatched_ci=(float(mismatched_interval.low), float(mismatched_interval.high)),
        difference=float(difference_interval.estimate),
        difference_ci=(float(difference_interval.low), float(difference_interval.high)),
        n_units=int(difference_interval.units),
    )


def attach(finding: Finding, report: ControlReport) -> Finding:
    """Finding 의 detail 에 통제값을 붙인 사본을 돌려준다."""

    detail = dict(finding.detail)
    detail[CONTROL_KEY] = report.to_payload()
    return Finding(
        claim_id=finding.claim_id,
        name=finding.name,
        evidence=finding.evidence,
        estimate=finding.estimate,
        unit=finding.unit,
        n_units=finding.n_units,
        role=finding.role,
        ci_low=finding.ci_low,
        ci_high=finding.ci_high,
        p_value=finding.p_value,
        null_mean=finding.null_mean,
        null_draws=finding.null_draws,
        method=finding.method,
        detail=detail,
    )


def needs_control(finding: Finding) -> bool:
    haystack = f"{finding.name} {finding.method}".lower()
    return any(token in haystack for token in CORRELATIONAL_METHODS)


def enforce_controls(findings: Sequence[Finding]) -> None:
    """통제 없는 상관 주장이 있으면 리포트를 만들지 못하게 죽인다.

    ``claims.adjudicate`` 앞에 세워 쓴다.  규율을 문서가 아니라 실행 경로에 건다.
    """

    offenders = [
        f.name for f in findings if needs_control(f) and CONTROL_KEY not in (f.detail or {})
    ]
    if offenders:
        raise MissingControl(
            "상관 근거를 쓰는 finding 에 cross-model 통제값이 없다:\n  "
            + "\n  ".join(offenders)
            + "\n\nC5 가 바로 이 방식으로 무너졌다(저널 §6). control.crossmodel_control 로 "
            "통제를 계산해 control.attach 로 붙일 것."
        )
