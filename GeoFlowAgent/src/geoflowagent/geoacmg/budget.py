"""벽시계 예산 관리자 — 20시간 안에 끝내기.

지난 실행의 실패 방식은 "느렸다"가 아니라 **"18시간째에 남은 작업이 8시간짜리인 걸
알게 됐다"** 였다.  flow 전수 평가는 8.1시간 뒤에도 미완이었고 root-only 로도
3시간을 넘겼다.  그러니 예산은 사후 보고가 아니라 **시작 전 게이트**여야 한다.

원칙 셋:

1. **소요 시간을 지어내지 않는다.**  각 스테이지는 작은 표본으로 자기 속도를
   측정(``calibrate``)하고, 관리자는 그 측정값으로 외삽한다.  측정 없이 등록된
   스테이지는 "미지"로 표시되며 미지가 남은 채로는 계획을 확정하지 않는다.
2. **싸고 결정적인 것을 먼저.**  단계 2·3·4 는 기존 체크포인트 위의 순수 분석이라
   분 단위다.  비싼 것이 못 끝나도 논문의 핵심은 손에 남는다.
3. **못 들어가면 줄이거나 거른다.**  마감을 넘길 스테이지는 축소 스위치를 켜거나
   건너뛰고, 그 사실을 결과에 남긴다. 조용히 시작해서 마감에 잘리는 것이 최악이다.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


class DeadlineExceeded(RuntimeError):
    """남은 예산으로는 이 스테이지를 시작할 수 없다."""


@dataclass(frozen=True)
class Estimate:
    seconds: float
    basis: str
    measured: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "seconds": round(self.seconds, 1),
            "minutes": round(self.seconds / 60.0, 1),
            "basis": self.basis,
            "measured": self.measured,
        }


@dataclass
class StagePlan:
    key: str
    title: str
    decisive: bool
    needs_gpu: bool
    units: int
    calibrate: Callable[[int], float] | None = None
    reduction: str | None = None
    estimate: Estimate | None = None
    skipped_reason: str | None = None

    def measure(self, *, probe_units: int = 8) -> Estimate:
        """작은 표본으로 속도를 재고 외삽한다."""

        if self.calibrate is None:
            self.estimate = Estimate(float("nan"), "no calibration registered", False)
            return self.estimate
        probe = max(1, min(probe_units, self.units))
        seconds = float(self.calibrate(probe))
        per_unit = seconds / probe
        self.estimate = Estimate(
            per_unit * self.units,
            f"measured {seconds:.1f}s over {probe} units -> {per_unit:.3f}s/unit x {self.units}",
            True,
        )
        return self.estimate


@dataclass
class Governor:
    """남은 시간을 아는 실행 관리자."""

    total_seconds: float
    started: float = field(default_factory=time.time)
    spent: dict[str, float] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def for_hours(cls, hours: float) -> Governor:
        return cls(total_seconds=hours * 3600.0)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float:
        return self.total_seconds - self.elapsed

    def plan(self, stages: Sequence[StagePlan], *, probe_units: int = 8) -> dict[str, Any]:
        """전부 측정하고, 싸고 결정적인 것부터 배치하고, 안 들어가는 것을 표시한다."""

        for stage in stages:
            if stage.estimate is None:
                stage.measure(probe_units=probe_units)

        def sort_key(stage: StagePlan) -> tuple[int, float]:
            seconds = stage.estimate.seconds if stage.estimate else float("inf")
            if seconds != seconds:  # NaN
                seconds = float("inf")
            return (0 if stage.decisive else 1, seconds)

        ordered = sorted(stages, key=sort_key)

        budget_left = self.remaining
        rows: list[dict[str, Any]] = []
        for stage in ordered:
            seconds = stage.estimate.seconds if stage.estimate else float("nan")
            fits = seconds == seconds and seconds <= budget_left
            if fits:
                budget_left -= seconds
            elif seconds != seconds:
                stage.skipped_reason = None  # 미지 — 거르지는 않되 계획을 확정하지 않는다
            else:
                stage.skipped_reason = (
                    f"예상 {seconds / 3600:.1f}h > 남은 예산 {budget_left / 3600:.1f}h"
                    + (f"; 축소 스위치: {stage.reduction}" if stage.reduction else "")
                )
            rows.append(
                {
                    "stage": stage.key,
                    "title": stage.title,
                    "decisive": stage.decisive,
                    "needs_gpu": stage.needs_gpu,
                    "estimate": stage.estimate.to_payload() if stage.estimate else None,
                    "fits": fits,
                    "skipped_reason": stage.skipped_reason,
                    "reduction": stage.reduction,
                }
            )

        unknown = [r["stage"] for r in rows if r["estimate"] and not r["estimate"]["measured"]]
        return {
            "budget_hours": round(self.total_seconds / 3600.0, 2),
            "remaining_hours": round(self.remaining / 3600.0, 2),
            "order": [r["stage"] for r in rows],
            "rows": rows,
            "unmeasured_stages": unknown,
            "slack_hours": round(budget_left / 3600.0, 2),
            "note": (
                "결정적 스테이지를 먼저 둔다. 비싼 것이 못 끝나도 핵심 결과는 남는다. "
                "미측정 스테이지가 있으면 계획은 확정이 아니다."
            ),
        }

    def guard(self, stage: StagePlan, *, safety: float = 1.25) -> None:
        """스테이지 시작 직전 게이트. 못 들어가면 시작하지 않는다."""

        if stage.estimate is None:
            stage.measure()
        seconds = stage.estimate.seconds if stage.estimate else float("nan")
        if seconds != seconds:
            return  # 측정 불가는 경고만 — plan() 이 이미 표시했다
        needed = seconds * safety
        if needed > self.remaining:
            raise DeadlineExceeded(
                f"{stage.key}: 예상 {seconds / 3600:.1f}h (안전계수 {safety}) > "
                f"남은 {self.remaining / 3600:.1f}h.\n"
                + (f"축소 스위치를 켜고 다시 시도: {stage.reduction}" if stage.reduction else
                   "이 스테이지를 건너뛰고 결과에 남길 것.")
            )

    def record(self, key: str, seconds: float) -> None:
        self.spent[key] = self.spent.get(key, 0.0) + seconds
        self.log.append(
            {
                "stage": key,
                "seconds": round(seconds, 1),
                "elapsed_hours": round(self.elapsed / 3600.0, 2),
                "remaining_hours": round(self.remaining / 3600.0, 2),
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "budget_hours": round(self.total_seconds / 3600.0, 2),
            "elapsed_hours": round(self.elapsed / 3600.0, 2),
            "remaining_hours": round(self.remaining / 3600.0, 2),
            "spent_by_stage_minutes": {k: round(v / 60.0, 1) for k, v in self.spent.items()},
            "log": self.log,
        }


#: 지난 실행에서 실제로 관측된 시간. 계획의 출발점이지 상수가 아니다 —
#: ``StagePlan.calibrate`` 가 이번 하드웨어에서 다시 잰다.
OBSERVED_LAST_RUN: dict[str, str] = {
    "oracle": "2,921 task 기준 약 47분 (238 task 282초에서 외삽)",
    "flow_eval_exhaustive": "8.1시간 실행 후에도 미완 — 쓰지 않는다",
    "flow_eval_root_only": "3시간 초과 (CPU 1.2코어에 묶인 파이썬 루프)",
    "geometry_sweep": "시드별 배치로 store 적재 3회",
    "store_load": "약 12분 / 회",
}
