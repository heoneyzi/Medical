"""단계 6 — 사전등록 수정본.

지금이 수정할 수 있는 **유일하게 정당한 시점**이다.  test 를 한 번도 열지 않았기
때문이다(test task 0개).  개봉 이후의 수정은 결과를 보고 규칙을 바꾸는 것이 된다.

수정본에 반드시 들어가는 것:

* 기존 fingerprint ``143e496ce323074c`` 와의 diff
* 수정 시점의 봉인 상태(감사 가능한 수치로)
* C4 구조적 검정 불가 선언 + ``delta_p95`` 를 거부한 이유
* C5 철회 + 통제에서 살아남은 해리
* 자유도를 상수가 아니라 **곡선**으로 등록
* 저널의 옛 dev 수치표를 **사전 기록된 예측**으로 첨부

마지막 항목이 중요하다.  산출물을 잃었기 때문에 재실행을 하는데, 옛 수치가 먼저
기록돼 있으므로 재실행은 사후 보고가 아니라 **사전 기록된 예측에 대한 독립 재현**이
된다.  단일 실행이었던 원래 상태보다 입증력이 높다.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PRIOR_FINGERPRINT = "143e496ce323074c"

#: 저널 §3 의 dev 평균. 재현 대상이지 통과 기준이 아니다.
PRIOR_DEV_TABLE: Mapping[str, Mapping[str, float]] = {
    "pair_mlp": {"policy": 0.8699, "joint": 0.7890, "regret_at_1": 0.1438, "params": 369_479},
    "directed_quasimetric": {"policy": 0.8622, "joint": 0.7870, "regret_at_1": 0.1540, "params": 338_022},
    "cosine": {"policy": 0.8613, "joint": 0.7813, "regret_at_1": 0.1444, "params": 335_942},
    "poincare": {"policy": 0.8603, "joint": 0.7867, "regret_at_1": 0.1533, "params": 340_038},
    "euclidean": {"policy": 0.8512, "joint": 0.7786, "regret_at_1": 0.1649, "params": 335_942},
}


@dataclass(frozen=True)
class Retraction:
    claim_id: str
    what_it_looked_like: str
    what_the_control_showed: str
    what_survives: str


@dataclass(frozen=True)
class Untestable:
    claim_id: str
    registered_definition: str
    why_unestimable: str
    alternatives_and_why_refused: str


@dataclass(frozen=True)
class DegreeOfFreedom:
    """상수 대신 곡선으로 등록하는 자유도.

    값을 고르면 그 값을 방어해야 한다.  대신 곡선 전체에서 결론이 유지되는 것을
    요구사항으로 등록하면 방어할 상수가 사라진다.
    """

    key: str
    was: str
    now: str
    sweep: tuple[Any, ...]
    requirement: str


DEFAULT_DOF: tuple[DegreeOfFreedom, ...] = (
    DegreeOfFreedom(
        key="max_tasks_per_gene",
        was="캡 100 (단일 값)",
        now="캡 곡선 전체",
        sweep=(25, 50, 100, 200, None),
        requirement="결론의 방향이 곡선 전 구간에서 유지되어야 한다",
    ),
    DegreeOfFreedom(
        key="task_length_weighting",
        was="dev 자연 분포 (root V*=3.5 에 47.1% 쏠림)",
        now="세 가중치 전부 보고",
        sweep=("natural", "uniform", "length_stratified"),
        requirement=(
            "가중치에 따라 승자가 바뀌면 단일 승자를 보고하지 않는다. "
            "selected_energy=pair_mlp 는 자연 분포의 함수이므로 단독 보고 금지."
        ),
    ),
    DegreeOfFreedom(
        key="energy_family_policy",
        was="dev 에서 최고 family 를 선택",
        now="family 축 자체를 결과로 보고",
        sweep=("cosine", "euclidean", "directed_quasimetric", "pair_mlp", "poincare"),
        requirement="하나를 '그 모델'로 승격하지 않는다",
    ),
    DegreeOfFreedom(
        key="seeds",
        was="family 당 3",
        now="시드 분산과 유전자 분산을 분리 보고, 시드 수는 검정력에서 계산",
        sweep=(3, 5, 7),
        requirement="시드 구간과 유전자 구간을 하나로 합치지 않는다",
    ),
)

DEFAULT_RETRACTIONS: tuple[Retraction, ...] = (
    Retraction(
        claim_id="C5",
        what_it_looked_like=(
            "within-model 상관이 지지처럼 보였다. 단 범위를 정확히 적는다 — "
            "15런 전체는 −0.1277 ~ +0.6142 이고 euclidean 3런은 전부 음수다. "
            "구간이 0을 배제하는 11런만 보면 +0.3114 ~ +0.6142 이며, "
            "그 11런이라는 선별 사실과 개수를 함께 적는다(무언의 필터 금지)."
        ),
        what_the_control_showed=(
            "세 그룹을 모두 적는다: A=B +0.3282(n=15) · A≠B 같은 에너지 +0.3206(n=30) · "
            "A≠B 다른 에너지 +0.3289(n=180). 일치쌍이 불일치쌍보다 높지 않으므로 "
            "잰 것은 모델 사이의 관계가 아니라 과제 난이도다. "
            "독립 확인: R4_crossmodel_control.json 이 다른 통계량(정렬×정책 곱)에서 "
            "차이 +0.0002 [−0.00016, +0.00051] 로 UNRESOLVED 를 낸다."
        ),
        what_survives=(
            "해리는 남는다: 정렬 원천 효과가 계획 원천 효과보다 약 10배 크다. "
            "단 euclidean −0.044 를 단독 점추정으로 쓰지 않는다 — 그 자체로는 0을 포함해 "
            "UNRESOLVED 다. 대비(pair_mlp − euclidean) 형태로만 인용하고, "
            "그 구간은 findings 에 산출해 둔 뒤 참조한다."
        ),
    ),
)

DEFAULT_UNTESTABLE: tuple[Untestable, ...] = (
    Untestable(
        claim_id="C4",
        registered_definition=(
            "Gromov 4-point δ 와 비가역성 ρ 를 조절변수로. "
            "출처는 docs/01_EXPERIMENT_PLAN.md 이며, **지문이 찍힌 "
            "preregistration.json(143e496ce323074c) 본문에는 이 정의가 없다** — "
            "그 파일의 C4 에는 statement 와 falsifier 만 있다. 이 사실을 숨기지 않는다."
        ),
        why_unestimable=(
            "엣지 1,458,964 개가 전부 prefix-increasing, 감소 0개 → ρ = 1.0 상수. "
            "노드 수 192 는 전 task 동일하나 **엣지 수는 608~639 로 변동한다"
            "(평균 624.29)** — '그래프가 전부 동일'은 사실이 아니므로 쓰지 않는다. "
            "불변인 것은 delta_max 뿐이다(sd 0.0, 고유값 1). delta_mean 은 고유값 578 · "
            "sd 0.004863, delta_p95 는 고유값 2 · sd 0.05842 로 변동한다."
        ),
        alternatives_and_why_refused=(
            "delta_p95 로는 poincare +0.02709 [+0.01782, +0.03684] 가 나오지만 "
            "거부한다. 이유 셋: (1) 세 operationalization 이 서로 어긋나는데 "
            "결과가 나오는 정의를 고르는 것이 된다. (2) delta_p95 의 상위 수준(2.0)은 "
            "dev 584 task 중 **2개**뿐이다. (3) 유일한 연속 조작화인 delta_mean 에서는 "
            "dqm +0.68755 [−0.08324, +1.49714], poincare +1.04622 [−0.02424, +1.93597] 로 "
            "둘 다 0을 포함한다. 등록된 반증조건('기울기 구간이 0을 포함')이 충족되므로 "
            "REFUTED 로 읽을 여지가 있으나, 조절변수 자체가 퇴화(ρ 상수)라 "
            "**검정이 성립하지 않았다**고 본다. 두 읽기를 모두 적어 둔다."
        ),
    ),
)


@dataclass
class Amendment:
    study: str
    amends: str = PRIOR_FINGERPRINT
    reason: str = ""
    seal_state: dict[str, Any] = field(default_factory=dict)
    primary_reassignment: dict[str, str] = field(default_factory=dict)
    retractions: tuple[Retraction, ...] = DEFAULT_RETRACTIONS
    untestable: tuple[Untestable, ...] = DEFAULT_UNTESTABLE
    degrees_of_freedom: tuple[DegreeOfFreedom, ...] = DEFAULT_DOF
    prior_dev_table: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: dict(PRIOR_DEV_TABLE)
    )
    registered_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "amends": self.amends,
            "registered_at": self.registered_at,
            "reason": self.reason,
            "seal_state_at_amendment": self.seal_state,
            "primary_reassignment": self.primary_reassignment,
            "retractions": [r.__dict__ for r in self.retractions],
            "structurally_untestable": [u.__dict__ for u in self.untestable],
            "degrees_of_freedom": [
                {
                    "key": d.key,
                    "was": d.was,
                    "now": d.now,
                    "sweep": list(d.sweep),
                    "requirement": d.requirement,
                }
                for d in self.degrees_of_freedom
            ],
            "prior_prediction": {
                "source": "EXPERIMENT_JOURNAL_20260919.md §3 (dev means, test sealed)",
                "table": {k: dict(v) for k, v in self.prior_dev_table.items()},
                "status": (
                    "**'독립 재현'이라고 쓰지 않는다.** 시드 17·29·43 은 그 표를 만든 "
                    "바로 그 학습 실행이고 산출물이 살아 있다(metric_profile.json, 2026-09-19). "
                    "따라서 정확한 서술은 '결정적 재도출 + 신규 시드 59·71 확장'이다. "
                    "그리고 정렬 축은 재현됐으나 **이 표의 지표(policy/joint/regret)는 "
                    "신규 시드에서 부호가 뒤집혔다**(R1b_new_seed_planning_axis.json). "
                    "이 표는 통과 기준이 아니다."
                ),
            },
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        """수정본을 봉인하는 지문.

        **본문 전체를 해시한다.**  claim_id 와 sweep 값만 해시하면 철회 사유나
        검정 불가 근거를 사후에 고쳐도 지문이 움직이지 않아, 수정본의 존재 이유인
        변조 탐지가 무력해진다.  시각 필드(``verified_at``)만 제외한다 — 그것은
        내용이 아니라 실행 시점이고 포함하면 매 실행마다 지문이 바뀐다.
        """

        seal = {k: v for k, v in self.seal_state.items() if k != "verified_at"}
        payload = {
            "study": self.study,
            "amends": self.amends,
            "reason": self.reason,
            "primary": self.primary_reassignment,
            "retractions": [
                {
                    "claim_id": r.claim_id,
                    "what_it_looked_like": r.what_it_looked_like,
                    "what_the_control_showed": r.what_the_control_showed,
                    "what_survives": r.what_survives,
                }
                for r in self.retractions
            ],
            "untestable": [
                {
                    "claim_id": u.claim_id,
                    "registered_definition": u.registered_definition,
                    "why_unestimable": u.why_unestimable,
                    "alternatives_and_why_refused": u.alternatives_and_why_refused,
                }
                for u in self.untestable
            ],
            "dof": [
                {"key": d.key, "was": d.was, "now": d.now,
                 "sweep": list(d.sweep), "requirement": d.requirement}
                for d in self.degrees_of_freedom
            ],
            "prior_dev_table": {k: dict(v) for k, v in self.prior_dev_table.items()},
            "seal_state": seal,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def analysis_plan_sha256(self) -> str:
        blob = json.dumps(self.to_payload(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def require_sealed(
    test_task_count: int,
    *,
    where: str = "amendment",
    available: int | None = None,
    counted_from: str | None = None,
) -> dict[str, Any]:
    """수정 시점에 test 가 봉인돼 있었음을 감사 가능한 형태로 남긴다."""

    if test_task_count != 0:
        raise RuntimeError(
            f"{where}: test task 가 {test_task_count} 개다. 봉인이 깨진 뒤의 수정은 "
            "결과를 보고 규칙을 바꾸는 것이므로 거부한다."
        )
    return {
        # "열린 test 가 0" 과 "test 가 존재하지 않음" 은 다르다. 둘을 분리해 적는다.
        # ``test_tasks`` 는 기존 계약(테스트가 검사한다)이라 유지한다.
        "test_tasks": 0,
        "test_tasks_opened": 0,
        "test_tasks_available": available if available is not None else "unrecorded",
        "counted_from": counted_from or "caller did not record the source",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statement": "수정 시점에 test 는 한 번도 열리지 않았다",
    }
