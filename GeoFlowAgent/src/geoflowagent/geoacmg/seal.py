"""단계 8 — test 개봉 가드.

test 는 봉인이다(현재 test task 0개).  개봉은 되돌릴 수 없고 한 번뿐이므로,
실수로 열리는 경로가 있으면 안 된다.  이 모듈은 개봉 전에 세 가지를 강제한다.

1. **무엇을 볼지 먼저 적는다.**  개봉 의향서(토큰 파일)에 볼 finding 이름을
   전부 적어야 하고, 적히지 않은 것은 개봉 후에도 계산되지 않는다.
2. **자유도가 먼저 얼어 있어야 한다.**  가중치·캡·family 선택이 확정되지
   않은 상태의 개봉은 갈림길을 열어둔 채 여는 것이다.
3. **의향서가 개봉보다 먼저 쓰였어야 한다.**  파일 mtime 과 해시를 기록한다.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SealViolation(RuntimeError):
    """봉인 규약을 어긴 개봉 시도."""


@dataclass(frozen=True)
class OpeningIntent:
    written_at: str
    reason: str
    findings_to_read: tuple[str, ...]
    frozen_degrees_of_freedom: dict[str, Any]
    analysis_plan_sha256: str

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "reason": self.reason,
                "findings": sorted(self.findings_to_read),
                "dof": self.frozen_degrees_of_freedom,
                "plan": self.analysis_plan_sha256,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


REQUIRED_DOF = ("max_tasks_per_gene", "task_length_weighting", "energy_family_policy", "seeds")


def write_intent(
    path: str | Path,
    *,
    reason: str,
    findings_to_read: Sequence[str],
    frozen_degrees_of_freedom: dict[str, Any],
    analysis_plan_sha256: str,
) -> OpeningIntent:
    missing = [key for key in REQUIRED_DOF if key not in frozen_degrees_of_freedom]
    if missing:
        raise SealViolation(
            "개봉 전에 얼려야 할 자유도가 비어 있다: " + ", ".join(missing) + "\n"
            "이 값들이 정해지지 않은 채 test 를 열면 개봉 후에 고르는 것이 된다."
        )
    if not findings_to_read:
        raise SealViolation("무엇을 볼지 적지 않은 개봉은 허용되지 않는다")
    if not reason.strip():
        raise SealViolation("개봉 사유가 비어 있다")

    intent = OpeningIntent(
        written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reason=reason.strip(),
        findings_to_read=tuple(findings_to_read),
        frozen_degrees_of_freedom=dict(frozen_degrees_of_freedom),
        analysis_plan_sha256=analysis_plan_sha256,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "written_at": intent.written_at,
                "reason": intent.reason,
                "findings_to_read": list(intent.findings_to_read),
                "frozen_degrees_of_freedom": intent.frozen_degrees_of_freedom,
                "analysis_plan_sha256": intent.analysis_plan_sha256,
                "fingerprint": intent.fingerprint(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        "utf-8",
    )
    return intent


def require_intent(path: str | Path, *, analysis_plan_sha256: str) -> OpeningIntent:
    """개봉 직전에 호출한다. 의향서가 없거나 계획이 바뀌었으면 죽는다."""

    target = Path(path)
    if not target.exists():
        raise SealViolation(
            f"개봉 의향서가 없다: {target}\n"
            "seal.write_intent 로 무엇을 볼지 먼저 적고, 그 다음에 열 것."
        )
    payload = json.loads(target.read_text("utf-8"))
    intent = OpeningIntent(
        written_at=payload["written_at"],
        reason=payload["reason"],
        findings_to_read=tuple(payload["findings_to_read"]),
        frozen_degrees_of_freedom=payload["frozen_degrees_of_freedom"],
        analysis_plan_sha256=payload["analysis_plan_sha256"],
    )
    if intent.analysis_plan_sha256 != analysis_plan_sha256:
        raise SealViolation(
            "의향서를 쓴 뒤 분석 계획이 바뀌었다.\n"
            f"  의향서: {intent.analysis_plan_sha256[:16]}\n"
            f"  현재:   {analysis_plan_sha256[:16]}\n"
            "바뀐 계획으로 열려면 의향서를 다시 쓰고, 왜 바뀌었는지 남길 것."
        )
    return intent


def filter_to_intent(findings: Sequence[Any], intent: OpeningIntent) -> list[Any]:
    """의향서에 적힌 finding 만 통과시킨다. 적지 않은 것은 개봉 후에도 안 본다."""

    allowed = set(intent.findings_to_read)
    return [f for f in findings if getattr(f, "name", None) in allowed]
