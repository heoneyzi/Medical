"""Findings in, verdicts and a readable report out.

The last step of the study is mechanical on purpose.  Every experiment has
already written its findings to disk; this module loads them, checks them against
the frozen preregistration, adjudicates, and renders.  Nothing is decided here
that was not decided when the plan was registered.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from geoflowagent.geoacmg.cited import Cited
from geoflowagent.geoacmg.claims import (
    ALPHA,
    Claim,
    Evidence,
    Finding,
    Outcome,
    Preregistration,
    Role,
    adjudicate,
)
from geoflowagent.geoacmg.inference import adjust_exploratory


def write_findings(path: str | Path, findings: Iterable[Finding]) -> int:
    rows = [finding.to_dict() for finding in findings]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def read_findings(path: str | Path) -> list[Finding]:
    findings: list[Finding] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ci = row.get("ci") or [None, None]
        findings.append(
            Finding(
                claim_id=row["claim_id"],
                name=row["name"],
                evidence=Evidence(row["evidence"]),
                estimate=row["estimate"],
                unit=row["unit"],
                n_units=row["n_units"],
                role=Role(row["role"]),
                ci_low=ci[0],
                ci_high=ci[1],
                p_value=row.get("p_value"),
                null_mean=row.get("null_mean"),
                null_draws=row.get("null_draws"),
                method=row.get("method", ""),
                detail=row.get("detail", {}),
            )
        )
    return findings


def _claims_from(payload: dict[str, Any]) -> tuple[Claim, ...]:
    return tuple(
        Claim(
            claim_id=row["claim_id"],
            statement=row["statement"],
            falsifier=row["falsifier"],
            direction=int(row.get("direction", 1)),
        )
        for row in payload["claims"]
    )


def load_preregistration(path: str | Path) -> Preregistration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prereg = Preregistration(
        study=payload["study"],
        claims=_claims_from(payload),
        primary_findings=tuple(payload["primary_findings"]),
        # An amendment may move alpha.  Reading the module default instead would
        # adjudicate at a level the registration does not say, and the recomputed
        # fingerprint would silently stop matching the one stored in the file.
        alpha=Cited.from_dict(payload["alpha"]) if "alpha" in payload else ALPHA,
        registered_at=payload.get("registered_at", ""),
        amends=payload.get("amends"),
        note=payload.get("note", ""),
    )
    stored = payload.get("fingerprint")
    if stored and stored != prereg.fingerprint():
        raise ValueError(
            f"preregistration fingerprint mismatch: file says {stored!r}, the plan it "
            f"describes hashes to {prereg.fingerprint()!r}. The registration was edited "
            "in place; amend it instead of editing it."
        )
    return prereg


_SYMBOL = {
    Outcome.SUPPORTED: "지지",
    Outcome.REFUTED: "반증",
    Outcome.UNRESOLVED: "미결",
    Outcome.NOT_TESTED: "미측정",
}


def render(
    prereg: Preregistration,
    findings: Sequence[Finding],
    *,
    corpus_card: dict[str, Any] | None = None,
) -> str:
    """A report whose conclusions a reader can check line by line."""

    checked = prereg.enforce_roles(findings)
    verdicts = adjudicate(prereg.claims, checked, alpha=prereg.alpha)
    adjusted = adjust_exploratory(checked)
    by_claim: dict[str, list[Finding]] = {}
    for finding in checked:
        by_claim.setdefault(finding.claim_id, []).append(finding)

    lines: list[str] = [
        f"# {prereg.study}",
        "",
        f"사전등록 {prereg.registered_at} · fingerprint `{prereg.fingerprint()}`"
        + (f" · amends `{prereg.amends}`" if prereg.amends else ""),
        "",
        "판정은 세 가지 증거만으로 내려집니다 — 짝지은 신뢰구간이 0을 배제하는가, "
        "순열 귀무분포 대비 위치, TOST 동등성. 임계값은 쓰이지 않았습니다.",
        "",
        "## 판정",
        "",
        "| 주장 | 결론 | 근거 |",
        "|---|---|---|",
    ]
    for claim, verdict in zip(prereg.claims, verdicts, strict=True):
        lines.append(
            f"| **{claim.claim_id}** {claim.statement} | {_SYMBOL[verdict.outcome]} | "
            f"{verdict.reason} |"
        )

    lines += ["", "## 주검정", "", "| finding | 추정 | 구간 | 단위 | n |", "|---|---:|---|---|---:|"]
    for finding in checked:
        if finding.role is not Role.PRIMARY:
            continue
        interval = (
            f"[{finding.ci_low:.4g}, {finding.ci_high:.4g}]"
            if finding.ci_low is not None
            else f"p={finding.p_value:.4g}"
        )
        lines.append(
            f"| `{finding.name}` | {finding.estimate:.4g} | {interval} | "
            f"{finding.unit} | {finding.n_units} |"
        )

    exploratory = [f for f in checked if f.role is Role.EXPLORATORY]
    if exploratory:
        lines += [
            "",
            "## 탐색적 결과",
            "",
            "사전등록되지 않았으므로 주장을 지지하지 않습니다. p값은 Holm 보정 후 함께 적습니다.",
            "",
            "| finding | 추정 | 구간 또는 p | Holm |",
            "|---|---:|---|---|",
        ]
        for finding in exploratory:
            interval = (
                f"[{finding.ci_low:.4g}, {finding.ci_high:.4g}]"
                if finding.ci_low is not None
                else f"p={finding.p_value:.4g}"
            )
            holm = adjusted.get(finding.name)
            lines.append(
                f"| `{finding.name}` | {finding.estimate:.4g} | {interval} | "
                f"{format(holm, '.4g') if holm is not None else '—'} |"
            )

    diagnostics = [f for f in checked if f.role is Role.DIAGNOSTIC]
    if diagnostics or corpus_card:
        lines += ["", "## 벤치마크 진단", "", "장비에 대한 서술이며 주장이 아닙니다.", ""]
        for finding in diagnostics:
            lines.append(
                f"- `{finding.name}` = {finding.estimate:.4g} "
                + (
                    f"[{finding.ci_low:.4g}, {finding.ci_high:.4g}]"
                    if finding.ci_low is not None
                    else ""
                )
            )
        if corpus_card:
            card = corpus_card
            lines += [
                f"- task {card.get('tasks')} · 유전자 {card.get('genes')} · "
                f"패널 {card.get('expert_panels')}",
                f"- 유효 클러스터 "
                f"{card.get('tasks_per_gene', {}).get('effective_number_of_clusters', 0):.1f} "
                "(유전자 수가 아니라 이 값으로 구간을 읽을 것)",
                f"- 최빈 클래스 "
                f"{card.get('target_balance', {}).get('majority_class_rate', 0):.3f}",
                f"- 단일 도구군 천장 대비 격차 "
                f"{card.get('single_family_ceiling', {}).get('gap', 0):.3f}",
            ]

    missing = [v.claim_id for v in verdicts if v.outcome is Outcome.NOT_TESTED]
    if missing:
        lines += [
            "",
            "## 아직 측정되지 않은 주장",
            "",
            "주검정 finding이 보고되지 않았습니다: " + ", ".join(missing) + ".",
        ]
    return "\n".join(lines) + "\n"


def build(
    prereg_path: str | Path,
    findings_path: str | Path,
    output_path: str | Path,
    *,
    corpus_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prereg = load_preregistration(prereg_path)
    findings = read_findings(findings_path)
    text = render(prereg, findings, corpus_card=corpus_card)
    Path(output_path).write_text(text, encoding="utf-8")
    verdicts = adjudicate(prereg.claims, prereg.enforce_roles(findings), alpha=prereg.alpha)
    return {
        "output": str(output_path),
        "findings": len(findings),
        "verdicts": {v.claim_id: v.outcome.value for v in verdicts},
    }
