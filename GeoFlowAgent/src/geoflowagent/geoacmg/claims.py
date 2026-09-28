"""Claims, findings and verdicts as first-class objects.

Every experiment in this study exists to move one pre-registered claim.  Rather
than leaving that mapping in a document, it lives in the code: an experiment
emits :class:`Finding` records tagged with a claim id, and :func:`adjudicate`
turns findings into :class:`Verdict` objects.

The adjudicator is deliberately narrow.  It accepts exactly three kinds of
evidence, which are the three forms of judgement the analysis plan permits:

``PAIRED_INTERVAL``
    A paired contrast with a cluster bootstrap interval.  Supported when the
    interval excludes zero in the pre-registered direction.
``NULL_POSITION``
    A statistic compared against a permutation or label-shuffling null.
    Supported when the null-referenced p-value clears the pre-registered alpha,
    which is itself a cited constant.
``EQUIVALENCE``
    A TOST result, for the cases where "no difference" is the claim.

There is no code path that accepts a bare threshold, so a verdict cannot be
manufactured by choosing a number that makes it come out right.  A finding whose
interval straddles zero yields ``UNRESOLVED``, never a downgraded "trend".
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from geoflowagent.geoacmg.cited import Cited, assert_decision_safe, guideline
from geoflowagent.utils.io import canonical_json, sha256_text

ALPHA = guideline(
    0.05,
    "conventional two-sided level; pre-registered before any test split was opened",
    "used for every interval and null-referenced p-value in this study",
)


class Evidence(str, Enum):
    PAIRED_INTERVAL = "paired_interval"
    NULL_POSITION = "null_position"
    EQUIVALENCE = "equivalence"


class Outcome(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"
    NOT_TESTED = "not_tested"


class Role(str, Enum):
    PRIMARY = "primary"
    EXPLORATORY = "exploratory"
    DIAGNOSTIC = "diagnostic"
    """A diagnostic describes the benchmark, it does not decide a claim."""


@dataclass(frozen=True)
class Claim:
    """A statement that the study can be wrong about."""

    claim_id: str
    statement: str
    falsifier: str
    direction: int = 1
    """Expected sign of the primary contrast: +1, -1, or 0 for an equivalence claim."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "falsifier": self.falsifier,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class Finding:
    """One measured quantity, with everything needed to judge it."""

    claim_id: str
    name: str
    evidence: Evidence
    estimate: float
    unit: str
    """The independent unit the interval is over, e.g. ``gene``."""
    n_units: int
    role: Role = Role.EXPLORATORY
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    null_mean: float | None = None
    null_draws: int | None = None
    method: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence is Evidence.PAIRED_INTERVAL and (
            self.ci_low is None or self.ci_high is None
        ):
            raise ValueError(f"{self.name}: a paired-interval finding needs an interval")
        if self.evidence is Evidence.NULL_POSITION and self.p_value is None:
            raise ValueError(f"{self.name}: a null-position finding needs a null-referenced p")
        if self.evidence is Evidence.EQUIVALENCE and self.p_value is None:
            raise ValueError(f"{self.name}: an equivalence finding needs a TOST p")
        if self.n_units <= 0:
            raise ValueError(f"{self.name}: n_units must be positive")

    @property
    def interval_excludes_zero(self) -> bool:
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def sign(self) -> int:
        if self.ci_low is not None and self.ci_low > 0.0:
            return 1
        if self.ci_high is not None and self.ci_high < 0.0:
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "name": self.name,
            "evidence": self.evidence.value,
            "role": self.role.value,
            "estimate": self.estimate,
            "ci": [self.ci_low, self.ci_high],
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_draws": self.null_draws,
            "unit": self.unit,
            "n_units": self.n_units,
            "method": self.method,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Verdict:
    claim_id: str
    outcome: Outcome
    reason: str
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "findings": list(self.findings),
        }


def _judge_one(claim: Claim, finding: Finding, alpha: float) -> tuple[Outcome, str]:
    if finding.evidence is Evidence.PAIRED_INTERVAL:
        if not finding.interval_excludes_zero:
            return (
                Outcome.UNRESOLVED,
                f"interval [{finding.ci_low:.4g}, {finding.ci_high:.4g}] includes zero",
            )
        if claim.direction == 0:
            return (
                Outcome.REFUTED,
                "an equivalence claim is contradicted by an interval that excludes zero",
            )
        if finding.sign == claim.direction:
            return (
                Outcome.SUPPORTED,
                f"interval [{finding.ci_low:.4g}, {finding.ci_high:.4g}] excludes zero "
                f"in the pre-registered direction",
            )
        return (
            Outcome.REFUTED,
            f"interval [{finding.ci_low:.4g}, {finding.ci_high:.4g}] excludes zero "
            f"in the direction opposite to the pre-registered one",
        )
    if finding.evidence is Evidence.NULL_POSITION:
        assert finding.p_value is not None
        if finding.p_value <= alpha:
            return (
                Outcome.SUPPORTED,
                f"p={finding.p_value:.4g} against a null of {finding.null_draws} draws",
            )
        return (
            Outcome.UNRESOLVED,
            f"p={finding.p_value:.4g} does not separate from the permutation null",
        )
    assert finding.p_value is not None
    if finding.p_value <= alpha:
        return (Outcome.SUPPORTED, f"TOST p={finding.p_value:.4g}: equivalence established")
    return (Outcome.UNRESOLVED, f"TOST p={finding.p_value:.4g}: equivalence not established")


def adjudicate(
    claims: Sequence[Claim],
    findings: Iterable[Finding],
    *,
    alpha: Cited[float] = ALPHA,
) -> list[Verdict]:
    """Map findings onto claim outcomes.

    Only findings whose role is PRIMARY decide a claim.  Exploratory findings are
    reported but never promoted into a verdict, which is what keeps a sweep over
    many configurations from becoming a result.
    """

    assert_decision_safe(alpha, "adjudicate(alpha=...)")
    level = float(alpha.value)
    by_claim: dict[str, list[Finding]] = {claim.claim_id: [] for claim in claims}
    for finding in findings:
        if finding.claim_id not in by_claim:
            raise KeyError(
                f"finding {finding.name!r} references unregistered claim {finding.claim_id!r}"
            )
        by_claim[finding.claim_id].append(finding)

    verdicts: list[Verdict] = []
    for claim in claims:
        primaries = [f for f in by_claim[claim.claim_id] if f.role is Role.PRIMARY]
        if not primaries:
            verdicts.append(
                Verdict(claim.claim_id, Outcome.NOT_TESTED, "no primary finding was reported", ())
            )
            continue
        judged = [(_judge_one(claim, f, level), f) for f in primaries]
        outcomes = {outcome for (outcome, _), _ in judged}
        names = tuple(f.name for _, f in judged)
        if Outcome.REFUTED in outcomes:
            reason = "; ".join(r for (o, r), _ in judged if o is Outcome.REFUTED)
            verdicts.append(Verdict(claim.claim_id, Outcome.REFUTED, reason, names))
        elif outcomes == {Outcome.SUPPORTED}:
            reason = "; ".join(r for (_, r), _ in judged)
            verdicts.append(Verdict(claim.claim_id, Outcome.SUPPORTED, reason, names))
        else:
            reason = "; ".join(r for (o, r), _ in judged if o is not Outcome.SUPPORTED)
            verdicts.append(Verdict(claim.claim_id, Outcome.UNRESOLVED, reason, names))
    return verdicts


@dataclass(frozen=True)
class Preregistration:
    """The frozen analysis plan. Amended, never edited."""

    study: str
    claims: tuple[Claim, ...]
    primary_findings: tuple[str, ...]
    """Names of the findings allowed to carry PRIMARY role. Anything else is exploratory."""
    alpha: Cited[float] = ALPHA
    registered_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    amends: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        ids = [claim.claim_id for claim in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate claim ids in a preregistration")
        if not self.primary_findings:
            raise ValueError(
                "a preregistration must name its primary findings; leaving the set open "
                "lets the largest observed effect become the headline"
            )

    def fingerprint(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))[:16]

    def enforce_roles(self, findings: Iterable[Finding]) -> list[Finding]:
        """Demote any finding that claims PRIMARY without being pre-registered."""

        allowed = set(self.primary_findings)
        out: list[Finding] = []
        for finding in findings:
            if finding.role is Role.PRIMARY and finding.name not in allowed:
                raise ValueError(
                    f"finding {finding.name!r} claims primary role but was not "
                    f"pre-registered; pre-registered primaries are {sorted(allowed)}"
                )
            out.append(finding)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "claims": [claim.to_dict() for claim in self.claims],
            "primary_findings": list(self.primary_findings),
            "alpha": self.alpha.to_dict(),
            "registered_at": self.registered_at,
            "amends": self.amends,
            "note": self.note,
        }
