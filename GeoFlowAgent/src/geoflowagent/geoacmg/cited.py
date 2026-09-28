"""Constants that carry their provenance.

The research plan forbids thresholds chosen because they seemed reasonable.  A
promise in a document does not survive contact with a deadline, so the rule is
enforced by the type system instead: every numeric constant that can influence a
decision is a :class:`Cited`, and a :class:`Cited` cannot be constructed without
a source.  Code that wants a bare float has to call :meth:`Cited.value`, which is
greppable, and the source travels with the number into every report.

Three kinds of provenance are allowed, and they are exactly the three forms of
judgement the analysis plan permits:

``GUIDELINE``
    A published clinical or statistical convention (ACMG/AMP, ClinGen SVI, a
    named paper).  The number is not ours.
``MEASURED``
    Estimated from data we collected, with the estimation procedure named.  A
    measured constant must record the sample it came from.
``CONVENTION``
    A value with no decision content: a random seed, a display precision, a
    hard resource bound.  These may not be used in a comparison that produces a
    verdict, and :func:`assert_decision_safe` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T", int, float)


class Provenance(str, Enum):
    GUIDELINE = "guideline"
    MEASURED = "measured"
    CONVENTION = "convention"


@dataclass(frozen=True)
class Cited(Generic[T]):
    """A number that knows where it came from."""

    value: T
    provenance: Provenance
    source: str
    note: str = ""
    sample: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("a cited constant needs a non-empty source")
        if self.provenance is Provenance.MEASURED and not self.sample.strip():
            raise ValueError(
                f"measured constant {self.source!r} must record the sample it was "
                "estimated from"
            )

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "source": self.source,
            "note": self.note,
            "sample": self.sample,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Cited[Any]:
        """Rebuild a cited constant written by :meth:`to_dict`.

        A registration that was amended to a different alpha has to come back as
        that alpha.  Without this, reading a preregistration silently substitutes
        the module default and the recomputed fingerprint stops matching the one
        on disk, which is the failure the fingerprint exists to catch.
        """

        return cls(
            value=payload["value"],
            provenance=Provenance(payload["provenance"]),
            source=payload["source"],
            note=payload.get("note", ""),
            sample=payload.get("sample", ""),
        )


def guideline(value: T, source: str, note: str = "") -> Cited[T]:
    return Cited(value=value, provenance=Provenance.GUIDELINE, source=source, note=note)


def measured(value: T, source: str, sample: str, note: str = "") -> Cited[T]:
    return Cited(
        value=value,
        provenance=Provenance.MEASURED,
        source=source,
        note=note,
        sample=sample,
    )


def convention(value: T, source: str, note: str = "") -> Cited[T]:
    return Cited(value=value, provenance=Provenance.CONVENTION, source=source, note=note)


def assert_decision_safe(constant: Cited[Any], where: str) -> Cited[Any]:
    """Reject a convention used where a verdict depends on it.

    A seed or a display width may be arbitrary.  A number that decides whether a
    claim is supported may not be.
    """

    if constant.provenance is Provenance.CONVENTION:
        raise ValueError(
            f"{where} would let a convention decide an outcome: "
            f"{constant.source!r}={constant.value!r}. Use a guideline citation or a "
            "measured estimate, or compare against a null distribution instead."
        )
    return constant
