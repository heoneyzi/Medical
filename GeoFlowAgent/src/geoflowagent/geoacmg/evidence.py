"""ACMG/AMP evidence codes, strength modifiers, and the point scale.

The point scale is the reason this benchmark has a *step-wise* notion of
progress at all.  Every intermediate state has a score, the score implies a
classification band, and the distance to the band is the amount of evidence
still missing.  None of those numbers are ours:

* strength -> points, and the band boundaries, are Tavtigian et al. 2020,
  *Hum Mutat* 41:1734, "Fitting a naturally scaled point system to the ACMG/AMP
  variant classification guidelines".  That paper is the log-scaled integer form
  of the 2018 Bayesian framework (Tavtigian et al., *Genet Med* 20:1054) and is
  the basis of the forthcoming ACMG SVC v4.
* the criteria themselves are Richards et al. 2015, *Genet Med* 17:405.
* strength modifiers (``PM2_Supporting``, ``PP1_Strong``, ...) are the ClinGen
  SVI convention for a VCEP re-weighting a criterion for its gene.

The code grammar was derived from the 13,278 published ClinGen Evidence
Repository records, not guessed: 26 base codes and five modifier spellings,
including the two that contain a space (``PS4_Very Strong``, ``BS2_Stand Alone``).
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from geoflowagent.geoacmg.cited import Cited, guideline

TAVTIGIAN_2020 = "Tavtigian et al. 2020, Hum Mutat 41:1734 (naturally scaled point system)"
RICHARDS_2015 = "Richards et al. 2015, Genet Med 17:405 (ACMG/AMP criteria)"
SVI_STRENGTH = "ClinGen SVI strength-modification convention"


class Direction(str, Enum):
    PATHOGENIC = "pathogenic"
    BENIGN = "benign"


class Strength(str, Enum):
    SUPPORTING = "Supporting"
    MODERATE = "Moderate"
    STRONG = "Strong"
    VERY_STRONG = "Very Strong"
    STAND_ALONE = "Stand Alone"


#: Magnitude of one evidence line, in points. Sign comes from the direction.
STRENGTH_POINTS: dict[Strength, Cited[int]] = {
    Strength.SUPPORTING: guideline(1, TAVTIGIAN_2020, "Supporting = 1 point"),
    Strength.MODERATE: guideline(2, TAVTIGIAN_2020, "Moderate = 2 points"),
    Strength.STRONG: guideline(4, TAVTIGIAN_2020, "Strong = 4 points"),
    Strength.VERY_STRONG: guideline(8, TAVTIGIAN_2020, "Very Strong = 8 points"),
    Strength.STAND_ALONE: guideline(8, TAVTIGIAN_2020, "Stand-alone benign (BA1) = -8 points"),
}


class Classification(str, Enum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    UNCERTAIN = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"

    @property
    def is_decisive(self) -> bool:
        """Anything other than VUS ends the curation."""

        return self is not Classification.UNCERTAIN


#: Inclusive point bounds per band, ordered from pathogenic to benign.
BAND_BOUNDS: tuple[tuple[Classification, Cited[int], Cited[int]], ...] = (
    (
        Classification.PATHOGENIC,
        guideline(10, TAVTIGIAN_2020, "Pathogenic at >= 10 points"),
        guideline(10**6, TAVTIGIAN_2020, "no upper bound"),
    ),
    (
        Classification.LIKELY_PATHOGENIC,
        guideline(6, TAVTIGIAN_2020, "Likely Pathogenic spans 6..9 points"),
        guideline(9, TAVTIGIAN_2020, "Likely Pathogenic spans 6..9 points"),
    ),
    (
        Classification.UNCERTAIN,
        guideline(0, TAVTIGIAN_2020, "Uncertain spans 0..5 points"),
        guideline(5, TAVTIGIAN_2020, "Uncertain spans 0..5 points"),
    ),
    (
        Classification.LIKELY_BENIGN,
        guideline(-6, TAVTIGIAN_2020, "Likely Benign spans -6..-1 points"),
        guideline(-1, TAVTIGIAN_2020, "Likely Benign spans -6..-1 points"),
    ),
    (
        Classification.BENIGN,
        guideline(-(10**6), TAVTIGIAN_2020, "no lower bound"),
        guideline(-7, TAVTIGIAN_2020, "Benign at <= -7 points"),
    ),
)


@dataclass(frozen=True)
class Criterion:
    """One ACMG/AMP criterion at its guideline-default strength."""

    base: str
    direction: Direction
    default_strength: Strength
    summary: str


def _p(base: str, strength: Strength, summary: str) -> Criterion:
    return Criterion(base, Direction.PATHOGENIC, strength, summary)


def _b(base: str, strength: Strength, summary: str) -> Criterion:
    return Criterion(base, Direction.BENIGN, strength, summary)


#: The 26 base criteria that occur in the ClinGen Evidence Repository.
CRITERIA: dict[str, Criterion] = {
    c.base: c
    for c in (
        _p("PVS1", Strength.VERY_STRONG, "null variant in a gene where loss of function is a known mechanism"),
        _p("PS1", Strength.STRONG, "same amino acid change as an established pathogenic variant"),
        _p("PS2", Strength.STRONG, "de novo, with maternity and paternity confirmed"),
        _p("PS3", Strength.STRONG, "well-established functional study shows a damaging effect"),
        _p("PS4", Strength.STRONG, "prevalence in affecteds significantly exceeds controls"),
        _p("PM1", Strength.MODERATE, "located in a mutational hot spot or critical functional domain"),
        _p("PM2", Strength.MODERATE, "absent or extremely rare in population databases"),
        _p("PM3", Strength.MODERATE, "detected in trans with a pathogenic variant (recessive)"),
        _p("PM4", Strength.MODERATE, "protein length change from in-frame indel or stop-loss"),
        _p("PM5", Strength.MODERATE, "novel missense at a residue where another change is pathogenic"),
        _p("PM6", Strength.MODERATE, "assumed de novo without confirmation of parentage"),
        _p("PP1", Strength.SUPPORTING, "cosegregation with disease in affected family members"),
        _p("PP2", Strength.SUPPORTING, "missense in a gene with a low rate of benign missense variation"),
        _p("PP3", Strength.SUPPORTING, "computational evidence supports a deleterious effect"),
        _p("PP4", Strength.SUPPORTING, "patient phenotype is highly specific for the gene's disease"),
        _b("BA1", Strength.STAND_ALONE, "allele frequency too high for the disorder (stand-alone benign)"),
        _b("BS1", Strength.STRONG, "allele frequency greater than expected for the disorder"),
        _b("BS2", Strength.STRONG, "observed in healthy adults in the expected inheritance state"),
        _b("BS3", Strength.STRONG, "well-established functional study shows no damaging effect"),
        _b("BS4", Strength.STRONG, "lack of segregation in affected family members"),
        _b("BP1", Strength.SUPPORTING, "missense in a gene where only truncating variants cause disease"),
        _b("BP2", Strength.SUPPORTING, "observed in trans or in cis with a pathogenic variant"),
        _b("BP3", Strength.SUPPORTING, "in-frame indel in a repeat region without known function"),
        _b("BP4", Strength.SUPPORTING, "computational evidence suggests no impact"),
        _b("BP5", Strength.SUPPORTING, "found in a case with an alternate molecular basis for disease"),
        _b("BP7", Strength.SUPPORTING, "synonymous with no predicted splice impact"),
    )
}

_CODE_RE = re.compile(r"^([PB][A-Z]{1,2}\d+)(?:_(.+))?$")
_STRENGTH_BY_SPELLING = {s.value.lower().replace(" ", ""): s for s in Strength}

#: The published export is internally inconsistent: the same panel writes both
#: ``PM3_Very Strong`` and ``PM3_Very``, and both ``BS1_Stand Alone`` and
#: ``BS1_Stand``.  The truncations are unambiguous -- ACMG has exactly one
#: strength beginning with "Very" and one beginning with "Stand" -- so they are
#: normalised rather than dropped.  Every normalisation is counted and surfaced
#: in the ingest report, because a silent repair is indistinguishable from a bug.
_STRENGTH_ALIASES: dict[str, Strength] = {
    "very": Strength.VERY_STRONG,
    "verystr": Strength.VERY_STRONG,
    "stand": Strength.STAND_ALONE,
    "standalone": Strength.STAND_ALONE,
    "sa": Strength.STAND_ALONE,
}

#: Records every alias applied during a run, for the ingest report.
NORMALISATIONS: collections.Counter[str] = collections.Counter()


@dataclass(frozen=True)
class EvidenceCode:
    """A criterion applied at a specific strength, e.g. ``PM2_Supporting``."""

    base: str
    strength: Strength

    @property
    def direction(self) -> Direction:
        return CRITERIA[self.base].direction

    @property
    def label(self) -> str:
        if self.strength is CRITERIA[self.base].default_strength:
            return self.base
        return f"{self.base}_{self.strength.value}"

    @property
    def points(self) -> int:
        magnitude = STRENGTH_POINTS[self.strength].value
        return magnitude if self.direction is Direction.PATHOGENIC else -magnitude

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base": self.base,
            "strength": self.strength.value,
            "direction": self.direction.value,
            "points": self.points,
        }


class UnknownEvidenceCode(ValueError):
    """Raised instead of silently dropping a code we cannot score."""


def parse_code(text: str) -> EvidenceCode:
    """Parse ``PM2`` / ``PM2_Supporting`` / ``PS4_Very Strong``.

    Unparseable input raises. A benchmark that silently discards evidence codes
    it does not understand reports a score that is quietly wrong.
    """

    raw = text.strip()
    match = _CODE_RE.match(raw)
    if match is None:
        raise UnknownEvidenceCode(f"not an ACMG evidence code: {text!r}")
    base, modifier = match.group(1), match.group(2)
    criterion = CRITERIA.get(base)
    if criterion is None:
        raise UnknownEvidenceCode(f"unknown ACMG criterion {base!r} (from {text!r})")
    if modifier is None:
        return EvidenceCode(base, criterion.default_strength)
    key = modifier.strip().lower().replace(" ", "").replace("_", "")
    strength = _STRENGTH_BY_SPELLING.get(key)
    if strength is None:
        strength = _STRENGTH_ALIASES.get(key)
        if strength is not None:
            NORMALISATIONS[f"{modifier.strip()} -> {strength.value}"] += 1
    if strength is None:
        raise UnknownEvidenceCode(f"unknown strength modifier {modifier!r} (from {text!r})")
    return EvidenceCode(base, strength)


def parse_codes(text: str) -> tuple[EvidenceCode, ...]:
    """Parse a comma-separated code list as published by ClinGen."""

    return tuple(parse_code(part) for part in text.split(",") if part.strip())


def score(codes: Iterable[EvidenceCode]) -> int:
    """Total points. Additive by construction of the point scale."""

    return sum(code.points for code in codes)


def band(points: int) -> Classification:
    """The classification implied by a point total."""

    for classification, low, high in BAND_BOUNDS:
        if low.value <= points <= high.value:
            return classification
    raise AssertionError(f"point total {points} fell outside every band")


def points_to_band(points: int, target: Classification) -> int:
    """Evidence still missing, in points, before ``target`` is reached.

    Zero when the total is already inside the band.  This is the quantity the
    goal-conditioned distance head is asked to represent: not an invented cost,
    but how much more evidence a curator still has to find.
    """

    for classification, low, high in BAND_BOUNDS:
        if classification is not target:
            continue
        if points < low.value:
            return low.value - points
        if points > high.value:
            return points - high.value
        return 0
    raise AssertionError(f"unknown band {target!r}")


def band_of_assertion(assertion: str) -> Classification:
    """Map a ClinGen ``Assertion`` string onto the classification enum."""

    key = assertion.strip().lower()
    for classification in Classification:
        if classification.value.lower() == key:
            return classification
    aliases = {
        "likely pathogenic": Classification.LIKELY_PATHOGENIC,
        "likely benign": Classification.LIKELY_BENIGN,
        "uncertain significance": Classification.UNCERTAIN,
        "vus": Classification.UNCERTAIN,
    }
    if key in aliases:
        return aliases[key]
    raise UnknownEvidenceCode(f"unrecognised ClinGen assertion {assertion!r}")


def reconstruct(codes: Iterable[EvidenceCode]) -> tuple[int, Classification]:
    """Points and the band they imply."""

    total = score(codes)
    return total, band(total)


def citations() -> dict[str, str]:
    return {
        "points_and_bands": TAVTIGIAN_2020,
        "criteria": RICHARDS_2015,
        "strength_modifiers": SVI_STRENGTH,
    }
