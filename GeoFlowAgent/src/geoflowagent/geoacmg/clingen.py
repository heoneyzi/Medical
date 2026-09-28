"""The ClinGen Evidence Repository as a corpus of curated variants.

Source: ``https://erepo.clinicalgenome.org/evrepo/api/summary/classifications/download``
(TSV, CC0).  Each row is one variant curated by a Variant Curation Expert Panel
and carries three things this study needs and cannot obtain elsewhere:

1. the panel's final classification -- the endpoint answer;
2. the evidence codes the panel **applied** (``Met``) -- the step-level answer;
3. the codes the panel **considered and rejected** (``Not Met``) -- which is the
   part that makes this a search problem rather than a lookup.  A rejected code
   cost the curator a query and returned nothing.  Without it, a policy that only
   ever calls the tools that end up mattering looks optimal, and it is not,
   because nobody knows in advance which those are.

Nothing here invents an answer.  Parsing is fail-loud: a code we cannot score is
counted and reported, never dropped into silence.
"""

from __future__ import annotations

import collections
import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from geoflowagent.geoacmg import evidence as _evidence
from geoflowagent.geoacmg.evidence import (
    Classification,
    EvidenceCode,
    UnknownEvidenceCode,
    band,
    band_of_assertion,
    parse_codes,
    score,
)

COLUMNS = (
    "Variation",
    "ClinVar Variation Id",
    "Allele Registry Id",
    "HGVS Expressions",
    "HGNC Gene Symbol",
    "Disease",
    "Mondo Id",
    "Mode of Inheritance",
    "Assertion",
    "Applied Evidence Codes (Met)",
    "Applied Evidence Codes (Not Met)",
    "Summary of interpretation",
    "PubMed Articles",
    "Expert Panel",
    "Guideline",
    "Approval Date",
    "Published Date",
    "Retracted",
    "Evidence Repo Link",
    "Uuid",
)


@dataclass(frozen=True)
class CuratedVariant:
    """One expert-curated variant: the endpoint, the steps, and the identifiers."""

    uuid: str
    gene: str
    disease: str
    mondo_id: str
    variation: str
    hgvs: tuple[str, ...]
    clinvar_id: str
    allele_registry_id: str
    assertion: Classification
    met: tuple[EvidenceCode, ...]
    not_met: tuple[EvidenceCode, ...]
    expert_panel: str
    guideline: str
    approval_date: date | None
    mode_of_inheritance: str

    @property
    def points(self) -> int:
        """Total ACMG points implied by the codes the panel applied."""

        return score(self.met)

    @property
    def implied(self) -> Classification:
        return band(self.points)

    @property
    def reconstructs(self) -> bool:
        """Whether the point scale reproduces the panel's own classification.

        Reported as a corpus-level fidelity number, never used to filter: a
        record where the points and the panel disagree is a record about which
        the panel knew something the point scale does not encode, and dropping
        those would quietly make the benchmark easier than the real task.
        """

        return self.implied is self.assertion

    @property
    def evaluated(self) -> tuple[EvidenceCode, ...]:
        """Every code the curator actually spent a query on."""

        return self.met + self.not_met

    @property
    def primary_hgvs(self) -> str:
        return self.hgvs[0] if self.hgvs else self.variation

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "gene": self.gene,
            "disease": self.disease,
            "mondo_id": self.mondo_id,
            "variation": self.variation,
            "hgvs": list(self.hgvs),
            "clinvar_id": self.clinvar_id,
            "allele_registry_id": self.allele_registry_id,
            "assertion": self.assertion.value,
            "met": [c.label for c in self.met],
            "not_met": [c.label for c in self.not_met],
            "points": self.points,
            "implied": self.implied.value,
            "reconstructs": self.reconstructs,
            "expert_panel": self.expert_panel,
            "guideline": self.guideline,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "mode_of_inheritance": self.mode_of_inheritance,
        }


@dataclass
class IngestReport:
    """What the corpus looks like, and everything that was skipped and why."""

    rows_read: int = 0
    kept: int = 0
    skipped: collections.Counter = field(default_factory=collections.Counter)
    unparseable_codes: collections.Counter = field(default_factory=collections.Counter)
    reconstructs: int = 0
    assertion_counts: collections.Counter = field(default_factory=collections.Counter)
    confusion: collections.Counter = field(default_factory=collections.Counter)
    met_code_counts: collections.Counter = field(default_factory=collections.Counter)
    evaluated_per_record: list[int] = field(default_factory=list)
    normalisations: collections.Counter = field(default_factory=collections.Counter)
    """Upstream spelling repairs applied, so a silent fix cannot hide a bug."""

    @property
    def reconstruction_rate(self) -> float:
        return self.reconstructs / self.kept if self.kept else 0.0

    def to_dict(self) -> dict[str, Any]:
        evaluated = sorted(self.evaluated_per_record)
        median = evaluated[len(evaluated) // 2] if evaluated else 0
        return {
            "rows_read": self.rows_read,
            "kept": self.kept,
            "skipped": dict(self.skipped),
            "unparseable_codes": dict(self.unparseable_codes.most_common(50)),
            "normalised_spellings": dict(self.normalisations),
            "point_scale_fidelity": {
                "reconstructs": self.reconstructs,
                "rate": self.reconstruction_rate,
                "note": (
                    "fraction of records where the Tavtigian point total lands in the "
                    "panel's own band; reported, never used to filter"
                ),
                "disagreements": {
                    f"{expert} -> {implied}": count
                    for (expert, implied), count in self.confusion.most_common()
                    if expert != implied
                },
            },
            "assertions": dict(self.assertion_counts),
            "met_codes": dict(self.met_code_counts.most_common()),
            "codes_evaluated_per_record": {
                "median": median,
                "mean": sum(evaluated) / len(evaluated) if evaluated else 0.0,
                "max": evaluated[-1] if evaluated else 0,
            },
        }


def _parse_date(text: str) -> date | None:
    text = text.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def read_corpus(
    path: str | Path,
    *,
    include_retracted: bool = False,
    require_met: bool = True,
) -> tuple[list[CuratedVariant], IngestReport]:
    """Read the Evidence Repository TSV.

    ``require_met`` drops records with no applied code: those carry an endpoint
    but no step-level answer, so they cannot serve as tasks for this study.  They
    are counted in the report rather than vanishing.
    """

    path = Path(path)
    _evidence.NORMALISATIONS.clear()
    report = IngestReport()
    records: list[CuratedVariant] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [name for name in COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is not a ClinGen Evidence Repository export; missing columns {missing}"
            )
        for row in reader:
            report.rows_read += 1
            if not include_retracted and row["Retracted"].strip().lower() == "true":
                report.skipped["retracted"] += 1
                continue
            try:
                met = parse_codes(row["Applied Evidence Codes (Met)"])
                not_met = parse_codes(row["Applied Evidence Codes (Not Met)"])
            except UnknownEvidenceCode as error:
                report.skipped["unparseable_code"] += 1
                report.unparseable_codes[str(error)] += 1
                continue
            try:
                assertion = band_of_assertion(row["Assertion"])
            except UnknownEvidenceCode:
                report.skipped["unparseable_assertion"] += 1
                continue
            if require_met and not met:
                report.skipped["no_applied_code"] += 1
                continue
            gene = row["HGNC Gene Symbol"].strip()
            if not gene:
                report.skipped["no_gene_symbol"] += 1
                continue
            record = CuratedVariant(
                uuid=row["Uuid"].strip(),
                gene=gene,
                disease=row["Disease"].strip(),
                mondo_id=row["Mondo Id"].strip(),
                variation=row["Variation"].strip(),
                hgvs=tuple(h.strip() for h in row["HGVS Expressions"].split(",") if h.strip()),
                clinvar_id=row["ClinVar Variation Id"].strip(),
                allele_registry_id=row["Allele Registry Id"].strip(),
                assertion=assertion,
                met=met,
                not_met=not_met,
                expert_panel=row["Expert Panel"].strip(),
                guideline=row["Guideline"].strip(),
                approval_date=_parse_date(row["Approval Date"]),
                mode_of_inheritance=row["Mode of Inheritance"].strip(),
            )
            records.append(record)
            report.kept += 1
            report.reconstructs += record.reconstructs
            report.assertion_counts[assertion.value] += 1
            report.confusion[(assertion.value, record.implied.value)] += 1
            report.evaluated_per_record.append(len(record.evaluated))
            for code in met:
                report.met_code_counts[code.label] += 1
    report.normalisations.update(_evidence.NORMALISATIONS)
    return records, report


def gene_clusters(records: Sequence[CuratedVariant]) -> dict[str, str]:
    """Map record uuid -> cluster key.

    The gene is the independent unit.  Two variants in the same gene share the
    gene's constraint, its dosage curation, its VCEP's specification and often
    its literature, so they are not independent evidence about a method.
    """

    return {record.uuid: record.gene for record in records}


def temporal_split(
    records: Sequence[CuratedVariant], cutoff: date
) -> tuple[list[CuratedVariant], list[CuratedVariant]]:
    """Split on panel approval date.

    The later half is the contamination control: a variant a frozen encoder could
    not have read about, because the panel had not published it yet.  Records with
    no date are placed in the earlier half, which is the conservative direction --
    it never inflates the clean split.
    """

    before: list[CuratedVariant] = []
    after: list[CuratedVariant] = []
    for record in records:
        if record.approval_date is not None and record.approval_date >= cutoff:
            after.append(record)
        else:
            before.append(record)
    return before, after


def iter_by_gene(records: Iterable[CuratedVariant]) -> Iterator[tuple[str, list[CuratedVariant]]]:
    grouped: dict[str, list[CuratedVariant]] = {}
    for record in records:
        grouped.setdefault(record.gene, []).append(record)
    for gene in sorted(grouped):
        yield gene, grouped[gene]
