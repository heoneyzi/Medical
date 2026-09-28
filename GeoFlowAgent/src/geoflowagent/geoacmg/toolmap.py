"""Which tool can decide which ACMG criterion, and which criteria we cannot decide.

The mapping is read off the criterion definitions in Richards et al. 2015, not
invented here: PM2 is defined in terms of population databases, PP3 in terms of
computational predictors, PVS1 in terms of the variant's molecular consequence
and the gene's mechanism.  A VCEP specification may change the *threshold* for a
criterion in its gene; it does not change which kind of data answers it.  That is
why the base mapping is fixed and spec refinements are layered on top.

Some criteria are honestly out of reach.  PS3 needs a functional assay from the
literature; PP1 needs a pedigree; PS2 needs confirmed trio data.  Pretending an
API can decide those would be the same mistake as phase 1's synthetic facts.  The
coverage report states the scope explicitly, weighted by how often each criterion
actually occurs, so a reader can see how much of real curation this benchmark
covers -- and how much it does not.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from geoflowagent.geoacmg.clingen import CuratedVariant
from geoflowagent.geoacmg.evidence import CRITERIA, RICHARDS_2015


class ToolFamily(str, Enum):
    POPULATION_FREQUENCY = "population_frequency"
    MOLECULAR_CONSEQUENCE = "molecular_consequence"
    GENE_MECHANISM = "gene_mechanism"
    IN_SILICO = "in_silico"
    CLINICAL_ASSERTION = "clinical_assertion"
    PHENOTYPE = "phenotype"
    LITERATURE = "literature"
    PEDIGREE = "pedigree"
    COHORT = "cohort"


#: The concrete endpoint behind each family. Verified reachable on 2026-09-19.
FAMILY_SOURCES: dict[ToolFamily, tuple[str, ...]] = {
    ToolFamily.POPULATION_FREQUENCY: ("gnomAD GraphQL (r4)", "MARRVEL /data/gnomAD"),
    ToolFamily.MOLECULAR_CONSEQUENCE: ("Ensembl REST VEP (r116)", "Ensembl /lookup/id"),
    ToolFamily.GENE_MECHANISM: (
        "ClinGen gene curation list (FTP)",
        "ClinGen dosage sensitivity",
        "gnomAD constraint (LOEUF)",
    ),
    ToolFamily.IN_SILICO: ("AlphaMissense precomputed (CC BY 4.0)", "MARRVEL /data/dbNSFP"),
    ToolFamily.CLINICAL_ASSERTION: ("ClinVar weekly release", "MARRVEL /data/clinVar"),
    ToolFamily.PHENOTYPE: ("HPO / phenotype.hpoa", "Geno2MP", "MARRVEL /data/Geno2MP"),
    ToolFamily.LITERATURE: (),
    ToolFamily.PEDIGREE: (),
    ToolFamily.COHORT: (),
}


class Decidability(str, Enum):
    DECIDABLE = "decidable"
    """A tool in our registry returns the value the criterion is defined on."""

    PARTIAL = "partial"
    """A tool returns a proxy; the criterion also needs judgement we do not have."""

    OUT_OF_SCOPE = "out_of_scope"
    """No API answers this. It needs a paper, a pedigree or a cohort."""


@dataclass(frozen=True)
class CriterionBinding:
    """Which query yields a criterion, and what else must be known to read it.

    ``primary`` is the call that returns the value.  ``requires`` are the calls
    whose answers are needed to *interpret* that value, and they matter: PVS1 is
    not "the variant is truncating", it is "the variant is truncating **and**
    loss of function is an established mechanism for this gene".  Modelling the
    prerequisite is what stops a one-call policy from claiming PVS1, and it is
    where the multi-step structure of real curation actually lives.

    Each criterion contributes its points exactly once, through its primary
    family, and only once every prerequisite has also been queried.
    """

    base: str
    primary: ToolFamily
    decidability: Decidability
    rationale: str
    requires: tuple[ToolFamily, ...] = ()
    source: str = RICHARDS_2015

    @property
    def families(self) -> tuple[ToolFamily, ...]:
        return (self.primary, *self.requires)

    def resolved_by(self, queried: frozenset[ToolFamily] | set[ToolFamily]) -> bool:
        return self.primary in queried and all(f in queried for f in self.requires)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "primary": self.primary.value,
            "requires": [f.value for f in self.requires],
            "decidability": self.decidability.value,
            "rationale": self.rationale,
            "source": self.source,
        }


def _bind(
    base: str,
    families: Sequence[ToolFamily],
    decidability: Decidability,
    rationale: str,
) -> CriterionBinding:
    if base not in CRITERIA:
        raise KeyError(f"{base} is not a registered ACMG criterion")
    if not families:
        raise ValueError(f"{base} needs at least a primary family")
    return CriterionBinding(
        base=base,
        primary=families[0],
        requires=tuple(families[1:]),
        decidability=decidability,
        rationale=rationale,
    )


F = ToolFamily
D = Decidability

BINDINGS: dict[str, CriterionBinding] = {
    b.base: b
    for b in (
        _bind("PVS1", [F.MOLECULAR_CONSEQUENCE, F.GENE_MECHANISM], D.DECIDABLE,
              "null variant plus a loss-of-function mechanism: VEP consequence and exon "
              "position give the NMD arm of the SVI decision tree; ClinGen dosage gives "
              "whether LoF is an established mechanism for the gene"),
        _bind("PM2", [F.POPULATION_FREQUENCY], D.DECIDABLE,
              "defined on absence or extreme rarity in population databases"),
        _bind("BA1", [F.POPULATION_FREQUENCY], D.DECIDABLE,
              "defined on a population allele frequency above the stand-alone cut-off"),
        _bind("BS1", [F.POPULATION_FREQUENCY], D.DECIDABLE,
              "defined on a frequency greater than expected for the disorder"),
        _bind("PP3", [F.IN_SILICO], D.DECIDABLE,
              "defined on computational predictors; AlphaMissense class or score"),
        _bind("BP4", [F.IN_SILICO], D.DECIDABLE,
              "the benign-direction counterpart of PP3, same predictors"),
        _bind("BP7", [F.MOLECULAR_CONSEQUENCE, F.IN_SILICO], D.DECIDABLE,
              "synonymous variant with no predicted splice effect: consequence plus a "
              "splice predictor"),
        _bind("PM4", [F.MOLECULAR_CONSEQUENCE], D.DECIDABLE,
              "protein length change from an in-frame indel or stop-loss, read off the "
              "consequence annotation"),
        _bind("BP3", [F.MOLECULAR_CONSEQUENCE], D.DECIDABLE,
              "in-frame indel in a repeat region without known function"),
        _bind("PS1", [F.CLINICAL_ASSERTION], D.DECIDABLE,
              "same amino acid change as a previously established pathogenic variant"),
        _bind("PM5", [F.CLINICAL_ASSERTION], D.DECIDABLE,
              "a different change at a residue where another change is pathogenic"),
        _bind("PP2", [F.GENE_MECHANISM], D.DECIDABLE,
              "gene-level missense constraint"),
        _bind("BP1", [F.GENE_MECHANISM], D.DECIDABLE,
              "gene where only truncating variants cause disease"),
        _bind("PP4", [F.PHENOTYPE], D.PARTIAL,
              "phenotype specificity can be scored against HPO disease annotations, but "
              "the curator's judgement about this patient is not in any database"),
        _bind("PM1", [F.MOLECULAR_CONSEQUENCE], D.PARTIAL,
              "domain and hotspot annotation is retrievable; whether a region is a "
              "*mutational hotspot* for this disease usually needs curation"),
        _bind("PM3", [F.CLINICAL_ASSERTION], D.PARTIAL,
              "in-trans observations appear in ClinVar case-level data when submitters "
              "provide phase, which is often missing"),
        _bind("BP2", [F.CLINICAL_ASSERTION], D.PARTIAL,
              "same phase limitation as PM3"),
        _bind("BP5", [F.CLINICAL_ASSERTION], D.PARTIAL,
              "requires knowing that a case had an alternate molecular cause"),
        _bind("BS2", [F.POPULATION_FREQUENCY], D.PARTIAL,
              "observation in healthy adults needs zygosity and age, only partly present "
              "in population releases"),
        _bind("PS3", [F.LITERATURE], D.OUT_OF_SCOPE,
              "needs a well-established functional assay reported in a paper"),
        _bind("BS3", [F.LITERATURE], D.OUT_OF_SCOPE, "functional assay, benign direction"),
        _bind("PS2", [F.PEDIGREE], D.OUT_OF_SCOPE, "de novo with confirmed parentage"),
        _bind("PM6", [F.PEDIGREE], D.OUT_OF_SCOPE, "assumed de novo without confirmation"),
        _bind("PP1", [F.PEDIGREE], D.OUT_OF_SCOPE, "cosegregation in a family"),
        _bind("BS4", [F.PEDIGREE], D.OUT_OF_SCOPE, "lack of segregation in a family"),
        _bind("PS4", [F.COHORT], D.OUT_OF_SCOPE,
              "case-control prevalence needs an affected cohort we do not have"),
    )
}

missing = sorted(set(CRITERIA) - set(BINDINGS))
if missing:  # pragma: no cover - guards a future criterion added without a binding
    raise RuntimeError(f"ACMG criteria with no tool binding: {missing}")


def families_for(base: str) -> tuple[ToolFamily, ...]:
    return BINDINGS[base].families


def primary_family(base: str) -> ToolFamily:
    return BINDINGS[base].primary


def decidable_bases() -> frozenset[str]:
    return frozenset(b.base for b in BINDINGS.values() if b.decidability is D.DECIDABLE)


def coverage(records: Sequence[CuratedVariant]) -> dict[str, Any]:
    """How much of real curation this tool registry can actually decide.

    Two views, because they answer different questions.  *By code* says which
    criteria we cover.  *By record* says what fraction of an expert's applied
    evidence we could reproduce, and how many records are fully inside scope --
    which is the number that determines the benchmark's usable size.
    """

    per_code: collections.Counter = collections.Counter()
    per_decidability: collections.Counter = collections.Counter()
    fully_covered = 0
    covered_points = 0
    total_points = 0
    families_per_record: list[int] = []
    for record in records:
        bases = [code.base for code in record.met]
        per_code.update(bases)
        decidable = True
        families: set[ToolFamily] = set()
        for code in record.met:
            binding = BINDINGS[code.base]
            per_decidability[binding.decidability.value] += 1
            total_points += abs(code.points)
            if binding.decidability is D.DECIDABLE:
                covered_points += abs(code.points)
                families.update(binding.families)
            else:
                decidable = False
        fully_covered += decidable
        families_per_record.append(len(families))
    counts = collections.Counter(families_per_record)
    return {
        "records": len(records),
        "records_fully_decidable": fully_covered,
        "records_fully_decidable_fraction": fully_covered / len(records) if records else 0.0,
        "applied_codes_by_decidability": dict(per_decidability),
        "evidence_points_decidable_fraction": (
            covered_points / total_points if total_points else 0.0
        ),
        "distinct_tool_families_per_record": {
            "histogram": {str(k): v for k, v in sorted(counts.items())},
            "mean": (
                sum(families_per_record) / len(families_per_record) if families_per_record else 0.0
            ),
            "at_least_two": sum(1 for n in families_per_record if n >= 2),
            "at_least_three": sum(1 for n in families_per_record if n >= 3),
            "note": (
                "tool diversity is a property of the data, not a design choice: this is "
                "how many distinct kinds of evidence a real curation actually used"
            ),
        },
        "bindings": {base: binding.to_dict() for base, binding in sorted(BINDINGS.items())},
    }
