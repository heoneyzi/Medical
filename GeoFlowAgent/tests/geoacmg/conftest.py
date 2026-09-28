from __future__ import annotations

import csv
from pathlib import Path

import pytest

HEADER = [
    "Variation", "ClinVar Variation Id", "Allele Registry Id", "HGVS Expressions",
    "HGNC Gene Symbol", "Disease", "Mondo Id", "Mode of Inheritance", "Assertion",
    "Applied Evidence Codes (Met)", "Applied Evidence Codes (Not Met)",
    "Summary of interpretation", "PubMed Articles", "Expert Panel", "Guideline",
    "Approval Date", "Published Date", "Retracted", "Evidence Repo Link", "Uuid",
]


def _row(**kwargs) -> dict[str, str]:
    row = dict.fromkeys(HEADER, "")
    row.update(kwargs)
    return row


#: Rows chosen to exercise one behaviour each, so a failure names its cause.
FIXTURE_ROWS = [
    # PVS1 needs the gene mechanism as well as the consequence: two families.
    _row(Uuid="u-pvs1", **{"HGNC Gene Symbol": "GENEA", "Assertion": "Pathogenic",
         "Applied Evidence Codes (Met)": "PVS1,PM2_Supporting,PP3",
         "Applied Evidence Codes (Not Met)": "BA1,BP4",
         "HGVS Expressions": "NM_000001.1:c.10C>T", "Approval Date": "2025-01-10",
         "Expert Panel": "Panel A", "Disease": "disease a"}),
    # Stand-alone benign from one query.
    _row(Uuid="u-ba1", **{"HGNC Gene Symbol": "GENEA", "Assertion": "Benign",
         "Applied Evidence Codes (Met)": "BA1", "Applied Evidence Codes (Not Met)": "PM2,PP3",
         "HGVS Expressions": "NM_000001.1:c.20A>G", "Approval Date": "2020-03-01",
         "Expert Panel": "Panel A", "Disease": "disease a"}),
    # Truncated upstream spelling that must be normalised, not dropped.
    _row(Uuid="u-trunc", **{"HGNC Gene Symbol": "GENEB", "Assertion": "Likely Pathogenic",
         "Applied Evidence Codes (Met)": "PM3_Very,PM2_Supporting",
         "HGVS Expressions": "NM_000002.1:c.5G>A", "Approval Date": "2026-02-02",
         "Expert Panel": "Panel B", "Disease": "disease b"}),
    # Out of scope: PS3 needs a functional assay, so this record cannot become a task.
    _row(Uuid="u-ps3", **{"HGNC Gene Symbol": "GENEB", "Assertion": "Pathogenic",
         "Applied Evidence Codes (Met)": "PS3,PVS1", "HGVS Expressions": "NM_000002.1:c.9T>C",
         "Approval Date": "2025-06-06", "Expert Panel": "Panel B", "Disease": "disease b"}),
    # Retracted rows are excluded.
    _row(Uuid="u-retracted", Retracted="true", **{"HGNC Gene Symbol": "GENEC",
         "Assertion": "Benign", "Applied Evidence Codes (Met)": "BA1",
         "HGVS Expressions": "NM_000003.1:c.1A>T", "Expert Panel": "Panel C"}),
    # No applied code: an endpoint with no step-level answer.
    _row(Uuid="u-empty", **{"HGNC Gene Symbol": "GENEC", "Assertion": "Uncertain Significance",
         "HGVS Expressions": "NM_000003.1:c.2A>T", "Expert Panel": "Panel C"}),
]


@pytest.fixture
def corpus_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "erepo.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        for row in FIXTURE_ROWS:
            writer.writerow(row)
    return path
