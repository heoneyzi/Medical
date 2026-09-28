"""Minimal pairs, from real records rather than from construction.

Phase 1 reported this diagnostic as empty: *"명시적 minimal-pair corpus가 없어 해당
intrinsic 항목은 아직 0개다"*.  The repository already has the slot -- the embedding
cache encodes ``minimal_pairs.jsonl`` and ``minimal_pair_metrics`` scores it -- so
the only thing missing was pairs.

Clinical genomics supplies them for free, in two relations the existing schema
already understands:

``invariant``
    Two spellings of the *same* variant.  ClinGen publishes several HGVS
    expressions per record (``NM_000260.3:c.3503+12_3503+33del``,
    ``...del22``, ``...delGAGGCGGGGA``).  The surface differs, the biology does
    not.  An encoder whose representation moves between these is reading
    notation, not variants.

``functional_sensitive``
    Two variants in the *same gene* with *opposite* classifications.  The
    identifiers look nearly identical -- same transcript, same gene, adjacent
    coordinates -- and the biology is as different as it gets.  An encoder that
    cannot separate these is not carrying variant-level meaning.

Both relations are scored against a null built from the same pool, so "far" and
"close" are positions in a distribution rather than judgements against a number
somebody picked.
"""

from __future__ import annotations

import collections
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from geoflowagent.geoacmg.clingen import CuratedVariant
from geoflowagent.geoacmg.evidence import Classification
from geoflowagent.utils.io import sha256_text

#: Classifications treated as opposite ends for a functional-sensitive pair.
#: VUS is excluded on purpose: it is the absence of a conclusion, not a pole.
PATHOGENIC_SIDE = (Classification.PATHOGENIC, Classification.LIKELY_PATHOGENIC)
BENIGN_SIDE = (Classification.BENIGN, Classification.LIKELY_BENIGN)


def _pair_id(kind: str, left: str, right: str) -> str:
    return f"{kind}:{sha256_text(f'{left}|{right}')[:16]}"


def _variant_text(record: CuratedVariant, hgvs: str) -> str:
    """The surface an encoder sees for one variant.

    Deliberately the same shape as the task's visible state, minus anything that
    would give the answer away: no classification, no evidence codes.
    """

    return (
        f"variant {hgvs} in gene {record.gene}"
        + (f" for {record.disease}" if record.disease else "")
    )


def invariant_pairs(
    records: Sequence[CuratedVariant],
    splits: Mapping[str, str],
    *,
    limit_per_record: int = 1,
) -> list[dict[str, Any]]:
    """Same variant, different published spelling.

    Only records carrying at least two HGVS expressions contribute.  The pair is
    built from the two most dissimilar spellings, so the test is not trivially
    passed by two strings that differ in a character.
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        spellings = sorted(set(record.hgvs))
        if len(spellings) < 2:
            continue
        chosen = sorted(spellings, key=len)
        selected = [(chosen[0], chosen[-1])]
        for left, right in selected[:limit_per_record]:
            rows.append(
                {
                    "pair_id": _pair_id("inv", left, right),
                    "relation": "invariant",
                    "changed_fields": ["hgvs_notation"],
                    "split": splits.get(record.uuid, "train"),
                    "source_task_id": record.uuid,
                    "left_text": _variant_text(record, left),
                    "right_text": _variant_text(record, right),
                    "provenance": {
                        "gene": record.gene,
                        "kind": "hgvs_spelling",
                        "note": "same allele, different published HGVS expression",
                    },
                }
            )
    return rows


def functional_pairs(
    records: Sequence[CuratedVariant],
    splits: Mapping[str, str],
    *,
    max_per_gene: int = 4,
) -> list[dict[str, Any]]:
    """Same gene, opposite classification.

    Pairs are formed within a gene so that everything an encoder could use as a
    shortcut -- gene name, transcript, disease -- is held constant, and only the
    variant differs.
    """

    by_gene: dict[str, list[CuratedVariant]] = collections.defaultdict(list)
    for record in records:
        by_gene[record.gene].append(record)
    rows: list[dict[str, Any]] = []
    for gene in sorted(by_gene):
        members = by_gene[gene]
        pathogenic = [r for r in members if r.assertion in PATHOGENIC_SIDE]
        benign = [r for r in members if r.assertion in BENIGN_SIDE]
        if not pathogenic or not benign:
            continue
        pathogenic.sort(key=lambda r: r.uuid)
        benign.sort(key=lambda r: r.uuid)
        for left, right in itertools.islice(zip(pathogenic, benign, strict=False), max_per_gene):
            # A pair is only usable when both sides sit in the same split.
            left_split = splits.get(left.uuid, "train")
            if left_split != splits.get(right.uuid, "train"):
                continue
            rows.append(
                {
                    "pair_id": _pair_id("fun", left.uuid, right.uuid),
                    "relation": "functional_sensitive",
                    "changed_fields": ["variant", "classification"],
                    "split": left_split,
                    "source_task_id": left.uuid,
                    "left_text": _variant_text(left, left.primary_hgvs),
                    "right_text": _variant_text(right, right.primary_hgvs),
                    "provenance": {
                        "gene": gene,
                        "kind": "opposite_classification",
                        "left_assertion": left.assertion.value,
                        "right_assertion": right.assertion.value,
                    },
                }
            )
    return rows


def control_pairs(
    records: Sequence[CuratedVariant],
    splits: Mapping[str, str],
    *,
    max_per_gene: int = 4,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Same gene, *same* classification: the null both relations are read against.

    Without this, "functional pairs are far apart" is unfalsifiable -- two
    different variants are always somewhat far apart.  The question is whether
    they are further apart than two variants that agree.
    """

    by_gene: dict[str, list[CuratedVariant]] = collections.defaultdict(list)
    for record in records:
        by_gene[record.gene].append(record)
    rows: list[dict[str, Any]] = []
    for gene in sorted(by_gene):
        by_label: dict[str, list[CuratedVariant]] = collections.defaultdict(list)
        for record in by_gene[gene]:
            by_label[record.assertion.value].append(record)
        taken = 0
        for label in sorted(by_label):
            members = sorted(by_label[label], key=lambda r: sha256_text(f"{seed}:{r.uuid}"))
            for left, right in zip(members[0::2], members[1::2], strict=False):
                if taken >= max_per_gene:
                    break
                split = splits.get(left.uuid, "train")
                if split != splits.get(right.uuid, "train"):
                    continue
                rows.append(
                    {
                        "pair_id": _pair_id("ctl", left.uuid, right.uuid),
                        "relation": "invariant",
                        "changed_fields": ["variant"],
                        "split": split,
                        "source_task_id": left.uuid,
                        "left_text": _variant_text(left, left.primary_hgvs),
                        "right_text": _variant_text(right, right.primary_hgvs),
                        "provenance": {
                            "gene": gene,
                            "kind": "same_classification_control",
                            "assertion": label,
                            "note": (
                                "not an invariance claim: this is the null distribution "
                                "for the functional pairs, carried under the schema's "
                                "invariant relation because the schema has two labels"
                            ),
                        },
                    }
                )
                taken += 1
    return rows


def build_pairs(
    records: Sequence[CuratedVariant], splits: Mapping[str, str], *, seed: int = 17
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All three families of pair, plus a report of what was available."""

    invariant = invariant_pairs(records, splits)
    functional = functional_pairs(records, splits)
    control = control_pairs(records, splits, seed=seed)
    rows = invariant + functional + control
    by_split: collections.Counter = collections.Counter(row["split"] for row in rows)
    return rows, {
        "invariant_hgvs_spelling": len(invariant),
        "functional_opposite_classification": len(functional),
        "control_same_classification": len(control),
        "total": len(rows),
        "by_split": dict(by_split),
        "genes_with_functional_pairs": len(
            {row["provenance"]["gene"] for row in functional}
        ),
        "note": (
            "functional pairs are read against the same-classification control, not "
            "against a chosen distance; an encoder passes when the two distributions "
            "separate, and the separation is reported with an interval"
        ),
    }
