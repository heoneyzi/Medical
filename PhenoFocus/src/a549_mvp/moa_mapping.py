"""Map LINCS A549 (cpg0004) ``Metadata_moa`` strings onto the six PhenoCompass
anchor MoA clusters.

PhenoCompass ships anchor embeddings for six mechanism-of-action clusters
(see ``phenocompass.inference.scorer._CLUSTER_IDX_TO_MOA``)::

    mtor_pi3k, hsp90, jak_rock, hdac, mapk, cdk

The LINCS Cell Painting dataset annotates each compound with a free-text
``Metadata_moa`` field sourced from the Drug Repurposing Hub (e.g.
``"PI3K inhibitor"``, ``"HDAC inhibitor"``, ``"MEK inhibitor"``).  This module
provides an *editable* lookup that maps those free-text strings onto the six
anchor clusters so that A549 compounds can be labelled in PhenoCompass' own MoA
taxonomy.

The mapping is deliberately simple and transparent: matching is done by
lower-cased substring so that variants like ``"PI3K inhibitor|mTOR inhibitor"``
still resolve.  Edit ``MOA_SYNONYMS`` if you want to widen/narrow a cluster.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# The six PhenoCompass anchor clusters, in a stable order.
ANCHOR_MOAS: List[str] = ["mtor_pi3k", "hsp90", "jak_rock", "hdac", "mapk", "cdk"]

# Lower-cased substrings that, if found in a LINCS ``Metadata_moa`` string,
# assign the compound to the given anchor cluster.  First match wins, in the
# order clusters are listed here.  These are intentionally conservative; add
# synonyms as needed for your own annotation set.
MOA_SYNONYMS: Dict[str, List[str]] = {
    "mtor_pi3k": [
        "pi3k",
        "pi3-kinase",
        "pi 3-kinase",
        "pi3 kinase",
        "phosphoinositide 3",
        "phosphoinositide-3",
        "phosphatidylinositol 3",
        "phosphatidylinositol-4,5",
        "mtor",
        "mechanistic target of rapamycin",
        "target of rapamycin",
        "pikk",
        "akt",  # PI3K/AKT/mTOR axis; drop if you want a stricter cluster
    ],
    "hsp90": [
        "hsp90",
        "hsp 90",
        "hsp inhibitor",
        "heat shock protein 90",
        "heat shock protein",
    ],
    "jak_rock": [
        "jak inhibitor",
        "jak ",
        "jak1", "jak2", "jak3", "tyk2",
        "janus kinase",
        "rock inhibitor",
        "rho associated",
        "rho-associated",
        "rho kinase",
        "rho-kinase",
    ],
    "hdac": [
        "hdac",
        "histone deacetylase",
    ],
    "mapk": [
        "mek inhibitor",
        "mek1",
        "mek2",
        "mapkk",
        "erk inhibitor",
        "erk1", "erk2",
        "mapk",
        "map kinase",
        "mitogen-activated protein kinase",
        "mitogen activated protein kinase",
        "raf inhibitor",
        "b-raf",
        "braf",
    ],
    "cdk": [
        "cdk inhibitor",
        "cyclin-dependent kinase",
        "cyclin dependent kinase",
        "cdk1", "cdk2", "cdk4", "cdk6", "cdk7", "cdk9",
    ],
}


def map_moa_string(moa: Optional[str]) -> Optional[str]:
    """Return the anchor cluster for a single LINCS ``Metadata_moa`` string.

    Parameters
    ----------
    moa : str or None
        Free-text MoA annotation (may be ``"unknown"``, ``None``, or a
        pipe-delimited multi-MoA string).

    Returns
    -------
    str or None
        One of :data:`ANCHOR_MOAS`, or ``None`` if no cluster matches.
    """
    if moa is None:
        return None
    text = str(moa).strip().lower()
    if not text or text == "unknown" or text == "nan":
        return None
    for cluster in ANCHOR_MOAS:
        for token in MOA_SYNONYMS[cluster]:
            if token in text:
                return cluster
    return None


def annotate_clusters(moa_series) -> "list":
    """Vectorised :func:`map_moa_string` over an iterable/Series of MoA strings."""
    return [map_moa_string(m) for m in moa_series]


def summarize_mapping(moa_series) -> Dict[str, int]:
    """Count how many rows map to each anchor cluster (plus ``None``)."""
    counts: Dict[str, int] = {c: 0 for c in ANCHOR_MOAS}
    counts["(unmapped)"] = 0
    for m in moa_series:
        c = map_moa_string(m)
        counts[c if c is not None else "(unmapped)"] += 1
    return counts
