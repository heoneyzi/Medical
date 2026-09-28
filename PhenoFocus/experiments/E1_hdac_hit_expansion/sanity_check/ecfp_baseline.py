#!/usr/bin/env python3
"""Portfolio sanity check for E1 (added during portfolio curation, Sept 2026).

This script is NOT part of the original team run. It only re-uses the files
committed next to it, so anyone can re-check the E1 numbers without the
PhenoCompass checkpoints:

1. Re-computes the ECFP4 Tanimoto values of the reported top-5 candidates.
2. Re-derives AUROC / top-5 fold-enrichment / hypergeometric p from the reported
   ranks of the two HDAC compounds (ranks 1 and 4 of 31).
3. Runs a fingerprint-only baseline: rank the same 31 compounds by plain ECFP4
   Tanimoto to the hit (classic "find look-alikes" similarity search) and scores
   it with the same metrics.

Fingerprint settings match ``src/a549_mvp/mvp_core.py``: Morgan radius 2, 2048 bits.

Usage (from this folder):
    pip install rdkit scikit-learn scipy
    python ecfp_baseline.py            # writes ecfp_baseline.json
"""

from __future__ import annotations

import csv
import json
import os

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(HERE, "..", "inputs", "a549_compounds.csv")
CANDIDATES = os.path.join(HERE, "..", "results", "hit_JHSXDAWGLCZYSM_UHFFFAOYSA_N_candidates.csv")
TOP_K = 5

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smiles: str):
    return _gen.GetFingerprint(Chem.MolFromSmiles(smiles))


def metrics(labels, scores, top_k=TOP_K):
    """AUROC, top-k same-MoA count, fold-enrichment and hypergeometric p."""
    n, n_pos = len(labels), sum(labels)
    order = sorted(range(n), key=lambda i: -scores[i])
    k_pos = sum(labels[i] for i in order[:top_k])
    expected = top_k * n_pos / n
    return {
        "auroc": roc_auc_score(labels, scores),
        "top_k": top_k,
        "same_moa_in_top_k": k_pos,
        "expected_by_chance": expected,
        "fold_enrichment": k_pos / expected,
        "hypergeom_p": float(hypergeom.sf(k_pos - 1, n, n_pos, top_k)),
        "ranks_of_same_moa": [order.index(i) + 1 for i in range(n) if labels[i]],
    }


def main() -> None:
    rows = list(csv.DictReader(open(LIBRARY)))
    hit, library = rows[0], rows[1:]  # row 0 is the query hit (auto:hdac picks it)
    hit_fp = fp(hit["smiles"])
    labels = [int(r["moa_cluster"] == hit["moa_cluster"]) for r in library]

    # 1) Tanimoto re-check of the reported candidates
    recheck = []
    for c in csv.DictReader(open(CANDIDATES)):
        t = DataStructs.TanimotoSimilarity(hit_fp, fp(c["smiles"]))
        recheck.append({"rank": int(c["rank"]), "moa": c["moa_cluster"],
                        "reported": round(float(c["tanimoto"]), 4), "recomputed": round(t, 4)})

    # 2) Model metrics re-derived from the reported ranks (HDAC at #1 and #4 of 31)
    n = len(library)
    model_rank_scores = [n - r for r in range(1, n + 1)]  # higher = better rank
    model_labels = [1 if r in (1, 4) else 0 for r in range(1, n + 1)]
    model = metrics(model_labels, model_rank_scores)

    # 3) Fingerprint-only baseline on the same library
    tanimoto = [DataStructs.TanimotoSimilarity(hit_fp, fp(r["smiles"])) for r in library]
    baseline = metrics(labels, tanimoto)

    out = {
        "note": "Portfolio sanity check added during curation; not part of the original team run.",
        "library_size": n,
        "same_moa_in_library": sum(labels),
        "fingerprint": "Morgan/ECFP4, radius 2, 2048 bits",
        "tanimoto_recheck_top5": recheck,
        "phenocompass_structure_ranking_from_reported_ranks": model,
        "ecfp4_nearest_neighbour_baseline": baseline,
    }
    with open(os.path.join(HERE, "ecfp_baseline.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
