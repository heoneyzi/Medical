"""T-B — ClinVar variant classification (10-fold stratified AUROC, DeLong).

Reuses gDTR-PoC results/phase3_main/variants_features.csv (10,910 ClinVar
variants in 15 cancer genes). Adds the new TDiG feature columns from the
14-cell settling table. Reports AUROC and DeLong p vs Paper 2 baseline 0.926.
"""

from __future__ import annotations


def build_variant_features(settling_table_ref, settling_table_alt, variant_table):
    """Per-variant features: settling depths + delta(ref->alt) for each cell.

    Returns:
        X: [n_variants, n_features].
        y: [n_variants] in {0=B/LB, 1=P/LP}.
        variant_meta: gene, chr, pos, consequence.
    """
    raise NotImplementedError("S6 implementation")


def stratified_cv_auroc(X, y, groups=None, n_splits=10):
    """10-fold stratified CV; if groups provided, also report LOGO-CV (gene)."""
    raise NotImplementedError("S6 implementation")


def delong_vs_baseline(scores_ours, scores_baseline, labels):
    """DeLong paired AUROC test against Paper 2 ||Delta h||_2 baseline (0.926)."""
    raise NotImplementedError("S6 implementation")
