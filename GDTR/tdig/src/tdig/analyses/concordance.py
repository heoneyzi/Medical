"""S3 — cross-metric concordance and dissociation token mapping."""

from __future__ import annotations


def pairwise_spearman_matrix(settling_table):
    """14x14 Spearman rho matrix across all metric x ref cells.

    Args:
        settling_table: long-form DataFrame with columns
            (token_id, metric_id, ref_id, c_value).

    Returns:
        DataFrame [14, 14] of Spearman correlations.
    """
    raise NotImplementedError("S3 implementation")


def dissociation_tokens(settling_table, cell_a, cell_b, threshold_layers=5):
    """Identify tokens where cell_a and cell_b disagree by >= threshold_layers.

    Returns:
        DataFrame with disagreeing tokens, their contexts, and the gap.
    """
    raise NotImplementedError("S3 implementation")


def settling_vector_pca(settling_table):
    """PCA on the 14-d settling-depth vector across all tokens.

    Returns:
        (projections [n_tokens, 2], explained_var [14], loadings).
    """
    raise NotImplementedError("S3 implementation")
