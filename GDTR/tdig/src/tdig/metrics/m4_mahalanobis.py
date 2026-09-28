"""M4 — c_M Mahalanobis residual (distribution-aware).

D_M(ell, t) = sqrt( (h_ell - h_ref)^T Sigma_ell^{-1} (h_ell - h_ref) )

Sigma_ell estimated empirically on chr22 sanity at each layer.
Three estimation options (S1 ablation):
  - Diagonal (fast, stable, ignores correlations)         <-- recommended start
  - Ledoit-Wolf shrinkage (full Sigma + shrinkage)
  - PCA-top-k (top-k eigen-directions, k in {64, 128, 256})

Reference variants A, B, C all valid; Sigma_ell estimated on the variant's
representation space.

Diagnostic obligation: report condition number kappa(Sigma_ell) per layer.
If kappa > 1e6, switch to diagonal or strengthen shrinkage.
"""

from __future__ import annotations


def estimate_sigma(hidden_states_sanity, layer, method="diagonal", **kwargs):
    """Estimate per-layer covariance Sigma_ell.

    Args:
        hidden_states_sanity: [n_layers, n_sanity_tokens, d_model].
        layer: layer index.
        method: 'diagonal' | 'ledoit_wolf' | 'pca_topk'.
        **kwargs: method-specific (e.g. k=128 for pca_topk).

    Returns:
        Sigma_ell or its inverse (depending on method) for D_M computation.
        Also returns condition number for diagnostic.
    """
    raise NotImplementedError("S1 implementation")


def compute_d_mahalanobis(hidden_states, ref, sigma_inv_per_layer):
    """Per-layer Mahalanobis distance to reference.

    Args:
        hidden_states: [n_layers, n_tokens, d_model].
        ref: [n_tokens, d_model].
        sigma_inv_per_layer: sequence of per-layer Sigma^{-1} (or diagonals).

    Returns:
        D_M: [n_layers, n_tokens].
    """
    raise NotImplementedError("S1 implementation")


def condition_number_report(sigma_per_layer):
    """Diagnostic: per-layer kappa(Sigma_ell). Returns [n_layers] array."""
    raise NotImplementedError("S1 diagnostic")
