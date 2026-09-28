"""M3 — c_geo trajectory dynamics (reference-free).

Velocity:  v_ell(t) = ||h_{ell+1} - h_ell|| / ||h_ell||
Curvature: kappa_ell(t) = 1 - cos(h_{ell+1} - h_ell, h_{ell+2} - h_{ell+1})

Standardize per-population, combine: g_ell = alpha * v_tilde + beta * kappa_tilde
Settling: first ell such that g_k <= tau for ALL k in [ell, L-2] (strict).

No reference — variant matrix does not apply.
Defaults: (alpha, beta) = (1, 1). Tune after S1 calibration.

Open questions (PLAN.md #3, #4, #6):
  #3 (alpha, beta) tuning grid
  #4 "all k" range — strict [ell, L-2] vs rolling [ell, ell+5]
  #6 CKA control threshold

Provenance:
  Velocity-curvature decomposition follows Neural ODE (Chen et al. 2018),
  Geshkovski et al. 2023, Transformer Dynamics (arXiv:2502.12131), and the
  curvature-naming convention of arXiv:2510.06640.
"""

from __future__ import annotations


def compute_velocity(hidden_states):
    """v_ell = ||h_{ell+1} - h_ell|| / ||h_ell|| for ell = 0..L-2.

    Returns:
        v: [n_layers - 1, n_tokens].
    """
    raise NotImplementedError("S1 implementation")


def compute_curvature(hidden_states):
    """kappa_ell = 1 - cos(h_{ell+1} - h_ell, h_{ell+2} - h_{ell+1}).

    Returns:
        kappa: [n_layers - 2, n_tokens].
    """
    raise NotImplementedError("S1 implementation")


def standardize_per_layer(x):
    """Z-score per layer over the population."""
    raise NotImplementedError("S1 implementation")


def compute_g(v_tilde, kappa_tilde, alpha=1.0, beta=1.0):
    """Combined gauge g_ell = alpha * v_tilde + beta * kappa_tilde.

    Aligns v and kappa to the same layer index (v covers 0..L-2,
    kappa covers 0..L-3; truncate to common range).
    """
    raise NotImplementedError("S1 implementation")


def settling_depth_strict(g, tau, k_range="all"):
    """First ell such that g[k, t] <= tau for all k in [ell, end].

    Args:
        g: [n_layers - 2, n_tokens] combined gauge.
        tau: scalar threshold.
        k_range: 'all' for [ell, L-2], or int N for rolling [ell, ell+N].

    Returns:
        c_geo: [n_tokens] integer settling depths.
    """
    raise NotImplementedError("S1 implementation")


def calibrate_tau(g_sanity, layer_idx, quantile=0.70):
    """q70 of g at calibration layer over chr22 sanity."""
    raise NotImplementedError("S1 implementation")
