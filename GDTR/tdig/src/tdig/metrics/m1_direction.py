"""M1 — c_dir direction settling (cosine).

D_dir(ell, t) = 1 - cos(h_ell, h_ref)
Running-min envelope, q70 calibration.

Reference variants: A, B, C all valid (see references/).
"""

from __future__ import annotations


def compute_d_dir(hidden_states, ref, variant):
    """Compute per-layer cosine distance to reference.

    Args:
        hidden_states: cache [n_layers, n_tokens, d_model] (or accessor).
        ref: reference vector [n_tokens, d_model] from references module.
        variant: 'A' | 'B' | 'C'. If 'B', also normalize h_ell symmetrically.

    Returns:
        D: [n_layers, n_tokens] with D[ell, t] = 1 - cos(h_ell[t], ref[t]).
    """
    raise NotImplementedError("S1 implementation")


def compute_settling_depth(D, gamma):
    """First layer at which running-min(D) drops to or below gamma.

    Args:
        D: per-layer distance matrix [n_layers, n_tokens].
        gamma: scalar threshold.

    Returns:
        c: [n_tokens] integer settling depths. -1 indicates never-settled.
    """
    raise NotImplementedError("S1 implementation")


def calibrate_gamma(D_sanity, layer_idx, quantile=0.70):
    """Compute gamma via q-th percentile of D at a calibration layer.

    Args:
        D_sanity: per-layer D for chr22 sanity sequences.
        layer_idx: layer at which to take the quantile (e.g. L* - 1 = 28).
        quantile: 0.70 by default (regional q70 protocol).

    Returns:
        gamma: scalar threshold.
    """
    raise NotImplementedError("S1 implementation")
