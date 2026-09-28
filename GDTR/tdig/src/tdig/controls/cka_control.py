"""CKA cross-validation for M3 c_geo.

Velocity and curvature are cosine-based; residual skip connections bias
adjacent-layer cosine similarity upward. CKA (Kornblith et al. 2019) is
rotation-invariant and compares relational structure rather than vectors,
so it is robust to skip-connection bias.

If c_geo declares 'settled at ell' but CKA(h_ell, h_{ell+k}) is low for
small k, skip-bias contamination is likely.
"""

from __future__ import annotations


def cka(x, y):
    """Centered Kernel Alignment between two [n_samples, d] matrices."""
    raise NotImplementedError("S1 implementation")


def cross_validate_c_geo(c_geo, hidden_states, k_max=5):
    """For each token's c_geo, verify CKA(h_{c_geo}, h_{c_geo + k}) for k = 1..k_max.

    Returns:
        Fraction of tokens passing the CKA consistency check at the chosen threshold.
    """
    raise NotImplementedError("S1 implementation")
