"""M6 — D_L2 residual (consistency check, function of M1 + M2).

D_L2(ell, t) = ||h_ell - h_ref|| / ||h_ref||

Algebraic decomposition:
  D_L2^2 = r^2 + 1 - 2*r*c = (r - c)^2 + (1 - c^2)
  where r = ||h_ell|| / ||h_ref||, c = cos(h_ell, h_ref).

(r - c)^2  — magnitude-mismatch term (captured by M2 |r - 1|)
(1 - c^2)  — angular term (captured by M1 1 - cos)

D_L2 -> 0 requires both r -> 1 AND c -> 1.

Role: appendix-level consistency check. The joint (c_dir, c_mag) analysis
should explain D_L2 behavior; if it doesn't, family decomposition is wrong.

Bonus: rotation-invariant (cosine isn't, under per-dimension gamma weighting).
Useful audit when M2's antiparallel edge case occurs.
"""

from __future__ import annotations


def compute_d_l2(hidden_states, ref):
    """D_L2(ell, t) = ||h_ell - ref|| / ||ref||.

    Returns:
        D: [n_layers, n_tokens].
    """
    raise NotImplementedError("S1 implementation")


def decompose_to_m1_m2(hidden_states, ref):
    """Return (r - c)^2 and (1 - c^2) terms separately for audit.

    Should reproduce D_L2^2 = (r - c)^2 + (1 - c^2) exactly.
    """
    raise NotImplementedError("S1 implementation")
