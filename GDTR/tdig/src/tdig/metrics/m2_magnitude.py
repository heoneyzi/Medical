"""M2 — c_mag magnitude settling.

D_mag(ell, t) = |r - 1| with r = ||h_ell|| / ||h_ref||
Running-min envelope, q70 calibration.

Reference variants: A and C valid; B is mathematically degenerate
(RMSNorm flattens magnitudes — reported only for diagnostic completeness).

Open question (PLAN.md #1): |r - 1| vs |log r|. Decide after S1 distribution
inspection on chr22 sanity, given Evo 2's huge hidden-state norms
(h_30.std ~ 2e10).

Open question (PLAN.md #2): Variant B handling. Options:
  (a) compute and report as trivial baseline
  (b) define a gamma-only variant (gamma multiplication without
      the RMS normalization step) to give B a meaningful magnitude axis
"""

from __future__ import annotations


def compute_d_mag(hidden_states, ref, variant, mode="abs"):
    """Per-layer magnitude-ratio deviation from 1.

    Args:
        hidden_states: cache [n_layers, n_tokens, d_model].
        ref: reference vector [n_tokens, d_model].
        variant: 'A' | 'B' | 'C'. Variant B is degenerate.
        mode: 'abs' for |r - 1|, 'log' for |log r|. PLAN.md #1.

    Returns:
        D: [n_layers, n_tokens].
    """
    raise NotImplementedError("S1 implementation")


def antiparallel_diagnostic(hidden_states, ref):
    """Fraction of (layer, token) where cos(h_ell, ref) < -0.9.

    Reports how often the antiparallel edge case occurs in chr22 sanity.
    If this fraction is non-negligible, M2 (|r - 1|) without a direction check
    can mis-call settling — use M6 D_L2 as audit.

    Returns:
        Per-layer fraction array [n_layers] of antiparallel tokens.
    """
    raise NotImplementedError("S1 diagnostic")
