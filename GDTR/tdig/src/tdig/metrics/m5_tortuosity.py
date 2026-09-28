"""M5 — c_tau path tortuosity (cumulative path efficiency).

tau(ell, t) = sum_{k=ell..L*-1} ||h_{k+1} - h_k||  /  (||h_ell - h_ref|| + eps)
c_tau(t) = first ell with tau <= gamma_tau.

Properties:
  - tau >= 1 always (triangle inequality)
  - tau = 1 iff path is collinear (perfectly direct)
  - Denominator -> 0 as ell -> L*  =>  epsilon smoothing required

Reference variants A, B, C all valid — only the denominator is
reference-dependent; the cumulative path in the numerator is invariant.

Open questions (PLAN.md #10, #11, #12):
  #10 epsilon: distribution-based (1st percentile of denominator) vs constant 1e-6
  #11 monotonicity-fail handling: running-min fallback? or report as the metric's outcome?
  #12 high-correlation with M3 curvature: drop one in feature engineering? regularize?

Complementarity with M3:
  - M3 velocity / curvature: pointwise local trajectory dynamics
  - M5 tau: cumulative global path efficiency
  Different things — a low-curvature trajectory can have high tortuosity
  (slow consistent drift); a high-curvature trajectory can have low
  tortuosity (zigzag that net-cancels).
"""

from __future__ import annotations


def compute_remaining_path(hidden_states):
    """Cumulative remaining path length from ell to L*.

    path[ell, t] = sum_{k=ell..L*-1} ||h_{k+1}[t] - h_k[t]||

    Reference-invariant — same across all variants.

    Returns:
        path: [n_layers, n_tokens] (path[L*] = 0).
    """
    raise NotImplementedError("S1 implementation")


def compute_d_tortuosity(hidden_states, ref, eps):
    """tau(ell, t) = remaining_path[ell, t] / (||h_ell - ref|| + eps).

    Args:
        hidden_states: cache [n_layers, n_tokens, d_model].
        ref: reference vector [n_tokens, d_model] (variant-dependent).
        eps: numerical smoothing for the denominator.

    Returns:
        tau: [n_layers, n_tokens], values >= 1.
    """
    raise NotImplementedError("S1 implementation")


def calibrate_eps(hidden_states_sanity, ref):
    """Distribution-based epsilon: 1st percentile of the denominator.

    Returns:
        eps scalar.
    """
    raise NotImplementedError("S1 implementation")


def monotonicity_diagnostic(tau):
    """Fraction of tokens whose tau(ell) is monotone non-increasing.

    Target >= 95% per PLAN.md #11. If below, strengthen eps or switch ref.

    Returns:
        Fraction in [0, 1].
    """
    raise NotImplementedError("S1 diagnostic")
