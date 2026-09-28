"""Entropy decoupling gate (S2.2) — E8 protocol carryover from gDTR-PoC.

Tests whether a settling cell is just per-position next-token entropy in
disguise. Two conditions:
  1. Spearman rho(c, H_per_pos) magnitude <= 0.20
  2. Entropy-residualized Cohen's d (splice donor vs intron) not weaker than raw

Reference: gDTR-PoC scripts/exp1_entropy_correlation.py and Paper 1 sec 3.1.
"""

from __future__ import annotations


def entropy_correlation(c_per_token, h_per_token):
    """Spearman rho between settling depth and per-position entropy.

    Args:
        c_per_token: [n_tokens].
        h_per_token: [n_tokens] per-position Shannon entropy of next-token logits.

    Returns:
        rho (scalar) and p-value.
    """
    raise NotImplementedError("S2 implementation")


def residualized_d(c_per_token, h_per_token, context_labels):
    """Partial out per-position entropy from c, then recompute splice-vs-intron Cohen's d.

    Args:
        c_per_token: [n_tokens].
        h_per_token: [n_tokens].
        context_labels: [n_tokens].

    Returns:
        dict with 'd_raw', 'd_residualized', 'pass' (True if |d_resid| >= |d_raw|).
    """
    raise NotImplementedError("S2 implementation")
