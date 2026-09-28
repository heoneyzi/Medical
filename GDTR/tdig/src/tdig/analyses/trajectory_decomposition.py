"""Local vs cumulative trajectory comparison (M3 c_geo vs M5 c_tau).

A trajectory can have:
- low M3 curvature + high M5 tortuosity  (slow consistent drift)
- high M3 curvature + low M5 tortuosity  (zigzag that net-cancels)

This module identifies tokens in each regime and reports their biological-context
enrichment. Used to defend the inclusion of both metrics rather than just one.
"""

from __future__ import annotations


def regime_classification(c_geo, c_tau, hidden_states):
    """Classify each token into one of four regimes by (c_geo, c_tau) split.

    Returns:
        labels: [n_tokens] in {'low-low', 'low-high', 'high-low', 'high-high'}.
    """
    raise NotImplementedError("S3 implementation")


def context_enrichment(regime_labels, context_labels):
    """Hypergeometric enrichment of context classes per regime."""
    raise NotImplementedError("S3 implementation")
