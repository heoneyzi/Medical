"""T-A — splice site prediction (per-position recall@k, AUC-PR).

Train on chr22 splice positions (GENCODE v44 +/- 10bp), evaluate held-out
chr17 splice positions. Features = per-cell settling depth + per-layer D
vector concatenations (PLAN.md sec 4.7).
"""

from __future__ import annotations


def build_features(settling_table, position_labels):
    """Assemble per-position feature matrix from the 14-cell settling table.

    Returns:
        X: [n_positions, n_features] feature matrix.
        y: [n_positions] binary splice/non-splice label.
        position_meta: chr, pos, strand.
    """
    raise NotImplementedError("S6 implementation")


def evaluate_recall_at_k(scores, labels, k_values=(10, 100, 1000)):
    """Per-position recall@k for selected k values."""
    raise NotImplementedError("S6 implementation")
