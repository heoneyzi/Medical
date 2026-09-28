"""V2 — velocity heatmap. Rows = context classes, cols = layers, value = mean v_ell."""

from __future__ import annotations


def velocity_heatmap_data(v, context_labels):
    """Aggregate v[ell, t] by context class -> [n_contexts, n_layers] mean matrix."""
    raise NotImplementedError("S5 implementation")


def plot_heatmap(matrix, row_labels, col_labels, savepath=None):
    """Render the heatmap."""
    raise NotImplementedError("S5 implementation")
