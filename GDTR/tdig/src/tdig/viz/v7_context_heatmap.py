"""V7 — 14 x 7 = 98-cell context heatmap. Headline figure.

Densest single biological-mapping view in the project. Feeds from S4
context_stratify.heatmap_data().

Rows: 14 (metric x ref) cells.
Cols: 7 context classes.
Cell value: Cohen's d vs intron baseline.
"""

from __future__ import annotations


def plot_98cell_heatmap(matrix, row_labels, col_labels,
                        cmap="RdBu_r", center=0.0, savepath=None):
    """Render the 14 x 7 d-vs-intron heatmap."""
    raise NotImplementedError("S5 implementation")
