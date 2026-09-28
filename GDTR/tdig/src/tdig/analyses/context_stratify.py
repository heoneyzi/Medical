"""S4 — context-stratified analysis. The 14 x 7 = 98-cell heatmap.

For every (metric x ref) cell and every biological context class:
  distribution, mean, std, Cohen's d vs intron baseline.

Output is the densest single biological-mapping artifact in the project.
Feeds V7 visualization.
"""

from __future__ import annotations


CONTEXT_CLASSES = (
    "intergenic",
    "intron",
    "coding_exon",
    "5utr",
    "3utr",
    "splice_donor",
    "splice_acceptor",
)


def per_cell_per_context_stats(settling_table, context_labels):
    """Build the 98-cell summary table.

    Args:
        settling_table: long-form (token_id, metric_id, ref_id, c_value).
        context_labels: [n_tokens] context class strings.

    Returns:
        DataFrame indexed by (metric_id, ref_id, context) with columns
        mean, median, std, n, d_vs_intron, mannwhitney_p.
    """
    raise NotImplementedError("S4 implementation")


def heatmap_data(per_cell_per_context):
    """Pivot the long table into a 14 x 7 matrix for V7 heatmap."""
    raise NotImplementedError("S4 implementation")
