"""V8 — dissociation map. Optional figure.

x = chromosome position (chr22).
y = metric-pair disagreement magnitude (e.g. |c_dir - c_geo|).
color or annotation = context class.

Reveals whether metric disagreements cluster at biologically meaningful loci
(splice junctions, repeat regions, etc.).
"""

from __future__ import annotations


def plot_dissociation_track(positions, disagreement, context_labels=None, savepath=None):
    raise NotImplementedError("S5 implementation")
