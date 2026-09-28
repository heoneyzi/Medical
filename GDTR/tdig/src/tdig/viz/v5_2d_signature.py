"""V5 — 2D signature scatter. Headline figure.

PLAN.md #8: choose default pair from
  - (c_dir^A, c_mag^A)   : direction vs magnitude  (recommended start)
  - (c_dir^A, c_geo)     : reference-dependent vs reference-free
  - (c_geo, c_tau)       : pointwise vs cumulative trajectory

Quadrant interpretation depends on pair; see docs/thesis.md for the
direction/magnitude commitment story.
"""

from __future__ import annotations


def plot_2d_signature(c_x, c_y, context_labels=None, x_label="c_dir", y_label="c_mag",
                      quadrant_split="median", savepath=None):
    """Scatter c_y vs c_x with optional quadrant overlay and color by context."""
    raise NotImplementedError("S5 implementation")
