"""V4 — cumulative path length. Headline figure.

x = layer, y = sum_{k <= ell} ||h_{k+1} - h_k|| / ||h_0||.
Plateau onset gives c_geo intuition.
"""

from __future__ import annotations


def cumulative_path(hidden_states):
    """Per-token cumulative path normalized by ||h_0||.

    Returns:
        cum: [n_layers, n_tokens].
    """
    raise NotImplementedError("S5 implementation")


def plot_paths_by_context(cum, context_labels, savepath=None):
    """Plot mean + std cumulative path per context class."""
    raise NotImplementedError("S5 implementation")
