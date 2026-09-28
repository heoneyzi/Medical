"""V1 — single-token trajectory in 2D PCA. Headline figure.

PLAN.md #7: PCA mode is open.
  - raw: population PCA across all (layer, token) hidden states; PC1
         likely dominated by layer-systematic magnitude shift
  - per-layer-centered: subtract per-layer mean before PCA; isolates
                        per-token variation
  - tuned-lens: project via Phase-1-follow-up tuned-lens to common frame,
                then PCA (most informative but requires the affine fit)

Recommended: compute both 'raw' and 'per-layer-centered'; present both.
"""

from __future__ import annotations


def population_pca(hidden_states, mode="per_layer_centered", n_components=2):
    """PCA over (layer, token) -> 2D projection.

    Args:
        hidden_states: [n_layers, n_tokens, d_model].
        mode: 'raw' | 'per_layer_centered' | 'tuned_lens'.

    Returns:
        projection: [n_layers, n_tokens, 2].
        pca_model: fitted sklearn PCA.
    """
    raise NotImplementedError("S5 implementation")


def plot_trajectories(projection, token_ids, context_labels=None, savepath=None):
    """Plot N selected tokens' layer-trajectories in 2D."""
    raise NotImplementedError("S5 implementation")
