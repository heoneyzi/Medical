"""Figure generation for the A549 cross-line MVP. Matplotlib only."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2c6fbb"
GREY = "#b7b7b7"
RED = "#c1432c"


def _watermark(fig, text: str = "SYNTHETIC / MOCK — not real model output") -> None:
    fig.text(0.5, 0.5, text, fontsize=22, color="red", alpha=0.18,
             ha="center", va="center", rotation=30, weight="bold")


def _save(fig, out_base: str, mock: bool) -> None:
    if mock:
        _watermark(fig)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auroc_bar(
    per_moa: pd.DataFrame, value_col: str, out_base: str, title: str,
    macro: Optional[float] = None, mock: bool = False,
) -> None:
    d = per_moa.dropna(subset=[value_col])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(d))
    ax.bar(x, d[value_col], color=BLUE, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, ls="--", c=GREY, lw=1, label="chance (0.5)")
    if macro is not None:
        ax.axhline(macro, ls="-", c=RED, lw=1.2, label=f"macro = {macro:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(d["moa"], rotation=30, ha="right")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out_base, mock)


def plot_score_heatmap(
    score_df: pd.DataFrame, moa_list: Sequence[str], out_base: str,
    title: str, mock: bool = False,
) -> None:
    """Mean anchor-similarity by true MoA group (rows) x anchor MoA (cols)."""
    labelled = score_df[score_df["true_moa"].notna()]
    mat = labelled.groupby("true_moa")[list(moa_list)].mean().reindex(list(moa_list))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(moa_list)))
    ax.set_xticklabels(moa_list, rotation=40, ha="right")
    ax.set_yticks(range(len(moa_list)))
    ax.set_yticklabels(moa_list)
    ax.set_xlabel("anchor MoA (scored against)")
    ax.set_ylabel("true MoA (A549 compound)")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < np.nanmean(mat.values) else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean cosine")
    fig.tight_layout()
    _save(fig, out_base, mock)


def plot_score_vs_tanimoto(
    max_tanimoto, anchor_score, same_moa, out_base: str, title: str,
    mock: bool = False,
) -> None:
    """Anchor score (y) vs max structural similarity to the anchors (x).

    The money quadrant is top-LEFT: high MoA/anchor score at LOW Tanimoto =
    a phenotypically related but structurally independent chemotype.
    """
    import numpy as np
    x = np.asarray(max_tanimoto, dtype=float)
    y = np.asarray(anchor_score, dtype=float)
    same = np.asarray(same_moa, dtype=bool)
    ok = ~np.isnan(x) & ~np.isnan(y)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(x[ok & ~same], y[ok & ~same], s=10, c=GREY, alpha=0.4, label="other MoA")
    ax.scatter(x[ok & same], y[ok & same], s=20, c=BLUE, edgecolor="black",
               linewidth=0.3, label="same MoA as anchors")
    ax.axvline(0.4, ls="--", c=RED, lw=1, label="Tanimoto 0.4 (novelty)")
    ax.set_xlabel("max ECFP Tanimoto to customer anchors")
    ax.set_ylabel("anchor MoA score")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, out_base, mock)


def plot_embedding_scatter(
    latent: np.ndarray, true_moa: Sequence, moa_list: Sequence[str],
    out_base: str, title: str, mock: bool = False,
) -> None:
    """2-D PCA of A549 morphology embeddings, coloured by mapped MoA."""
    from sklearn.decomposition import PCA

    X = np.asarray(latent, dtype=np.float64)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    labels = np.array([t if t is not None else "unmapped" for t in true_moa])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(xy[labels == "unmapped", 0], xy[labels == "unmapped", 1],
               s=6, c=GREY, alpha=0.35, label="unmapped")
    cmap = plt.get_cmap("tab10")
    for i, moa in enumerate(moa_list):
        m = labels == moa
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1], s=18, color=cmap(i % 10),
                       edgecolor="black", linewidth=0.3, label=f"{moa} (n={m.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=1.2, loc="best")
    fig.tight_layout()
    _save(fig, out_base, mock)
