"""S1 — HP sensitivity (supplementary).

Heatmap of mean AUROC across 9-cell penalty × C grid (l1_ratio=0.5 used for
elasticnet). Annotates each cell, plus mean ± std band overlay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402

ROOT = Path("/path/to/ICML")
HP_GRID = ROOT / "results/tier2_sensitivity/hp_grid.csv"
OUT = ROOT / "results/figures_v2/S1_hp_sensitivity"


def main() -> None:
    fs.setup()
    df = pd.read_csv(HP_GRID)
    penalties = ["l2", "l1", "elasticnet"]
    Cs = [0.1, 1.0, 10.0]
    A = np.full((len(penalties), len(Cs)), np.nan)
    S = np.full_like(A, np.nan)
    for i, p in enumerate(penalties):
        for j, c in enumerate(Cs):
            row = df[(df["penalty"] == p) & (df["C"] == c)]
            if len(row):
                A[i, j] = row["auroc_mean"].values[0]
                S[i, j] = row["auroc_std"].values[0]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    vmin, vmax = float(np.nanmin(A)), float(np.nanmax(A))
    pad = max((vmax - vmin) * 0.5, 0.001)
    im = ax.imshow(A, cmap="Blues", vmin=vmin - pad, vmax=vmax + pad, aspect="auto")
    ax.set_xticks(range(len(Cs)))
    ax.set_xticklabels([f"C={c}" for c in Cs])
    ax.set_yticks(range(len(penalties)))
    ax.set_yticklabels(penalties)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            txt = f"{A[i, j]:.4f}\n±{S[i, j]:.4f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="#222")
    ax.set_xlabel("Inverse-regularization C")
    ax.set_ylabel("Penalty")
    ax.set_title(f"HP sensitivity (mean ± std AUROC across 10 folds)\n"
                 f"range = [{vmin:.4f}, {vmax:.4f}]  (Δ = {vmax-vmin:.4f})",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC mean")
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
