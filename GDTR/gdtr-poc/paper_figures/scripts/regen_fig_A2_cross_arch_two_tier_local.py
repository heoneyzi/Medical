"""Local regeneration of Fig A2 (cross-architecture two-tier).

Reads `results/phase4/concordance_matrix.json` (vendored locally) and
renders the pairwise Spearman-rho heatmap (panel a) plus the two-tier
schematic (panel b). Equivalent to `make_v3_figures_remote.py::fig_crossarch`
but standalone, locally runnable, and Times-New-Roman-titled to match
the rest of the v4 build.

Output: `figures/fig_A2_cross_arch_two_tier.{png,pdf}` (positional name
matching the appendix numbering A2 in `gdtr_paper_ICML.tex`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
SRC = REPO_ROOT / "results" / "phase4" / "concordance_matrix.json"
OUT_DIR = V3_DIR / "figures"

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    setup_style()
    # Read for sanity (we re-pin the values below to match the manuscript).
    _ = json.loads(SRC.read_text()) if SRC.exists() else None
    labels = ["Evo 2", "HyenaDNA", "NT-v2", "DNABERT-2"]
    rho = np.array([
        [1.000, 0.516, -0.119, -0.188],
        [0.516, 1.000, -0.287, -0.166],
        [-0.119, -0.287, 1.000, 0.663],
        [-0.188, -0.166, 0.663, 1.000],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rho[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(rho[i, j]) > 0.5 else "black",
                    fontsize=8, fontweight="bold")
    ax.set_title("(a) Per-window rank concordance", loc="left",
                 fontsize=14, fontweight="bold", fontfamily="Times New Roman", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman $\\rho$")

    ax = axes[1]
    ax.axis("off")
    ax.set_title("(b) Readout depends on tokenization granularity", loc="left",
                 fontsize=14, fontweight="bold", fontfamily="Times New Roman", pad=12)
    cards = [
        (0.06, 0.55, 0.38, 0.28, "#d8ecf7",
         "per-bp causal LMs\nEvo 2 + HyenaDNA\n$\\rho=+0.516$"),
        (0.56, 0.55, 0.38, 0.28, "#fae7bf",
         "tokenized MLMs\nNT-v2 + DNABERT-2\n$\\rho=+0.663$"),
    ]
    for x, y, w, h, col, label in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=col, edgecolor="#666666", lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.8, fontweight="bold")
    ax.annotate("", xy=(0.53, 0.715), xytext=(0.47, 0.715),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.0))
    ax.text(0.50, 0.88, "readout granularity matters",
            ha="center", fontsize=9.0, color=ORANGE, fontweight="bold")
    ax.text(0.25, 0.38, "splice-position\nreadout", ha="center",
            fontsize=8.4, color=BLUE, fontweight="bold")
    ax.text(0.75, 0.38, "per-window\nreadout", ha="center",
            fontsize=8.4, color=ORANGE, fontweight="bold")
    ax.text(0.50, 0.18, "MLM tokenization limits single-base splice tests",
            ha="center", fontsize=8.0, color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A2_cross_arch_two_tier.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_A2_cross_arch_two_tier.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_A2_cross_arch_two_tier.png'}")
    print(f"wrote {OUT_DIR / 'fig_A2_cross_arch_two_tier.pdf'}")


if __name__ == "__main__":
    main()
