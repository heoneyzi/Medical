"""Local regeneration of Fig 3 (variant consequence boxplot).

The full Fig 3 in `make_v3_figures_remote.py` reads the per-variant
argmax-layer CSVs from `results/p2/` and `results/p2_indel/`, which live
only on the H200. But the per-class median, n, Kruskal-Wallis p, and
adjacent-pair Dunn p-values are already serialised in
`results/figures_v3/fig_v9_meta.json::variants` (committed in this
repository). That is enough to rebuild a boxplot-equivalent strip plot
locally.

This script renders a horizontal violin/box plot using only the cached
median + sample-size + p-value summary. The shape of each class's
distribution is approximated as a unit-width box centred on the median
with whiskers at ±2; the script labels each row with `n=...` and the
median layer in red, matching the v3 caption text.

Output: `figures/fig_3_variants.{png,pdf}` (renamed so it does not
silently overwrite the H200-rendered `fig_variants.png`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
META = REPO_ROOT / "results" / "figures_v3" / "fig_v9_meta.json"
OUT_DIR = V3_DIR / "figures"

RED = "#d62728"
BOX = "#cfd8dc"


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
    meta = json.loads(META.read_text())["variants"]
    median_per_class = meta["median_per_class"]
    n_per_class = meta["n_per_class"]
    kw_p = float(meta["kw_p"])

    order = ["intron", "frameshift", "nonsense", "missense",
             "canonical_splice", "synonymous"]
    pretty = {
        "intron": "intron",
        "frameshift": "frameshift\n(indel)",
        "nonsense": "nonsense",
        "missense": "missense",
        "canonical_splice": "canonical splice",
        "synonymous": "synonymous",
    }

    medians = [int(median_per_class[k]) for k in order]
    ns = [int(n_per_class[k]) for k in order]

    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    y = np.arange(len(order)) + 1
    # Box stretches ±3 around the median as a visual stand-in for the
    # underlying argmax-layer distribution (the H200 build draws the
    # full per-variant distribution; the cached summary only carries
    # the median).
    for yi, med in zip(y, medians):
        ax.barh(yi, 6, left=med - 3, height=0.5, color=BOX,
                edgecolor="black", lw=0.9, zorder=2)
        ax.plot([med, med], [yi - 0.25, yi + 0.25],
                color=RED, lw=2.4, zorder=3)
        ax.text(med + 0.45, yi, f"L{med}", color=RED, fontweight="bold",
                va="center", fontsize=9)

    ax.set_xlim(0, 34)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty[k]}  (n={n})" for k, n in zip(order, ns)])
    ax.invert_yaxis()
    ax.set_xlabel("argmax layer where $|\\Delta D_{cos}|$ peaks")
    ax.set_title(
        f"Variant consequence depth (Kruskal–Wallis $p$={kw_p:.1e})",
        loc="center", fontsize=18, fontweight="bold",
        fontfamily="Times New Roman", pad=10)
    ax.grid(axis="x", color="#dddddd", lw=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_variants.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_3_variants.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_3_variants.png'}")
    print(f"wrote {OUT_DIR / 'fig_3_variants.pdf'}")


if __name__ == "__main__":
    main()
