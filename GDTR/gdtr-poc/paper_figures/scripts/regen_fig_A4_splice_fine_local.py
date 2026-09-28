"""Local regeneration of Fig A4 (splice positional fine-profile).

Reads `results/phase1.6_sub/splice_distance_profile.json` (vendored
locally) and renders the donor / acceptor mean-settling-depth profile
as a function of distance from the splice junction.

Output: `figures/fig_A4_splice_fine.{png,pdf}` (positional name
matching the appendix numbering A4 in `gdtr_paper_ICML.tex`).

Title font: Times New Roman to match the rest of the v4 figures.
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
SRC = REPO_ROOT / "results" / "phase1.6_sub" / "splice_distance_profile.json"
OUT_DIR = V3_DIR / "figures"

BLUE = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    setup_style()
    data = json.loads(SRC.read_text())
    bins = list(data["distance_bins"])
    intron_bg = float(data["intron_mean_c_background"])

    donor_means = [float(data["donor"][str(b)]["mean_c"]) for b in bins]
    acceptor_means = [float(data["acceptor"][str(b)]["mean_c"]) for b in bins]
    donor_n = int(data["donor"][str(bins[0])]["n"])
    acceptor_n = int(data["acceptor"][str(bins[0])]["n"])

    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    ax.plot(bins, donor_means, marker="o", color=BLUE, lw=1.6,
            label=f"splice donor (n={donor_n:,})")
    ax.plot(bins, acceptor_means, marker="s", color=GREEN, lw=1.6,
            label=f"splice acceptor (n={acceptor_n:,})")
    ax.axhline(intron_bg, color=RED, ls="--", lw=1.0, alpha=0.8,
               label=f"intron baseline ($\\bar c={intron_bg:.2f}$)")
    ax.axvline(0, color="#888888", lw=0.6)

    # Annotate the per-side argmin (deepest dip) for donor and acceptor.
    # Place donor annotation ABOVE its dip and acceptor annotation BELOW its
    # dip so the two label boxes never collide even though the two minima
    # sit at almost identical (x, y) coordinates.
    donor_min_idx = int(np.argmin(donor_means))
    acceptor_min_idx = int(np.argmin(acceptor_means))
    ax.scatter([bins[donor_min_idx]], [donor_means[donor_min_idx]],
               s=70, color=BLUE, edgecolor="black", zorder=4)
    ax.scatter([bins[acceptor_min_idx]], [acceptor_means[acceptor_min_idx]],
               s=70, color=GREEN, edgecolor="black", zorder=4)
    ax.annotate(f"donor min: +{bins[donor_min_idx]} bp ($\\bar c={donor_means[donor_min_idx]:.2f}$)",
                xy=(bins[donor_min_idx], donor_means[donor_min_idx]),
                xytext=(bins[donor_min_idx] - 80, donor_means[donor_min_idx] + 1.0),
                fontsize=7.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.annotate(f"acceptor min: +{bins[acceptor_min_idx]} bp ($\\bar c={acceptor_means[acceptor_min_idx]:.2f}$)",
                xy=(bins[acceptor_min_idx], acceptor_means[acceptor_min_idx]),
                xytext=(bins[acceptor_min_idx] + 30, acceptor_means[acceptor_min_idx] + 0.4),
                fontsize=7.5, color=GREEN,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))

    ax.set_xlabel("distance from splice junction (bp; negative = exonic side)")
    ax.set_ylabel("mean settling depth $\\bar c$")
    ax.set_xlim(min(bins) - 10, max(bins) + 10)
    ax.set_title("Splice positional fine-profile",
                 loc="center", fontsize=14, fontweight="bold",
                 fontfamily="Times New Roman", pad=10)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A4_splice_fine.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_A4_splice_fine.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_A4_splice_fine.png'}")
    print(f"wrote {OUT_DIR / 'fig_A4_splice_fine.pdf'}")


if __name__ == "__main__":
    main()
