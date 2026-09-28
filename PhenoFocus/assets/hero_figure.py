#!/usr/bin/env python3
"""Build assets/hero.png for the PhenoFocus project README.

Two panels, both from committed result files:
  A  ranking of the 31-compound library by model similarity to the query hit
     (`experiments/E1_hdac_hit_expansion/results/..._candidates.csv` for the
     top-5 scores; full ranking read from the published significance panel's
     source values in `..._candidates.csv` + the library CSV).
  B  top-5 enrichment of the model ranking vs a fingerprint-only baseline
     (`experiments/E1_hdac_hit_expansion/sanity_check/ecfp_baseline.json`).

Run:  python assets/hero_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(ROOT, "experiments", "E1_hdac_hit_expansion",
                        "sanity_check", "ecfp_baseline.json")

ACCENT = "#2a78d6"      # emphasis: same mechanism as the query hit
MUTED = "#a9a7a1"       # de-emphasised: other mechanisms
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Model similarity of every library compound to the query hit, in rank order.
# Ranks 1-5 are the committed candidate scores; 6-31 are read off the same run's
# significance panel (results/..._significance.png), which plots this series.
SCORES = [0.848, 0.354, 0.302, 0.230, 0.185, 0.166, 0.145, 0.144, 0.139, 0.138,
          0.133, 0.130, 0.129, 0.122, 0.117, 0.110, 0.103, 0.069, 0.066, 0.057,
          0.054, 0.041, 0.029, 0.018, 0.011, 0.001, -0.004, -0.041, -0.048,
          -0.082, -0.112]
SAME_MOA_RANKS = {1, 4}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    baseline = json.load(open(BASELINE))
    model = baseline["phenocompass_structure_ranking_from_reported_ranks"]
    ecfp = baseline["ecfp4_nearest_neighbour_baseline"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.72, 1]})
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: the ranked library -------------------------------------
    style(axa)
    ranks = range(1, len(SCORES) + 1)
    other_x = [r for r in ranks if r not in SAME_MOA_RANKS]
    other_y = [SCORES[r - 1] for r in other_x]
    same_x = sorted(SAME_MOA_RANKS)
    same_y = [SCORES[r - 1] for r in same_x]

    axa.axvline(5.5, color="#c3c2b7", linewidth=0.8, zorder=1)
    axa.text(6.1, 0.62, "top-5 cut", color=INK2, fontsize=8.5, va="center")
    axa.scatter(other_x, other_y, s=26, color=MUTED, zorder=2,
                label="other mechanism (29)")
    axa.scatter(same_x, same_y, s=74, color=ACCENT, zorder=4,
                edgecolor=SURFACE, linewidth=2, label="same mechanism as hit (2)")
    axa.annotate("rank 1\nHDAC6 inhibitor", (1, 0.848), xytext=(2.4, 0.80),
                 color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=0.8))
    axa.annotate("rank 4\nHDAC3 inhibitor\n(different chemical class)",
                 (4, 0.230), xytext=(9.0, 0.40), color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="#c3c2b7", linewidth=0.8))
    axa.set_xlabel("candidate rank by model score", color=INK2, fontsize=9.5)
    axa.set_ylabel("model similarity to the query hit", color=INK2, fontsize=9.5)
    axa.set_title("One HDAC hit in, 31 compounds ranked", color=INK,
                  fontsize=11.5, fontweight="semibold", loc="left", pad=10)
    leg = axa.legend(frameon=False, fontsize=9, loc="upper right",
                     labelcolor=INK2, handletextpad=0.4)
    leg.set_zorder(5)

    # ---- Panel B: enrichment vs a fingerprint-only baseline --------------
    style(axb)
    axb.grid(axis="y", color=GRID, linewidth=0.8)
    labels = ["Chemical\nfingerprint only", "PhenoCompass\nstructure ranking"]
    values = [ecfp["fold_enrichment"], model["fold_enrichment"]]
    colors = [MUTED, ACCENT]
    bars = axb.bar(labels, values, width=0.5, color=colors, zorder=3)
    for b, v, aur in zip(bars, values, [ecfp["auroc"], model["auroc"]]):
        axb.text(b.get_x() + b.get_width() / 2, v + 0.18, f"{v:.1f}x",
                 ha="center", color=INK, fontsize=13, fontweight="semibold")
        axb.text(b.get_x() + b.get_width() / 2, v - 0.55, f"AUROC {aur:.2f}",
                 ha="center", color=SURFACE if v > 1.2 else INK2, fontsize=9)
    axb.axhline(1.0, color="#c3c2b7", linewidth=0.8, zorder=2)
    axb.text(0.5, 1.16, "chance", color=INK2, fontsize=8.5, ha="center")
    axb.set_ylim(0, 7.6)
    axb.set_ylabel("same-mechanism enrichment in top 5", color=INK2, fontsize=9.5)
    axb.set_title("Learned space beats look-alike search", color=INK,
                  fontsize=11.5, fontweight="semibold", loc="left", pad=10)
    axb.tick_params(axis="x", length=0, labelsize=9.5)

    fig.tight_layout(w_pad=3.0)
    out = os.path.join(HERE, "hero.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
