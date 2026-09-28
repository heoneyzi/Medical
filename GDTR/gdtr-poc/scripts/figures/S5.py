"""S5 — Bootstrap stability of AUROC.

4 violin plots (one per model): ΔD_cos vector (32-d), ΔD_jsd vector (32-d),
Evo2 ΔLL (1-d), Ensemble (33-d). 1000 bootstrap iterations each.
Shows mean line + 95 % CI shaded box from auroc_distributions.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402

ROOT = Path("/path/to/ICML")
DIST = ROOT / "results/tier1_bootstrap/auroc_distributions.csv"
SUMMARY = ROOT / "results/tier1_bootstrap/summary.json"
OUT = ROOT / "results/figures_v2/S5_bootstrap_stability"


def main() -> None:
    fs.setup()
    df = pd.read_csv(DIST)
    summary = json.loads(SUMMARY.read_text())

    order = ["dD_cos_vec_32d", "dD_jsd_vec_32d", "evo2_dLL_1d", "ensemble_cos_evo2_33d"]
    pretty = {
        "dD_cos_vec_32d":      "ΔD_cos vector",
        "dD_jsd_vec_32d":      "ΔD_jsd vector",
        "evo2_dLL_1d":         "Evo2 ΔLL",
        "ensemble_cos_evo2_33d":"Ensemble (ΔD+ΔLL)",
    }
    palette = [fs.COLORS["dD_cos"], fs.COLORS["dD_jsd"],
               fs.COLORS["evo2"],   fs.COLORS["ensemble"]]
    data = [df[df["model"] == m]["auroc"].values for m in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    parts = ax.violinplot(data, positions=range(len(order)), widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("#222")
        body.set_alpha(0.55)
        body.set_linewidth(0.7)

    # Overlay mean ± 95% CI
    for i, m in enumerate(order):
        s = summary[m]
        ax.errorbar([i], [s["mean"]],
                    yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                    fmt="o", color="#222", ms=5, lw=1.0, capsize=3, zorder=5)
        ax.text(i, s["ci_high"] + 0.003,
                f"{s['mean']:.3f}\n[{s['ci_low']:.3f}, {s['ci_high']:.3f}]",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pretty[m] for m in order])
    ax.set_ylabel("AUROC (1000 stratified bootstraps)")
    ax.set_title("Bootstrap stability of AUROC across models")
    fs.grid_y(ax)
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
