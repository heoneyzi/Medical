"""S6 — Failure case stratification.

Panels:
  (a) Failure rate by gene: stacked bars FN+FP per gene, sorted by total error.
  (b) Failure rate by CADD-disagreement strata + score quintile strata.
Reads failure_breakdown.csv.
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
FAIL = ROOT / "results/tier2_failure/failure_breakdown.csv"
THRESH = ROOT / "results/tier2_failure/youden_threshold.json"
OUT = ROOT / "results/figures_v2/S6_failure_analysis"


def panel_gene(ax, df) -> None:
    g = df[df["stratum_type"] == "gene"].copy()
    g["total_err"] = g["fn_rate"] + g["fp_rate"]
    g = g.sort_values("total_err", ascending=False).reset_index(drop=True)
    x = np.arange(len(g))
    ax.bar(x, g["fn_rate"], color=fs.COLORS["dD_cos"], edgecolor="#222",
           lw=0.5, label="FN rate")
    ax.bar(x, g["fp_rate"], bottom=g["fn_rate"], color=fs.COLORS["evo2"],
           edgecolor="#222", lw=0.5, label="FP rate")
    ax.set_xticks(x)
    ax.set_xticklabels(g["stratum"], rotation=40, ha="right")
    ax.set_ylabel("Failure rate at Youden threshold")
    ax.set_title("Failure rate by gene (sorted by total error)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(g["fn_rate"] + g["fp_rate"]) * 1.15)
    fs.grid_y(ax)


def panel_strata(ax, df) -> None:
    rows = []
    for src, label_prefix in [("cadd_disagreement", "CADD "),
                              ("score_quintile",    "Quintile "),
                              ("variant_type",      "")]:
        sub = df[df["stratum_type"] == src]
        for _, r in sub.iterrows():
            rows.append((f"{label_prefix}{r['stratum']}", r["fn_rate"], r["fp_rate"]))
    names = [r[0] for r in rows]
    fn = np.array([r[1] for r in rows])
    fp = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]

    ax.barh(y, fn, color=fs.COLORS["dD_cos"], edgecolor="#222", lw=0.5,
            label="FN rate")
    ax.barh(y, fp, left=fn, color=fs.COLORS["evo2"], edgecolor="#222", lw=0.5,
            label="FP rate")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Failure rate at Youden threshold")
    ax.set_title("Failure rate by CADD-disagreement, quintile, and variant type")
    ax.legend(loc="lower right", frameon=False)
    fs.grid_y(ax)


def main() -> None:
    fs.setup()
    df = pd.read_csv(FAIL)
    thr = json.loads(THRESH.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    panel_gene(axes[0], df)
    panel_strata(axes[1], df)
    fs.label_panel(axes[0], "(a)")
    fs.label_panel(axes[1], "(b)")
    fig.suptitle(f"Failure-case stratification (Youden threshold "
                 f"= {thr['threshold']:.3f}; AUROC = {thr['auroc']:.4f}; "
                 f"FN = {thr['n_FN']}, FP = {thr['n_FP']})",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
