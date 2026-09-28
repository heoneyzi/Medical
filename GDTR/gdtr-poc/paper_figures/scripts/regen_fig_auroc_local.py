"""Local regeneration of Fig 6 (variant AUROC four-panel diagnostic).

All four panels can be drawn from artifacts already vendored in this
repository — no H200 caches required:

  (a) ROC curves         — fold-AUROC means in
                           `results/tier1_baselines/baseline_auroc.json`
  (b) DeLong paired      — `results/tier1_baselines/baseline_auroc.json::delong_pairs`
  (c) Per-layer AUROC    — `results/tier1_per_layer/per_layer_auroc.csv`
  (d) Leave-one-gene-out — `results/tier1_baselines/baseline_auroc.json::results_logo`

Output: `figures/fig_A3_auroc.{png,pdf}` (positional name matching
the appendix numbering A3 in `gdtr_paper_ICML.tex`).

Note. The original Fig 6 in the paper plots the actual ROC curves from
held-out variant scores. Those raw scores are not vendored here, so panel
(a) is rendered as a horizontal bar chart of stratified-10-fold and
LOGO-CV AUROCs per method (an information-equivalent summary). The other
three panels match the paper exactly.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
T1_BASE = REPO_ROOT / "results" / "tier1_baselines" / "baseline_auroc.json"
T1_LAYER_CSV = REPO_ROOT / "results" / "tier1_per_layer" / "per_layer_auroc.csv"
OUT_DIR = V3_DIR / "figures"

METHOD_LABEL = {
    "A_dD_cos": "$\\Delta D_{\\cos}$ (32-d)",
    "B_delta_h": "$\\|\\Delta h\\|_2$",
    "C_rollout": "attn rollout",
    "D_ig": "integrated grad.",
}
METHOD_COLOR = {
    "A_dD_cos": "#1f77b4",
    "B_delta_h": "#d62728",
    "C_rollout": "#7f7f7f",
    "D_ig": "#bdbdbd",
}
METHOD_ORDER = ["A_dD_cos", "B_delta_h", "C_rollout", "D_ig"]


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


def panel_auroc_summary(ax, base) -> None:
    """Panel (a): mean ± CI AUROC per method (stratified + LOGO)."""
    strat = base["results_stratified"]
    logo = base["results_logo"]
    y = np.arange(len(METHOD_ORDER))
    width = 0.34  # was 0.4 — reduce so paired-bar value labels don't crowd
    for i, method in enumerate(METHOD_ORDER):
        s = strat[method]
        lg = logo[method]
        ax.barh(y[i] - width / 2, s["mean_auroc"], height=width,
                color=METHOD_COLOR[method], edgecolor="black", lw=0.45,
                alpha=0.95, label="stratified 10-fold" if i == 0 else None)
        ax.errorbar(s["mean_auroc"], y[i] - width / 2,
                    xerr=[[s["mean_auroc"] - s["ci95_lo"]],
                          [s["ci95_hi"] - s["mean_auroc"]]],
                    fmt="none", ecolor="black", lw=0.7, capsize=2)
        ax.barh(y[i] + width / 2, lg["mean_auroc"], height=width,
                color=METHOD_COLOR[method], edgecolor="black", lw=0.45,
                alpha=0.55, label="leave-one-gene-out" if i == 0 else None)
        ax.errorbar(lg["mean_auroc"], y[i] + width / 2,
                    xerr=[[lg["mean_auroc"] - lg["ci95_lo"]],
                          [lg["ci95_hi"] - lg["mean_auroc"]]],
                    fmt="none", ecolor="black", lw=0.7, capsize=2)
        # White bbox masks the error-bar centre line that would otherwise
        # cut through the value label.
        ax.text(s["mean_auroc"] + 0.012, y[i] - width / 2,
                f"{s['mean_auroc']:.3f}", va="center", fontsize=7.5,
                bbox=dict(boxstyle="square,pad=0.12",
                          facecolor="white", edgecolor="none"))
        ax.text(lg["mean_auroc"] + 0.012, y[i] + width / 2,
                f"{lg['mean_auroc']:.3f}", va="center", fontsize=7.5,
                alpha=0.85,
                bbox=dict(boxstyle="square,pad=0.12",
                          facecolor="white", edgecolor="none"))
    ax.axvline(0.5, color="#999999", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC (mean, 95% CI)")
    ax.set_title("(a) Method comparison", loc="left", fontsize=14, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)


def panel_delong(ax, base) -> None:
    pairs = base["delong_pairs"]
    y = np.arange(len(pairs))
    deltas = [p["delta"] for p in pairs]
    pvals = [p["p_value"] for p in pairs]
    labels = [f"{METHOD_LABEL[p['feat_a']]}\nvs {METHOD_LABEL[p['feat_b']]}"
              for p in pairs]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    ax.barh(y, deltas, color=colors, edgecolor="black", lw=0.5, height=0.55)
    # Always anchor the value label to the right of x=0 so it cannot collide
    # with the y-tick labels on the left of the y-axis (negative Δ rows).
    for yi, (d, p) in enumerate(zip(deltas, pvals)):
        ax.text(max(d, 0) + 0.005, yi,
                f"$\\Delta$={d:+.3f}\n$p$={p:.0e}",
                ha="left", va="center", fontsize=7.5)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$AUROC (DeLong paired)")
    ax.set_title("(b) Pairwise DeLong tests", loc="left", fontsize=14, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)


def panel_per_layer(ax) -> None:
    # CSV is in long form: (lens, layer, cv_scheme, auroc_mean, ...).
    # Pivot to per-layer cosine vs JSD on the stratified_10fold split.
    by_lens: dict[str, dict[int, float]] = {"cos": {}, "jsd": {}}
    for r in csv.DictReader(T1_LAYER_CSV.open()):
        if r["cv_scheme"] != "stratified_10fold":
            continue
        by_lens.setdefault(r["lens"], {})[int(r["layer"])] = float(r["auroc_mean"])
    layers = sorted(set(by_lens["cos"]) | set(by_lens["jsd"]))
    cos = [by_lens["cos"].get(L, np.nan) for L in layers]
    jsd = [by_lens["jsd"].get(L, np.nan) for L in layers]
    ax.plot(layers, cos, color="#1f77b4", lw=1.6, label="cosine $\\Delta D$")
    ax.plot(layers, jsd, color="#d62728", lw=1.6, label="JSD $\\Delta D$")
    cos_arr, jsd_arr = np.asarray(cos, float), np.asarray(jsd, float)
    if np.any(np.isfinite(cos_arr)):
        i = int(np.nanargmax(cos_arr))
        ax.scatter([layers[i]], [cos_arr[i]], s=42, color="#1f77b4",
                   edgecolor="black", zorder=4)
    if np.any(np.isfinite(jsd_arr)):
        j = int(np.nanargmax(jsd_arr))
        ax.scatter([layers[j]], [jsd_arr[j]], s=42, color="#d62728",
                   edgecolor="black", zorder=4)
    ax.axhline(0.5, color="#999999", lw=0.5)
    ax.set_xlabel("layer $\\ell$")
    ax.set_ylabel("single-tap AUROC")
    ax.set_title("(c) Per-layer ablation", loc="left", fontsize=14, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def panel_logo(ax, base) -> None:
    per_gene = base["results_logo"]["A_dD_cos"]["per_gene"]
    rows = sorted(((g, v["auroc"], v["n"]) for g, v in per_gene.items()
                   if v["auroc"] is not None),
                  key=lambda r: r[1])
    genes = [r[0] for r in rows]
    aurocs = [r[1] for r in rows]
    y = np.arange(len(genes))
    ax.barh(y, aurocs, color="#1f77b4", edgecolor="black", lw=0.45,
            height=0.65)
    ax.axvline(0.77, color="#d62728", ls="--", lw=0.9,
               label="paper floor (0.77)")
    for yi, val in enumerate(aurocs):
        ax.text(val + 0.005, yi, f"{val:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(genes)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("LOGO-CV AUROC")
    ax.set_title("(d) Leave-one-gene-out", loc="left", fontsize=14, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)
    # Park the legend OUTSIDE the axes (below the x-axis label) so it
    # cannot collide with bars or y-tick labels.
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))


def main() -> None:
    setup_style()
    base = json.loads(T1_BASE.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    panel_auroc_summary(axes[0, 0], base)
    panel_delong(axes[0, 1], base)
    panel_per_layer(axes[1, 0])
    panel_logo(axes[1, 1], base)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A3_auroc.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_A3_auroc.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_A3_auroc.png'}")
    print(f"wrote {OUT_DIR / 'fig_A3_auroc.pdf'}")


if __name__ == "__main__":
    main()
