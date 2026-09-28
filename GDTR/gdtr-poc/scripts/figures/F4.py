"""F4 — Interpretability baseline comparison + compute cost.

Produced from:
  - results/tier1_baselines/baseline_auroc.json
  - results/tier1_baselines/baseline_spearman.csv
  - results/tier1_baselines/delong_pairs.csv
  - results/tier2_compute/cost_benchmark.csv  (optional; cost panel skipped if absent)

4 panels:
(a) Per-method AUROC + 95 % CI (stratified 10-fold AND LOGO-CV grouped bars)
(b) Pairwise Spearman ρ heatmap (4×4) on out-of-fold pooled scores
(c) DeLong forest plot: ΔD_cos vs each baseline (paired ΔAUROC + 95 % CI + p)
(d) Incremental info (residualized ΔD AUROC after subtracting each baseline)
    AND compute cost (ms/variant + peak VRAM) if T2.4 outputs landed.

The script is idempotent — re-running after T1.2/T2.4 land just regenerates.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _figstyle import setup, COLORS, label_panel, grid_y, save

setup()
RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "figures_v2"
OUT.mkdir(exist_ok=True)


# Method ordering and display names — matches script 48 method_specs keys
METHOD_ORDER = ["A_dD_cos", "B_delta_h", "C_rollout", "D_ig"]
METHOD_DISP = {
    "A_dD_cos":  "ΔD_cos (gDTR)",
    "B_delta_h": "‖Δh‖₂",
    "C_rollout": "attn rollout",
    "D_ig":      "Integrated\ngradients",
}
METHOD_COLORS = {
    "A_dD_cos":  COLORS["dD_cos"],
    "B_delta_h": COLORS["evo2"],
    "C_rollout": COLORS["alphamis"],
    "D_ig":      COLORS["ensemble"],
}


def load_inputs():
    auroc_p = RES / "tier1_baselines" / "baseline_auroc.json"
    spear_p = RES / "tier1_baselines" / "baseline_spearman.csv"
    delong_p = RES / "tier1_baselines" / "delong_pairs.csv"
    cost_p = RES / "tier2_compute" / "cost_benchmark.csv"

    missing = [str(p) for p in (auroc_p, spear_p, delong_p) if not p.exists()]
    if missing:
        print("[F4] Skipping — required T1.2 outputs not present:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(0)

    auroc = json.loads(auroc_p.read_text())
    spear = pd.read_csv(spear_p, index_col=0)
    delong = pd.read_csv(delong_p)
    cost = pd.read_csv(cost_p) if cost_p.exists() else None
    return auroc, spear, delong, cost


def panel_a(ax, auroc):
    """Per-method AUROC bar chart, stratified vs LOGO grouped."""
    methods = METHOD_ORDER
    strat = [auroc["results_stratified"][m]["mean_auroc"] for m in methods]
    strat_ci_lo = [auroc["results_stratified"][m]["ci95_lo"] for m in methods]
    strat_ci_hi = [auroc["results_stratified"][m]["ci95_hi"] for m in methods]
    logo = [auroc["results_logo"][m]["mean_auroc"] for m in methods]
    logo_ci_lo = [auroc["results_logo"][m]["ci95_lo"] for m in methods]
    logo_ci_hi = [auroc["results_logo"][m]["ci95_hi"] for m in methods]

    xs = np.arange(len(methods))
    w = 0.38
    yerr_strat = [
        [s - lo for s, lo in zip(strat, strat_ci_lo)],
        [hi - s for s, hi in zip(strat, strat_ci_hi)],
    ]
    yerr_logo = [
        [s - lo for s, lo in zip(logo, logo_ci_lo)],
        [hi - s for s, hi in zip(logo, logo_ci_hi)],
    ]
    ax.bar(xs - w/2, strat, w, yerr=yerr_strat, capsize=3,
           color=[METHOD_COLORS[m] for m in methods],
           edgecolor="white", linewidth=0.7, label="stratified 10-fold")
    ax.bar(xs + w/2, logo, w, yerr=yerr_logo, capsize=3,
           color=[METHOD_COLORS[m] for m in methods],
           edgecolor="black", linewidth=0.9, alpha=0.55, label="leave-one-gene-out")

    for i, (s, h) in enumerate(zip(strat, strat_ci_hi)):
        ax.text(i - w/2, h + 0.005, f"{s:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_DISP[m] for m in methods], fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7.5, loc="upper right")
    grid_y(ax)
    label_panel(ax, "(a)", x=-0.13)
    ax.set_title("Per-method AUROC on 8,008 ClinVar P_LP/B_LB", fontsize=10, loc="left")


def panel_b(ax, spear):
    """4×4 Spearman ρ heatmap."""
    spear = spear.loc[METHOD_ORDER, METHOD_ORDER]
    im = ax.imshow(spear.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_DISP[m] for m in METHOD_ORDER], rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels([METHOD_DISP[m] for m in METHOD_ORDER], fontsize=8)
    for i in range(len(METHOD_ORDER)):
        for j in range(len(METHOD_ORDER)):
            v = spear.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black",
                    fontweight="bold" if i != j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Spearman ρ")
    label_panel(ax, "(b)", x=-0.13)
    ax.set_title("Pairwise concordance (out-of-fold)", fontsize=10, loc="left")


def panel_c(ax, delong):
    """DeLong forest: ΔD_cos vs each baseline."""
    df = delong[delong["feat_a"] == "A_dD_cos"].copy()
    df["disp"] = df["feat_b"].map(METHOD_DISP)
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)
    ys = np.arange(len(df))

    # Approx 95% CI from z and delta: SE = |delta / z|
    se = np.where(np.abs(df["z"]) > 1e-6, np.abs(df["delta"] / df["z"]), 0.01)
    ci_lo = df["delta"] - 1.96 * se
    ci_hi = df["delta"] + 1.96 * se

    ax.errorbar(df["delta"], ys, xerr=[df["delta"] - ci_lo, ci_hi - df["delta"]],
                fmt="o", color=COLORS["dD_cos"], markersize=8, capsize=4,
                ecolor=COLORS["dD_cos"], linewidth=1.5)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"ΔD_cos vs {d}" for d in df["disp"]], fontsize=8)
    ax.set_xlabel("ΔAUROC (ΔD_cos − baseline) ± 95 % CI")
    grid_y(ax)
    for i, (delta, p) in enumerate(zip(df["delta"], df["p_value"])):
        ax.text(max(ci_hi[i], delta) + 0.005, i,
                f"p = {p:.2g}",
                fontsize=7.5, va="center", color="#222")
    label_panel(ax, "(c)", x=-0.18)
    ax.set_title("DeLong paired tests: ΔD_cos beats baselines?", fontsize=10, loc="left")


def panel_d(ax, auroc, cost):
    """Bottom-right: incremental info bars (always) + cost annotation if available."""
    incr = pd.DataFrame(auroc["incremental_info"])
    if len(incr):
        incr["disp"] = incr["baseline"].map(METHOD_DISP)
        incr = incr.sort_values("auroc_residual_dD_minus_baseline", ascending=False)
        ys = np.arange(len(incr))
        ax.barh(ys, incr["auroc_residual_dD_minus_baseline"],
                color=COLORS["dD_cos"], edgecolor="white", linewidth=0.7,
                label="ΔD residualized on baseline")
        ax.barh(ys, incr["auroc_baseline_alone"],
                left=incr["auroc_residual_dD_minus_baseline"],
                color=COLORS["control"], edgecolor="white", linewidth=0.7, alpha=0.6,
                label="baseline alone")
        ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(incr["disp"], fontsize=8)
        ax.set_xlabel("AUROC")
        ax.set_xlim(0.45, 1.0)
        ax.legend(fontsize=7.5, loc="lower right")
        grid_y(ax)
        label_panel(ax, "(d)", x=-0.18)
        ax.set_title("Incremental info: ΔD score after residualizing on baseline",
                     fontsize=10, loc="left")

    # If cost data available, overlay as annotation
    if cost is not None and len(cost):
        cost_map = dict(zip(cost["method"], cost["mean_per_variant_ms"]))
        vram_map = dict(zip(cost["method"], cost["peak_vram_gb"]))
        annot_text = "Compute cost (ms/var | GB):\n"
        for m in METHOD_ORDER:
            # cost csv may use different method names — try common mappings
            for key in (m, m.split("_", 1)[1] if "_" in m else m,
                        METHOD_DISP[m].split()[0]):
                if key in cost_map:
                    annot_text += f"  {METHOD_DISP[m].splitlines()[0]:>14}  {cost_map[key]:>5.1f} | {vram_map[key]:>4.1f}\n"
                    break
        ax.text(0.55, 0.05, annot_text.rstrip(),
                transform=ax.transAxes, fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="grey", linewidth=0.6, alpha=0.95))


def main():
    auroc, spear, delong, cost = load_inputs()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    panel_a(axes[0, 0], auroc)
    panel_b(axes[0, 1], spear)
    panel_c(axes[1, 0], delong)
    panel_d(axes[1, 1], auroc, cost)

    headline = (auroc["results_stratified"]["A_dD_cos"]["mean_auroc"]
                if "A_dD_cos" in auroc["results_stratified"] else float("nan"))
    fig.suptitle(
        f"Interpretability baselines: ΔD_cos AUROC = {headline:.3f}  "
        f"(n = {auroc['n_train_variants']:,})",
        fontsize=11.5, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    save(fig, OUT / "F4_baselines")
    print(f"saved {OUT/'F4_baselines.pdf'} and .png")
    print(f"  ΔD_cos AUROC: {headline:.4f}")
    print(f"  n train variants: {auroc['n_train_variants']:,}")
    if cost is not None:
        print(f"  cost benchmark methods: {len(cost)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
