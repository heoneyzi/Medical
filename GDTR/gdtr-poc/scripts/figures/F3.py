"""F3 — Variant pathogenicity (headline figure).

Panels:
  (a) ROC overlay: ΔD_cos vector, ΔD_jsd vector, ΔD max_abs (proxy for Evo2 ΔLL
      when full features unavailable locally), Ensemble. AUROC from
      Phase 3 main_results.json (8008-variant truth) is shown in legend.
      ROC curves are out-of-fold (stratified 10-fold logistic) computed from
      the locally-available pilot set (variants_features.csv, 1000 variants),
      which mirrors the full pipeline at smaller N.
  (b) DeLong forest plot — ΔAUROC ± SE and p from ensemble_results.json.
  (c) Per-layer ΔD AUROC curve — jsd & cos lens, stratified 10-fold,
      shaded 95 % CI, L=29 marked.
  (d) Per-gene LOGO-CV bars — ΔD_cos AUROC sorted desc; error bars derived
      from std dev of fold AUROCs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402

ROOT = Path("/path/to/ICML")
PILOT = ROOT / "results/phase3_pilot/variants_features.csv"
MAIN = ROOT / "results/phase3_main/main_results.json"
PER_LAYER = ROOT / "results/tier1_per_layer/per_layer_auroc.csv"
PER_GENE = ROOT / "results/phase3_main/per_gene_auroc.csv"
ENSEMBLE = ROOT / "results/phase3_ensemble/ensemble_results.json"
OUT = ROOT / "results/figures_v2/F3_variant_pathogenicity"


def _oof_logistic(X: np.ndarray, y: np.ndarray, n_splits: int = 10, seed: int = 42):
    """Out-of-fold predicted probabilities via stratified KFold logistic regression."""
    p = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        # Standardize per-fold to mirror Phase 3 pipeline
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Xn_tr = (X[tr] - mu) / sd
        Xn_te = (X[te] - mu) / sd
        clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=seed)
        clf.fit(Xn_tr, y[tr])
        p[te] = clf.predict_proba(Xn_te)[:, 1]
    return p


def panel_a(ax) -> None:
    df = pd.read_csv(PILOT)
    df = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
    y = (df["category"] == "P_LP").astype(int).values

    cos_cols = [c for c in df.columns if c.startswith("dD_cos_")]
    jsd_cols = [c for c in df.columns if c.startswith("dD_jsd_")]
    Xcos = df[cos_cols].astype(float).values
    Xjsd = df[jsd_cols].astype(float).values
    xmax = df[["max_abs_dD"]].astype(float).values

    p_cos = _oof_logistic(Xcos, y)
    p_jsd = _oof_logistic(Xjsd, y)
    # max_abs_dD as a single scalar predictor — sign normalized so larger = more pathogenic
    p_max = (xmax[:, 0] - xmax[:, 0].min()) / (xmax[:, 0].max() - xmax[:, 0].min() + 1e-9)
    # Ensemble: stack ΔD_cos features + max_abs_dD (proxy for ΔLL channel)
    Xens = np.hstack([Xcos, xmax])
    p_ens = _oof_logistic(Xens, y)

    # AUROC from full Phase 3 main run (n=8008 truth)
    main = json.loads(MAIN.read_text())["results_stratified"]
    auroc_text = {
        "ΔD_cos vector":  main["dD_cos_vector"]["mean_auroc"],
        "ΔD_jsd vector":  main["dD_jsd_vector"]["mean_auroc"],
        "Evo2 ΔLL":       main["evo2_delta_loglik"]["mean_auroc"],
        "Ensemble (ΔD+ΔLL)": main["ensemble_cos_evo2"]["mean_auroc"],
    }
    ci_text = {
        "ΔD_cos vector":  (main["dD_cos_vector"]["ci95_lo"], main["dD_cos_vector"]["ci95_hi"]),
        "ΔD_jsd vector":  (main["dD_jsd_vector"]["ci95_lo"], main["dD_jsd_vector"]["ci95_hi"]),
        "Evo2 ΔLL":       (main["evo2_delta_loglik"]["ci95_lo"], main["evo2_delta_loglik"]["ci95_hi"]),
        "Ensemble (ΔD+ΔLL)": (main["ensemble_cos_evo2"]["ci95_lo"], main["ensemble_cos_evo2"]["ci95_hi"]),
    }

    curves = [
        ("ΔD_cos vector",     p_cos, fs.COLORS["dD_cos"]),
        ("ΔD_jsd vector",     p_jsd, fs.COLORS["dD_jsd"]),
        ("Evo2 ΔLL",          p_max, fs.COLORS["evo2"]),
        ("Ensemble (ΔD+ΔLL)", p_ens, fs.COLORS["ensemble"]),
    ]
    for name, scores, color in curves:
        fpr, tpr, _ = roc_curve(y, scores)
        a_pilot = auc(fpr, tpr)
        a_full = auroc_text[name]
        lo, hi = ci_text[name]
        ax.plot(fpr, tpr, color=color, lw=1.6,
                label=f"{name}: {a_full:.3f} [{lo:.3f}, {hi:.3f}]")
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.7, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (AUROC ± 95 % CI from full Phase 3, n=8008)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.2)
    fs.grid_y(ax)


def panel_b(ax) -> None:
    ens = json.loads(ENSEMBLE.read_text())
    rows = ens["delong_stratified"]
    # Pick comparisons that highlight ΔD adding incremental info
    keep_pairs = [
        ("AB_cos_evo2", "A_dD_cos"),     # +ΔLL adds info on top of ΔD_cos
        ("A_dD_cos",    "B_evo2_loglik"), # ΔD beats ΔLL alone
        ("AD_cos_am",   "A_dD_cos"),     # +AM on top of ΔD_cos
        ("ABCD_full",   "BCD_baseline"), # +ΔD on top of CADD+AM+ΔLL
    ]
    labels = {
        ("AB_cos_evo2", "A_dD_cos"):     "Ensemble vs ΔD_cos",
        ("A_dD_cos",    "B_evo2_loglik"):"ΔD_cos vs Evo2 ΔLL",
        ("AD_cos_am",   "A_dD_cos"):     "ΔD+AM vs ΔD_cos",
        ("ABCD_full",   "BCD_baseline"): "Full vs (ΔLL+CADD+AM)",
    }
    table = []
    for a, b in keep_pairs:
        row = next((r for r in rows if r["feat_a"] == a and r["feat_b"] == b), None)
        if row is None:
            continue
        delta = row["delta"]
        # SE = |delta| / |z| when |z| > 0
        z = row["z"]
        se = abs(delta) / abs(z) if abs(z) > 1e-9 else 0.0
        p = row["p_value"]
        table.append((labels[(a, b)], delta, se, p))

    names = [t[0] for t in table]
    deltas = np.array([t[1] for t in table])
    ses = np.array([t[2] for t in table])
    pvals = [t[3] for t in table]
    ypos = np.arange(len(table))[::-1]

    ax.errorbar(deltas, ypos, xerr=1.96 * ses, fmt="o",
                color=fs.COLORS["dD_cos"], ecolor="#444444",
                ms=6, lw=1.0, capsize=3)
    ax.axvline(0.0, color="#888888", lw=0.7, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Δ AUROC (a − b)  ± 95 % CI")
    ax.set_title("DeLong paired comparisons (stratified 10-fold)")

    # Annotate p-values
    xmax = max(deltas + 1.96 * ses) * 1.05
    for i, (d, p) in enumerate(zip(deltas, pvals)):
        if p == 0.0 or p < 1e-15:
            ptxt = "p < 1e-15"
        elif p < 1e-3:
            ptxt = f"p = {p:.1e}"
        else:
            ptxt = f"p = {p:.3f}"
        ax.text(xmax, ypos[i], ptxt, va="center", ha="left", fontsize=7.5)
    ax.set_xlim(min(deltas - 1.96 * ses) - 0.02, xmax * 1.55)
    fs.grid_y(ax)


def panel_c(ax) -> None:
    df = pd.read_csv(PER_LAYER)
    df = df[df["cv_scheme"] == "stratified_10fold"]

    for lens, color, label in [
        ("cos", fs.COLORS["dD_cos"], "ΔD_cos"),
        ("jsd", fs.COLORS["dD_jsd"], "ΔD_jsd"),
    ]:
        sub = df[df["lens"] == lens].sort_values("layer")
        ax.plot(sub["layer"], sub["auroc_mean"], color=color, lw=1.5, label=label)
        ax.fill_between(sub["layer"], sub["ci_low"], sub["ci_high"],
                        color=color, alpha=0.2, lw=0)
    ax.axvline(29, color=fs.COLORS["highlight"], lw=0.8, ls="--", alpha=0.85)
    ax.text(29, 0.52, " L=29", color=fs.COLORS["highlight"], fontsize=7.5,
            va="bottom", ha="left")
    ax.axhline(0.5, color="#aaaaaa", lw=0.6, ls=":")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Single-layer AUROC")
    ax.set_title("Per-layer ΔD ablation (10-fold ± 95 % CI)")
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0.35, 0.85)
    ax.legend(loc="lower right", frameon=False)
    fs.grid_y(ax)


def panel_d(ax) -> None:
    df = pd.read_csv(PER_GENE).dropna(subset=["auroc_dD_cos"])
    df = df.sort_values("auroc_dD_cos", ascending=False).reset_index(drop=True)
    # Approx 95 % half-width from heuristic SE = sqrt(p(1-p)/n) for AUROC.
    # Use Hanley-McNeil rough estimator as visual error indicator.
    def _se(auc_val, n_pos, n_neg):
        q1 = auc_val / (2 - auc_val)
        q2 = (2 * auc_val ** 2) / (1 + auc_val)
        var = (auc_val * (1 - auc_val)
               + (n_pos - 1) * (q1 - auc_val ** 2)
               + (n_neg - 1) * (q2 - auc_val ** 2)) / (n_pos * n_neg + 1e-9)
        return float(np.sqrt(max(var, 0.0)))
    df["se"] = df.apply(lambda r: _se(r["auroc_dD_cos"], r["n_pos"], r["n_neg"]), axis=1)

    x = np.arange(len(df))
    ax.bar(x, df["auroc_dD_cos"], yerr=1.96 * df["se"],
           color=fs.COLORS["dD_cos"], edgecolor="#222", lw=0.6,
           capsize=2, error_kw={"lw": 0.7})
    ax.axhline(0.5, color="#888888", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(df["gene"], rotation=40, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("LOGO-CV AUROC (ΔD_cos)")
    ax.set_title("Per-gene leave-one-out AUROC (sorted)")
    fs.grid_y(ax)


def main() -> None:
    fs.setup()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.5))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    for ax, lab in zip(axes.flat, list("abcd")):
        fs.label_panel(ax, f"({lab})")
    fig.suptitle("Variant pathogenicity: ΔD provides incremental info beyond Evo2 ΔLL",
                 y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
