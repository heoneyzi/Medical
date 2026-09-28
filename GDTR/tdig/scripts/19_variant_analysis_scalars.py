"""Phase 1.1a — Variant analysis using cached scalars.

Uses variant_scalars.parquet (already computed during variant_forward) to derive:
  • Per-layer AUROC for ΔH norm (L2/L1), Δcos discriminating P/LP vs B/LB
  • Per-gene AUROC table (15 cancer genes)
  • Stars-stratified analysis (ClinVar review confidence)
  • Headline curve: full-cohort 32-layer AUROC profile

This is the scalar-only fast pass. Full 17-cell ΔC analysis (loading h_ell)
will be a follow-up if scalar results merit deeper investigation.

Outputs under --out-dir (default: results/variant_analysis_scalars/):
  per_layer_auroc.csv       (layer, feature, AUROC, n_pos, n_neg)
  per_layer_auroc.png       curves
  per_gene_auroc.csv        (gene, feature, best_layer, best_AUROC, n_pos, n_neg)
  per_gene_auroc.png        per-gene comparison
  stars_stratified.csv      (stars, feature, layer, AUROC)
  summary.json              headline numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32

FEATURES = ("delta_h_norm_2", "delta_h_norm_1", "delta_cos")


def list_to_array(df_col):
    """Convert column-of-lists to 2-D float array (n_variants, n_layers)."""
    return np.asarray(df_col.tolist(), dtype=np.float32)


def auroc_per_layer(X, y, n_layers=N_LAYERS):
    """For each layer, compute AUROC of X[:, ell] discriminating y∈{0,1}."""
    aurocs = np.zeros(n_layers)
    for ell in range(n_layers):
        x_ell = X[:, ell]
        # Use the score directly; larger value = pathogenic by convention
        aurocs[ell] = roc_auc_score(y, x_ell)
    return aurocs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_analysis_scalars"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print(f"[load] {args.variants}")
    df = pd.read_parquet(args.variants)
    print(f"[load]   {len(df):,} variants, {df.gene.nunique()} genes, {sorted(df.chrom.unique())} chroms")
    print(f"[load]   categories: {df.category.value_counts().to_dict()}")

    # Binary labeling: P_LP=1, B_LB=0; drop VUS
    df_bin = df[df.category.isin({"P_LP", "B_LB"})].copy().reset_index(drop=True)
    df_bin["y"] = (df_bin.category == "P_LP").astype(int)
    n_pos, n_neg = int(df_bin.y.sum()), int((1 - df_bin.y).sum())
    print(f"[binary] P_LP n={n_pos}, B_LB n={n_neg}")

    # === Per-layer AUROC, full cohort ===
    print("\n[A] per-layer AUROC, full cohort ===")
    per_layer_records = []
    for feat in FEATURES:
        X = list_to_array(df_bin[feat])  # (n, 32)
        # For delta_cos, the column might be float16 list — check dtype
        if X.shape[1] != N_LAYERS:
            print(f"  WARN {feat} unexpected layer count {X.shape[1]}")
            continue
        aurocs = auroc_per_layer(X, df_bin.y.values)
        for ell, a in enumerate(aurocs):
            per_layer_records.append({
                "layer": ell, "feature": feat,
                "AUROC": float(a), "n_pos": n_pos, "n_neg": n_neg,
            })
        print(f"  {feat:18s}  best L={int(np.argmax(aurocs)):2d}  best AUROC={float(aurocs.max()):.3f}  worst={float(aurocs.min()):.3f}")
    pl_df = pd.DataFrame(per_layer_records)
    pl_df.to_csv(args.out_dir / "per_layer_auroc.csv", index=False)
    print(f"[A] saved per_layer_auroc.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {"delta_h_norm_2": "#d62728", "delta_h_norm_1": "#ff7f0e",
              "delta_cos": "#1f77b4"}
    for feat in FEATURES:
        g = pl_df[pl_df.feature == feat]
        ax.plot(g.layer, g.AUROC, marker="o", ms=4, lw=1.8,
                color=colors[feat], label=feat)
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4, label=f"L*={L_STAR}")
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("AUROC (P_LP vs B_LB)")
    ax.set_title(f"(1.1a) Variant pathogenicity AUROC per layer — full cohort\n"
                 f"n_PLP={n_pos:,}, n_BLB={n_neg:,} across {df_bin.gene.nunique()} cancer genes")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_layer_auroc.{ext}", dpi=150)
    plt.close(fig)
    print(f"[A] figure saved")

    # === Per-gene AUROC ===
    print("\n[B] per-gene AUROC ===")
    per_gene_records = []
    for gene, g_df in df_bin.groupby("gene"):
        if g_df.y.sum() < 10 or (1 - g_df.y).sum() < 10:
            print(f"  skip {gene}: insufficient class balance (P={int(g_df.y.sum())}, B={int((1-g_df.y).sum())})")
            continue
        n_p, n_b = int(g_df.y.sum()), int((1 - g_df.y).sum())
        for feat in FEATURES:
            X = list_to_array(g_df[feat])
            aurocs = auroc_per_layer(X, g_df.y.values)
            best_l = int(np.argmax(aurocs))
            per_gene_records.append({
                "gene": gene, "feature": feat,
                "best_layer": best_l, "best_AUROC": float(aurocs[best_l]),
                "AUROC_L29": float(aurocs[L_STAR]),
                "AUROC_L27": float(aurocs[27]),
                "n_pos": n_p, "n_neg": n_b,
            })
        print(f"  {gene:8s} n=({n_p},{n_b})  best ΔH2 AUROC={per_gene_records[-3]['best_AUROC']:.3f} at L={per_gene_records[-3]['best_layer']}")
    pg_df = pd.DataFrame(per_gene_records)
    pg_df.to_csv(args.out_dir / "per_gene_auroc.csv", index=False)
    print(f"[B] saved per_gene_auroc.csv")

    # Plot — bar chart of best AUROC per gene, ΔH norm L2 only
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sub = pg_df[pg_df.feature == "delta_h_norm_2"].sort_values("best_AUROC", ascending=False)
    ax.bar(sub.gene, sub.best_AUROC, color="#d62728", alpha=0.8)
    for i, (g_, a_) in enumerate(zip(sub.gene, sub.best_AUROC)):
        ax.text(i, a_ + 0.005, f"{a_:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_ylabel("Best AUROC (ΔH norm L2)")
    ax.set_title(f"(1.1a) Per-gene best AUROC — ΔH norm L2, P_LP vs B_LB")
    ax.set_ylim(0.4, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_gene_auroc.{ext}", dpi=150)
    plt.close(fig)
    print(f"[B] figure saved")

    # === Stars-stratified ===
    print("\n[C] stars-stratified AUROC ===")
    stars_records = []
    for stars, s_df in df_bin.groupby("stars"):
        if s_df.y.sum() < 10 or (1 - s_df.y).sum() < 10:
            continue
        n_p, n_b = int(s_df.y.sum()), int((1 - s_df.y).sum())
        for feat in FEATURES:
            X = list_to_array(s_df[feat])
            aurocs = auroc_per_layer(X, s_df.y.values)
            for ell, a in enumerate(aurocs):
                stars_records.append({
                    "stars": int(stars), "feature": feat, "layer": ell,
                    "AUROC": float(a), "n_pos": n_p, "n_neg": n_b,
                })
        print(f"  stars={int(stars)}, n=({n_p},{n_b})")
    s_df = pd.DataFrame(stars_records)
    s_df.to_csv(args.out_dir / "stars_stratified.csv", index=False)
    print(f"[C] saved stars_stratified.csv")

    # === Summary ===
    headline = {
        "n_variants_total": len(df),
        "n_binary_PLP": n_pos, "n_binary_BLB": n_neg,
        "n_genes": int(df.gene.nunique()),
        "best_overall": {},
    }
    for feat in FEATURES:
        sub = pl_df[pl_df.feature == feat]
        best_row = sub.loc[sub.AUROC.idxmax()]
        headline["best_overall"][feat] = {
            "best_layer": int(best_row.layer),
            "best_AUROC": float(best_row.AUROC),
            "L29_AUROC": float(sub[sub.layer == L_STAR].AUROC.iloc[0]),
            "L27_AUROC": float(sub[sub.layer == 27].AUROC.iloc[0]),
            "L0_AUROC": float(sub[sub.layer == 0].AUROC.iloc[0]),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(headline, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print("Headline summary:")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
