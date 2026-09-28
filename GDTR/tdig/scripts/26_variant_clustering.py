"""Exp 1.1 — Variant mechanism clustering (unsupervised).

Builds per-variant feature vector from per-layer ΔH norm (32-D) + Δcos (32-D),
UMAP to 2D, HDBSCAN cluster. Goal: discover if P_LP variants segregate into
distinct mechanism sub-clusters interpretable as gain/loss-of-function /
splicing / regulatory etc.

Outputs: results/variant_clustering/
  umap_coordinates.csv         per-variant (UMAP1, UMAP2, cluster_id, metadata)
  cluster_summary.csv          per-cluster (n, % P_LP, % B_LB, dominant gene, dominant consequence)
  cluster_signatures.csv       per-cluster mean ΔH per layer (the "mechanism signature")
  scatter_category.png         UMAP colored by category
  scatter_consequence.png      UMAP colored by consequence
  scatter_cluster.png          UMAP colored by HDBSCAN cluster
  cluster_signatures.png       per-cluster ΔH layer profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scalars", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--consequence", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_per_consequence/variant_with_consequence.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_clustering"))
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.9)

    print(f"[load] {args.scalars}")
    df = pd.read_parquet(args.scalars)
    cons = pd.read_csv(args.consequence)[["chrom", "pos", "ref", "alt", "consequence"]]
    cons["chrom"] = cons.chrom.astype(str)
    df["chrom"] = df.chrom.astype(str)
    df = df.merge(cons, on=["chrom", "pos", "ref", "alt"], how="left")
    print(f"[load]   {len(df):,} variants  consequence-matched={int(df.consequence.notna().sum()):,}")

    # Feature matrix: per-layer dh_norm_2 (32) + delta_cos (32) = 64-D
    X_dh = np.asarray(df.delta_h_norm_2.tolist(), dtype=np.float32)   # (N, 32)
    X_dc = np.asarray(df.delta_cos.tolist(), dtype=np.float32)        # (N, 32)
    # log-transform dH magnitude for scale
    X = np.concatenate([np.log1p(X_dh), X_dc], axis=1)                # (N, 64)
    print(f"[feat] feature shape {X.shape}")

    # Standardize
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    # UMAP
    print(f"[UMAP] fitting (n_neighbors={args.n_neighbors})...")
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed; falling back to PCA")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=args.seed).fit_transform(Xs)
    else:
        reducer = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors,
                             min_dist=0.1, random_state=args.seed)
        emb = reducer.fit_transform(Xs)
    print(f"[UMAP] emb shape {emb.shape}")

    # HDBSCAN cluster
    print(f"[HDBSCAN] clustering (min_cluster_size={args.min_cluster_size})...")
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                      cluster_selection_method="eom")
        labels = clusterer.fit_predict(emb)
    except ImportError:
        print("ERROR: hdbscan not available; falling back to KMeans k=8")
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=8, random_state=args.seed, n_init=10).fit_predict(emb)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[HDBSCAN] {n_clusters} clusters, noise={int((labels == -1).sum())}")

    # Save per-variant coords
    coords = df[["chrom", "pos", "ref", "alt", "gene", "category", "stars", "consequence"]].copy()
    coords["UMAP1"] = emb[:, 0]; coords["UMAP2"] = emb[:, 1]
    coords["cluster"] = labels
    coords.to_csv(args.out_dir / "umap_coordinates.csv", index=False)
    print(f"[save] umap_coordinates.csv")

    # Cluster summary
    summary_rows = []
    for cl in sorted(set(labels)):
        sub = coords[coords.cluster == cl]
        n = len(sub)
        n_PLP = int((sub.category == "P_LP").sum())
        n_BLB = int((sub.category == "B_LB").sum())
        n_VUS = int((sub.category == "VUS").sum())
        dom_gene = sub.gene.value_counts().idxmax() if n > 0 else "—"
        dom_cons = sub.consequence.value_counts().idxmax() if sub.consequence.notna().any() else "—"
        summary_rows.append({
            "cluster": int(cl), "n": n,
            "n_PLP": n_PLP, "n_BLB": n_BLB, "n_VUS": n_VUS,
            "pct_PLP": float(100 * n_PLP / max(n, 1)),
            "pct_BLB": float(100 * n_BLB / max(n, 1)),
            "dom_gene": dom_gene, "dom_consequence": dom_cons,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "cluster_summary.csv", index=False)
    print(f"[save] cluster_summary.csv")
    print(summary_df.to_string(index=False))

    # Cluster signatures (mean ΔH per layer per cluster)
    sig_rows = []
    for cl in sorted(set(labels)):
        sub_idx = np.where(labels == cl)[0]
        if len(sub_idx) == 0:
            continue
        mean_dh = X_dh[sub_idx].mean(axis=0)
        mean_dc = X_dc[sub_idx].mean(axis=0)
        for ell in range(32):
            sig_rows.append({"cluster": int(cl), "layer": ell,
                              "mean_dh_norm_2": float(mean_dh[ell]),
                              "mean_delta_cos": float(mean_dc[ell]),
                              "n": int(len(sub_idx))})
    pd.DataFrame(sig_rows).to_csv(args.out_dir / "cluster_signatures.csv", index=False)
    print(f"[save] cluster_signatures.csv")

    # === Figures ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # By category
    ax = axes[0]
    cat_colors = {"P_LP": "#d62728", "B_LB": "#1f77b4", "VUS": "#7f7f7f"}
    for cat, color in cat_colors.items():
        mask = coords.category == cat
        ax.scatter(coords.UMAP1[mask], coords.UMAP2[mask], s=2, alpha=0.4,
                    color=color, label=f"{cat} (n={int(mask.sum())})")
    ax.set_title("UMAP colored by ClinVar category")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(markerscale=4)

    # By consequence
    ax = axes[1]
    cons_present = coords.consequence.value_counts().head(8).index.tolist()
    cmap = plt.cm.tab10
    for i, c in enumerate(cons_present):
        mask = coords.consequence == c
        ax.scatter(coords.UMAP1[mask], coords.UMAP2[mask], s=2, alpha=0.4,
                    color=cmap(i), label=f"{c} (n={int(mask.sum())})")
    ax.set_title("UMAP colored by molecular consequence (top 8)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(markerscale=4, fontsize=8)

    # By HDBSCAN cluster
    ax = axes[2]
    cmap = plt.cm.tab20
    for cl in sorted(set(labels)):
        mask = labels == cl
        color = "#000000" if cl == -1 else cmap(cl % 20)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.5,
                    color=color, label=f"C{cl} (n={int(mask.sum())})")
    ax.set_title(f"UMAP colored by HDBSCAN cluster ({n_clusters} clusters)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(markerscale=4, fontsize=7, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"clustering_overview.{ext}", dpi=150)
    plt.close(fig)
    print(f"[plot] clustering_overview saved")

    # Cluster ΔH signatures
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sig_df = pd.DataFrame(sig_rows)
    for cl in sorted(set(labels)):
        if cl == -1:
            continue
        g = sig_df[sig_df.cluster == cl]
        ax.plot(g.layer, g.mean_dh_norm_2, marker="o", ms=3, lw=1.5,
                color=plt.cm.tab20(cl % 20),
                label=f"C{cl} (n={int(g.n.iloc[0])})")
    ax.axvline(29, color="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean ΔH norm L2")
    ax.set_title("Per-cluster ΔH layer profile — \"mechanism signature\"\n"
                  "Distinct shapes ⇒ distinct mechanisms")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"cluster_signatures.{ext}", dpi=150)
    plt.close(fig)
    print(f"[plot] cluster_signatures saved")

    summary_json = {
        "n_variants": len(coords),
        "n_clusters": n_clusters,
        "n_noise": int((labels == -1).sum()),
        "PLP_enriched_clusters": summary_df[summary_df.pct_PLP > 70].cluster.tolist(),
        "BLB_enriched_clusters": summary_df[summary_df.pct_BLB > 70].cluster.tolist(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
