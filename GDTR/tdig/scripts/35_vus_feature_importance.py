"""Exp — VUS classifier feature importance + per-layer single-layer baseline.

Extends script 27: identifies which (feature_type, layer) is most important in
the gbm_n100 VUS classifier, plus per-layer-only ΔH classifier AUROC curve
for direct interpretation.

Outputs: results/vus_feature_importance/
  gbm_feature_importance.csv     (layer, feature_type, importance)
  per_layer_only_auroc.csv       (layer, AUROC) using only that layer's features
  feature_importance.png         layer-resolved feature importance heatmap
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_features(df):
    X_dh = np.asarray(df.delta_h_norm_2.tolist(), dtype=np.float32)
    X_dc = np.asarray(df.delta_cos.tolist(), dtype=np.float32)
    return np.concatenate([np.log1p(X_dh), X_dc], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scalars", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/vus_feature_importance"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print(f"[load] variants ...")
    df = pd.read_parquet(args.scalars)
    df_bin = df[df.category.isin({"P_LP", "B_LB"})].reset_index(drop=True)
    df_bin["y"] = (df_bin.category == "P_LP").astype(int)
    print(f"  n_PLP={int(df_bin.y.sum())}, n_BLB={int((1-df_bin.y).sum())}")

    X = build_features(df_bin)
    Xs = StandardScaler().fit_transform(X)
    y = df_bin.y.values

    # Fit GBM
    print("[fit] GBM ...")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=args.seed)
    clf.fit(Xs, y)

    # Feature importance
    fi = clf.feature_importances_
    fi_records = []
    for i in range(32):
        fi_records.append({"feature_type": "log_dh_norm_2", "layer": i, "importance": float(fi[i])})
    for i in range(32):
        fi_records.append({"feature_type": "delta_cos", "layer": i, "importance": float(fi[32 + i])})
    fi_df = pd.DataFrame(fi_records)
    fi_df.to_csv(args.out_dir / "gbm_feature_importance.csv", index=False)
    print(f"[save] gbm_feature_importance.csv")
    top10 = fi_df.sort_values("importance", ascending=False).head(10)
    print("Top 10 features:")
    print(top10.to_string(index=False))

    # Per-layer-only AUROC (using only 2-D feature: log_dh + delta_cos at that layer)
    print("\n[fit] per-layer-only 2-D classifier ...")
    rows = []
    X_dh_all = np.asarray(df_bin.delta_h_norm_2.tolist(), dtype=np.float32)
    X_dc_all = np.asarray(df_bin.delta_cos.tolist(), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for ell in range(32):
        X_ell = np.column_stack([np.log1p(X_dh_all[:, ell]), X_dc_all[:, ell]])
        cv_pred = cross_val_predict(LogisticRegression(max_iter=300, C=1.0),
                                      X_ell, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        auroc = roc_auc_score(y, cv_pred)
        rows.append({"layer": ell, "AUROC": float(auroc)})
        print(f"  L={ell:2d}  AUROC={auroc:.3f}")
    pd.DataFrame(rows).to_csv(args.out_dir / "per_layer_only_auroc.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    # Pivot for heatmap
    piv = fi_df.pivot(index="feature_type", columns="layer", values="importance")
    sns.heatmap(piv, ax=ax, cmap="viridis", cbar_kws={"label": "GBM feature importance"})
    ax.set_title("GBM VUS classifier — feature importance by (feature_type, layer)")
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Feature type")

    ax = axes[1]
    rdf = pd.DataFrame(rows)
    ax.plot(rdf.layer, rdf.AUROC, marker="o", ms=4, lw=1.8, color="#d62728")
    ax.axhline(0.949, color="green", ls="--", lw=0.8, label="full 64-D GBM (0.949)")
    ax.axhline(0.855, color="gray", ls=":", lw=0.8, label="baseline ΔH L=8 (0.855)")
    ax.axvline(29, color="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("AUROC")
    ax.set_title("Per-layer-only 2-D classifier AUROC (log ΔH + Δcos)")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"feature_importance.{ext}", dpi=150)
    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
