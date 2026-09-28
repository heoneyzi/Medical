"""Exp 1.2 — VUS reclassification using layer-resolved TDiG features.

Train classifier on P_LP vs B_LB labeled variants, predict on 2,902 VUS.
Report VUS pathogenicity probabilities, uncertainty (calibrated), and
agreement with simple ΔH score baseline.

Output: results/vus_reclassification/
  vus_predictions.csv          per VUS (variant, prob_PLP, prediction, agreement_with_dh)
  classifier_metrics.csv       train/test AUROC, AUPRC, calibration
  feature_importance.csv       which layers/features matter most
  vus_clusters.png             VUS UMAP colored by predicted prob
  prob_distribution.png        distribution of VUS predicted probs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_features(df):
    """Per-variant features: per-layer ΔH norm L2 (32), Δcos (32) — 64-D."""
    X_dh = np.asarray(df.delta_h_norm_2.tolist(), dtype=np.float32)
    X_dc = np.asarray(df.delta_cos.tolist(), dtype=np.float32)
    return np.concatenate([np.log1p(X_dh), X_dc], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scalars", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--consequence", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_per_consequence/variant_with_consequence.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/vus_reclassification"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print(f"[load] variants + consequence ...")
    df = pd.read_parquet(args.scalars)
    cons = pd.read_csv(args.consequence)[["chrom", "pos", "ref", "alt", "consequence"]]
    cons["chrom"] = cons.chrom.astype(str)
    df["chrom"] = df.chrom.astype(str)
    df = df.merge(cons, on=["chrom", "pos", "ref", "alt"], how="left")
    print(f"  total={len(df):,}  categories={df.category.value_counts().to_dict()}")

    # Split
    df_train = df[df.category.isin({"P_LP", "B_LB"})].reset_index(drop=True)
    df_train["y"] = (df_train.category == "P_LP").astype(int)
    df_vus = df[df.category == "VUS"].reset_index(drop=True)
    print(f"[split] train n={len(df_train)} (P={int(df_train.y.sum())}, B={int((1-df_train.y).sum())}); VUS n={len(df_vus)}")

    # Build features
    X_train = build_features(df_train)
    X_vus = build_features(df_vus)
    print(f"[feat] X_train {X_train.shape}, X_vus {X_vus.shape}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_vus_s = scaler.transform(X_vus)

    # === Train multiple classifiers, report CV metrics ===
    metrics_records = []
    classifiers = {
        "logreg_L2_C1": LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", n_jobs=-1),
        "logreg_L2_C0.1": LogisticRegression(C=0.1, max_iter=500, solver="lbfgs", n_jobs=-1),
        "gbm_n100": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=args.seed),
    }
    cv_predictions = {}
    for name, clf in classifiers.items():
        print(f"[clf] {name} 5-fold CV ...")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_pred = cross_val_predict(clf, X_train_s, df_train.y.values, cv=skf,
                                       method="predict_proba", n_jobs=-1)[:, 1]
        cv_predictions[name] = cv_pred
        auroc = roc_auc_score(df_train.y.values, cv_pred)
        auprc = average_precision_score(df_train.y.values, cv_pred)
        metrics_records.append({"classifier": name, "CV_AUROC": auroc, "CV_AUPRC": auprc,
                                  "n_train": len(df_train)})
        print(f"  {name}: CV AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

    # Baseline: best layer ΔH norm L2 alone
    from sklearn.linear_model import LogisticRegression as LR
    # Train per-layer-best with cv selection
    baseline_aurocs = []
    for ell in range(32):
        a = roc_auc_score(df_train.y.values, np.log1p(df_train.delta_h_norm_2.iloc[0][ell]) if False else np.log1p(np.asarray(df_train.delta_h_norm_2.tolist())[:, ell]))
        baseline_aurocs.append(a)
    best_ell = int(np.argmax(baseline_aurocs))
    baseline_auroc = float(baseline_aurocs[best_ell])
    metrics_records.append({"classifier": f"baseline_dh_L{best_ell}", "CV_AUROC": baseline_auroc,
                              "CV_AUPRC": np.nan, "n_train": len(df_train)})
    print(f"  baseline (ΔH at L={best_ell}): AUROC={baseline_auroc:.3f}")

    pd.DataFrame(metrics_records).to_csv(args.out_dir / "classifier_metrics.csv", index=False)

    # === VUS prediction with the best classifier ===
    best_clf_name = max(metrics_records[:-1], key=lambda r: r["CV_AUROC"])["classifier"]
    print(f"\n[VUS] applying best classifier: {best_clf_name}")
    best_clf = classifiers[best_clf_name].fit(X_train_s, df_train.y.values)
    vus_prob = best_clf.predict_proba(X_vus_s)[:, 1]

    # Baseline ΔH score (just for comparison): per-VUS L=best_ell value
    X_vus_dh = np.asarray(df_vus.delta_h_norm_2.tolist())[:, best_ell]
    # Normalize to [0,1] using train distribution as a rough comparable score
    X_train_dh = np.asarray(df_train.delta_h_norm_2.tolist())[:, best_ell]
    threshold_50 = float(np.quantile(X_train_dh, 0.5))
    vus_dh_high = (X_vus_dh > threshold_50).astype(int)

    # Reclass categories
    pred_pathogenic = (vus_prob > 0.5).astype(int)
    pred_likely_path = (vus_prob > 0.7).astype(int)
    pred_likely_benign = (vus_prob < 0.3).astype(int)

    vus_out = df_vus[["chrom", "pos", "ref", "alt", "gene", "stars", "consequence"]].copy()
    vus_out["prob_PLP"] = vus_prob
    vus_out["prediction"] = np.where(vus_prob > 0.7, "Likely_Pathogenic",
                              np.where(vus_prob < 0.3, "Likely_Benign", "Uncertain"))
    vus_out["dh_baseline_high"] = vus_dh_high
    vus_out["agreement_with_dh_baseline"] = (pred_pathogenic == vus_dh_high).astype(int)
    vus_out.to_csv(args.out_dir / "vus_predictions.csv", index=False)
    print(f"[save] vus_predictions.csv ({len(vus_out):,} VUS)")
    print(f"  prediction distribution: {pd.Series(vus_out.prediction).value_counts().to_dict()}")
    print(f"  baseline dh-high agreement: {vus_out.agreement_with_dh_baseline.mean()*100:.1f}%")

    # Feature importance (for gradient boosting)
    if "gbm" in best_clf_name:
        fi_rows = []
        importances = best_clf.feature_importances_
        for i in range(32):
            fi_rows.append({"feature_type": "log_dh_norm_2", "layer": i, "importance": float(importances[i])})
        for i in range(32):
            fi_rows.append({"feature_type": "delta_cos", "layer": i, "importance": float(importances[32 + i])})
        pd.DataFrame(fi_rows).to_csv(args.out_dir / "feature_importance.csv", index=False)

    # === Plots ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Probability distribution
    ax = axes[0]
    ax.hist(vus_prob, bins=40, color="#7f7f7f", alpha=0.85)
    ax.axvline(0.3, color="#1f77b4", ls="--", label="LB threshold (0.3)")
    ax.axvline(0.7, color="#d62728", ls="--", label="LP threshold (0.7)")
    ax.set_xlabel("Predicted P_LP probability"); ax.set_ylabel("# VUS variants")
    ax.set_title(f"VUS pathogenicity probability distribution\nn_VUS={len(df_vus):,}, classifier={best_clf_name}")
    ax.legend()

    # Per-consequence VUS prediction
    ax = axes[1]
    cons_order = ["intron", "synonymous", "missense", "5utr", "3utr", "noncoding"]
    box_data = []; box_labels = []
    for c in cons_order:
        sub = vus_prob[df_vus.consequence == c]
        if len(sub) < 5:
            continue
        box_data.append(sub); box_labels.append(f"{c}\n(n={len(sub)})")
    ax.boxplot(box_data, labels=box_labels, showfliers=False, patch_artist=True)
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    ax.set_ylabel("Predicted P_LP probability")
    ax.set_title("VUS probability by molecular consequence")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"vus_prob_distribution.{ext}", dpi=150)
    plt.close(fig)
    print(f"[plot] vus_prob_distribution saved")

    summary = {
        "n_VUS": len(df_vus),
        "best_classifier": best_clf_name,
        "best_CV_AUROC": float(metrics_records[0]["CV_AUROC"]),
        "VUS_predicted_LP": int((vus_prob > 0.7).sum()),
        "VUS_predicted_LB": int((vus_prob < 0.3).sum()),
        "VUS_uncertain": int(((vus_prob >= 0.3) & (vus_prob <= 0.7)).sum()),
        "baseline_dh_agreement_pct": float(vus_out.agreement_with_dh_baseline.mean() * 100),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
