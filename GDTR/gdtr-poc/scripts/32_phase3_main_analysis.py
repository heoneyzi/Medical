"""Phase 3 main analysis — per-gene-stratified CV + ensemble + VUS ranking.

Reads `/path/to/gDTR/results/phase3_main/variants_features.csv` produced by
`31_phase3_main.py`, runs:

  1) Standard 10-fold StratifiedKFold (by class) — pilot-style baseline.
  2) Per-gene-stratified 10-fold CV (variants of the same gene stay together
     within a fold) via GroupKFold on `gene`.
  3) Leave-One-Gene-Out (LOGO) CV.

For each split it evaluates 4 feature sets:
  - dD_jsd vector (32-d)
  - dD_cos vector (32-d)            <-- primary
  - Evo 2 likelihood Δ (1-d)
  - Ensemble: dD_cos ⊕ Evo 2 likelihood Δ (33-d)

VUS ranking: train on all P/LP+B/LB with dD_cos+Evo2 ensemble, predict
P_pathogenic on VUS variants, output top-100 / bottom-100.

Outputs:
  results/phase3_main/main_results.json
  results/phase3_main/vus_ranking.csv
  results/phase3_main/per_gene_auroc.csv
  results/phase3_main/F_phase3_main_auroc.{pdf,png}
  results/phase3_main/F_per_gene_auroc.{pdf,png}
  results/phase3_main/F_vus_ranking.{pdf,png}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase3_main_analysis"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase3_main"
LOG = ru.setup_logging(PHASE)

CSV_PATH = PHASE_OUT_DIR / "variants_features.csv"
SEED = 42
N_FOLDS = 10
N_LAYERS = 32


def load_features(csv_path: Path):
    df = pd.read_csv(csv_path)
    LOG.info("loaded %d rows / %d cols", len(df), len(df.columns))
    return df


def block_features(df: pd.DataFrame):
    feat_jsd = np.stack(
        [df[f"dD_jsd_{l}"].to_numpy(dtype=float) for l in range(N_LAYERS)], axis=1
    )
    feat_cos = np.stack(
        [df[f"dD_cos_{l}"].to_numpy(dtype=float) for l in range(N_LAYERS)], axis=1
    )
    feat_evo = df[["evo2_delta_loglik"]].to_numpy(dtype=float)
    feat_ens = np.concatenate([feat_cos, feat_evo], axis=1)
    feat_max = df[["max_abs_dD"]].to_numpy(dtype=float)
    return {
        "dD_jsd_vector": feat_jsd,
        "dD_cos_vector": feat_cos,
        "evo2_delta_loglik": feat_evo,
        "ensemble_cos_evo2": feat_ens,
        "max_abs_dD": feat_max,
    }


def _fit_pred_logreg(Xtr, ytr, Xte, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")),
    ])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1], pipe


def cv_stratified(X, y, name):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aurocs = []; ys, ss = [], []
    for tr, te in skf.split(X, y):
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        aurocs.append(roc_auc_score(y[te], s))
        ys.append(y[te]); ss.append(s)
    yt = np.concatenate(ys); st = np.concatenate(ss)
    fpr, tpr, _ = roc_curve(yt, st)
    mean_au = float(np.mean(aurocs)); std_au = float(np.std(aurocs, ddof=1))
    se = std_au / np.sqrt(N_FOLDS)
    return {
        "name": name, "split": "stratified_kfold",
        "fold_aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_au, "std_auroc": std_au,
        "ci95_lo": float(mean_au - 1.96 * se),
        "ci95_hi": float(mean_au + 1.96 * se),
        "n_features": int(X.shape[1]), "n_samples": int(X.shape[0]),
        "pooled_auroc": float(roc_auc_score(yt, st)),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


def cv_group_kfold(X, y, groups, name, n_splits=10):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    n_groups = len(np.unique(groups))
    n_eff = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_eff)
    aurocs = []; ys, ss = [], []
    for tr, te in gkf.split(X, y, groups):
        # Need both classes in train for AUROC
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        aurocs.append(roc_auc_score(y[te], s))
        ys.append(y[te]); ss.append(s)
    if not aurocs:
        return {
            "name": name, "split": "group_kfold",
            "mean_auroc": float("nan"), "n_folds_used": 0,
            "n_features": int(X.shape[1]), "n_samples": int(X.shape[0]),
            "fpr": [], "tpr": [], "pooled_auroc": float("nan"),
            "fold_aurocs": [], "std_auroc": float("nan"),
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
        }
    yt = np.concatenate(ys); st = np.concatenate(ss)
    fpr, tpr, _ = roc_curve(yt, st)
    mean_au = float(np.mean(aurocs))
    std_au = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0
    se = std_au / max(np.sqrt(len(aurocs)), 1.0)
    return {
        "name": name, "split": "group_kfold",
        "fold_aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_au, "std_auroc": std_au,
        "ci95_lo": float(mean_au - 1.96 * se),
        "ci95_hi": float(mean_au + 1.96 * se),
        "n_features": int(X.shape[1]), "n_samples": int(X.shape[0]),
        "n_folds_used": int(len(aurocs)),
        "pooled_auroc": float(roc_auc_score(yt, st)),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


def cv_logo(X, y, groups, name):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, roc_curve
    logo = LeaveOneGroupOut()
    aurocs = []; per_gene = {}; ys, ss = [], []
    for tr, te in logo.split(X, y, groups):
        gene_held = str(groups[te][0])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            per_gene[gene_held] = {"auroc": None, "n": int(len(te)),
                                   "reason": "single class in heldout"}
            continue
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        au = float(roc_auc_score(y[te], s))
        aurocs.append(au); ys.append(y[te]); ss.append(s)
        per_gene[gene_held] = {"auroc": au, "n": int(len(te))}
    if not aurocs:
        return {"name": name, "split": "logo", "mean_auroc": float("nan"),
                "per_gene": per_gene, "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0])}
    yt = np.concatenate(ys); st = np.concatenate(ss)
    fpr, tpr, _ = roc_curve(yt, st)
    mean_au = float(np.mean(aurocs))
    std_au = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0
    se = std_au / max(np.sqrt(len(aurocs)), 1.0)
    return {
        "name": name, "split": "logo",
        "fold_aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_au, "std_auroc": std_au,
        "ci95_lo": float(mean_au - 1.96 * se),
        "ci95_hi": float(mean_au + 1.96 * se),
        "per_gene": per_gene,
        "n_features": int(X.shape[1]), "n_samples": int(X.shape[0]),
        "pooled_auroc": float(roc_auc_score(yt, st)),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


def per_gene_auroc(X, y, groups, name):
    """Train on all-other-gene variants, evaluate on held-out gene only.
    Same as LOGO but reported per-gene as a table."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    logo = LeaveOneGroupOut()
    rows = []
    for tr, te in logo.split(X, y, groups):
        gene_held = str(groups[te][0])
        n_pos = int((y[te] == 1).sum()); n_neg = int((y[te] == 0).sum())
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            rows.append({"gene": gene_held, "auroc": None,
                         "n_pos": n_pos, "n_neg": n_neg, "n": int(len(te))})
            continue
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        au = float(roc_auc_score(y[te], s))
        rows.append({"gene": gene_held, "auroc": au,
                     "n_pos": n_pos, "n_neg": n_neg, "n": int(len(te))})
    return rows


def vus_ranking(df_train, df_vus, X_train, X_vus):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    y_train = (df_train["category"] == "P_LP").astype(int).to_numpy()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    p_path = pipe.predict_proba(X_vus)[:, 1]
    out = df_vus.copy()
    out["p_pathogenic"] = p_path
    out = out.sort_values("p_pathogenic", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="analysis"):
        t0 = time.time()
        df = load_features(CSV_PATH)

        # Train subset (P/LP + B/LB), VUS subset
        df_train = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        df_vus = df[df["category"] == "VUS"].reset_index(drop=True)
        LOG.info("train rows: %d  VUS rows: %d", len(df_train), len(df_vus))

        feats_train = block_features(df_train)
        feats_vus = block_features(df_vus) if len(df_vus) > 0 else None
        y_train = (df_train["category"] == "P_LP").astype(int).to_numpy()
        groups_train = df_train["gene"].to_numpy()

        feature_sets = [
            "dD_jsd_vector",
            "dD_cos_vector",
            "evo2_delta_loglik",
            "ensemble_cos_evo2",
            "max_abs_dD",
        ]

        # ---- 1) Stratified 10-fold (by class) ----
        results_strat = {}
        for name in feature_sets:
            r = cv_stratified(feats_train[name], y_train, name)
            results_strat[name] = r
            LOG.info("[strat] %-22s mean_auroc=%.4f CI=[%.4f, %.4f]",
                     name, r["mean_auroc"], r["ci95_lo"], r["ci95_hi"])

        # ---- 2) Per-gene-stratified GroupKFold ----
        results_group = {}
        for name in feature_sets:
            r = cv_group_kfold(
                feats_train[name], y_train, groups_train, name, n_splits=N_FOLDS)
            results_group[name] = r
            LOG.info("[group] %-22s mean_auroc=%.4f CI=[%.4f, %.4f] n_folds=%d",
                     name, r["mean_auroc"], r["ci95_lo"], r["ci95_hi"],
                     r.get("n_folds_used", 0))

        # ---- 3) LOGO ----
        results_logo = {}
        for name in feature_sets:
            r = cv_logo(feats_train[name], y_train, groups_train, name)
            results_logo[name] = r
            LOG.info("[LOGO] %-22s mean_auroc=%.4f", name, r["mean_auroc"])

        # ---- 4) Per-gene AUROC table for primary feature (dD_cos) and ensemble ----
        per_gene_cos = per_gene_auroc(
            feats_train["dD_cos_vector"], y_train, groups_train, "dD_cos_vector")
        per_gene_ens = per_gene_auroc(
            feats_train["ensemble_cos_evo2"], y_train, groups_train, "ensemble_cos_evo2")
        per_gene_evo = per_gene_auroc(
            feats_train["evo2_delta_loglik"], y_train, groups_train, "evo2_delta_loglik")

        per_gene_df = pd.DataFrame(per_gene_cos).rename(
            columns={"auroc": "auroc_dD_cos"})
        ens_df = pd.DataFrame(per_gene_ens).rename(columns={"auroc": "auroc_ensemble"})
        evo_df = pd.DataFrame(per_gene_evo).rename(columns={"auroc": "auroc_evo2_loglik"})
        per_gene_df = per_gene_df.merge(
            ens_df[["gene", "auroc_ensemble"]], on="gene", how="left")
        per_gene_df = per_gene_df.merge(
            evo_df[["gene", "auroc_evo2_loglik"]], on="gene", how="left")
        per_gene_df.to_csv(PHASE_OUT_DIR / "per_gene_auroc.csv", index=False)
        LOG.info("wrote per_gene_auroc.csv")

        # ---- 5) VUS ranking ----
        vus_top = None
        if feats_vus is not None and len(df_vus) > 0:
            ranked = vus_ranking(df_train, df_vus,
                                 feats_train["ensemble_cos_evo2"],
                                 feats_vus["ensemble_cos_evo2"])
            ranked.to_csv(PHASE_OUT_DIR / "vus_ranking.csv", index=False)
            LOG.info("wrote vus_ranking.csv  n=%d", len(ranked))
            vus_top = ranked

        # ---- 6) Verdict ----
        primary_strat = results_strat["dD_cos_vector"]["mean_auroc"]
        primary_group = results_group["dD_cos_vector"]["mean_auroc"]
        primary_logo = results_logo["dD_cos_vector"]["mean_auroc"]
        ensemble_strat = results_strat["ensemble_cos_evo2"]["mean_auroc"]
        ensemble_group = results_group["ensemble_cos_evo2"]["mean_auroc"]
        evo_strat = results_strat["evo2_delta_loglik"]["mean_auroc"]
        delta_ensemble_strat = ensemble_strat - primary_strat
        delta_ensemble_group = ensemble_group - primary_group

        verdict = {
            "primary_feature": "dD_cos_vector",
            "primary_strat_auroc": primary_strat,
            "primary_group_auroc": primary_group,
            "primary_logo_auroc": primary_logo,
            "evo2_loglik_strat_auroc": evo_strat,
            "ensemble_strat_auroc": ensemble_strat,
            "ensemble_group_auroc": ensemble_group,
            "ensemble_minus_primary_strat": delta_ensemble_strat,
            "ensemble_minus_primary_group": delta_ensemble_group,
            "ensemble_adds_incremental_info": bool(delta_ensemble_group > 0.005),
            "primary_passes_0.55_group": bool(primary_group >= 0.55),
            "primary_passes_0.65_group": bool(primary_group >= 0.65),
            "primary_passes_0.55_strat": bool(primary_strat >= 0.55),
        }

        out = {
            "phase": PHASE,
            "config": {"seed": SEED, "n_folds": N_FOLDS, "n_layers": N_LAYERS},
            "n_train": int(len(df_train)),
            "n_vus": int(len(df_vus)),
            "category_counts": {
                k: int(v) for k, v in df["category"].value_counts().items()
            },
            "per_gene_train_counts": {
                g: int(v) for g, v in df_train["gene"].value_counts().items()
            },
            "per_gene_vus_counts": {
                g: int(v) for g, v in df_vus["gene"].value_counts().items()
            },
            "results_stratified": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
                for k, v in results_strat.items()
            },
            "results_group_kfold": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
                for k, v in results_group.items()
            },
            "results_logo": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
                for k, v in results_logo.items()
            },
            "verdict": verdict,
            "wall_time_sec": float(time.time() - t0),
        }
        json_path = PHASE_OUT_DIR / "main_results.json"
        json_path.write_text(json.dumps(out, indent=2, default=float))
        LOG.info("wrote %s", json_path)
        LOG.info("VERDICT: %s", verdict)

        # ---- 7) Plots ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 7a) ROC overlay (stratified) for 4 feature sets
            fig, ax = plt.subplots(figsize=(7, 6))
            for name in ("dD_jsd_vector", "dD_cos_vector",
                         "evo2_delta_loglik", "ensemble_cos_evo2"):
                r = results_strat[name]
                ax.plot(r["fpr"], r["tpr"],
                        label=f"{name} (AUROC={r['mean_auroc']:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"Phase 3 main ROC (n={len(df_train)})")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_phase3_main_auroc.{ext}", dpi=150)
            plt.close(fig)

            # 7b) per-gene AUROC bar chart
            pdf = per_gene_df.dropna(subset=["auroc_dD_cos"]).copy()
            pdf = pdf.sort_values("auroc_dD_cos")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(pdf))
            width = 0.4
            ax.bar(x - width/2, pdf["auroc_dD_cos"], width=width,
                   label="dD_cos vector")
            ax.bar(x + width/2, pdf["auroc_ensemble"], width=width,
                   label="ensemble (cos+Evo2)")
            ax.set_xticks(x); ax.set_xticklabels(pdf["gene"], rotation=45, ha="right")
            ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
            ax.set_ylabel("LOGO AUROC")
            ax.set_title("Per-gene LOGO AUROC")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_per_gene_auroc.{ext}", dpi=150)
            plt.close(fig)

            # 7c) VUS top-20 visualization
            if vus_top is not None and len(vus_top) > 0:
                top = vus_top.head(20).copy()
                top["label"] = top.apply(
                    lambda r: f"{r['gene']}:{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}", axis=1)
                fig, ax = plt.subplots(figsize=(7, 6))
                yy = np.arange(len(top))[::-1]
                ax.barh(yy, top["p_pathogenic"], color="steelblue")
                ax.set_yticks(yy); ax.set_yticklabels(top["label"], fontsize=8)
                ax.set_xlabel("Predicted P(pathogenic)")
                ax.set_title("Top-20 VUS by ensemble model")
                ax.set_xlim(0, 1); ax.grid(alpha=0.3)
                fig.tight_layout()
                for ext in ("pdf", "png"):
                    fig.savefig(PHASE_OUT_DIR / f"F_vus_ranking.{ext}", dpi=150)
                plt.close(fig)
            LOG.info("plots saved")
        except Exception as e:  # noqa: BLE001
            LOG.warning("figure failed: %s", e)


if __name__ == "__main__":
    main()
