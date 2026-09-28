"""T1.1 — Per-layer ΔD AUROC ablation.

For each lens in {jsd, cos} and each layer l in {0..31}, fit a single-feature
LogisticRegression and evaluate AUROC under (a) stratified 10-fold and (b)
leave-one-gene-out CV. Compute a 1000-resample bootstrap 95% CI on the
out-of-fold pooled scores. Also compute the 32-d vector-model AUROC for
both lenses (reference reproduction).

Outputs to /path/to/gDTR/results/tier1_per_layer/:
  - per_layer_auroc.csv
  - summary.json
  - _done

Usage: python scripts/40_t11_per_layer_ablation.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _runner_utils as ru

PHASE = "tier1_per_layer"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_per_layer"
SRC_CSV = ru.GDTR_ROOT / "results" / "phase3_ensemble" / "variants_features_full.csv"
LOG = ru.setup_logging(PHASE)

SEED = 42
N_FOLDS = 10
N_LAYERS = 32
N_BOOT = 1000


def _fit_pred_logreg(Xtr, ytr, Xte, max_iter=1000, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")),
    ])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def cv_pooled_strat(X, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pooled = np.full(len(y), np.nan, dtype=np.float64)
    py = np.full(len(y), -1, dtype=int)
    fold_aurocs = []
    from sklearn.metrics import roc_auc_score
    for tr, te in skf.split(X, y):
        s = _fit_pred_logreg(X[tr], y[tr], X[te])
        pooled[te] = s
        py[te] = y[te]
        fold_aurocs.append(roc_auc_score(y[te], s))
    return pooled, py, fold_aurocs


def cv_pooled_logo(X, y, groups):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    logo = LeaveOneGroupOut()
    pooled = np.full(len(y), np.nan, dtype=np.float64)
    py = np.full(len(y), -1, dtype=int)
    fold_aurocs = []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        s = _fit_pred_logreg(X[tr], y[tr], X[te])
        pooled[te] = s
        py[te] = y[te]
        fold_aurocs.append(roc_auc_score(y[te], s))
    return pooled, py, fold_aurocs


def bootstrap_ci(y_true, y_score, n_boot=N_BOOT, seed=SEED):
    """1000-resample bootstrap on pooled OOF scores.

    Returns (mean_auroc, ci_low, ci_high)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_score) & (y_true >= 0)
    yt = y_true[valid]
    ys = y_score[valid]
    n = len(yt)
    aucs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yt[idx])) < 2:
            aucs[i] = np.nan
            continue
        try:
            aucs[i] = roc_auc_score(yt[idx], ys[idx])
        except ValueError:
            aucs[i] = np.nan
    aucs = aucs[~np.isnan(aucs)]
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=None):
        t0 = time.time()
        LOG.info("loading %s", SRC_CSV)
        df = pd.read_csv(SRC_CSV)
        df = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        y = (df["category"] == "P_LP").astype(int).to_numpy()
        groups = df["gene"].to_numpy()
        LOG.info("train rows: %d (P_LP=%d, B_LB=%d, genes=%d)",
                 len(df), int(y.sum()), int((1 - y).sum()), len(np.unique(groups)))

        rows = []
        per_layer_summary = {"jsd": {}, "cos": {}}
        n_train_strat = int(len(df) * (N_FOLDS - 1) / N_FOLDS)
        n_test_strat = len(df) - n_train_strat

        # ---- per-layer single-feature models ----
        for lens in ("jsd", "cos"):
            for l in range(N_LAYERS):
                col = f"dD_{lens}_{l}"
                X = df[[col]].to_numpy(dtype=float)

                # stratified
                p, py, fold_aurocs = cv_pooled_strat(X, y)
                m, lo, hi = bootstrap_ci(py, p)
                rows.append({
                    "lens": lens, "layer": l, "cv_scheme": "stratified_10fold",
                    "auroc_mean": m, "ci_low": lo, "ci_high": hi,
                    "n_train": n_train_strat, "n_test": n_test_strat,
                })
                per_layer_summary[lens][l] = {"strat": m}

                # LOGO
                p2, py2, fold_aurocs2 = cv_pooled_logo(X, y, groups)
                m2, lo2, hi2 = bootstrap_ci(py2, p2)
                # n_train/n_test for LOGO: report mean held-out fold size
                from sklearn.model_selection import LeaveOneGroupOut
                logo = LeaveOneGroupOut()
                fold_test_sizes = [int(len(te)) for _, te in logo.split(X, y, groups)]
                rows.append({
                    "lens": lens, "layer": l, "cv_scheme": "logo",
                    "auroc_mean": m2, "ci_low": lo2, "ci_high": hi2,
                    "n_train": int(len(df) - np.mean(fold_test_sizes)),
                    "n_test": int(np.mean(fold_test_sizes)),
                })
                per_layer_summary[lens][l]["logo"] = m2
                LOG.info("%s layer %2d  strat AUROC=%.4f [%.4f,%.4f]  LOGO AUROC=%.4f [%.4f,%.4f]",
                         lens, l, m, lo, hi, m2, lo2, hi2)

        # ---- 32-d vector model (reference) ----
        vec_strat = {}
        vec_logo = {}
        for lens in ("jsd", "cos"):
            X = np.stack([df[f"dD_{lens}_{l}"].to_numpy(dtype=float)
                          for l in range(N_LAYERS)], axis=1)
            p, py, fa = cv_pooled_strat(X, y)
            m, lo, hi = bootstrap_ci(py, p)
            vec_strat[lens] = {"auroc_mean": m, "ci_low": lo, "ci_high": hi,
                                "fold_mean": float(np.mean(fa))}
            LOG.info("VECTOR %s 32-d  strat AUROC=%.4f [%.4f,%.4f]  fold_mean=%.4f",
                     lens, m, lo, hi, float(np.mean(fa)))
            p2, py2, fa2 = cv_pooled_logo(X, y, groups)
            m2, lo2, hi2 = bootstrap_ci(py2, p2)
            vec_logo[lens] = {"auroc_mean": m2, "ci_low": lo2, "ci_high": hi2,
                              "fold_mean": float(np.mean(fa2))}
            LOG.info("VECTOR %s 32-d  LOGO AUROC=%.4f [%.4f,%.4f]  fold_mean=%.4f",
                     lens, m2, lo2, hi2, float(np.mean(fa2)))

        # write CSV
        out_csv = PHASE_OUT_DIR / "per_layer_auroc.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        LOG.info("wrote %s (%d rows)", out_csv, len(rows))

        # best layer per lens (use stratified)
        best_jsd_layer = max(per_layer_summary["jsd"].items(),
                              key=lambda kv: kv[1]["strat"])[0]
        best_cos_layer = max(per_layer_summary["cos"].items(),
                              key=lambda kv: kv[1]["strat"])[0]

        # argmax_layer column from CSV — most-common across train rows
        argmax_jsd = int(df["argmax_layer"].mode().iloc[0]) if "argmax_layer" in df.columns else None
        argmax_cos = argmax_jsd  # the column applies to whichever lens drove signed_argmax

        summary = {
            "vector_auroc_jsd": vec_strat["jsd"],
            "vector_auroc_cos": vec_strat["cos"],
            "vector_auroc_jsd_logo": vec_logo["jsd"],
            "vector_auroc_cos_logo": vec_logo["cos"],
            "best_single_layer_jsd": int(best_jsd_layer),
            "best_single_layer_cos": int(best_cos_layer),
            "best_single_layer_jsd_auroc": per_layer_summary["jsd"][best_jsd_layer]["strat"],
            "best_single_layer_cos_auroc": per_layer_summary["cos"][best_cos_layer]["strat"],
            "argmax_layer_jsd": argmax_jsd,
            "argmax_layer_cos": argmax_cos,
            "n_train": int(len(df)),
            "p_lp": int(y.sum()),
            "b_lb": int((1 - y).sum()),
            "n_genes": int(len(np.unique(groups))),
            "elapsed_sec": float(time.time() - t0),
        }
        out_summary = PHASE_OUT_DIR / "summary.json"
        out_summary.write_text(json.dumps(summary, indent=2))
        LOG.info("wrote %s", out_summary)
        LOG.info("summary: %s", json.dumps(summary, indent=2))

        ru.write_done(PHASE, PHASE_OUT_DIR, extra={"elapsed_sec": time.time() - t0})


if __name__ == "__main__":
    main()
