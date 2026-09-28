"""T1.2 Comparison pipeline.

Inputs (from previous T1.2 stages):
  - /path/to/gDTR/results/phase3_main/variants_features.csv         (ΔD vectors)
  - /path/to/gDTR/results/tier1_baselines/delta_h_features.csv      (‖Δh‖ 32-d)
  - /path/to/gDTR/results/tier1_baselines/rollout_features.csv      (rollout 5-d × 3 cols)
  - /path/to/gDTR/results/tier1_baselines/ig_features.csv           (IG 1-d × 3 cols)

Pipeline (P_LP + B_LB train rows only):
  1. Train scalar/vector LR for each method on stratified 10-fold + LOGO-CV.
     Report mean AUROC, std, 95% CI, pooled AUROC.
  2. Pairwise Spearman ρ across the 4 method scores using each method's
     out-of-fold prediction as the per-variant score. We use the pooled
     stratified-CV OOF score so all 4 share the same row order.
  3. DeLong test ΔD_cos vs each baseline (paired, two-sided).
  4. Incremental info: residualize ΔD score on each baseline, recompute AUROC
     of the residual.

Outputs:
  /path/to/gDTR/results/tier1_baselines/baseline_auroc.json
  /path/to/gDTR/results/tier1_baselines/baseline_spearman.csv
  /path/to/gDTR/results/tier1_baselines/delong_pairs.csv
  /path/to/gDTR/results/tier1_baselines/_done                       (final marker)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sst
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "tier1_baselines_compare"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_baselines"
LOG = ru.setup_logging(PHASE)

PHASE3_CSV = ru.GDTR_ROOT / "results" / "phase3_main" / "variants_features.csv"
DELTA_H_CSV = PHASE_OUT_DIR / "delta_h_features.csv"
ROLLOUT_CSV = PHASE_OUT_DIR / "rollout_features.csv"
IG_CSV = PHASE_OUT_DIR / "ig_features.csv"

OUT_AUROC = PHASE_OUT_DIR / "baseline_auroc.json"
OUT_SPEAR = PHASE_OUT_DIR / "baseline_spearman.csv"
OUT_DELONG = PHASE_OUT_DIR / "delong_pairs.csv"
OUT_DONE = PHASE_OUT_DIR / "_done"

SEED = 42
N_FOLDS = 10
N_LAYERS = 32
ATTN_LAYERS = [3, 10, 17, 24, 31]
TRAIN_CATS = ("P_LP", "B_LB")
GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)


# ---------------- DeLong (copied from 33_phase3_ensemble.py) -----------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, score1, score2):
    y = np.asarray(y_true).astype(int)
    s1 = np.asarray(score1, dtype=np.float64)
    s2 = np.asarray(score2, dtype=np.float64)
    order = np.argsort(-y)
    y_sorted = y[order]
    label_1_count = int(y_sorted.sum())
    preds = np.vstack([s1[order], s2[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - sst.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ---------------- Data loading ----------------------------------------------

def variant_key(df: pd.DataFrame) -> pd.Index:
    return df.apply(lambda r: f"{r['chrom']}:{r['pos']}:{r['ref']}>{r['alt']}", axis=1)


def load_features() -> pd.DataFrame:
    LOG.info("loading phase3_main features ...")
    p3 = pd.read_csv(PHASE3_CSV)
    LOG.info("phase3_main rows: %d", len(p3))

    LOG.info("loading delta_h ...")
    dh = pd.read_csv(DELTA_H_CSV)
    LOG.info("delta_h rows: %d", len(dh))

    LOG.info("loading rollout ...")
    rl = pd.read_csv(ROLLOUT_CSV)
    LOG.info("rollout rows: %d", len(rl))

    LOG.info("loading IG ...")
    ig = pd.read_csv(IG_CSV)
    LOG.info("ig rows: %d", len(ig))

    p3["_key"] = variant_key(p3)
    dh["_key"] = variant_key(dh)
    rl["_key"] = variant_key(rl)
    ig["_key"] = variant_key(ig)

    # Inner-join all four on _key
    join_cols_p3 = ["_key"] + [f"dD_cos_{i}" for i in range(N_LAYERS)] \
                   + ["chrom", "pos", "ref", "alt", "gene", "category", "stars"]
    join_cols_dh = ["_key"] + [f"delta_h_norm_{i}" for i in range(N_LAYERS)]
    join_cols_rl = ["_key"] + [f"rollout_{e}_delta" for e in ATTN_LAYERS]
    join_cols_ig = ["_key"] + ["ig_score_delta"]

    df = p3[join_cols_p3].merge(dh[join_cols_dh], on="_key", how="inner")
    df = df.merge(rl[join_cols_rl], on="_key", how="inner")
    df = df.merge(ig[join_cols_ig], on="_key", how="inner")
    LOG.info("after inner-join: %d", len(df))
    return df


# ---------------- Modelling utilities ----------------------------------------

def _prepare_xy(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_train = df[df["category"].isin(TRAIN_CATS)].copy()
    X = df_train[feat_cols].values.astype(np.float64)
    y = (df_train["category"] == "P_LP").astype(int).values
    groups = df_train["gene"].values
    return X, y, groups


def stratified_kfold_oof(
    X: np.ndarray, y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED,
) -> Tuple[np.ndarray, dict]:
    """Run stratified-kfold LR; return (OOF score [n], summary dict)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float64)
    fold_aurocs = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(Xtr, y[tr])
        score = clf.predict_proba(Xte)[:, 1]
        oof[te] = score
        try:
            fold_aurocs.append(float(roc_auc_score(y[te], score)))
        except ValueError:
            fold_aurocs.append(float("nan"))
    pooled = float(roc_auc_score(y, oof))
    mean = float(np.nanmean(fold_aurocs))
    std = float(np.nanstd(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(max(len(fold_aurocs), 1))
    return oof, {
        "split": "stratified_kfold",
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "fold_aurocs": fold_aurocs,
        "mean_auroc": mean,
        "std_auroc": std,
        "ci95_lo": mean - ci,
        "ci95_hi": mean + ci,
        "pooled_auroc": pooled,
    }


def logo_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = SEED) -> dict:
    """Leave-one-gene-out CV. Per-gene AUROC is None if no two-class split."""
    uniq = sorted(np.unique(groups).tolist())
    per_gene = {}
    fold_aurocs = []
    pooled_y = []; pooled_p = []
    for g in uniq:
        tr = groups != g
        te = groups == g
        if y[te].sum() == 0 or y[te].sum() == y[te].shape[0]:
            per_gene[g] = {"auroc": None, "n": int(te.sum())}
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        try:
            au = float(roc_auc_score(y[te], p))
        except ValueError:
            au = None
        per_gene[g] = {"auroc": au, "n": int(te.sum())}
        if au is not None:
            fold_aurocs.append(au)
        pooled_y.append(y[te]); pooled_p.append(p)
    pooled_y = np.concatenate(pooled_y); pooled_p = np.concatenate(pooled_p)
    pooled = float(roc_auc_score(pooled_y, pooled_p)) if pooled_y.size else float("nan")
    mean = float(np.nanmean(fold_aurocs)) if fold_aurocs else float("nan")
    std = float(np.nanstd(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(max(len(fold_aurocs), 1))
    return {
        "split": "logo",
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "fold_aurocs": fold_aurocs,
        "mean_auroc": mean,
        "std_auroc": std,
        "ci95_lo": mean - ci if not np.isnan(mean) else None,
        "ci95_hi": mean + ci if not np.isnan(mean) else None,
        "pooled_auroc": pooled,
        "per_gene": per_gene,
    }


def main() -> None:
    PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    df = load_features()
    df_tr = df[df["category"].isin(TRAIN_CATS)].copy().reset_index(drop=True)
    LOG.info("train rows (P_LP + B_LB): %d  (P_LP=%d B_LB=%d)",
             len(df_tr),
             int((df_tr["category"] == "P_LP").sum()),
             int((df_tr["category"] == "B_LB").sum()))

    method_specs = {
        "A_dD_cos": [f"dD_cos_{i}" for i in range(N_LAYERS)],
        "B_delta_h": [f"delta_h_norm_{i}" for i in range(N_LAYERS)],
        "C_rollout": [f"rollout_{e}_delta" for e in ATTN_LAYERS],
        "D_ig": ["ig_score_delta"],
    }

    results_strat = {}
    results_logo = {}
    oof_scores = {}
    y_train = (df_tr["category"] == "P_LP").astype(int).values
    groups = df_tr["gene"].values

    for name, cols in method_specs.items():
        X = df_tr[cols].values.astype(np.float64)
        # NaN guard: drop any NaN rows from this method only
        nan_mask = ~np.isnan(X).any(axis=1)
        if not nan_mask.all():
            LOG.warning("%s: dropping %d rows with NaN", name, int((~nan_mask).sum()))
        X_use = X[nan_mask]; y_use = y_train[nan_mask]; g_use = groups[nan_mask]
        oof, stats_strat = stratified_kfold_oof(X_use, y_use)
        stats_strat["name"] = name
        stats_logo = logo_cv(X_use, y_use, g_use)
        stats_logo["name"] = name
        results_strat[name] = stats_strat
        results_logo[name] = stats_logo
        # Embed OOF back into full-length array (NaN where dropped)
        oof_full = np.full(len(df_tr), np.nan, dtype=np.float64)
        oof_full[nan_mask] = oof
        oof_scores[name] = oof_full
        LOG.info("%s: stratified mean AUROC=%.4f (CI %.4f-%.4f) | LOGO mean=%.4f",
                 name, stats_strat["mean_auroc"], stats_strat["ci95_lo"],
                 stats_strat["ci95_hi"], stats_logo["mean_auroc"])

    # ----- Spearman matrix -----
    methods = list(method_specs.keys())
    n = len(methods)
    rho = np.zeros((n, n))
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            si = oof_scores[mi]; sj = oof_scores[mj]
            valid = (~np.isnan(si)) & (~np.isnan(sj))
            r, _ = sst.spearmanr(si[valid], sj[valid])
            rho[i, j] = float(r)
    spear_df = pd.DataFrame(rho, index=methods, columns=methods)
    spear_df.to_csv(OUT_SPEAR, float_format="%.4f")
    LOG.info("Spearman matrix:\n%s", spear_df.to_string())

    # ----- DeLong vs ΔD_cos -----
    rows_dl = []
    s_a = oof_scores["A_dD_cos"]
    for name in methods:
        if name == "A_dD_cos":
            continue
        s_b = oof_scores[name]
        valid = (~np.isnan(s_a)) & (~np.isnan(s_b))
        au_a, au_b, z, p = delong_roc_test(y_train[valid], s_a[valid], s_b[valid])
        rows_dl.append({
            "feat_a": "A_dD_cos",
            "feat_b": name,
            "auc_a": au_a, "auc_b": au_b,
            "delta": au_a - au_b,
            "z": z, "p_value": p,
            "n": int(valid.sum()),
        })
        LOG.info("DeLong A vs %s: AUC %.4f vs %.4f  z=%.2f  p=%.3g",
                 name, au_a, au_b, z, p)

    # ----- Incremental info: residualize ΔD on each baseline -----
    incr_rows = []
    for name in methods:
        if name == "A_dD_cos":
            continue
        s_a = oof_scores["A_dD_cos"]; s_b = oof_scores[name]
        valid = (~np.isnan(s_a)) & (~np.isnan(s_b))
        # Linear regression: s_a = b0 + b1 * s_b + e ; use residuals as new score
        sb = s_b[valid]; sa = s_a[valid]
        b1, b0 = np.polyfit(sb, sa, 1)
        resid = sa - (b0 + b1 * sb)
        au_resid = float(roc_auc_score(y_train[valid], resid))
        incr_rows.append({
            "baseline": name,
            "auroc_residual_dD_minus_baseline": au_resid,
            "auroc_dD_alone": float(roc_auc_score(y_train[valid], sa)),
            "auroc_baseline_alone": float(roc_auc_score(y_train[valid], sb)),
            "n": int(valid.sum()),
        })

    pd.DataFrame(rows_dl).to_csv(OUT_DELONG, index=False, float_format="%.6g")

    out = {
        "phase": PHASE,
        "config": {"n_folds": N_FOLDS, "seed": SEED,
                   "method_specs": {k: len(v) for k, v in method_specs.items()}},
        "n_train_variants": int(len(df_tr)),
        "results_stratified": results_strat,
        "results_logo": results_logo,
        "delong_pairs": rows_dl,
        "incremental_info": incr_rows,
        "wall_time_sec": float(time.time() - t_start),
    }
    OUT_AUROC.write_text(json.dumps(out, indent=2))
    OUT_DONE.write_text(json.dumps({"ok": True, "n": int(len(df_tr))}, indent=2))
    LOG.info("compare pipeline done in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
