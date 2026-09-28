"""T1.4 — Bootstrap stability for AUROC.

For each model in:
  (a) ΔD_cos vector (32-d)
  (b) ΔD_jsd vector (32-d)
  (c) Evo 2 Δ-LL alone (1-d)
  (d) Ensemble: ΔD_cos vector + Evo 2 Δ-LL (33-d)
run stratified 10-fold to get pooled OOF scores, then 1000-bootstrap
the (y_true, y_score) pairs to compute the AUROC distribution.

Outputs to /path/to/gDTR/results/tier1_bootstrap/:
  - auroc_distributions.csv  (model, bootstrap_idx, auroc)
  - summary.json
  - _done

Usage: python scripts/41_t14_bootstrap.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _runner_utils as ru

PHASE = "tier1_bootstrap"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_bootstrap"
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


def pooled_oof(X, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pooled = np.full(len(y), np.nan, dtype=np.float64)
    py = np.full(len(y), -1, dtype=int)
    for tr, te in skf.split(X, y):
        s = _fit_pred_logreg(X[tr], y[tr], X[te])
        pooled[te] = s
        py[te] = y[te]
    return pooled, py


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=None):
        t0 = time.time()
        LOG.info("loading %s", SRC_CSV)
        df = pd.read_csv(SRC_CSV)
        df = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        y = (df["category"] == "P_LP").astype(int).to_numpy()
        LOG.info("train rows: %d (P_LP=%d, B_LB=%d)", len(df), int(y.sum()), int((1 - y).sum()))

        # build feature blocks
        feat_cos = np.stack([df[f"dD_cos_{l}"].to_numpy(dtype=float)
                             for l in range(N_LAYERS)], axis=1)
        feat_jsd = np.stack([df[f"dD_jsd_{l}"].to_numpy(dtype=float)
                             for l in range(N_LAYERS)], axis=1)
        feat_evo = df[["evo2_delta_loglik"]].to_numpy(dtype=float)
        # impute nan in evo2 with column median (training-set)
        if np.isnan(feat_evo).any():
            feat_evo = np.where(np.isnan(feat_evo),
                                np.nanmedian(feat_evo, axis=0), feat_evo)
        feat_ens = np.concatenate([feat_cos, feat_evo], axis=1)

        models = {
            "dD_cos_vec_32d": feat_cos,
            "dD_jsd_vec_32d": feat_jsd,
            "evo2_dLL_1d": feat_evo,
            "ensemble_cos_evo2_33d": feat_ens,
        }

        # collect pooled OOF
        oof = {}
        for name, X in models.items():
            LOG.info("fitting CV for %s d=%d ...", name, X.shape[1])
            p, py = pooled_oof(X, y)
            oof[name] = (p, py)

        # Bootstrap
        from sklearn.metrics import roc_auc_score
        rows = []
        rng = np.random.default_rng(SEED)
        summary = {}
        for name, (p, py) in oof.items():
            valid = ~np.isnan(p) & (py >= 0)
            yt = py[valid]
            ys = p[valid]
            n = len(yt)
            aucs = np.empty(N_BOOT, dtype=np.float64)
            for i in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                if len(np.unique(yt[idx])) < 2:
                    aucs[i] = np.nan
                    continue
                try:
                    aucs[i] = roc_auc_score(yt[idx], ys[idx])
                except ValueError:
                    aucs[i] = np.nan
            valid_aucs = aucs[~np.isnan(aucs)]
            for i, a in enumerate(aucs):
                rows.append({"model": name, "bootstrap_idx": i, "auroc": float(a)})
            point = float(roc_auc_score(yt, ys))
            summary[name] = {
                "point_auroc": point,
                "mean": float(np.mean(valid_aucs)),
                "ci_low": float(np.percentile(valid_aucs, 2.5)),
                "ci_high": float(np.percentile(valid_aucs, 97.5)),
                "std": float(np.std(valid_aucs, ddof=1)),
                "n": int(len(valid_aucs)),
            }
            LOG.info("%s point=%.4f  mean=%.4f  CI=[%.4f,%.4f]  std=%.4f",
                     name, point, summary[name]["mean"],
                     summary[name]["ci_low"], summary[name]["ci_high"],
                     summary[name]["std"])

        out_csv = PHASE_OUT_DIR / "auroc_distributions.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        LOG.info("wrote %s (%d rows)", out_csv, len(rows))

        out_sum = PHASE_OUT_DIR / "summary.json"
        out_sum.write_text(json.dumps(summary, indent=2))
        LOG.info("wrote %s", out_sum)

        # sanity check vs Phase 3 (cos vector should overlap [0.831, 0.857])
        cos_summary = summary["dD_cos_vec_32d"]
        target_lo, target_hi = 0.831, 0.857
        lo_diff = abs(cos_summary["ci_low"] - target_lo)
        hi_diff = abs(cos_summary["ci_high"] - target_hi)
        LOG.info("Phase-3 CI verification: target [%.3f, %.3f] vs got [%.4f, %.4f]"
                 "  lo_diff=%.4f  hi_diff=%.4f",
                 target_lo, target_hi, cos_summary["ci_low"], cos_summary["ci_high"],
                 lo_diff, hi_diff)
        if lo_diff > 0.01 or hi_diff > 0.01:
            LOG.warning("CI deviates by >0.01 from Phase 3 ref — investigate.")

        ru.write_done(PHASE, PHASE_OUT_DIR, extra={"elapsed_sec": time.time() - t0})


if __name__ == "__main__":
    main()
