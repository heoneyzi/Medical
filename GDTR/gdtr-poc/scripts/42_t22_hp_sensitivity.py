"""T2.2 — Hyperparameter sensitivity for ΔD_cos vector model.

Sweep classifier (penalty, C):
  l1: C in [0.1, 1, 10]                 (solver=saga)
  l2: C in [0.1, 1, 10]                 (solver=lbfgs)
  elasticnet (l1_ratio=0.5): C in [0.1, 1, 10]   (solver=saga)
For each combo: stratified 10-fold AUROC mean ± std.

Outputs to /path/to/gDTR/results/tier2_sensitivity/:
  - hp_grid.csv
  - _done
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _runner_utils as ru

PHASE = "tier2_sensitivity"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier2_sensitivity"
SRC_CSV = ru.GDTR_ROOT / "results" / "phase3_ensemble" / "variants_features_full.csv"
LOG = ru.setup_logging(PHASE)

SEED = 42
N_FOLDS = 10
N_LAYERS = 32


def cv_logreg(X, y, penalty, C, l1_ratio=None, solver="lbfgs"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, te in skf.split(X, y):
        kwargs = dict(max_iter=5000, random_state=SEED, solver=solver,
                      penalty=penalty, C=C)
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio
        clf = LogisticRegression(**kwargs)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
            s = pipe.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], s))
    return float(np.mean(aurocs)), float(np.std(aurocs, ddof=1)), aurocs


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=None):
        t0 = time.time()
        df = pd.read_csv(SRC_CSV)
        df = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        y = (df["category"] == "P_LP").astype(int).to_numpy()
        X = np.stack([df[f"dD_cos_{l}"].to_numpy(dtype=float)
                      for l in range(N_LAYERS)], axis=1)
        LOG.info("train rows: %d  d=%d  P_LP=%d  B_LB=%d",
                 len(df), X.shape[1], int(y.sum()), int((1 - y).sum()))

        grid = []
        # l2
        for C in [0.1, 1.0, 10.0]:
            grid.append(("l2", C, None, "lbfgs"))
        # l1
        for C in [0.1, 1.0, 10.0]:
            grid.append(("l1", C, None, "saga"))
        # elasticnet
        for C in [0.1, 1.0, 10.0]:
            grid.append(("elasticnet", C, 0.5, "saga"))

        rows = []
        outliers = []
        for penalty, C, l1r, solver in grid:
            LOG.info("running penalty=%s C=%g l1r=%s solver=%s ...",
                     penalty, C, l1r, solver)
            mean, std, fold_au = cv_logreg(X, y, penalty, C, l1_ratio=l1r,
                                            solver=solver)
            rows.append({
                "penalty": penalty, "C": C,
                "l1_ratio": l1r if l1r is not None else "",
                "auroc_mean": mean, "auroc_std": std,
            })
            LOG.info("  AUROC mean=%.4f  std=%.4f  folds=%s",
                     mean, std,
                     ", ".join(f"{a:.3f}" for a in fold_au))
            if mean < 0.82 or mean > 0.86:
                outliers.append({"penalty": penalty, "C": C, "auroc": mean})

        out_csv = PHASE_OUT_DIR / "hp_grid.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        LOG.info("wrote %s", out_csv)

        all_means = [r["auroc_mean"] for r in rows]
        LOG.info("AUROC range: [%.4f, %.4f]", min(all_means), max(all_means))
        if outliers:
            LOG.warning("Outliers (outside [0.82, 0.86]): %s", outliers)
        else:
            LOG.info("All AUROC in [0.82, 0.86] band — PASS")

        ru.write_done(PHASE, PHASE_OUT_DIR, extra={
            "elapsed_sec": time.time() - t0,
            "auroc_min": float(min(all_means)),
            "auroc_max": float(max(all_means)),
            "n_outliers": len(outliers),
        })


if __name__ == "__main__":
    main()
