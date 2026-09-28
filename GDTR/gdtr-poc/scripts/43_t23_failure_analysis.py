"""T2.3 — Failure case analysis for ΔD_cos vector model.

Use stratified 10-fold OOF predicted probabilities. Define misclassification
at threshold = Youden's J optimum (max TPR - FPR on full pooled scores).
Stratify FN (true=P_LP, pred=B_LB) and FP (true=B_LB, pred=P_LP) by:
  1. Gene
  2. Variant type — SNV transitions (A<->G, C<->T), transversions, indels
  3. CADD-disagreement flag: |z(cadd_phred) - z(dD_cos_score)| > 0.5
  4. dD_cos prob quintile

Outputs to /path/to/gDTR/results/tier2_failure/:
  - failure_breakdown.csv
  - youden_threshold.json
  - _done
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _runner_utils as ru

PHASE = "tier2_failure"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier2_failure"
SRC_CSV = ru.GDTR_ROOT / "results" / "phase3_ensemble" / "variants_features_full.csv"
LOG = ru.setup_logging(PHASE)

SEED = 42
N_FOLDS = 10
N_LAYERS = 32


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


def variant_type(ref: str, alt: str) -> str:
    r, a = str(ref).upper(), str(alt).upper()
    if len(r) == 1 and len(a) == 1 and r in "ACGT" and a in "ACGT":
        pair = frozenset((r, a))
        if pair == frozenset(("A", "G")) or pair == frozenset(("C", "T")):
            return "SNV_transition"
        return "SNV_transversion"
    return "indel_or_other"


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=None):
        t0 = time.time()
        df = pd.read_csv(SRC_CSV)
        df = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        y = (df["category"] == "P_LP").astype(int).to_numpy()
        groups = df["gene"].to_numpy()
        X = np.stack([df[f"dD_cos_{l}"].to_numpy(dtype=float)
                      for l in range(N_LAYERS)], axis=1)
        LOG.info("train rows: %d", len(df))

        # OOF predictions
        pooled, py = pooled_oof(X, y)
        valid = (py >= 0) & ~np.isnan(pooled)
        # by construction every row should be covered, but assert
        assert valid.all(), f"some rows uncovered: {(~valid).sum()}"
        scores = pooled
        labels = py.astype(int)

        # Youden's J optimum
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thr = roc_curve(labels, scores)
        j = tpr - fpr
        idx_max = int(np.argmax(j))
        thr_opt = float(thr[idx_max])
        sens = float(tpr[idx_max])
        spec = float(1.0 - fpr[idx_max])
        auc = float(roc_auc_score(labels, scores))
        LOG.info("AUROC=%.4f  Youden_J_threshold=%.4f  sens=%.4f  spec=%.4f",
                 auc, thr_opt, sens, spec)

        pred = (scores >= thr_opt).astype(int)
        is_fn = (labels == 1) & (pred == 0)
        is_fp = (labels == 0) & (pred == 1)
        LOG.info("FN=%d  FP=%d  total=%d", int(is_fn.sum()), int(is_fp.sum()), len(df))

        # Build per-row metadata
        df = df.copy()
        df["score"] = scores
        df["pred"] = pred
        df["is_fn"] = is_fn
        df["is_fp"] = is_fp
        df["vtype"] = [variant_type(r, a) for r, a in zip(df["ref"], df["alt"])]

        # CADD-disagreement flag (z-score normalize cadd_phred and score)
        from scipy.stats import zscore
        cadd = df["cadd_phred"].to_numpy(dtype=float)
        z_cadd = zscore(cadd, nan_policy="omit")
        z_score = zscore(scores, nan_policy="omit")
        # for any nan cadd, mark as nan (we'll skip in stratification)
        diff = np.abs(z_cadd - z_score)
        df["cadd_disagree"] = diff > 0.5
        df["cadd_disagree_str"] = np.where(np.isnan(diff), "nan",
                                            np.where(diff > 0.5, "disagree", "agree"))

        # Quintile of score
        df["score_quintile"] = pd.qcut(df["score"], 5,
                                       labels=["Q1_0-20", "Q2_20-40", "Q3_40-60",
                                               "Q4_60-80", "Q5_80-100"])

        # Build breakdown rows
        rows = []

        def add_stratum(stratum_type: str, stratum: str, sub_idx: np.ndarray):
            n_total = int(sub_idx.sum())
            if n_total == 0:
                return
            sub_fn = int((is_fn & sub_idx).sum())
            sub_fp = int((is_fp & sub_idx).sum())
            n_pos = int((labels[sub_idx] == 1).sum())
            n_neg = int((labels[sub_idx] == 0).sum())
            rows.append({
                "stratum_type": stratum_type,
                "stratum": stratum,
                "n_total": n_total,
                "n_FN": sub_fn,
                "n_FP": sub_fp,
                "fn_rate": float(sub_fn / n_pos) if n_pos else 0.0,
                "fp_rate": float(sub_fp / n_neg) if n_neg else 0.0,
            })

        # 1) Gene
        for gene in sorted(df["gene"].unique()):
            add_stratum("gene", str(gene), (df["gene"] == gene).to_numpy())

        # 2) Variant type
        for vt in sorted(df["vtype"].unique()):
            add_stratum("variant_type", vt, (df["vtype"] == vt).to_numpy())

        # 2b) finer transition split: A-G vs C-T transitions; transversion umbrella
        finer = []
        for r, a in zip(df["ref"], df["alt"]):
            r2, a2 = str(r).upper(), str(a).upper()
            if len(r2) == 1 and len(a2) == 1 and r2 in "ACGT" and a2 in "ACGT":
                pair = frozenset((r2, a2))
                if pair == frozenset(("A", "G")):
                    finer.append("trans_A_G")
                elif pair == frozenset(("C", "T")):
                    finer.append("trans_C_T")
                else:
                    finer.append("transversion")
            else:
                finer.append("indel_or_other")
        df["vtype_fine"] = finer
        for vt in sorted(set(finer)):
            add_stratum("variant_type_fine", vt,
                        (df["vtype_fine"] == vt).to_numpy())

        # 3) CADD-disagreement
        for label in ("agree", "disagree", "nan"):
            mask = (df["cadd_disagree_str"] == label).to_numpy()
            add_stratum("cadd_disagreement", label, mask)

        # 4) Quintile
        for q in df["score_quintile"].cat.categories:
            mask = (df["score_quintile"] == q).to_numpy()
            add_stratum("score_quintile", str(q), mask)

        out_csv = PHASE_OUT_DIR / "failure_breakdown.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        LOG.info("wrote %s (%d rows)", out_csv, len(rows))

        out_thr = PHASE_OUT_DIR / "youden_threshold.json"
        out_thr.write_text(json.dumps({
            "threshold": thr_opt,
            "sensitivity": sens,
            "specificity": spec,
            "youden_j": float(sens + spec - 1.0),
            "auroc": auc,
            "n_FN": int(is_fn.sum()),
            "n_FP": int(is_fp.sum()),
            "n_total": int(len(df)),
        }, indent=2))
        LOG.info("wrote %s", out_thr)

        # log top-3 FN-rate strata (excluding strata with very small n)
        df_strat = pd.DataFrame(rows)
        df_strat_filt = df_strat[df_strat["n_total"] >= 30]
        top_fn = df_strat_filt.sort_values("fn_rate", ascending=False).head(5)
        top_fp = df_strat_filt.sort_values("fp_rate", ascending=False).head(5)
        LOG.info("Top 5 FN-rate strata (n>=30):\n%s", top_fn.to_string(index=False))
        LOG.info("Top 5 FP-rate strata (n>=30):\n%s", top_fp.to_string(index=False))

        ru.write_done(PHASE, PHASE_OUT_DIR, extra={
            "elapsed_sec": time.time() - t0,
            "youden_threshold": thr_opt,
            "n_FN": int(is_fn.sum()),
            "n_FP": int(is_fp.sum()),
        })


if __name__ == "__main__":
    main()
