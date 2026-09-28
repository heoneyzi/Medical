"""Phase 3 ensemble — full 4-axis ensemble (gDTR ΔD_cos + Evo 2 LL + CADD + AlphaMissense)
plus DeLong tests for paired AUROC comparisons.

Reads the existing variants_features.csv from phase3_main, augments it with CADD and
AlphaMissense scores fetched from public sources, runs 10-fold stratified CV +
leave-one-gene-out for ten feature combinations, and reports DeLong p-values
for the central paper claims.

Outputs to /path/to/gDTR/results/phase3_ensemble/:
  - variants_features_full.csv       (10,910 x 80+ cols)
  - ensemble_results.json            (full AUROC + 95% CI + DeLong table)
  - F_phase3_ensemble_auroc.{pdf,png}
  - F_phase3_delong_table.{pdf,png}
  - _done

Usage: python scripts/33_phase3_ensemble.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import _runner_utils as ru

ru.add_repo_paths()

PHASE = "phase3_ensemble"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase3_ensemble"
SRC_CSV = ru.GDTR_ROOT / "results" / "phase3_main" / "variants_features.csv"
LOG = ru.setup_logging(PHASE)

CADD_URL = "https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh38/whole_genome_SNVs.tsv.gz"
AM_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
AM_LOCAL = ru.GDTR_ROOT / "data" / "annotation" / "AlphaMissense_hg38.tsv.gz"
AM_FILTERED = ru.GDTR_ROOT / "data" / "annotation" / "AlphaMissense_hg38.filtered.tsv.gz"

SEED = 42
N_FOLDS = 10
N_LAYERS = 32
REGION_PAD = 10_000  # ±10 kb around gene span — captures all variants


# -------------- DeLong test (Sun & Xu 2014 fast version) --------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank transform handling ties (DeLong 1988)."""
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
    """Fast DeLong (O(N log N)). Returns AUC array and covariance matrix.

    predictions_sorted_transposed: shape (k, n) — k models, n samples.
    label_1_count: number of positive samples; positives must be the FIRST
    label_1_count columns.
    """
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


def delong_roc_test(y_true: np.ndarray, score1: np.ndarray, score2: np.ndarray):
    """Two-sided DeLong test for two paired ROC curves.

    Returns (auc1, auc2, z_stat, p_value).
    """
    from scipy import stats as sst
    y = np.asarray(y_true).astype(int)
    s1 = np.asarray(score1, dtype=np.float64)
    s2 = np.asarray(score2, dtype=np.float64)
    order = np.argsort(-y)  # positives first
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


# ----------------------- CADD lookup via remote tabix ----------------------
def fetch_cadd_for_variants(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fetch CADD raw + PHRED for every (chrom, pos, ref, alt). Strategy:
    for each (chrom, gene), fetch the gene region once with pysam.TabixFile,
    then build a dict keyed by (chrom, pos, ref, alt).
    """
    import pysam
    log.info("Opening remote CADD tabix at %s", CADD_URL)
    tbx = pysam.TabixFile(CADD_URL)

    # Build per-(chrom,gene) query intervals
    cadd_raw = np.full(len(df), np.nan, dtype=np.float64)
    cadd_phred = np.full(len(df), np.nan, dtype=np.float64)
    df = df.reset_index(drop=True).copy()

    # Group by (chrom, gene) so we minimize the number of remote ranges
    grouped = df.groupby(["chrom", "gene"])
    n_groups = len(grouped)
    for gi, ((chrom, gene), sub) in enumerate(grouped):
        chrom_str = str(chrom)
        start = max(int(sub["pos"].min()) - REGION_PAD, 1)
        end = int(sub["pos"].max()) + REGION_PAD
        # Build local lookup dict
        lookup = {}
        try:
            for ln in tbx.fetch(chrom_str, start, end):
                # CADD whole-genome SNVs columns: chrom, pos, ref, alt, raw, phred
                f = ln.split("\t")
                if len(f) < 6:
                    continue
                key = (f[0], int(f[1]), f[2], f[3])
                try:
                    lookup[key] = (float(f[4]), float(f[5]))
                except ValueError:
                    continue
        except Exception as e:
            log.warning("CADD fetch failed for %s gene=%s region=%d-%d: %s",
                        chrom_str, gene, start, end, e)
            continue
        # Now match each variant
        for idx in sub.index:
            chrom_v = str(df.at[idx, "chrom"])
            pos_v = int(df.at[idx, "pos"])
            ref_v = str(df.at[idx, "ref"])
            alt_v = str(df.at[idx, "alt"])
            v = lookup.get((chrom_v, pos_v, ref_v, alt_v))
            if v is not None:
                cadd_raw[idx] = v[0]
                cadd_phred[idx] = v[1]
        n_hit = int(np.isfinite(cadd_phred[sub.index.to_numpy()]).sum())
        log.info("[CADD %d/%d] chr%s gene=%s region=%d-%d (n=%d) hit=%d",
                 gi + 1, n_groups, chrom_str, gene, start, end, len(sub), n_hit)

    df["cadd_raw"] = cadd_raw
    df["cadd_phred"] = cadd_phred
    return df


# ----------------------- AlphaMissense download + filter -------------------
def _download_alphamissense(log) -> Path:
    """Download AM file (643 MB) once; skip if AM_FILTERED already exists."""
    AM_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if AM_FILTERED.exists() and AM_FILTERED.stat().st_size > 1000:
        log.info("Filtered AM already exists at %s", AM_FILTERED)
        return AM_FILTERED
    if not AM_LOCAL.exists() or AM_LOCAL.stat().st_size < 100_000_000:
        log.info("Downloading AlphaMissense to %s ...", AM_LOCAL)
        import subprocess
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "-o", str(AM_LOCAL), AM_URL],
            check=True,
        )
        log.info("Download complete: %d bytes", AM_LOCAL.stat().st_size)
    return AM_LOCAL


def fetch_alphamissense_for_variants(df: pd.DataFrame, log) -> pd.DataFrame:
    """Build a (chrom,pos,ref,alt) -> (am_pathogenicity, am_class) dict by
    streaming the AM file once (gzip), filtering to our 10 chroms.
    """
    import gzip
    AM_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not AM_FILTERED.exists() or AM_FILTERED.stat().st_size < 1000:
        _download_alphamissense(log)
        chroms_keep = set(f"chr{c}" for c in sorted(df.chrom.unique()))
        log.info("Filtering AM to chroms %s -> %s", sorted(chroms_keep), AM_FILTERED)
        n_in = 0
        n_kept = 0
        with gzip.open(AM_LOCAL, "rt") as fin, gzip.open(AM_FILTERED, "wt") as fout:
            for line in fin:
                if line.startswith("#"):
                    fout.write(line)
                    continue
                n_in += 1
                if n_in % 5_000_000 == 0:
                    log.info("  AM stream: %d lines, %d kept", n_in, n_kept)
                fld = line[:line.find("\t")]
                if fld in chroms_keep:
                    fout.write(line)
                    n_kept += 1
        log.info("AM filter done: in=%d kept=%d", n_in, n_kept)

    # Build lookup dict from filtered file
    log.info("Building AM lookup from %s", AM_FILTERED)
    lookup: Dict[Tuple[int, int, str, str], Tuple[float, str]] = {}
    n_lines = 0
    with __import__("gzip").open(AM_FILTERED, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 10:
                continue
            try:
                ch_int = int(f[0].replace("chr", ""))
            except ValueError:
                continue
            try:
                key = (ch_int, int(f[1]), f[2], f[3])
                lookup[key] = (float(f[8]), f[9])
            except ValueError:
                continue
            n_lines += 1
    log.info("AM lookup built: %d entries (%d lines)", len(lookup), n_lines)

    am_score = np.full(len(df), np.nan, dtype=np.float64)
    am_class = np.full(len(df), "", dtype=object)
    df = df.reset_index(drop=True).copy()
    for idx in range(len(df)):
        key = (int(df.at[idx, "chrom"]), int(df.at[idx, "pos"]),
               str(df.at[idx, "ref"]), str(df.at[idx, "alt"]))
        v = lookup.get(key)
        if v is not None:
            am_score[idx] = v[0]
            am_class[idx] = v[1]
    df["am_pathogenicity"] = am_score
    df["am_class"] = am_class
    return df


# ----------------------- CV + AUROC ----------------------------------------
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


def cv_stratified_pooled(X, y, name):
    """Return per-fold AUROCs + pooled (out-of-fold) scores so DeLong can be
    run on paired predictions across feature sets."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aurocs = []
    pooled_scores = np.full(len(y), np.nan, dtype=np.float64)
    pooled_y = np.full(len(y), -1, dtype=int)
    for tr, te in skf.split(X, y):
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        aurocs.append(roc_auc_score(y[te], s))
        pooled_scores[te] = s
        pooled_y[te] = y[te]
    fpr, tpr, _ = roc_curve(pooled_y, pooled_scores)
    mean_au = float(np.mean(aurocs))
    std_au = float(np.std(aurocs, ddof=1))
    se = std_au / np.sqrt(N_FOLDS)
    return {
        "name": name,
        "split": "stratified_kfold",
        "fold_aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_au,
        "std_auroc": std_au,
        "ci95_lo": float(mean_au - 1.96 * se),
        "ci95_hi": float(mean_au + 1.96 * se),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "pooled_auroc": float(roc_auc_score(pooled_y, pooled_scores)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "_pooled_scores": pooled_scores,  # for DeLong
        "_pooled_y": pooled_y,
    }


def cv_logo_pooled(X, y, groups, name):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, roc_curve
    logo = LeaveOneGroupOut()
    aurocs = []
    pooled_scores = np.full(len(y), np.nan, dtype=np.float64)
    pooled_y = np.full(len(y), -1, dtype=int)
    per_gene = {}
    for tr, te in logo.split(X, y, groups):
        gene_held = str(groups[te][0])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            per_gene[gene_held] = {"auroc": None, "n": int(len(te))}
            continue
        s, _ = _fit_pred_logreg(X[tr], y[tr], X[te])
        au = float(roc_auc_score(y[te], s))
        aurocs.append(au)
        pooled_scores[te] = s
        pooled_y[te] = y[te]
        per_gene[gene_held] = {"auroc": au, "n": int(len(te))}
    if not aurocs:
        return {"name": name, "split": "logo", "mean_auroc": float("nan"),
                "per_gene": per_gene, "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0])}
    valid = pooled_y >= 0
    yp = pooled_y[valid]
    sp = pooled_scores[valid]
    fpr, tpr, _ = roc_curve(yp, sp)
    mean_au = float(np.mean(aurocs))
    std_au = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0
    se = std_au / max(np.sqrt(len(aurocs)), 1.0)
    return {
        "name": name,
        "split": "logo",
        "fold_aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_au,
        "std_auroc": std_au,
        "ci95_lo": float(mean_au - 1.96 * se),
        "ci95_hi": float(mean_au + 1.96 * se),
        "per_gene": per_gene,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "pooled_auroc": float(roc_auc_score(yp, sp)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "_pooled_scores": pooled_scores,
        "_pooled_y": pooled_y,
    }


def build_feature_blocks(df: pd.DataFrame):
    feat_cos = np.stack([df[f"dD_cos_{l}"].to_numpy(dtype=float)
                         for l in range(N_LAYERS)], axis=1)
    feat_evo = df[["evo2_delta_loglik"]].to_numpy(dtype=float)
    feat_cadd = df[["cadd_phred"]].to_numpy(dtype=float)
    feat_am = df[["am_pathogenicity"]].to_numpy(dtype=float)

    blocks = {
        "A_dD_cos": feat_cos,
        "B_evo2_loglik": feat_evo,
        "C_cadd_phred": feat_cadd,
        "D_am_score": feat_am,
        "AB_cos_evo2": np.concatenate([feat_cos, feat_evo], axis=1),
        "AC_cos_cadd": np.concatenate([feat_cos, feat_cadd], axis=1),
        "AD_cos_am": np.concatenate([feat_cos, feat_am], axis=1),
        "ABCD_full": np.concatenate([feat_cos, feat_evo, feat_cadd, feat_am], axis=1),
        "BCD_baseline": np.concatenate([feat_evo, feat_cadd, feat_am], axis=1),
        "CD_cadd_am": np.concatenate([feat_cadd, feat_am], axis=1),
    }
    return blocks


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=None):
        t0 = time.time()
        if not SRC_CSV.exists():
            raise SystemExit(f"missing input: {SRC_CSV}")
        df = pd.read_csv(SRC_CSV)
        LOG.info("loaded variants_features.csv: %d x %d", *df.shape)

        # ---- 1) CADD ----
        LOG.info("=== Phase A: CADD lookup ===")
        df = fetch_cadd_for_variants(df, LOG)

        # ---- 2) AlphaMissense ----
        LOG.info("=== Phase B: AlphaMissense lookup ===")
        df = fetch_alphamissense_for_variants(df, LOG)

        # ---- 2.5) Save augmented features ----
        out_csv = PHASE_OUT_DIR / "variants_features_full.csv"
        df.to_csv(out_csv, index=False)
        LOG.info("wrote %s (%d x %d)", out_csv, *df.shape)

        # Coverage diagnostics
        coverage = {}
        for cat in df["category"].unique():
            sub = df[df["category"] == cat]
            coverage[str(cat)] = {
                "n": int(len(sub)),
                "cadd_match_frac": float(np.isfinite(sub["cadd_phred"]).mean()),
                "am_match_frac": float(np.isfinite(sub["am_pathogenicity"]).mean()),
            }
        LOG.info("Coverage: %s", json.dumps(coverage, indent=2))

        # ---- 3) Train subset (P/LP + B/LB only) ----
        df_train = df[df["category"].isin(["P_LP", "B_LB"])].reset_index(drop=True)
        LOG.info("training rows: %d (P_LP=%d, B_LB=%d)", len(df_train),
                 (df_train["category"] == "P_LP").sum(),
                 (df_train["category"] == "B_LB").sum())

        # ---- 4) Build feature blocks; impute NaN with median (per col, on train) ----
        blocks = build_feature_blocks(df_train)
        # Impute NaN in CADD/AM using training-set median (column-wise)
        for name, X in blocks.items():
            if np.isnan(X).any():
                col_med = np.nanmedian(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_med, inds[1])
                blocks[name] = X

        y_train = (df_train["category"] == "P_LP").astype(int).to_numpy()
        groups_train = df_train["gene"].to_numpy()

        # ---- 5) Stratified 10-fold CV ----
        LOG.info("=== Phase C: Stratified 10-fold CV ===")
        results_strat: Dict[str, dict] = {}
        for name, X in blocks.items():
            r = cv_stratified_pooled(X, y_train, name)
            results_strat[name] = r
            LOG.info("[strat] %-18s d=%2d AUROC=%.4f CI=[%.4f, %.4f]",
                     name, X.shape[1], r["mean_auroc"], r["ci95_lo"], r["ci95_hi"])

        # ---- 6) LOGO CV ----
        LOG.info("=== Phase D: LOGO CV ===")
        results_logo: Dict[str, dict] = {}
        for name, X in blocks.items():
            r = cv_logo_pooled(X, y_train, groups_train, name)
            results_logo[name] = r
            LOG.info("[LOGO]  %-18s AUROC=%.4f CI=[%.4f, %.4f]",
                     name, r.get("mean_auroc", float("nan")),
                     r.get("ci95_lo", float("nan")), r.get("ci95_hi", float("nan")))

        # ---- 7) DeLong tests on stratified pooled scores ----
        LOG.info("=== Phase E: DeLong tests ===")
        comparisons = [
            ("ABCD_full", "BCD_baseline"),  # central claim — ΔD adds info
            ("A_dD_cos", "B_evo2_loglik"),
            ("A_dD_cos", "C_cadd_phred"),
            ("A_dD_cos", "D_am_score"),
            ("ABCD_full", "A_dD_cos"),
            ("AB_cos_evo2", "A_dD_cos"),
            ("AC_cos_cadd", "A_dD_cos"),
            ("AD_cos_am", "A_dD_cos"),
            ("ABCD_full", "AB_cos_evo2"),
            ("ABCD_full", "AC_cos_cadd"),
            ("ABCD_full", "AD_cos_am"),
        ]
        delong_strat = []
        for a, b in comparisons:
            ra = results_strat[a]; rb = results_strat[b]
            mask = (ra["_pooled_y"] == rb["_pooled_y"]) & (ra["_pooled_y"] >= 0)
            if mask.sum() < 10:
                LOG.warning("not enough overlap for DeLong %s vs %s", a, b)
                continue
            au_a, au_b, z, p = delong_roc_test(
                ra["_pooled_y"][mask], ra["_pooled_scores"][mask],
                rb["_pooled_scores"][mask],
            )
            delong_strat.append({
                "feat_a": a, "feat_b": b,
                "auroc_a": au_a, "auroc_b": au_b,
                "delta": au_a - au_b,
                "z": z, "p_value": p,
                "n_samples": int(mask.sum()),
            })
            LOG.info("DeLong[strat] %s vs %s  AUROC %.4f vs %.4f  Δ=%+.4f  z=%.3f  p=%.4g",
                     a, b, au_a, au_b, au_a - au_b, z, p)

        # Repeat DeLong on LOGO pooled
        delong_logo = []
        for a, b in comparisons:
            ra = results_logo[a]; rb = results_logo[b]
            if "_pooled_scores" not in ra or "_pooled_scores" not in rb:
                continue
            mask = (ra["_pooled_y"] == rb["_pooled_y"]) & (ra["_pooled_y"] >= 0)
            if mask.sum() < 10:
                continue
            au_a, au_b, z, p = delong_roc_test(
                ra["_pooled_y"][mask], ra["_pooled_scores"][mask],
                rb["_pooled_scores"][mask],
            )
            delong_logo.append({
                "feat_a": a, "feat_b": b,
                "auroc_a": au_a, "auroc_b": au_b,
                "delta": au_a - au_b,
                "z": z, "p_value": p,
                "n_samples": int(mask.sum()),
            })

        # ---- 8) Save ensemble_results.json ----
        def _strip(r):
            return {k: v for k, v in r.items() if not k.startswith("_")
                    and k not in ("fpr", "tpr")}

        out = {
            "phase": PHASE,
            "config": {"seed": SEED, "n_folds": N_FOLDS, "n_layers": N_LAYERS},
            "n_train": int(len(df_train)),
            "category_counts": {
                k: int(v) for k, v in df["category"].value_counts().items()
            },
            "coverage": coverage,
            "results_stratified": {k: _strip(v) for k, v in results_strat.items()},
            "results_logo": {
                k: {kk: vv for kk, vv in _strip(v).items() if kk != "per_gene"}
                | {"per_gene": v.get("per_gene", {})}
                for k, v in results_logo.items()
            },
            "delong_stratified": delong_strat,
            "delong_logo": delong_logo,
            "wall_time_sec": float(time.time() - t0),
        }
        json_path = PHASE_OUT_DIR / "ensemble_results.json"
        json_path.write_text(json.dumps(out, indent=2, default=float))
        LOG.info("wrote %s", json_path)

        # ---- 9) Plots ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 9a) ROC overlay (stratified) for all 10 features
            fig, ax = plt.subplots(figsize=(8, 7))
            color_map = plt.cm.tab10(np.linspace(0, 1, len(results_strat)))
            for (name, r), c in zip(results_strat.items(), color_map):
                ax.plot(r["fpr"], r["tpr"], color=c, lw=1.4,
                        label=f"{name} ({r['mean_auroc']:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"Phase 3 ensemble ROC — stratified 10-fold "
                         f"(n={len(df_train)})")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_phase3_ensemble_auroc.{ext}",
                            dpi=150)
            plt.close(fig)

            # 9b) Bar chart of mean AUROC + 95% CI
            fig, ax = plt.subplots(figsize=(10, 5))
            names = list(results_strat.keys())
            mus = [results_strat[n]["mean_auroc"] for n in names]
            lo = [results_strat[n]["mean_auroc"] - results_strat[n]["ci95_lo"] for n in names]
            hi = [results_strat[n]["ci95_hi"] - results_strat[n]["mean_auroc"] for n in names]
            x = np.arange(len(names))
            ax.bar(x, mus, yerr=[lo, hi], capsize=4, color="steelblue")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right",
                                                  fontsize=9)
            ax.set_ylabel("Mean AUROC ± 95 % CI")
            ax.set_ylim(0.5, 1.0)
            ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3, axis="y")
            ax.set_title("Phase 3 ensemble — stratified 10-fold AUROC")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_phase3_ensemble_bar.{ext}", dpi=150)
            plt.close(fig)

            # 9c) DeLong matrix visualization
            if delong_strat:
                fig, ax = plt.subplots(figsize=(9, 5))
                pairs = [f"{d['feat_a']}\nvs {d['feat_b']}" for d in delong_strat]
                deltas = [d["delta"] for d in delong_strat]
                pvals = [d["p_value"] for d in delong_strat]
                xs = np.arange(len(pairs))
                colors = ["#2ca02c" if p < 0.05 and d > 0
                          else ("#d62728" if p < 0.05 and d < 0 else "#aaaaaa")
                          for d, p in zip(deltas, pvals)]
                ax.bar(xs, deltas, color=colors)
                for i, (d, p) in enumerate(zip(deltas, pvals)):
                    ax.text(i, d + (0.002 if d >= 0 else -0.005),
                            f"p={p:.2g}", ha="center", fontsize=7,
                            va="bottom" if d >= 0 else "top")
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(xs); ax.set_xticklabels(pairs, rotation=45,
                                                       ha="right", fontsize=8)
                ax.set_ylabel("ΔAUROC (a − b)")
                ax.set_title("Phase 3 DeLong tests (stratified pooled)")
                ax.grid(alpha=0.3, axis="y")
                fig.tight_layout()
                for ext in ("pdf", "png"):
                    fig.savefig(PHASE_OUT_DIR / f"F_phase3_delong_table.{ext}", dpi=150)
                plt.close(fig)
            LOG.info("plots saved")
        except Exception as e:  # noqa: BLE001
            LOG.warning("figure failed: %s", e)

        LOG.info("=== DONE in %.1f s ===", time.time() - t0)


if __name__ == "__main__":
    main()
