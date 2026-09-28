"""Legacy ``cell-eval`` fingerprint and fast approximate local metrics.

This module is retained for historical diagnostics only.  It does *not* expose
the VCC 2026 six-component score.  Exact VCC 2026 evaluation delegates to the
official public ``cell-eval2`` package in :mod:`vcc_baselines.vcc2026_eval`.
"""
from __future__ import annotations

import subprocess
import math
from pathlib import Path

import anndata as ad
import pandas as pd

from . import metrics_local
from .config import Config
from .submit import executable, has

# cell-eval metric name -> fingerprint axis (from the strategy doc's table)
FINGERPRINT = {
    "PDS": ["discrimination_score_l1", "discrimination_score_cosine", "pds"],
    "MAE": ["mae", "mae_delta"],
    "MSE": ["mse", "mse_delta"],
    "PearsonDelta": ["pearson_delta"],
    "DirMatch": ["de_direction_match", "direction_match"],
    "OverlapN": ["overlap_at_N", "overlap_at_100"],
}

# Keep exactly the six public fingerprint ingredients while using the `full`
# pipeline. This avoids spending most runtime on unrelated cutoff variants,
# clustering, and AUC metrics during a large method matrix.
SIX_ONLY_SKIP = [
    "overlap_at_50", "overlap_at_100", "overlap_at_200", "overlap_at_500",
    "precision_at_N", "precision_at_50", "precision_at_100", "precision_at_200",
    "precision_at_500", "de_spearman_sig", "de_spearman_lfc_sig",
    "de_sig_genes_recall", "de_nsig_counts", "pr_auc", "roc_auc",
    "mse_delta", "mae_delta", "discrimination_score_l2",
    "discrimination_score_cosine", "pearson_edistance", "clustering_agreement",
]


def evaluate_local(pred: ad.AnnData, real: ad.AnnData, cfg: Config) -> dict[str, float]:
    return metrics_local.compute(pred, real, pert_col=cfg.data.pert_col,
                                 control_label=cfg.data.control_label)


def evaluate_cell_eval(pred_h5ad, real_h5ad, cfg: Config, outdir, profile=None, num_threads=None) -> dict[str, float]:
    if not has("cell-eval"):
        raise RuntimeError("cell-eval not found (`pip install cell-eval`).")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cmd = [executable("cell-eval") or "cell-eval", "run", "-ap", str(pred_h5ad), "-ar", str(real_h5ad),
           "--pert-col", cfg.data.pert_col, "--control-pert", cfg.data.control_label,
           "--profile", profile or cfg.eval.profile,
           "--num-threads", str(num_threads or cfg.eval.num_threads), "-o", str(outdir)]
    if cfg.submit.require_counts:
        cmd.append("--allow-discrete")
    if getattr(cfg, "_eval_ceiling", False):
        cmd.append("--ceiling")
    if getattr(cfg, "_eval_six_only", False):
        cmd.extend(["--skip-metrics", ",".join(SIX_ONLY_SKIP)])
    print("[evaluate] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    res: dict[str, float] = {}
    for name in ("agg_results.csv", "results.csv"):
        f = outdir / name
        if f.exists():
            for k, v in _flatten(pd.read_csv(f)).items():
                res.setdefault(k, v)
    if getattr(cfg, "_eval_ceiling", False):
        for name in ("agg_ceiling_results.csv", "ceiling_results.csv"):
            f = outdir / name
            if f.exists():
                for k, v in _flatten(pd.read_csv(f)).items():
                    res.setdefault(f"ceiling::{k}", v)
    return res


def fingerprint(metrics: dict[str, float]) -> dict[str, float]:
    out = {}
    for axis, names in FINGERPRINT.items():
        for n in names:
            if n in metrics:
                out[axis] = metrics[n]; break
        for n in names:
            if f"ceiling::{n}" in metrics:
                out[f"{axis}_ceiling"] = metrics[f"ceiling::{n}"]; break
    return out


def _flatten(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    cols = [c.lower() for c in df.columns]
    if {"metric", "value"} <= set(cols):
        mc, vc = df.columns[cols.index("metric")], df.columns[cols.index("value")]
        pairs = ((str(r[mc]), r[vc]) for _, r in df.iterrows())
    else:
        row = None
        if "statistic" in cols:
            stat_col = df.columns[cols.index("statistic")]
            mean_rows = df[df[stat_col].astype(str).str.lower() == "mean"]
            if not mean_rows.empty:
                row = mean_rows.iloc[0]
        if row is not None or len(df) == 1:
            row = row if row is not None else df.iloc[0]
            pairs = ((str(c), row[c]) for c in df.columns)
        else:
            # Per-perturbation results: aggregate numeric metric columns.
            pairs = ((str(c), pd.to_numeric(df[c], errors="coerce").mean()) for c in df.columns)
    for key, value in pairs:
        try:
            number = float(value)
            if math.isfinite(number):
                out[key] = number
        except (TypeError, ValueError):
            pass
    return out
