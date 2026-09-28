"""Lightweight LOCAL metrics for fast shadow-CV sanity checks.

These are *approximations* of the challenge axes (perturbation discrimination,
differential expression, expression error) so you can iterate without the full
cell-eval run. For anything you report or submit, use the official cell-eval
metrics (`evaluate.py --engine cell-eval`). Do not treat these numbers as the
leaderboard's.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .io import to_dense


def _pseudobulk_by_pert(adata: ad.AnnData, pert_col: str) -> dict[str, np.ndarray]:
    X = to_dense(adata.X).astype(np.float32)
    labels = adata.obs[pert_col].astype(str).values
    out = {}
    for p in np.unique(labels):
        out[p] = X[labels == p].mean(axis=0)
    return out


def compute(pred: ad.AnnData, real: ad.AnnData, pert_col: str = "target_gene",
            control_label: str = "non-targeting", top_n: int = 100) -> dict[str, float]:
    pb_p = _pseudobulk_by_pert(pred, pert_col)
    pb_r = _pseudobulk_by_pert(real, pert_col)
    ctrl_p = pb_p.get(control_label)
    ctrl_r = pb_r.get(control_label)
    perts = sorted((set(pb_p) & set(pb_r)) - {control_label})
    if not perts:
        return {"n_perts": 0}

    R = np.stack([pb_r[p] for p in perts])              # (P, G) real pseudobulks
    P = np.stack([pb_p[p] for p in perts])              # (P, G) pred pseudobulks

    # --- Perturbation discrimination (normalized L1 rank; 0=perfect, .5=random)
    ranks = []
    for i in range(len(perts)):
        d = np.abs(R - P[i]).sum(axis=1)                # L1 to every real pert
        r = int((d < d[i]).sum())                       # how many are closer than true
        ranks.append(r / max(len(perts) - 1, 1))
    pdisc = float(np.mean(ranks))

    # --- Expression error on pseudobulk
    mae = float(np.abs(P - R).mean())

    # --- Delta-based metrics (need controls)
    out = {"n_perts": len(perts), "pdisc_norm_rank": pdisc, "mae_pseudobulk": mae}
    if ctrl_p is not None and ctrl_r is not None:
        dP = P - ctrl_p
        dR = R - ctrl_r
        # Pearson of predicted vs real delta, averaged over perts
        pear = []
        for i in range(len(perts)):
            a, b = dP[i] - dP[i].mean(), dR[i] - dR[i].mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            pear.append(float(a @ b / (na * nb)) if na and nb else 0.0)
        out["pearson_delta"] = float(np.mean(pear))
        # top-N DE overlap by |delta|
        ov = []
        for i in range(len(perts)):
            tp = set(np.argsort(-np.abs(dP[i]))[:top_n])
            tr = set(np.argsort(-np.abs(dR[i]))[:top_n])
            ov.append(len(tp & tr) / top_n)
        out[f"de_overlap@{top_n}"] = float(np.mean(ov))
    return out
