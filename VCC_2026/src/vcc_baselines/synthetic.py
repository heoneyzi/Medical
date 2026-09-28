"""Synthetic, VCC-shaped RAW-COUNT dataset so the whole pipeline runs offline.

Multiplicative (fold-change) structure, matching the counts-space 2026 task:
    rate_perturbed = base_rate(context) * exp(shared_lfc[g] + residual)
    gwps_perturbed = base_rate(source)  * exp(shared_lfc[g] + source_residual)
The shared component transfers K562/RPE1 -> contexts (so GWPS beats no-effect);
the context/source residuals keep it non-trivial. Not real biology — plumbing only.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _pois(rate, n, rng):
    return rng.poisson(np.clip(rate, 0.02, None)[None, :].repeat(n, 0)).astype(np.float32)


def make(out_dir, *, n_genes=400, n_contexts=3, n_targets=40, frac_measured=0.7,
         n_gwps_extra=150, n_ctrl_cells=200, n_truth_cells=80, n_gwps_cells=25,
         cells_per_pert=50, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n_targets = min(n_targets, n_genes); n_gwps_extra = min(n_gwps_extra, n_genes)
    out = Path(out_dir); (out / "validation").mkdir(parents=True, exist_ok=True); (out / "gwps").mkdir(exist_ok=True)
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    pd.DataFrame(genes).to_csv(out / "gene_names.csv", index=False, header=False)

    def sparse_lfc(scale, gidx_self=None):
        d = np.zeros(n_genes, np.float32)
        k = rng.integers(3, 9); idx = rng.choice(n_genes, k, replace=False)
        d[idx] = rng.normal(0, scale, k)
        if gidx_self is not None:
            d[gidx_self] = np.log(0.2)   # knockdown suppresses the target itself
        return d

    gi = {g: i for i, g in enumerate(genes)}
    shared = {g: sparse_lfc(0.8, gi[g]) for g in genes}

    targets = list(rng.choice(genes, n_targets, replace=False))
    measured = set(targets[: int(frac_measured * n_targets)])
    pd.DataFrame({"target_gene": targets}).to_csv(out / "validation" / "targets.csv", index=False)
    pd.DataFrame({"target_gene": targets, "n_cells": cells_per_pert}).to_csv(
        out / "validation" / "pert_counts.csv", index=False)

    ctx_names = list("ABCDEFGH"[:n_contexts])
    for ci, ctx in enumerate(ctx_names):
        cdir = out / "validation" / ctx; cdir.mkdir(exist_ok=True)
        base = np.abs(rng.normal(3.0 + ci * 0.4, 1.3, n_genes)).astype(np.float32)
        ctrl = _pois(base, n_ctrl_cells, rng)
        a = ad.AnnData(X=ctrl, obs=pd.DataFrame({"target_gene": ["non-targeting"] * n_ctrl_cells}))
        a.var_names = genes; a.write_h5ad(cdir / "controls.h5ad")
        blocks, labels = [], []
        for g in targets:
            lfc = shared[g] + sparse_lfc(0.4)
            blocks.append(_pois(base * np.exp(lfc), n_truth_cells, rng)); labels += [g] * n_truth_cells
        blocks.append(ctrl); labels += ["non-targeting"] * n_ctrl_cells
        t = ad.AnnData(X=np.vstack(blocks).astype(np.float32), obs=pd.DataFrame({"target_gene": labels}))
        t.var_names = genes; t.write_h5ad(cdir / "truth.h5ad")

    gwps_genes = sorted(measured | set(rng.choice(genes, n_gwps_extra, replace=False)))
    for si, src in enumerate(["k562", "rpe1"]):
        base = np.abs(rng.normal(3.5 + si * 0.5, 1.1, n_genes)).astype(np.float32)
        blocks, labels = [_pois(base, n_gwps_cells * 3, rng)], ["non-targeting"] * (n_gwps_cells * 3)
        for g in gwps_genes:
            lfc = shared[g] + sparse_lfc(0.35)
            blocks.append(_pois(base * np.exp(lfc), n_gwps_cells, rng)); labels += [g] * n_gwps_cells
        a = ad.AnnData(X=np.vstack(blocks).astype(np.float32), obs=pd.DataFrame({"gene": labels}))
        a.var_names = genes; a.write_h5ad(out / "gwps" / f"{src}_pseudobulk.h5ad")

    return {"genes": genes, "targets": targets, "contexts": ctx_names,
            "measured": sorted(measured), "gwps_genes": gwps_genes, "n_genes": n_genes,
            "cells_per_pert": cells_per_pert}
