"""3-rung ladder over shadow-CV contexts, with coverage stratification.

Scores each method against held-out truth in every context; reports the per-
context mean and the WORST context, plus a coverage-group breakdown (how far
pure GWPS matching can reach vs the ESM-2 fallback).
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import evaluate, io, metrics_local, predict as P, vcc2026_eval
from .config import Config
from .effect_library import EffectLibrary
from .methods import build_method

DEFAULT = ["no_effect", "gwps_direct", "gwps_weighted", "esm2_knn"]


def get_library(cfg: Config, genes) -> EffectLibrary | None:
    if not cfg.gwps.enabled:
        return None
    cache = Path(cfg.gwps.cache)
    if cache.exists():
        return EffectLibrary.load(cache)
    lib = EffectLibrary.from_gwps(cfg.gwps, genes)
    if lib.sources:
        lib.save(cache)
    return lib


def _context_pbs(cfg, contexts, genes):
    out = {}
    for c in contexts:
        ctrl = io.align_to_genes(io.load_controls(cfg.data.contexts_dir, c, cfg.data.control_h5ad), genes)
        out[c] = io.to_dense(ctrl.X).astype(np.float32).mean(0)
    return out


def run_ladder(cfg: Config, methods=None, engine="local") -> pd.DataFrame:
    methods = methods or DEFAULT
    genes = io.read_gene_list(cfg.data.gene_list)
    targets = io.read_targets(cfg.data.targets)
    contexts = io.list_contexts(cfg.data.contexts_dir, cfg.data.control_h5ad)
    contexts = [c for c in contexts if io.load_truth(cfg.data.contexts_dir, c, cfg.data.truth_h5ad) is not None]
    if not contexts:
        raise RuntimeError("no shadow-CV contexts with truth found")
    lib = get_library(cfg, genes)
    ctx_pbs = _context_pbs(cfg, contexts, genes)
    embedder = P.make_embedder(cfg, lib, ctx_pbs)

    rows = []
    ceiling_requested = bool(getattr(cfg, "_eval_ceiling", False))
    for method_index, mname in enumerate(methods):
        cfg._eval_ceiling = ceiling_requested and method_index == 0
        cfg.predict.method = mname
        try:
            method = build_method(cfg, lib, genes)
        except Exception as e:
            print(f"[ladder] skip {mname}: {e}"); continue
        per = []
        for c in contexts:
            controls = io.load_controls(cfg.data.contexts_dir, c, cfg.data.control_h5ad)
            truth = io.load_truth(cfg.data.contexts_dir, c, cfg.data.truth_h5ad)
            pred, _ = P.build_prediction(cfg, controls, genes, targets, method, embedder, c,
                                         lib=lib, include_controls=True)
            if engine in {"cell-eval", "cell-eval2"}:
                od = Path(cfg.output_dir) / "ladder" / mname / c
                io.save_prediction(pred, od / "pred.h5ad"); io.save_prediction(truth, od / "truth.h5ad")
                if engine == "cell-eval2":
                    # Anchor identity must be tied to one stable reference path,
                    # not to the per-method convenience copy under ladder/.
                    truth_source = (Path(cfg.data.contexts_dir) / c /
                                    cfg.data.truth_h5ad)
                    scale_root = Path(getattr(
                        cfg, "_eval_scale_root",
                        Path(cfg.output_dir) / "vcc2026_local_scale_v2"))
                    m = vcc2026_eval.evaluate(
                        od / "pred.h5ad", truth_source, od / "vcc2026",
                        scale_root / c,
                        pert_col=cfg.data.pert_col, control=cfg.data.control_label,
                        target_gene_map_path=getattr(cfg, "_eval_target_gene_map", None),
                        require_gpu=not bool(getattr(cfg, "_eval_allow_cpu_scorer", False)),
                    )
                else:
                    m = evaluate.fingerprint(evaluate.evaluate_cell_eval(
                        od / "pred.h5ad", od / "truth.h5ad", cfg, od))
            else:
                m = metrics_local.compute(pred, truth, cfg.data.pert_col, cfg.data.control_label)
            m["context"] = c; per.append(m)
        if not per:
            continue
        df = pd.DataFrame(per).set_index("context")
        agg = {"method": mname}
        for col in df.columns:
            agg[f"{col}.mean"] = df[col].mean()
            agg[f"{col}.worst"] = df[col].max() if col in ("pdisc_norm_rank", "mae_pseudobulk", "MAE", "MSE") else df[col].min()
        rows.append(agg)
    return pd.DataFrame(rows).set_index("method")


def coverage_report(cfg: Config, method_name="gwps_weighted") -> pd.DataFrame:
    """Per coverage-group local pdisc — the DB approach's reach ceiling."""
    genes = io.read_gene_list(cfg.data.gene_list)
    targets = io.read_targets(cfg.data.targets)
    contexts = io.list_contexts(cfg.data.contexts_dir, cfg.data.control_h5ad)
    contexts = [c for c in contexts if io.load_truth(cfg.data.contexts_dir, c, cfg.data.truth_h5ad) is not None]
    lib = get_library(cfg, genes)
    if lib is None:
        raise RuntimeError("coverage report needs a GWPS library")
    groups = {g: lib.coverage_group(g) for g in targets}
    ctx_pbs = _context_pbs(cfg, contexts, genes)
    embedder = P.make_embedder(cfg, lib, ctx_pbs)
    cfg.predict.method = method_name
    method = build_method(cfg, lib, genes)
    rows = []
    for c in contexts:
        controls = io.load_controls(cfg.data.contexts_dir, c, cfg.data.control_h5ad)
        truth = io.load_truth(cfg.data.contexts_dir, c, cfg.data.truth_h5ad)
        pred, _ = P.build_prediction(cfg, controls, genes, targets, method, embedder, c, lib=lib, include_controls=True)
        for grp in sorted(set(groups.values())):
            gset = {g for g, gg in groups.items() if gg == grp}
            keep_p = pred[pred.obs[cfg.data.pert_col].astype(str).isin(gset | {cfg.data.control_label})]
            keep_t = truth[truth.obs[cfg.data.pert_col].astype(str).isin(gset | {cfg.data.control_label})]
            m = metrics_local.compute(keep_p, keep_t, cfg.data.pert_col, cfg.data.control_label)
            rows.append({"context": c, "group": grp, "n_targets": len(gset), **m})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith(".mean")]
    show = df[cols].copy(); show.columns = [c[:-5] for c in cols]
    try:
        return show.round(4).to_markdown()
    except Exception:
        return show.round(4).to_string()
