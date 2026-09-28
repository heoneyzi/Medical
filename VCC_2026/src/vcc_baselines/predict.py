"""Assemble a leaderboard prediction: effect -> raw-count cells (no controls)."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import generators, io
from .config import Config
from .context_embed import ContextEmbedder
from .effect_library import EffectLibrary
from .methods import Baseline
from .seeds import stable_seed


def make_embedder(cfg: Config, lib: EffectLibrary | None, context_pbs: dict[str, np.ndarray]) -> ContextEmbedder | None:
    if lib is None or not lib.sources:
        return None
    return ContextEmbedder(
        source_pb={s: lib.control_pb[s] for s in lib.sources},
        context_pb=context_pbs,
        space=cfg.predict.similarity_space,
        pca_components=cfg.predict.pca_components,
        precomputed=cfg.embed.context_embeddings,
    )


def _apply_scale_shrink(eff: np.ndarray, kind: str, scale: float, q: float) -> np.ndarray:
    if kind == "logfc":
        eff = np.power(np.clip(eff, 0, None), scale)   # D2b: s * logFC == relfold**s
        eff = 1.0 + q * (eff - 1.0)                     # D5 shrink toward no-effect
    else:
        eff = q * scale * eff
    return eff.astype(np.float32)


def build_prediction(cfg: Config, controls: ad.AnnData, genes: list[str], targets: list[str],
                     method: Baseline, embedder: ContextEmbedder | None, context_name: str,
                     lib: EffectLibrary | None = None,
                     include_controls: bool | None = None) -> tuple[ad.AnnData, pd.DataFrame]:
    rng = np.random.default_rng(cfg.predict.seed)
    controls = io.align_to_genes(controls, genes)
    ctrl_X = io.to_dense(controls.X).astype(np.float32)
    context_pb = ctrl_X.mean(0)
    gidx = {g: i for i, g in enumerate(genes)}

    weights = embedder.weights(context_name, lib.sources, cfg.predict.tau) if (embedder and lib) else {}
    q_ctx = embedder.max_cosine(context_name, lib.sources) if (embedder and lib and cfg.predict.shrink in ("context", "both")) else 1.0
    ctx = {"pb": context_pb, "weights": weights}
    med = np.median(context_pb[context_pb > 0]) if (context_pb > 0).any() else 1.0

    n = cfg.predict.cells_per_pert
    blocks, labels, cov = [], [], []
    for g in targets:
        eff = method.predict_effect(ctx, str(g))
        q = q_ctx
        if cfg.predict.shrink in ("targetexpr", "both") and str(g) in gidx:
            q *= float(min(1.0, context_pb[gidx[str(g)]] / (med + 1e-8)))
        eff = _apply_scale_shrink(eff, method.effect_kind, cfg.predict.scale, q)
        target_rng = (np.random.default_rng(stable_seed(
            cfg.predict.seed, cfg.run_name, context_name, str(g)))
            if cfg.predict.generator == "pseudobulk_multinomial" else rng)
        base = ctrl_X[target_rng.integers(0, ctrl_X.shape[0], size=n)]
        cells = generators.generate(base, eff, kind=method.effect_kind,
                                    generator=cfg.predict.generator, rng=target_rng,
                                    nb_theta=cfg.predict.nb_theta,
                                    multinomial_alpha=cfg.predict.multinomial_alpha)
        blocks.append(cells.astype(np.float32))
        labels.extend([str(g)] * n)
        grp = lib.coverage_group(str(g)) if lib else "NA"
        cov.append({"target_gene": str(g), "covered": bool(method.covered(str(g))),
                    "coverage_group": grp})

    inc = cfg.predict.include_controls if include_controls is None else include_controls
    if inc:
        blocks.append(ctrl_X)
        labels.extend([cfg.data.control_label] * ctrl_X.shape[0])

    if any(sp.issparse(block) for block in blocks):
        X = sp.vstack([
            block if sp.issparse(block) else sp.csr_matrix(block)
            for block in blocks
        ], format="csr", dtype=np.float32)
        X.eliminate_zeros(); X.sort_indices()
    else:
        X = np.vstack(blocks).astype(np.float32)
    if cfg.predict.clip_negative:
        if sp.issparse(X):
            np.clip(X.data, 0, None, out=X.data); X.eliminate_zeros()
        else:
            np.clip(X, 0, None, out=X)
    obs = pd.DataFrame({cfg.data.pert_col: pd.Categorical(labels)})
    obs[cfg.data.context_col] = context_name
    obs[cfg.data.celltype_col] = context_name
    pred = ad.AnnData(X=X, obs=obs)
    pred.var_names = list(map(str, genes))
    pred.obs_names = [f"{context_name}_{i}" for i in range(pred.n_obs)]
    return pred, pd.DataFrame(cov)


def cap_cells(adata: ad.AnnData, cap: int, seed: int = 0) -> ad.AnnData:
    if not cap or cap <= 0 or adata.n_obs <= cap:
        return adata
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(adata.n_obs, size=cap, replace=False))
    return adata[keep].copy()
