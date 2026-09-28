"""End-to-end smoke test (raw-counts space). Proves the pipeline runs and that
GWPS logFC transfer beats the no-effect anchor on perturbation discrimination."""
from __future__ import annotations

import numpy as np

from vcc_baselines import io, predict as P, generators, synthetic
from vcc_baselines.config import Config
from vcc_baselines.effect_library import EffectLibrary
from vcc_baselines.evaluate import evaluate_local
from vcc_baselines.methods import build_method


def _cfg(tmp):
    c = Config()
    c.output_dir = str(tmp / "out")
    c.data.contexts_dir = str(tmp / "validation")
    c.data.targets = str(tmp / "validation" / "targets.csv")
    c.data.perts = str(tmp / "validation" / "pert_counts.csv")
    c.data.gene_list = str(tmp / "gene_names.csv")
    c.gwps.sources = {"k562": str(tmp / "gwps" / "k562_pseudobulk.h5ad"),
                      "rpe1": str(tmp / "gwps" / "rpe1_pseudobulk.h5ad")}
    c.gwps.pert_col = "gene"
    c.predict.cells_per_pert = 40
    c.submit.gene_dim = 150
    return c


def test_pipeline(tmp_path):
    synthetic.make(tmp_path, n_genes=150, n_contexts=2, n_targets=20, cells_per_pert=40,
                   n_ctrl_cells=120, n_truth_cells=60, n_gwps_cells=15, seed=1)
    cfg = _cfg(tmp_path)
    genes = io.read_gene_list(cfg.data.gene_list)
    targets = io.read_targets(cfg.data.targets)
    assert len(genes) == 150 and len(targets) == 20

    lib = EffectLibrary.from_gwps(cfg.gwps, genes, verbose=False)
    assert set(lib.sources) == {"k562", "rpe1"}

    ctx = io.list_contexts(cfg.data.contexts_dir, cfg.data.control_h5ad)[0]
    controls = io.load_controls(cfg.data.contexts_dir, ctx, cfg.data.control_h5ad)
    truth = io.load_truth(cfg.data.contexts_dir, ctx, cfg.data.truth_h5ad)
    ctx_pbs = {c: io.to_dense(io.align_to_genes(io.load_controls(cfg.data.contexts_dir, c, cfg.data.control_h5ad), genes).X).mean(0)
               for c in io.list_contexts(cfg.data.contexts_dir, cfg.data.control_h5ad)}
    emb = P.make_embedder(cfg, lib, ctx_pbs)

    scores = {}
    for m in ["no_effect", "gwps_direct", "gwps_weighted", "esm2_knn"]:
        cfg.predict.method = m
        method = build_method(cfg, lib, genes)
        pred, cov = P.build_prediction(cfg, controls, genes, targets, method, emb, ctx, lib=lib, include_controls=False)
        X = io.to_dense(pred.X)
        assert pred.n_obs == len(targets) * 40                 # no controls
        assert np.allclose(X, np.round(X)) and X.min() >= 0    # raw non-negative integer counts
        assert cfg.data.control_label not in set(pred.obs[cfg.data.pert_col].astype(str))
        # add controls for local scoring
        predE, _ = P.build_prediction(cfg, controls, genes, targets, method, emb, ctx, lib=lib, include_controls=True)
        scores[m] = evaluate_local(predE, truth, cfg)["pdisc_norm_rank"]

    assert scores["no_effect"] > 0.35
    assert scores["gwps_direct"] < scores["no_effect"], scores
    assert scores["gwps_weighted"] < scores["no_effect"], scores


def test_generators(tmp_path):
    rng = np.random.default_rng(0)
    ctrl = rng.poisson(3.0, size=(30, 50)).astype(np.float32)
    relfold = np.full(50, 0.5, np.float32)
    for gen in ["multinomial", "poisson", "nbinom"]:
        out = generators.generate(ctrl, relfold, kind="logfc", generator=gen, rng=rng)
        assert out.shape == (30, 50)
        assert np.allclose(out, np.round(out)) and out.min() >= 0


def test_ladder(tmp_path):
    from vcc_baselines import ladder as L
    synthetic.make(tmp_path, n_genes=120, n_contexts=2, n_targets=15, cells_per_pert=30,
                   n_ctrl_cells=100, n_truth_cells=50, n_gwps_cells=12, seed=2)
    cfg = _cfg(tmp_path); cfg.submit.gene_dim = 120; cfg.predict.cells_per_pert = 30
    df = L.run_ladder(cfg, ["no_effect", "gwps_direct", "gwps_weighted"], engine="local")
    assert len(df) == 3 and "pdisc_norm_rank.mean" in df.columns
