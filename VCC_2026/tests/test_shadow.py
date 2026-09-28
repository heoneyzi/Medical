from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcc_baselines import evaluate, io, predict, shadow, vcc2026_eval
from vcc_baselines.config import Config
from vcc_baselines.methods import build_method


def _realish(path: Path, context: str, shift: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(30)]
    labels = ["control"] * 12
    nperts = [0] * 12
    blocks = [rng.poisson(2 + shift, (12, len(genes)))]
    for j, target in enumerate(["G1", "G2", "G3"]):
        rate = np.full(len(genes), 2 + shift)
        rate[j + 1] *= 0.2
        rate[10 + j] *= 2.0
        blocks.append(rng.poisson(rate, (8, len(genes))))
        labels += [target] * 8
        nperts += [1] * 8
    obs = pd.DataFrame({
        "perturbation": labels, "nperts": nperts,
        "ngenes": 500, "ncounts": 1000, "percent_mito": 1.0,
        "cell_line": context,
    })
    a = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks).astype(np.float32)), obs=obs)
    a.var_names = genes
    a.write_h5ad(path)


def test_sparse_align_preserves_sparse():
    a = ad.AnnData(X=sp.csr_matrix([[1, 0], [0, 2]], dtype=np.float32))
    a.var_names = ["b", "a"]
    out = io.align_to_genes(a, ["a", "missing", "b"])
    assert sp.issparse(out.X)
    assert out.X.toarray().tolist() == [[0, 0, 1], [2, 0, 0]]


def test_prepare_shadow_never_exposes_holdout_responses(tmp_path):
    src, held = tmp_path / "src.h5ad", tmp_path / "held.h5ad"
    _realish(src, "source", 0.0, 1)
    _realish(held, "alien", 2.0, 2)
    cfg_path = shadow.prepare_shadow(
        sources=[shadow.ContextSpec("source", str(src))],
        holdout=shadow.ContextSpec("alien", str(held)), out_dir=str(tmp_path / "shadow"),
        min_genes=0, min_counts=0, min_cells=3, max_targets=3, max_genes=20,
        max_controls=10, max_cells_per_pert=6, cells_per_pert=5,
    )
    controls = ad.read_h5ad(tmp_path / "shadow/validation/alien/controls.h5ad")
    truth = ad.read_h5ad(tmp_path / "shadow/validation/alien/truth.h5ad")
    reference = ad.read_h5ad(tmp_path / "shadow/references/source.h5ad")
    assert set(controls.obs.target_gene.astype(str)) == {"non-targeting"}
    assert set(truth.obs.target_gene.astype(str)) == {"non-targeting", "G1", "G2", "G3"}
    assert set(reference.obs.target_gene.astype(str)) == {"non-targeting", "G1", "G2", "G3"}
    manifest = json.loads((tmp_path / "shadow/manifest.json").read_text())
    assert manifest["challenge_validation_used"] is False
    assert manifest["holdout_response_visible_to_model"] is False
    result = shadow.benchmark(str(cfg_path), methods=["no_effect", "gwps_direct"])
    assert set(result.status) == {"ok"}


def test_prepare_shadow_retains_mapped_target_genes_for_vcc2026(tmp_path):
    src, held = tmp_path / "src.h5ad", tmp_path / "held.h5ad"
    _realish(src, "source", 0.0, 11)
    _realish(held, "alien", 1.0, 12)
    mapping = tmp_path / "target_map.json"
    mapping.write_text(json.dumps({"G1": "G27", "G2": "G28", "G3": "G29"}))
    shadow.prepare_shadow(
        sources=[shadow.ContextSpec("source", str(src))],
        holdout=shadow.ContextSpec("alien", str(held)), out_dir=str(tmp_path / "mapped"),
        min_genes=0, min_counts=0, min_cells=3, max_targets=3, max_genes=5,
        max_controls=10, max_cells_per_pert=6, cells_per_pert=5,
        target_gene_map_path=str(mapping),
    )
    genes = io.read_gene_list(tmp_path / "mapped/gene_names.csv")
    assert {"G27", "G28", "G29"} <= set(genes)
    manifest = json.loads((tmp_path / "mapped/manifest.json").read_text())
    assert manifest["target_gene_map"]["used_by_effect_prediction"] is False


def test_cell_eval_parser_uses_mean_row_and_ignores_nan():
    frame = pd.DataFrame({
        "statistic": ["count", "mean", "max"],
        "mae": [20.0, 0.25, 0.9], "discrimination_score_l1": [20.0, 0.6, 1.0],
    })
    assert evaluate._flatten(frame)["mae"] == 0.25
    ceiling = pd.DataFrame({"mae": [float("nan")], "discrimination_score_l1": [0.75]})
    assert "mae" not in evaluate._flatten(ceiling)
    assert evaluate._flatten(ceiling)["discrimination_score_l1"] == 0.75
    six = evaluate.fingerprint({
        "discrimination_score_l1": 0.6, "mae": 0.2, "mse": 0.1,
        "pearson_delta": 0.4, "de_direction_match": 0.7, "overlap_at_N": 0.3,
    })
    assert set(six) == {"PDS", "MAE", "MSE", "PearsonDelta", "DirMatch", "OverlapN"}


def test_compact_source_selects_filters_and_raw_counts_layer(tmp_path):
    obs = pd.DataFrame({"treatment": ["A", "B", "A"], "condition": ["control", "G1", "G1"]})
    a = ad.AnnData(X=sp.csr_matrix(np.log1p([[1, 2], [3, 4], [5, 6]])), obs=obs)
    a.layers["counts"] = sp.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    a.var_names = ["G1", "G2"]
    source, out = tmp_path / "large.h5ad", tmp_path / "compact.h5ad"
    a.write_h5ad(source)
    shadow.compact_source(str(source), str(out), {"treatment": "A"}, "counts")
    compact = ad.read_h5ad(out)
    assert compact.shape == (2, 2)
    assert compact.X.toarray().tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert set(compact.obs["treatment"]) == {"A"}
    assert compact.uns["shadow_compaction"]["matrix_origin"] == "layers[counts]"


def test_pool_embeddings_supports_x(tmp_path):
    paths = []
    for name, values in [("a", [[1, 2], [3, 4]]), ("b", [[10, 20]])]:
        p = tmp_path / f"{name}.h5ad"
        ad.AnnData(X=np.asarray(values, dtype=np.float32)).write_h5ad(p)
        paths.append(f"{name}={p}")
    out = tmp_path / "pooled.npz"
    shadow.pool_embeddings(paths, "X", str(out))
    with np.load(out) as z:
        assert z["a"].tolist() == [2.0, 3.0]
        assert z["b"].tolist() == [10.0, 20.0]


def test_effect_library_cache_source_order_is_stable(tmp_path):
    from vcc_baselines.effect_library import EffectLibrary
    lib = EffectLibrary(["G"])
    for source in ["z_source", "a_source"]:
        lib.control_pb[source] = np.ones(1, dtype=np.float32)
        lib.control_std[source] = np.zeros(1, dtype=np.float32)
        lib.pert_genes[source] = ["G"]
        lib.relfold[source] = np.ones((1, 1), dtype=np.float32)
        lib._row[source] = {"G": 0}
    cache = tmp_path / "effects.npz"
    lib.save(cache)
    assert EffectLibrary.load(cache).sources == ["a_source", "z_source"]


def test_prepare_shadow_rejects_repo_challenge_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    forbidden = tmp_path / "data" / "validation" / "context_A.h5ad"
    with np.testing.assert_raises_regex(ValueError, "data/ inputs are forbidden"):
        shadow.prepare_shadow(
            sources=[shadow.ContextSpec("source", str(forbidden))],
            holdout=shadow.ContextSpec("holdout", str(forbidden)),
            out_dir=str(tmp_path / "shadow"),
        )


def test_integer_count_check_covers_values_beyond_first_rows():
    X = sp.csr_matrix(np.vstack([np.ones((300, 2)), [[1.5, 2.0]]]))
    assert not shadow._is_integer_like(X)


def test_vcc2026_metric_members_are_the_challenge_six():
    assert vcc2026_eval.SCORED_METRICS == {
        "pds": "pds_cosine",
        "mse": "expr_mse_unbiased_capped_norm",
        "nmae": "de_wilcoxon_lfc_nmae",
        "fid": "de_wilcoxon_direction_fidelity_yield_raw",
        "reach": "de_wilcoxon_direction_reach_raw",
        "jac": "de_wilcoxon_sig_jaccard",
    }


def test_vcc2026_target_map_loader_rejects_conflicts(tmp_path):
    good = tmp_path / "map.csv"
    pd.DataFrame({
        "name": ["HAR1", "HAR1", "HAR2"],
        "target_gene_name": ["G1", "G1", "G2"],
    }).to_csv(good, index=False)
    assert vcc2026_eval.load_target_gene_map(good) == {"HAR1": "G1", "HAR2": "G2"}

    bad = tmp_path / "bad.csv"
    pd.DataFrame({
        "name": ["HAR1", "HAR1"],
        "target_gene_name": ["G1", "G2"],
    }).to_csv(bad, index=False)
    with np.testing.assert_raises_regex(ValueError, "conflicting genes"):
        vcc2026_eval.load_target_gene_map(bad)


def test_vcc2026_target_map_loader_adds_gse_control_alias(tmp_path):
    path = tmp_path / "gse_map.csv"
    pd.DataFrame({
        "name": ["cont_RAB1A"], "target_gene_name": ["RAB1A"],
    }).to_csv(path, index=False)
    mapping = vcc2026_eval.load_target_gene_map(path)
    assert mapping["cont_RAB1A"] == mapping["cont-RAB1A"] == "RAB1A"


def test_zero_shot_bundle_seals_all_responses_and_scores_existing_prediction(tmp_path):
    source = tmp_path / "alien.h5ad"
    _realish(source, "alien", 2.0, 7)
    config_path = shadow.prepare_zero_shot(
        input_path=str(source), context_name="alien", out_dir=str(tmp_path / "zero"),
        pert_col="perturbation", control_label="control", matrix="X",
        min_genes=0, min_counts=0, max_genes=20, max_controls=10,
        max_cells_per_pert=6, cells_per_pert=5, seed=2026)

    blind = tmp_path / "zero" / "blind"
    sealed = tmp_path / "zero" / "sealed"
    assert not any("truth.h5ad" in str(p) for p in blind.rglob("*"))
    controls = ad.read_h5ad(blind / "contexts/alien/controls.h5ad")
    truth = ad.read_h5ad(sealed / "alien/truth.h5ad")
    assert set(controls.obs.target_gene.astype(str)) == {"non-targeting"}
    assert set(truth.obs.target_gene.astype(str)) == {"non-targeting", "G1", "G2", "G3"}
    contract = json.loads((blind / "zero_shot_contract.json").read_text())
    assert contract["same_dataset_perturbation_responses_visible"] is False
    assert contract["truth_path_present"] is False

    cfg = Config.load(config_path).resolve(config_path.parent)
    assert cfg.gwps.enabled is False and cfg.gwps.sources == {}
    genes, targets = io.read_gene_list(cfg.data.gene_list), io.read_targets(cfg.data.targets)
    method = build_method(cfg, None, genes)
    pred, _ = predict.build_prediction(cfg, controls, genes, targets, method, None, "alien",
                                       lib=None, include_controls=True)
    pred_path = blind / "outputs/no_effect/prediction.h5ad"
    io.save_prediction(pred, pred_path)
    result = shadow.score_zero_shot(
        str(config_path), str(pred_path), str(sealed / "alien/truth.h5ad"),
        str(tmp_path / "zero/results"), engine="local")
    assert result.loc[0, "protocol"] == "strict-response-sealed-zero-shot-v1"
    audit = json.loads((tmp_path / "zero/results/scoring_audit.json").read_text())
    assert audit["gwps_enabled"] is False
    assert audit["same_dataset_response_sources"] == 0
