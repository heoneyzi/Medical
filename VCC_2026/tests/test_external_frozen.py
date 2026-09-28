from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vcc_baselines.external_frozen import (
    AuditRequired, ModelSpec, pool_model, preflight_model, run_context,
)
from vcc_baselines.generators import generate_pseudobulk_multinomial
from vcc_baselines.replogle import project_effect_library
from vcc_baselines.seeds import stable_seed


def test_stable_seed_is_order_sensitive_and_repeatable():
    assert stable_seed(2026, "A", "TP53") == stable_seed(2026, "A", "TP53")
    assert stable_seed(2026, "A", "TP53") != stable_seed(2026, "TP53", "A")


def test_pooled_generator_is_sparse_integer_deterministic_and_not_zero_locked():
    controls = np.array([[10, 0], [8, 0], [12, 0]], dtype=np.int32)
    effect = np.array([1, 1000], dtype=np.float32)
    one = generate_pseudobulk_multinomial(
        controls, effect, rng=np.random.default_rng(7), alpha=1.0)
    two = generate_pseudobulk_multinomial(
        controls, effect, rng=np.random.default_rng(7), alpha=1.0)
    assert sp.isspmatrix_csr(one)
    assert one.dtype == np.int32
    assert (one != two).nnz == 0
    assert np.array_equal(np.asarray(one.sum(1)).ravel(), controls.sum(1))
    assert one[:, 1].sum() > 0


def test_project_replogle_library_renormalized_exact_coverage(tmp_path):
    genes = np.array(["A", "B"])
    np.savez_compressed(
        tmp_path / "k.npz", genes=genes, targets=np.array(["T1", "T2"]),
        lnfc=np.array([[1, 0], [0, 1]], np.float32),
        ctrl_cpm=np.array([600000, 400000], np.float32),
        n_pert_cells=np.array([10, 10], np.float32))
    np.savez_compressed(
        tmp_path / "r.npz", genes=genes, targets=np.array(["T2"]),
        lnfc=np.array([[0, 2]], np.float32),
        ctrl_cpm=np.array([500000, 500000], np.float32),
        n_pert_cells=np.array([10], np.float32))
    lib, coverage = project_effect_library(
        {"k562": tmp_path / "k.npz", "rpe1": tmp_path / "r.npz"},
        ["A", "B"], ["T1", "T2", "T3"], tmp_path / "effect.npz")
    observed = dict(zip(coverage.target_gene, coverage.coverage_class))
    assert observed == {"T1": "K562_ONLY", "T2": "DUAL_SOURCE", "T3": "MISSING"}
    assert np.allclose(np.log(lib.get_relfold("k562", "T1")), [1, 0])


def test_external_command_adapter_validates_and_reuses(tmp_path):
    root = tmp_path
    repo = root / "repo"; repo.mkdir()
    checkpoint = root / "checkpoint.bin"; checkpoint.write_bytes(b"weights")
    script = repo / "embed.py"
    script.write_text(
        "import anndata as ad, numpy as np, pathlib, sys\n"
        "a=ad.read_h5ad(sys.argv[1]); a.obsm['X_test']=np.ones((a.n_obs,3),dtype='float32')\n"
        "pathlib.Path(sys.argv[2]).parent.mkdir(parents=True,exist_ok=True); a.write_h5ad(sys.argv[2])\n")
    inp = root / "input.h5ad"
    data = ad.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.int32)))
    data.obs_names = [f"c{i}" for i in range(4)]
    data.var_names = [f"g{i}" for i in range(4)]
    data.write_h5ad(inp)
    spec = ModelSpec(
        name="fake", display_name="fake", role="context_encoder", track="test",
        repo="repo", executable=sys.executable, checkpoint="checkpoint.bin",
        command=("{executable}", "{repo}/embed.py", "{input}", "{work_dir}/native/out.h5ad"),
        output_glob="{work_dir}/native/out.h5ad", output_type="h5ad",
        embedding_key="X_test", expected_dim=3)
    artifacts = root / "artifacts"
    first = run_context(spec, root, inp, "ctx", artifacts)
    second = run_context(spec, root, inp, "ctx", artifacts)
    assert first["context_sha256"] == second["context_sha256"]
    assert first["n_output_cells"] == 4
    pooled = pool_model(spec, {"ctx": inp}, artifacts, root / "pooled.npz")
    with np.load(pooled, allow_pickle=False) as values:
        assert np.array_equal(values["ctx"], np.ones(3, dtype=np.float32))


def test_external_preflight_fails_closed_on_runtime_asset_and_checksum(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    expected_md5 = hashlib.md5(b"weights", usedforsecurity=False).hexdigest()
    spec = ModelSpec(
        name="fake", display_name="fake", role="context_encoder", track="test",
        repo="repo", executable=sys.executable, checkpoint="checkpoint.bin",
        checkpoint_md5=expected_md5,
        required_runtime_files=("runtime/token.bin",),
    )
    missing = preflight_model(spec, tmp_path)
    assert not missing["ready"]
    assert any("missing runtime file" in error for error in missing["errors"])

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "token.bin").write_bytes(b"tokens")
    assert preflight_model(spec, tmp_path)["ready"]

    checkpoint.write_bytes(b"tampered")
    mismatch = preflight_model(spec, tmp_path)
    assert not mismatch["ready"]
    assert any("checkpoint MD5 mismatch" in error for error in mismatch["errors"])


def test_external_adapter_rejects_non_count_input_before_model_call(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    inp = tmp_path / "input.h5ad"
    data = ad.AnnData(X=np.array([[0.0, 0.25]], dtype=np.float32))
    data.obs_names = ["cell"]
    data.var_names = ["A", "B"]
    data.write_h5ad(inp)
    spec = ModelSpec(
        name="fake", display_name="fake", role="context_encoder", track="test",
        repo="repo", executable=sys.executable, checkpoint="checkpoint.bin",
        command=("{executable}", "-c", "raise SystemExit(99)"),
        output_glob="{work_dir}/native/out.npy", output_type="npy",
    )
    with pytest.raises(AuditRequired, match="raw integer-like counts"):
        run_context(spec, tmp_path, inp, "ctx", tmp_path / "artifacts")
