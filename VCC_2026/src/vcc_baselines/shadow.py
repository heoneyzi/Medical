"""Build and evaluate leakage-resistant, real-data shadow VCC benchmarks.

A shadow benchmark hides every perturbation response from one cellular context.
Only that context's non-targeting controls and the target-gene list are exposed to
prediction code; perturbation cells are written separately as ``truth.h5ad`` and
are read only by the evaluator.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from . import evaluate, io, ladder as L
from .config import Config


@dataclass(frozen=True)
class ContextSpec:
    name: str
    path: str
    context_col: str = ""
    context_value: str = ""


def parse_named_path(value: str) -> ContextSpec:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return ContextSpec(name=name, path=path)


def _parse_where(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected COLUMN=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def specs_from_single_file(path: str, context_col: str, holdout: str,
                           where: Iterable[str] = ()) -> tuple[list[ContextSpec], ContextSpec, dict[str, str]]:
    a = ad.read_h5ad(path, backed="r")
    if context_col not in a.obs:
        raise KeyError(f"{path} has no obs[{context_col!r}]; have {list(a.obs.columns)}")
    filters = _parse_where(where)
    mask = np.ones(a.n_obs, dtype=bool)
    for col, value in filters.items():
        if col not in a.obs:
            raise KeyError(f"{path} has no obs[{col!r}]")
        mask &= a.obs[col].astype(str).to_numpy() == value
    contexts = sorted(a.obs.loc[mask, context_col].astype(str).unique())
    if holdout not in contexts:
        raise ValueError(f"holdout {holdout!r} not in contexts after filters: {contexts}")
    refs = [ContextSpec(c, path, context_col, c) for c in contexts if c != holdout]
    result = (refs, ContextSpec(holdout, path, context_col, holdout), filters)
    a.file.close()
    del a
    gc.collect()
    return result


def _context_mask(obs: pd.DataFrame, spec: ContextSpec, filters: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    if spec.context_col:
        if spec.context_col not in obs:
            raise KeyError(f"{spec.path} has no obs[{spec.context_col!r}]")
        mask &= obs[spec.context_col].astype(str).to_numpy() == spec.context_value
    for col, value in filters.items():
        if col not in obs:
            raise KeyError(f"{spec.path} has no obs[{col!r}]")
        mask &= obs[col].astype(str).to_numpy() == value
    return mask


def _qc_mask(obs: pd.DataFrame, *, min_genes: int, min_counts: int,
             max_percent_mito: float) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    if "ngenes" in obs:
        mask &= pd.to_numeric(obs["ngenes"], errors="coerce").fillna(0).to_numpy() >= min_genes
    if "ncounts" in obs:
        mask &= pd.to_numeric(obs["ncounts"], errors="coerce").fillna(0).to_numpy() >= min_counts
    if "percent_mito" in obs:
        mask &= pd.to_numeric(obs["percent_mito"], errors="coerce").fillna(100).to_numpy() <= max_percent_mito
    return mask


def _label_masks(obs: pd.DataFrame, spec: ContextSpec, filters: dict[str, str],
                 pert_col: str, control_label: str, single_col: str,
                 min_genes: int, min_counts: int, max_percent_mito: float):
    if pert_col not in obs:
        raise KeyError(f"{spec.path} has no obs[{pert_col!r}]; have {list(obs.columns)}")
    labels = obs[pert_col].astype(str).to_numpy()
    eligible = _context_mask(obs, spec, filters)
    eligible &= _qc_mask(obs, min_genes=min_genes, min_counts=min_counts,
                         max_percent_mito=max_percent_mito)
    controls = eligible & (labels == control_label)
    single = eligible & (labels != control_label)
    if single_col and single_col in obs:
        single &= pd.to_numeric(obs[single_col], errors="coerce").fillna(-1).to_numpy() == 1
    return labels, controls, single


def _inspect(spec: ContextSpec, filters: dict[str, str], **kwargs) -> dict:
    p = Path(spec.path)
    if not p.exists():
        raise FileNotFoundError(p)
    a = ad.read_h5ad(p, backed="r")
    labels, controls, single = _label_masks(a.obs, spec, filters, **kwargs)
    counts = pd.Series(labels[single]).value_counts()
    result = {
        "spec": spec,
        "shape": [int(a.n_obs), int(a.n_vars)],
        "genes": a.var_names.astype(str).tolist(),
        "counts": counts,
        "n_controls": int(controls.sum()),
        "obs_columns": list(a.obs.columns),
        "layers": list(a.layers.keys()),
    }
    a.file.close()
    del a
    gc.collect()
    return result


def _raw_matrix(a: ad.AnnData, mode: str = "auto"):
    if mode not in {"auto", "X", "counts"}:
        raise ValueError("matrix mode must be auto, X, or counts")
    if mode == "counts" or (mode == "auto" and "counts" in a.layers):
        if "counts" not in a.layers:
            raise KeyError("matrix=counts requested but layers['counts'] is absent")
        X = a.layers["counts"]
        origin = "layers[counts]"
    else:
        X = a.X
        origin = "X"
    X = X.astype(np.float32)
    integer_like = _is_integer_like(X)
    if not integer_like:
        if mode != "auto":
            raise ValueError(f"{origin} is not raw integer counts")
        # Harmonized resources sometimes retain only log1p(CP10K). This is a
        # documented fallback, not a claim that the original UMI depth is known.
        if sp.issparse(X):
            X = X.copy()
            X.data = np.rint(np.expm1(X.data))
            X.eliminate_zeros()
        else:
            X = np.rint(np.expm1(X))
        origin += ":reconstructed_expm1"
    if sp.issparse(X):
        X = X.tocsr()
    return X, origin


def _is_integer_like(X, chunk_size: int = 1_000_000) -> bool:
    """Check every stored value with bounded temporary memory."""
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    for start in range(0, len(values), chunk_size):
        block = values[start:start + chunk_size]
        if not np.isfinite(block).all() or not np.allclose(block, np.round(block), atol=1e-5):
            return False
    return True


def _stable_seed(seed: int, name: str) -> int:
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(mask: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if cap > 0 and len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    return idx


def _choose_genes(holdout: ContextSpec, filters: dict[str, str], common_genes: list[str],
                  targets: list[str], max_genes: int, matrix: str, label_kwargs: dict,
                  max_controls: int, seed: int) -> tuple[list[str], str]:
    a = ad.read_h5ad(holdout.path, backed="r")
    _, controls, _ = _label_masks(a.obs, holdout, filters, **label_kwargs)
    ctrl_idx = np.flatnonzero(controls)
    if not len(ctrl_idx):
        raise ValueError(f"holdout {holdout.name} has no eligible controls")
    common = pd.Index(a.var_names.astype(str)).get_indexer(common_genes)
    valid = common >= 0
    common_genes = [g for g, ok in zip(common_genes, valid) if ok]
    common = common[valid]
    rng = np.random.default_rng(_stable_seed(seed, holdout.name + ":gene-selection"))
    ctrl_idx = _sample_indices(controls, max(max_controls, 2000), rng)
    col_order = np.argsort(common)
    sub = a[ctrl_idx, common[col_order]].to_memory()
    a.file.close()
    X, origin = _raw_matrix(sub, matrix)
    means_sorted = np.asarray(X.mean(axis=0)).ravel()
    means = np.empty_like(means_sorted)
    means[col_order] = means_sorted
    order = np.argsort(-means, kind="stable")
    if max_genes > 0:
        order = order[:max_genes]
    selected = [common_genes[i] for i in order]
    # Preserve assayed target genes even when lowly expressed.
    for g in targets:
        if g in common_genes and g not in selected:
            selected.append(g)
    return selected, origin


def _materialize(spec: ContextSpec, filters: dict[str, str], genes: list[str], targets: list[str],
                 out_path: Path, *, pert_col: str, control_label: str, single_col: str,
                 matrix: str, min_genes: int, min_counts: int, max_percent_mito: float,
                 max_controls: int, max_cells_per_pert: int, seed: int,
                 controls_only: bool = False) -> dict:
    a = ad.read_h5ad(spec.path, backed="r")
    label_kwargs = dict(pert_col=pert_col, control_label=control_label, single_col=single_col,
                        min_genes=min_genes, min_counts=min_counts,
                        max_percent_mito=max_percent_mito)
    labels, controls, single = _label_masks(a.obs, spec, filters, **label_kwargs)
    rng = np.random.default_rng(_stable_seed(seed, spec.name + str(out_path)))
    keep = [_sample_indices(controls, max_controls, rng)]
    if not controls_only:
        for target in targets:
            keep.append(_sample_indices(single & (labels == target), max_cells_per_pert, rng))
    idx = np.sort(np.concatenate(keep))
    source_index = pd.Index(a.var_names.astype(str))
    cols = source_index.get_indexer(genes)
    if (cols < 0).any():
        raise ValueError(f"{spec.name} is missing selected genes")
    col_order = np.argsort(cols)
    sub = a[idx, cols[col_order]].to_memory()
    a.file.close()
    X, origin = _raw_matrix(sub, matrix)
    inverse = np.argsort(col_order)
    block = X[:, inverse]
    if sp.issparse(block):
        block = block.tocsr().astype(np.float32)
        block.sum_duplicates()
        block.eliminate_zeros()
        block.sort_indices()
    else:
        block = np.asarray(block, dtype=np.float32)
    out_labels = np.where(labels[idx] == control_label, "non-targeting", labels[idx])
    obs = pd.DataFrame({
        "target_gene": pd.Categorical(out_labels),
        "context": spec.name,
        "celltype": spec.name,
        "source_dataset": Path(spec.path).name,
        "perturbation_type": "CRISPRi",
    }, index=[f"{spec.name}_{i}" for i in range(len(idx))])
    out = ad.AnnData(X=block, obs=obs)
    out.var_names = genes
    out.uns["shadow_provenance"] = {
        "source": str(Path(spec.path).resolve()), "matrix_origin": origin,
        "context": spec.name, "controls_only": controls_only,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path, compression="gzip")
    vc = pd.Series(out_labels).value_counts()
    return {"path": str(out_path.resolve()), "matrix_origin": origin,
            "n_cells": int(out.n_obs), "n_controls": int(vc.get("non-targeting", 0)),
            "per_target_min": int(vc.drop("non-targeting", errors="ignore").min()) if len(vc) > 1 else 0,
            "per_target_max": int(vc.drop("non-targeting", errors="ignore").max()) if len(vc) > 1 else 0}


def prepare_shadow(*, sources: list[ContextSpec], holdout: ContextSpec, out_dir: str,
                   filters: dict[str, str] | None = None, pert_col: str = "perturbation",
                   control_label: str = "control", single_col: str = "nperts",
                   matrix: str = "auto", min_genes: int = 200, min_counts: int = 500,
                   max_percent_mito: float = 20.0, min_cells: int = 30,
                   max_targets: int = 100, max_genes: int = 4000,
                   max_controls: int = 500, max_cells_per_pert: int = 100,
                   cells_per_pert: int = 50, seed: int = 0,
                   target_gene_map_path: str | None = None) -> Path:
    filters = filters or {}
    if not sources:
        raise ValueError("at least one reference context is required")
    names = [s.name for s in sources] + [holdout.name]
    if len(names) != len(set(names)):
        raise ValueError(f"reference and holdout context names must be distinct: {names}")
    challenge_data_root = (Path.cwd() / "data").resolve()
    if any(Path(s.path).resolve().is_relative_to(challenge_data_root)
           for s in sources + [holdout]):
        raise ValueError(
            "repo data/ inputs are forbidden in shadow benchmark preparation; "
            "put independent public data under data_public/"
        )

    label_kwargs = dict(pert_col=pert_col, control_label=control_label, single_col=single_col,
                        min_genes=min_genes, min_counts=min_counts,
                        max_percent_mito=max_percent_mito)
    audits = [_inspect(s, filters, **label_kwargs) for s in sources + [holdout]]
    if any(a["n_controls"] == 0 for a in audits):
        raise ValueError("every context must have eligible non-targeting controls")
    eligible = [set(a["counts"][a["counts"] >= min_cells].index.astype(str)) for a in audits]
    shared = set.intersection(*eligible)
    if not shared:
        raise ValueError(f"no target has >= {min_cells} cells in every context")
    min_support = {g: min(int(a["counts"].get(g, 0)) for a in audits) for g in shared}
    targets = sorted(shared, key=lambda g: (-min_support[g], g))
    if max_targets > 0:
        targets = targets[:max_targets]

    target_gene_map = None
    if target_gene_map_path:
        from .vcc2026_eval import load_target_gene_map
        target_gene_map = load_target_gene_map(target_gene_map_path)
        missing = sorted(set(targets) - set(target_gene_map))
        if missing:
            raise ValueError(
                f"target-gene map misses {len(missing)} selected perturbation(s): {missing[:10]}")
    target_features = ([target_gene_map[t] for t in targets]
                       if target_gene_map else targets)

    common = set(audits[0]["genes"])
    for audit in audits[1:]:
        common &= set(audit["genes"])
    ordered_common = [g for g in audits[0]["genes"] if g in common]
    genes, holdout_origin = _choose_genes(
        holdout, filters, ordered_common, target_features, max_genes, matrix,
        label_kwargs, max_controls, seed)

    out = Path(out_dir)
    refs_dir = out / "references"
    ref_controls_dir = out / "reference_controls"
    val_dir = out / "validation" / holdout.name
    materialized: dict[str, dict] = {}
    for spec in sources:
        materialized[spec.name] = _materialize(
            spec, filters, genes, targets, refs_dir / f"{spec.name}.h5ad",
            pert_col=pert_col, control_label=control_label, single_col=single_col,
            matrix=matrix, min_genes=min_genes, min_counts=min_counts,
            max_percent_mito=max_percent_mito, max_controls=max_controls,
            max_cells_per_pert=max_cells_per_pert, seed=seed,
        )
        materialized[f"{spec.name}:controls"] = _materialize(
            spec, filters, ordered_common, targets, ref_controls_dir / f"{spec.name}.h5ad",
            pert_col=pert_col, control_label=control_label, single_col=single_col,
            matrix=matrix, min_genes=min_genes, min_counts=min_counts,
            max_percent_mito=max_percent_mito, max_controls=max_controls,
            max_cells_per_pert=max_cells_per_pert, seed=seed, controls_only=True,
        )
    materialized[f"{holdout.name}:controls"] = _materialize(
        holdout, filters, ordered_common, targets, val_dir / "controls.h5ad",
        pert_col=pert_col, control_label=control_label, single_col=single_col,
        matrix=matrix, min_genes=min_genes, min_counts=min_counts,
        max_percent_mito=max_percent_mito, max_controls=max_controls,
        max_cells_per_pert=max_cells_per_pert, seed=seed, controls_only=True,
    )
    materialized[f"{holdout.name}:truth"] = _materialize(
        holdout, filters, genes, targets, val_dir / "truth.h5ad",
        pert_col=pert_col, control_label=control_label, single_col=single_col,
        matrix=matrix, min_genes=min_genes, min_counts=min_counts,
        max_percent_mito=max_percent_mito, max_controls=max_controls,
        max_cells_per_pert=max_cells_per_pert, seed=seed,
    )

    pd.Series(genes).to_csv(out / "gene_names.csv", index=False, header=False)
    pd.DataFrame({"target_gene": targets}).to_csv(out / "targets.csv", index=False)
    pd.DataFrame({"target_gene": targets, "n_cells": cells_per_pert}).to_csv(
        out / "pert_counts.csv", index=False)
    source_paths = {s.name: f"references/{s.name}.h5ad" for s in sources}
    cfg = {
        "run_name": f"shadow_{holdout.name}", "output_dir": "outputs",
        "data": {"contexts_dir": f"validation", "control_h5ad": "controls.h5ad",
                 "truth_h5ad": "truth.h5ad", "targets": "targets.csv",
                 "perts": "pert_counts.csv", "gene_list": "gene_names.csv",
                 "pert_col": "target_gene", "control_label": "non-targeting",
                 "context_col": "context", "celltype_col": "celltype"},
        "gwps": {"enabled": True, "sources": source_paths, "pert_col": "target_gene",
                 "control_label": "non-targeting", "counts_space": True,
                 "cache": "effect_library.npz"},
        "esm2": {"enabled": False, "allow_random_fallback": False},
        "embed": {"context_embeddings": ""},
        "predict": {"method": "gwps_weighted", "effect": "logfc",
                    "generator": "multinomial", "cells_per_pert": cells_per_pert,
                    "include_controls": False, "similarity_space": "raw",
                    "pca_components": min(50, max(1, len(sources))), "scale": 1.0,
                    "shrink": "none", "tau": 0.1, "knn_k": 5,
                    "nb_theta": 10.0, "clip_negative": True, "seed": seed},
        "submit": {"packer": "vcc", "gene_dim": len(genes), "max_cell_dim": 400000,
                   "max_counts_per_cell": 1000000, "encoding": 32,
                   "require_counts": True, "reject_controls": True,
                   "verify_targets": True, "check_cell_counts": True},
        "eval": {"profile": "full", "num_threads": 4},
    }
    (out / "shadow.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {
        "benchmark": "shadow-vcc-loco", "challenge_validation_used": False,
        "holdout_response_visible_to_model": False,
        "strict_complete_zero_shot": False,
        "same_dataset_response_references": any(
            Path(s.path).resolve() == Path(holdout.path).resolve() for s in sources),
        "holdout": asdict(holdout), "sources": [asdict(s) for s in sources],
        "filters": filters, "perturbation_modality": "CRISPRi",
        "targets": targets, "n_targets": len(targets), "n_genes": len(genes),
        "n_encoder_input_genes": len(ordered_common),
        "target_gene_map": {
            "role": "gene-axis retention and evaluation target exclusion only",
            "source": str(Path(target_gene_map_path).resolve()) if target_gene_map_path else None,
            "source_sha256": _sha256(target_gene_map_path) if target_gene_map_path else None,
            "used_by_effect_prediction": False,
            "n_mapped_labels": len(targets) if target_gene_map else 0,
            "n_mapped_genes_retained": sum(g in genes for g in set(target_features)),
        },
        "holdout_matrix_origin": holdout_origin, "qc": {
            "min_genes": min_genes, "min_counts": min_counts,
            "max_percent_mito": max_percent_mito, "min_cells_per_target_per_context": min_cells,
        }, "input_audit": [{k: v for k, v in a.items() if k not in {"genes", "counts", "spec"}}
                            | {"spec": asdict(a["spec"]), "n_targets": int(len(a["counts"]))}
                            for a in audits],
        "materialized": materialized,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[shadow] holdout={holdout.name} references={[s.name for s in sources]}")
    print(f"[shadow] {len(targets)} shared CRISPRi targets, {len(genes)} genes -> {out/'shadow.yaml'}")
    return out / "shadow.yaml"


def prepare_zero_shot(*, input_path: str, context_name: str, out_dir: str,
                      filters: dict[str, str] | None = None,
                      pert_col: str = "perturbation", control_label: str = "control",
                      single_col: str = "", matrix: str = "auto",
                      min_genes: int = 200, min_counts: int = 500,
                      max_percent_mito: float = 20.0, max_genes: int = 4000,
                      max_controls: int = 500, max_cells_per_pert: int = 100,
                      cells_per_pert: int = 50, seed: int = 0) -> Path:
    """Create a response-sealed benchmark with no same-dataset effect library.

    The prediction directory contains only controls, target identifiers, genes,
    and a config with GWPS disabled. Perturbed cells are materialized under the
    separate ``sealed`` directory and are only named on the scorer command line.
    Target inclusion is the experimental panel (every eligible non-control
    label); it is never ranked or filtered using response strength.
    """
    filters = filters or {}
    source = Path(input_path).resolve()
    challenge_data_root = (Path.cwd() / "data").resolve()
    if source.is_relative_to(challenge_data_root):
        raise ValueError("repo data/ inputs are forbidden; use an independent public dataset")
    spec = ContextSpec(context_name, str(source))
    label_kwargs = dict(pert_col=pert_col, control_label=control_label, single_col=single_col,
                        min_genes=min_genes, min_counts=min_counts,
                        max_percent_mito=max_percent_mito)
    audit = _inspect(spec, filters, **label_kwargs)
    if audit["n_controls"] == 0:
        raise ValueError("zero-shot context has no eligible controls")
    targets = sorted(map(str, audit["counts"].index))
    if not targets:
        raise ValueError("zero-shot context has no eligible perturbation targets")

    # Gene ranking sees controls only. Known target identifiers may be retained
    # if they happen to be expression-gene IDs (HAR IDs simply do not match).
    genes, matrix_origin = _choose_genes(
        spec, filters, audit["genes"], targets, max_genes, matrix,
        label_kwargs, max_controls, seed)
    root = Path(out_dir)
    blind, sealed = root / "blind", root / "sealed"
    controls_path = blind / "contexts" / context_name / "controls.h5ad"
    truth_path = sealed / context_name / "truth.h5ad"
    controls_info = _materialize(
        spec, filters, genes, targets, controls_path,
        pert_col=pert_col, control_label=control_label, single_col=single_col,
        matrix=matrix, min_genes=min_genes, min_counts=min_counts,
        max_percent_mito=max_percent_mito, max_controls=max_controls,
        max_cells_per_pert=max_cells_per_pert, seed=seed, controls_only=True)
    truth_info = _materialize(
        spec, filters, genes, targets, truth_path,
        pert_col=pert_col, control_label=control_label, single_col=single_col,
        matrix=matrix, min_genes=min_genes, min_counts=min_counts,
        max_percent_mito=max_percent_mito, max_controls=max_controls,
        max_cells_per_pert=max_cells_per_pert, seed=seed)

    blind.mkdir(parents=True, exist_ok=True)
    pd.Series(genes).to_csv(blind / "gene_names.csv", index=False, header=False)
    pd.DataFrame({"target_gene": targets}).to_csv(blind / "targets.csv", index=False)
    pd.DataFrame({"target_gene": targets, "n_cells": cells_per_pert}).to_csv(
        blind / "pert_counts.csv", index=False)
    cfg = {
        "run_name": f"strict_zero_shot_{context_name}", "output_dir": "outputs",
        "data": {"contexts_dir": "contexts", "control_h5ad": "controls.h5ad",
                 "truth_h5ad": "__SEALED_TRUTH_NOT_AVAILABLE__.h5ad",
                 "targets": "targets.csv", "perts": "pert_counts.csv",
                 "gene_list": "gene_names.csv", "pert_col": "target_gene",
                 "control_label": "non-targeting", "context_col": "context",
                 "celltype_col": "celltype"},
        "gwps": {"enabled": False, "sources": {}, "cache": "disabled.npz"},
        "esm2": {"enabled": False, "allow_random_fallback": False},
        "embed": {"context_embeddings": ""},
        "predict": {"method": "no_effect", "effect": "logfc",
                    "generator": "multinomial", "cells_per_pert": cells_per_pert,
                    "include_controls": True, "similarity_space": "raw",
                    "scale": 1.0, "shrink": "none", "seed": seed},
        "submit": {"packer": "cell-eval", "gene_dim": len(genes),
                   "max_cell_dim": 400000, "max_counts_per_cell": 1000000,
                   "encoding": 32, "require_counts": True,
                   "reject_controls": False, "verify_targets": True,
                   "check_cell_counts": True},
        "eval": {"profile": "full", "num_threads": 4},
    }
    config_path = blind / "predict.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    public_contract = {
        "protocol": "strict-response-sealed-zero-shot-v1",
        "context": context_name, "challenge_validation_used": False,
        "same_dataset_perturbation_responses_visible": False,
        "allowed_inputs": ["control raw counts", "target identifier list", "gene list"],
        "forbidden_inputs": ["perturbed cells", "DE statistics", "target-to-neighbor-gene mapping",
                             "same-dataset response references", "result-based tuning"],
        "n_controls": controls_info["n_controls"], "n_targets": len(targets),
        "n_genes": len(genes), "control_h5ad_sha256": _sha256(controls_path),
        "truth_path_present": False,
    }
    (blind / "zero_shot_contract.json").write_text(
        json.dumps(public_contract, indent=2, ensure_ascii=False))
    evaluation_manifest = {
        **public_contract, "truth_path_present": True,
        "source": str(source), "source_sha256": _sha256(source),
        "source_perturbation_column": pert_col, "source_control_label": control_label,
        "filters": filters, "matrix_origin": matrix_origin,
        "target_selection": "all eligible non-control labels; no response-effect filtering",
        "controls": controls_info, "truth": truth_info,
        "truth_h5ad_sha256": _sha256(truth_path),
    }
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, ensure_ascii=False))
    print(f"[zero-shot] blind inputs: {len(targets)} targets, {len(genes)} genes -> {config_path}")
    print(f"[zero-shot] responses sealed separately -> {truth_path}")
    return config_path


def score_zero_shot(config_path: str, pred_path: str, truth_path: str, out_dir: str,
                    engine: str = "cell-eval", profile: str = "full",
                    method_name: str | None = None,
                    target_gene_map: str | None = None,
                    allow_cpu_scorer: bool = False) -> pd.DataFrame:
    """Score an already-created prediction against explicitly supplied sealed truth."""
    cfg = Config.load(config_path).resolve(Path(config_path).parent)
    if cfg.gwps.enabled or cfg.gwps.sources:
        raise ValueError("strict zero-shot scoring requires GWPS/same-dataset references to be disabled")
    pred_file, truth_file = Path(pred_path), Path(truth_path)
    if not pred_file.is_file() or not truth_file.is_file():
        raise FileNotFoundError("prediction and sealed truth must already exist")
    pred, truth = ad.read_h5ad(pred_file), ad.read_h5ad(truth_file)
    genes, targets = io.read_gene_list(cfg.data.gene_list), io.read_targets(cfg.data.targets)
    if list(map(str, pred.var_names)) != list(map(str, genes)):
        raise ValueError("prediction genes/order differ from the blind gene list")
    allowed = set(map(str, targets)) | {cfg.data.control_label}
    pred_labels = set(pred.obs[cfg.data.pert_col].astype(str))
    if pred_labels != allowed:
        raise ValueError(f"prediction label set mismatch: missing={allowed-pred_labels}, extra={pred_labels-allowed}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg.eval.profile = profile
    if engine == "cell-eval2":
        from . import vcc2026_eval
        metrics = vcc2026_eval.evaluate(
            pred_file, truth_file, out / "vcc2026",
            out.parent / "vcc2026_local_scale",
            pert_col=cfg.data.pert_col, control=cfg.data.control_label,
            target_gene_map_path=target_gene_map,
            require_gpu=not allow_cpu_scorer,
        )
        values = metrics
        profile = "vcc2026"
    elif engine == "cell-eval":
        metrics = evaluate.evaluate_cell_eval(pred_file, truth_file, cfg, out, profile=profile)
        values = {**metrics, **{f"fingerprint::{k}": v for k, v in evaluate.fingerprint(metrics).items()}}
    elif engine == "local":
        values = evaluate.evaluate_local(pred, truth, cfg)
    else:
        raise ValueError("engine must be local or cell-eval")
    row = {"protocol": "strict-response-sealed-zero-shot-v1",
           "method": method_name or cfg.predict.method,
           "engine": engine, "profile": profile,
           "metric_protocol": ("cell-eval2-vcc2026-external-local-anchors-v1"
                               if engine == "cell-eval2" else "legacy-proxy"),
           "official_challenge_score": False, **values}
    frame = pd.DataFrame([row])
    frame.to_csv(out / "zero_shot_results.csv", index=False)
    audit = {"config_sha256": _sha256(config_path), "prediction_sha256": _sha256(pred_file),
             "truth_sha256": _sha256(truth_file), "gwps_enabled": cfg.gwps.enabled,
             "same_dataset_response_sources": len(cfg.gwps.sources),
             "prediction_created_before_truth_scoring": True}
    (out / "scoring_audit.json").write_text(json.dumps(audit, indent=2))
    return frame


def compact_source(input_path: str, out_path: str, filters: dict[str, str],
                   matrix: str = "counts") -> Path:
    """Write a filtered, counts-only H5AD without loading every H5AD layer."""
    source = Path(input_path)
    target = Path(out_path)
    with h5py.File(source, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
        mask = np.ones(len(obs), dtype=bool)
        for col, value in filters.items():
            if col not in obs:
                raise KeyError(f"{source} has no obs[{col!r}]")
            mask &= obs[col].astype(str).to_numpy() == value
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError(f"filters selected no cells: {filters}")
        if matrix == "counts" or (matrix == "auto" and "counts" in f.get("layers", {})):
            if "layers" not in f or "counts" not in f["layers"]:
                raise KeyError("matrix=counts requested but layers[counts] is absent")
            elem = f["layers/counts"]
            origin = "layers[counts]"
        elif matrix in {"auto", "X"}:
            elem = f["X"]
            origin = "X"
        else:
            raise ValueError("matrix must be auto, X, or counts")
        encoding = str(elem.attrs.get("encoding-type", ""))
        if encoding in {"csr_matrix", "csc_matrix"}:
            X = ad.io.sparse_dataset(elem)[rows, :]
        else:
            X = np.asarray(elem[rows, :])
    X = X.astype(np.float32)
    if not _is_integer_like(X):
        raise ValueError(f"selected {origin} is not raw integer-like counts")
    compact = ad.AnnData(X=X, obs=obs.iloc[rows].copy(), var=var.copy())
    compact.uns["shadow_compaction"] = {
        "source": str(source.resolve()), "matrix_origin": origin,
        "filters": filters, "lossless_cell_gene_slice": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    compact.write_h5ad(target, compression="gzip")
    print(f"[shadow] compacted {len(rows)} cells x {compact.n_vars} genes from {origin} -> {target}")
    return target


def _load_npz_keys(path: str) -> set[str]:
    with np.load(path, allow_pickle=False) as z:
        return set(z.files)


def benchmark(config_path: str, embeddings: list[str] | None = None,
              fm_predictions: list[str] | None = None,
              methods: list[str] | None = None, engine: str = "local",
              profile: str = "vcc", ceiling: bool = False,
              checkpoint_path: str | None = None, resume: bool = False,
              six_only: bool = False, target_gene_map: str | None = None,
              allow_cpu_scorer: bool = False,
              artifact_root: str | None = None) -> pd.DataFrame:
    """Compare classical and frozen representations with resumable progress."""
    methods = methods or ["no_effect", "global_mean", "gwps_direct", "gwps_nearest", "gwps_weighted"]
    specs = [("raw", "raw", ""), ("pca", "pca", "")]
    for item in embeddings or []:
        if "=" not in item:
            raise ValueError(f"expected MODEL=EMBEDDINGS.npz, got {item!r}")
        name, path = item.split("=", 1)
        specs.append((name, "embed", path))

    jobs: list[tuple[str, str, str, str, str]] = []
    for encoder, space, path in specs:
        use_methods = methods if encoder == "raw" else [
            m for m in methods if m in {"gwps_nearest", "gwps_weighted"}]
        jobs.extend(("representation", encoder, space, path, method)
                    for method in use_methods)
    for item in fm_predictions or []:
        if "=" not in item:
            raise ValueError(f"expected state|stack=PREDICTION.h5ad, got {item!r}")
        name, path = item.split("=", 1)
        if name not in {"state", "stack"}:
            raise ValueError("direct frozen predictions currently support state or stack")
        jobs.append(("direct", name, "direct", path, name))

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    rows: list[dict] = []
    if resume and checkpoint and checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
    completed = {(str(r.get("representation")), str(r.get("method")))
                 for r in rows if r.get("status") == "ok"}

    def persist() -> None:
        if not checkpoint:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        if not frame.empty and {"representation", "method"} <= set(frame.columns):
            frame = frame.drop_duplicates(["representation", "method"], keep="last")
        frame.to_csv(checkpoint, index=False)

    total = len(jobs)
    for job_index, (kind, encoder, space, path, method) in enumerate(jobs, 1):
        percent = 100.0 * job_index / max(total, 1)
        tag = f"{encoder}/{method}"
        if (encoder, method) in completed:
            print(f"[matrix {job_index:02d}/{total:02d} {percent:5.1f}%] "
                  f"SKIP completed {tag}", flush=True)
            continue
        started = time.monotonic()
        effective_profile = "vcc2026" if engine == "cell-eval2" else profile
        print(f"[matrix {job_index:02d}/{total:02d} {percent:5.1f}%] START {tag} "
              f"engine={engine} profile={effective_profile}", flush=True)
        cfg = Config.load(config_path).resolve(Path(config_path).parent)
        if artifact_root:
            root = Path(artifact_root).resolve()
            # Detailed predictions and official raw aggregates must not collide
            # between raw/PCA/frozen representations. The scale is shared because
            # it depends only on the real dataset, not on the predictor.
            cfg.output_dir = str(root / "representations" / encoder)
            cfg._eval_scale_root = str(root / "shared_vcc2026_local_scale")
        cfg.eval.profile = profile
        cfg._eval_ceiling = ceiling and job_index == 1
        cfg._eval_six_only = six_only
        cfg._eval_target_gene_map = target_gene_map
        cfg._eval_allow_cpu_scorer = allow_cpu_scorer
        try:
            if kind == "representation":
                cfg.predict.similarity_space = space
                cfg.embed.context_embeddings = str(Path(path).resolve()) if path else ""
                if space == "embed":
                    required = set(cfg.gwps.sources) | set(io.list_contexts(
                        cfg.data.contexts_dir, cfg.data.control_h5ad))
                    if not Path(path).exists():
                        raise FileNotFoundError(f"missing {path}")
                    missing = required - _load_npz_keys(path)
                    if missing:
                        raise KeyError(f"embedding artifact missing keys {sorted(missing)}")
            else:
                cfg._fm_pred = str(Path(path).resolve())
                if not Path(path).exists():
                    raise FileNotFoundError(f"missing {path}")
            result = L.run_ladder(cfg, [method], engine=engine).reset_index()
            for record in result.to_dict("records"):
                rows.append({"representation": encoder, "status": "ok", **record})
        except Exception as exc:
            rows.append({"representation": encoder, "method": method,
                         "status": "error", "error": str(exc)})
        persist()
        elapsed = time.monotonic() - started
        status = rows[-1].get("status", "unknown")
        print(f"[matrix {job_index:02d}/{total:02d} {percent:5.1f}%] DONE  {tag} "
              f"status={status} elapsed={elapsed:.1f}s checkpoint={checkpoint or '-'}", flush=True)
    frame = pd.DataFrame(rows)
    if not frame.empty and {"representation", "method"} <= set(frame.columns):
        frame = frame.drop_duplicates(["representation", "method"], keep="last")
    return frame


def pool_embeddings(named_h5ad: list[str], obsm_key: str, out: str) -> Path:
    pooled = {}
    for item in named_h5ad:
        spec = parse_named_path(item)
        a = ad.read_h5ad(spec.path)
        if obsm_key == "X":
            values = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        elif obsm_key not in a.obsm:
            raise KeyError(f"{spec.path} lacks obsm[{obsm_key!r}]; have {list(a.obsm.keys())}")
        else:
            values = np.asarray(a.obsm[obsm_key])
        pooled[spec.name] = values.mean(axis=0).astype(np.float32)
    path = Path(out); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pooled)
    location = "X" if obsm_key == "X" else f"obsm[{obsm_key}]"
    print(f"[shadow] pooled {len(pooled)} contexts from {location} -> {path}")
    return path
