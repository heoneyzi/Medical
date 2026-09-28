"""Command-isolated frozen model orchestration and artifact validation."""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from .config import Config


class AuditRequired(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    role: str
    track: str
    official_repo: str = ""
    repo: str = ""
    repo_commit: str = ""
    executable: str = ""
    checkpoint: str = ""
    checkpoint_md5: str = ""
    checkpoint_sha256: str = ""
    required_checkpoint_files: tuple[str, ...] = ()
    required_runtime_files: tuple[str, ...] = ()
    input_gene_id: str = "symbol"
    min_gene_mapping_fraction: float = 0.0
    min_input_genes: int = 0
    required_obs_defaults: tuple[tuple[str, str], ...] = ()
    cwd: str = ""
    command: tuple[str, ...] = ()
    output_glob: str = ""
    output_type: str = ""
    embedding_key: str = ""
    expected_dim: int | None = None
    output_cell_identity: str = "exact_ids"
    status: str = "enabled"
    reason: str = ""

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "ModelSpec":
        values = dict(raw)
        values["name"] = name
        values["command"] = tuple(map(str, values.get("command", ())))
        values["required_checkpoint_files"] = tuple(
            map(str, values.get("required_checkpoint_files", ())))
        values["required_runtime_files"] = tuple(
            map(str, values.get("required_runtime_files", ())))
        values["required_obs_defaults"] = tuple(
            (str(key), str(value))
            for key, value in values.get("required_obs_defaults", {}).items())
        return cls(**values)


def load_registry(path: str | Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("models"), dict):
        raise ValueError("external frozen registry must use schema_version: 1")
    return {name: ModelSpec.from_dict(name, spec)
            for name, spec in raw["models"].items()}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _executable(root: Path, value: str) -> Path | None:
    if "/" in value:
        path = _resolve(root, value)
        return path if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(value)
    return Path(found) if found else None


def preflight_model(spec: ModelSpec, root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    result: dict[str, Any] = {"model": spec.name, "role": spec.role,
                              "track": spec.track, "ready": False, "errors": []}
    if spec.status != "enabled" or spec.role != "context_encoder":
        result["status"] = spec.status
        result["errors"].append(spec.reason or "not enrolled in context matrix")
        return result
    exe = _executable(root, spec.executable)
    repo = _resolve(root, spec.repo)
    checkpoint = _resolve(root, spec.checkpoint)
    if exe is None:
        result["errors"].append(f"missing executable: {spec.executable}")
    if not repo.is_dir():
        result["errors"].append(f"missing official repo clone: {repo}")
    elif spec.repo_commit:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            capture_output=True)
        observed = revision.stdout.strip()
        result["repo_commit"] = observed
        if revision.returncode or observed != spec.repo_commit:
            result["errors"].append(
                f"repo commit mismatch: expected {spec.repo_commit}, found {observed or 'unknown'}")
        tracked_changes = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            text=True, capture_output=True)
        if tracked_changes.returncode or tracked_changes.stdout.strip():
            result["errors"].append(
                "official repo has tracked modifications; restore the pinned checkout")
    if not checkpoint.exists():
        result["errors"].append(f"missing checkpoint: {checkpoint}")
    elif checkpoint.is_file() and checkpoint.stat().st_size == 0:
        result["errors"].append(f"empty checkpoint: {checkpoint}")
    else:
        for relative in spec.required_checkpoint_files:
            if not (checkpoint / relative).is_file():
                result["errors"].append(f"missing checkpoint member: {checkpoint / relative}")
        if checkpoint.is_file() and spec.checkpoint_md5:
            observed_md5 = _md5_file(checkpoint)
            result["checkpoint_md5"] = observed_md5
            if observed_md5 != spec.checkpoint_md5:
                result["errors"].append(
                    f"checkpoint MD5 mismatch: expected {spec.checkpoint_md5}, found {observed_md5}")
        if spec.checkpoint_sha256:
            # Directory checkpoints (for example TranscriptFormer) are pinned by
            # a deterministic digest over relative paths and file contents.
            observed_sha256 = digest_path(checkpoint)
            result["checkpoint_sha256"] = observed_sha256
            if observed_sha256 != spec.checkpoint_sha256:
                result["errors"].append(
                    "checkpoint SHA-256 mismatch: expected "
                    f"{spec.checkpoint_sha256}, found {observed_sha256}")
    for value in spec.required_runtime_files:
        runtime_file = _resolve(root, value)
        if not runtime_file.is_file():
            result["errors"].append(f"missing runtime file: {runtime_file}")
        elif runtime_file.stat().st_size == 0:
            result["errors"].append(f"empty runtime file: {runtime_file}")
    result.update({"executable": str(exe) if exe else "", "repo": str(repo),
                   "checkpoint": str(checkpoint)})
    result["ready"] = not result["errors"]
    return result


def context_inputs(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    cfg = Config.load(config_path).resolve(config_path.parent)
    split = config_path.parent
    inputs: dict[str, Path] = {}
    control_dir = split / "reference_controls"
    if control_dir.is_dir():
        inputs.update({path.stem: path.resolve() for path in sorted(control_dir.glob("*.h5ad"))})
    else:
        inputs.update({name: Path(path).resolve() for name, path in cfg.gwps.sources.items()})
    contexts_dir = Path(cfg.data.contexts_dir)
    for directory in sorted(contexts_dir.iterdir()):
        path = directory / cfg.data.control_h5ad
        if directory.is_dir() and path.is_file():
            inputs[directory.name] = path.resolve()
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise AuditRequired(f"AUDIT_REQUIRED: missing context inputs {missing}")
    expected_sources = set(cfg.gwps.sources)
    if not expected_sources <= set(inputs):
        raise AuditRequired(
            f"AUDIT_REQUIRED: missing source control inputs {sorted(expected_sources - set(inputs))}")
    return inputs


def _stage_ensembl_input(input_path: Path, output: Path, gene_map: Path,
                         min_fraction: float,
                         obs_defaults: tuple[tuple[str, str], ...]) -> tuple[Path, dict[str, Any]]:
    if not gene_map.is_file():
        raise AuditRequired(
            f"AUDIT_REQUIRED: TranscriptFormer requires an explicit gene map: {gene_map}")
    mapping = pd.read_csv(gene_map)
    required = {"vcc_symbol", "ensembl_id", "mapping_status"}
    if not required <= set(mapping.columns):
        raise AuditRequired(f"AUDIT_REQUIRED: gene map lacks {sorted(required - set(mapping.columns))}")
    good = mapping[mapping["mapping_status"].isin(["exact", "alias"])].copy()
    good = good[good["ensembl_id"].notna() & good["vcc_symbol"].notna()]
    if good["vcc_symbol"].duplicated(False).any():
        raise AuditRequired("AUDIT_REQUIRED: one symbol has multiple accepted mapping rows")
    # If an approved symbol and one of its historical aliases are both present,
    # keep the approved exact symbol. Multiple exact symbols for one Ensembl ID
    # remain ambiguous and are excluded rather than selected arbitrarily.
    selected = []
    duplicate_aliases_dropped = []
    ambiguous_ensembl = []
    for ensembl, group in good.groupby("ensembl_id", sort=False):
        if len(group) == 1:
            selected.append(group.iloc[0])
            continue
        exact = group[group["mapping_status"] == "exact"]
        if len(exact) == 1:
            selected.append(exact.iloc[0])
            duplicate_aliases_dropped.extend(
                group.loc[group.index.difference(exact.index), "vcc_symbol"].astype(str).tolist())
        else:
            ambiguous_ensembl.append(str(ensembl))
    selected_frame = pd.DataFrame(selected, columns=good.columns)
    lookup = dict(zip(selected_frame["vcc_symbol"].astype(str),
                      selected_frame["ensembl_id"].astype(str)))
    data = ad.read_h5ad(input_path)
    keep = np.asarray([str(gene) in lookup for gene in data.var_names])
    fraction = float(keep.mean())
    if fraction < min_fraction:
        raise AuditRequired(
            f"AUDIT_REQUIRED: only {fraction:.1%} genes map uniquely to Ensembl; require {min_fraction:.1%}")
    staged = data[:, keep].copy()
    staged.var["ensembl_id"] = [lookup[str(gene)] for gene in staged.var_names]
    applied_obs_defaults = {}
    for column, value in obs_defaults:
        if column not in staged.obs:
            staged.obs[column] = value
            applied_obs_defaults[column] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    staged.write_h5ad(output, compression="gzip")
    return output, {"input_genes": int(data.n_vars), "mapped_genes": int(keep.sum()),
                    "mapping_fraction": fraction,
                    "duplicate_alias_symbols_dropped": sorted(duplicate_aliases_dropped),
                    "ambiguous_ensembl_ids_dropped": sorted(ambiguous_ensembl),
                    "applied_obs_defaults": applied_obs_defaults,
                    "gene_map_sha256": _sha256_file(gene_map)}


def _stage_zero_padded_symbol_input(input_path: Path, output: Path,
                                    gene_map: Path,
                                    min_input_genes: int) -> tuple[Path, dict[str, Any]]:
    """Pad a compact symbol panel with explicit zero genes for model input only.

    scPRINT-2's official Preprocessor rejects panels with fewer than 10,000
    mapped genes before it performs its own all-ontology zero padding.  Compact
    benchmark panels can therefore be biologically valid but fail that ordering
    constraint.  Padding approved HGNC symbols here changes neither observed
    counts nor cells and is recorded in provenance.
    """
    if not gene_map.is_file():
        raise AuditRequired(
            f"AUDIT_REQUIRED: zero-padding requires an explicit gene map: {gene_map}")
    mapping = pd.read_csv(gene_map)
    required = {"vcc_symbol", "ensembl_id", "mapping_status"}
    if not required <= set(mapping.columns):
        raise AuditRequired(f"AUDIT_REQUIRED: gene map lacks {sorted(required-set(mapping.columns))}")
    approved = mapping[
        (mapping["mapping_status"] == "exact")
        & mapping["vcc_symbol"].notna()
        & mapping["ensembl_id"].notna()
    ]["vcc_symbol"].astype(str)
    if approved.duplicated().any():
        raise AuditRequired("AUDIT_REQUIRED: exact HGNC symbol padding list is not unique")

    data = ad.read_h5ad(input_path)
    original_genes = set(data.var_names.astype(str))
    needed = max(0, min_input_genes - data.n_vars)
    candidates = sorted(set(approved) - original_genes)[:needed]
    if len(candidates) != needed:
        raise AuditRequired(
            f"AUDIT_REQUIRED: need {needed} zero-padding genes; found {len(candidates)}")
    if needed == 0:
        return input_path, {"input_genes_before_zero_padding": int(data.n_vars),
                            "zero_padded_genes": 0,
                            "gene_map_sha256": _sha256_file(gene_map)}
    source = data.X.tocsr() if sp.issparse(data.X) else sp.csr_matrix(data.X)
    zeros = sp.csr_matrix((data.n_obs, needed), dtype=source.dtype)
    matrix = sp.hstack([source, zeros], format="csr")
    var = pd.concat([data.var.copy(), pd.DataFrame(index=pd.Index(candidates))], axis=0)
    staged = ad.AnnData(X=matrix, obs=data.obs.copy(), var=var)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged.write_h5ad(output, compression="gzip")
    return output, {
        "input_genes_before_zero_padding": int(data.n_vars),
        "input_genes_after_zero_padding": int(staged.n_vars),
        "zero_padded_genes": needed,
        "zero_padding_contract": "HGNC-approved exact symbols; all added counts are zero",
        "zero_padding_genes_sha256": hashlib.sha256(
            "\n".join(candidates).encode()).hexdigest(),
        "gene_map_sha256": _sha256_file(gene_map),
    }


def _format_values(spec: ModelSpec, root: Path, input_path: Path,
                   work_dir: Path, context: str, executable: Path) -> dict[str, str]:
    return {
        "root": str(root), "repo": str(_resolve(root, spec.repo)),
        "checkpoint": str(_resolve(root, spec.checkpoint)),
        "executable": str(executable), "input": str(input_path),
        "work_dir": str(work_dir), "context": context,
    }


def _audit_count_input(path: Path) -> dict[str, Any]:
    """Validate the common raw-count contract before any official model command."""
    data = ad.read_h5ad(path)
    errors = []
    if not data.obs_names.is_unique:
        errors.append("cell IDs are not unique")
    if not data.var_names.is_unique:
        errors.append("gene IDs are not unique")
    values = data.X.data if sp.issparse(data.X) else np.asarray(data.X).ravel()
    if not np.issubdtype(values.dtype, np.number):
        errors.append(f"X is not numeric ({values.dtype})")
    elif not np.isfinite(values).all():
        errors.append("X contains non-finite values")
    elif (values < 0).any():
        errors.append("X contains negative values")
    elif not np.allclose(values, np.rint(values), rtol=0, atol=1e-5):
        errors.append("X is not raw integer-like counts")
    if data.n_obs == 0 or data.n_vars == 0:
        errors.append(f"empty matrix shape {data.shape}")
    if errors:
        raise AuditRequired(f"AUDIT_REQUIRED: invalid frozen-model input {path}: "
                            + "; ".join(errors))
    if sp.issparse(data.X):
        library_sizes = np.asarray(data.X.sum(axis=1)).ravel()
        detected = np.diff(data.X.tocsr().indptr)
        n_nonzero = int(data.X.nnz)
    else:
        dense = np.asarray(data.X)
        library_sizes = dense.sum(axis=1)
        detected = np.count_nonzero(dense, axis=1)
        n_nonzero = int(np.count_nonzero(dense))
    return {
        "matrix_contract": "X/raw-nonnegative-integer-counts",
        "matrix_storage": "sparse" if sp.issparse(data.X) else "dense",
        "n_nonzero": n_nonzero,
        "median_library_size": float(np.median(library_sizes)),
        "fraction_zero": float(1.0 - n_nonzero / (data.n_obs * data.n_vars)),
        "median_detected_genes": float(np.median(detected)),
        "unique_cell_ids": True,
        "unique_gene_ids": True,
    }


def _read_native(spec: ModelSpec, pattern: str,
                 input_obs: pd.Index) -> tuple[np.ndarray, str, float, str]:
    candidates = sorted(Path(path) for path in glob.glob(pattern))
    valid: list[tuple[np.ndarray, Path, float, str]] = []
    for path in candidates:
        if spec.output_type == "npy":
            values = np.load(path, allow_pickle=False)
            if values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0, :]
            retained = 1.0 if values.ndim == 2 and values.shape[0] == len(input_obs) else 0.0
            identity_validation = "positional_exact_count"
        elif spec.output_type == "h5ad":
            data = ad.read_h5ad(path)
            if spec.embedding_key not in data.obsm:
                continue
            values = np.asarray(data.obsm[spec.embedding_key])
            output_obs = pd.Index(data.obs_names.astype(str))
            if spec.output_cell_identity == "positional_exact_count":
                retained = 1.0 if len(output_obs) == len(input_obs) else 0.0
                identity_validation = "positional_exact_count"
            elif spec.output_cell_identity == "exact_ids":
                unknown = output_obs.difference(input_obs.astype(str))
                if len(unknown):
                    raise AuditRequired(
                        f"AUDIT_REQUIRED: {spec.name} output introduced unknown cell IDs: "
                        f"{unknown[:5].tolist()}")
                retained = len(output_obs) / max(len(input_obs), 1)
                identity_validation = "exact_ids"
            else:
                raise ValueError(
                    f"unknown output_cell_identity {spec.output_cell_identity!r}")
        else:
            raise ValueError(f"unknown output_type {spec.output_type!r}")
        if values.ndim == 2 and values.shape[0] > 0 and np.isfinite(values).all():
            valid.append((np.asarray(values, dtype=np.float32), path, retained,
                          identity_validation))
    if len(valid) != 1:
        raise AuditRequired(
            f"AUDIT_REQUIRED: expected one valid {spec.name} output for {pattern}; found {len(valid)}")
    values, path, retained, identity_validation = valid[0]
    if spec.expected_dim is not None and values.shape[1] != spec.expected_dim:
        raise AuditRequired(
            f"AUDIT_REQUIRED: {spec.name} dim {values.shape[1]} != verified {spec.expected_dim}")
    return values, str(path), retained, identity_validation


def run_context(spec: ModelSpec, root: str | Path, input_path: str | Path,
                context: str, artifact_dir: str | Path, *,
                gene_map: str | Path | None = None, force: bool = False,
                preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    root, input_path, artifact_dir = Path(root).resolve(), Path(input_path).resolve(), Path(artifact_dir)
    audit = preflight if preflight is not None else preflight_model(spec, root)
    if audit.get("model") != spec.name:
        raise ValueError(f"preflight model {audit.get('model')!r} != {spec.name!r}")
    if not audit["ready"]:
        raise AuditRequired(f"AUDIT_REQUIRED: {spec.name}: {'; '.join(audit['errors'])}")
    executable = Path(audit["executable"])
    checkpoint = Path(audit["checkpoint"])
    checkpoint_sha = audit.get("checkpoint_sha256") or digest_path(checkpoint)
    input_sha = _sha256_file(input_path)
    work_dir = artifact_dir / spec.name / context
    cells_path, context_path = work_dir / "cells.npy", work_dir / "context.npy"
    metadata_path = work_dir / "metadata.json"
    identity = {"model": spec.name, "input_sha256": input_sha,
                "checkpoint_sha256": checkpoint_sha, "repo_commit": audit.get("repo_commit", "")}
    if not force and all(path.is_file() for path in (cells_path, context_path, metadata_path)):
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in identity.items()):
            print(f"[external-frozen] REUSE {spec.name}/{context}", flush=True)
            return metadata

    work_dir.mkdir(parents=True, exist_ok=True)
    native_dir = work_dir / "native"; native_dir.mkdir(exist_ok=True)
    model_input = input_path
    mapping_meta: dict[str, Any] = {}
    if spec.input_gene_id == "ensembl":
        if gene_map is None:
            raise AuditRequired("AUDIT_REQUIRED: --gene-map is required for Ensembl models")
        model_input, mapping_meta = _stage_ensembl_input(
            input_path, work_dir / "input.ensembl.h5ad", Path(gene_map),
            spec.min_gene_mapping_fraction, spec.required_obs_defaults)
    elif spec.min_input_genes:
        if gene_map is None:
            raise AuditRequired("AUDIT_REQUIRED: --gene-map is required for gene-panel padding")
        model_input, mapping_meta = _stage_zero_padded_symbol_input(
            input_path, work_dir / "input.zero-padded-symbols.h5ad", Path(gene_map),
            spec.min_input_genes)
    input_audit = _audit_count_input(model_input)
    input_data = ad.read_h5ad(model_input, backed="r")
    input_obs = input_data.obs_names.astype(str).copy()
    n_input, n_genes = input_data.n_obs, input_data.n_vars
    input_data.file.close()

    values = _format_values(spec, root, model_input, work_dir, context, executable)
    command = [part.format_map(values) for part in spec.command]
    cwd = _resolve(root, spec.cwd) if spec.cwd else root
    log_path = work_dir / "run.log"
    print(f"[external-frozen] START {spec.name}/{context} cells={n_input} genes={n_genes}", flush=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{spec.name}/{context}] {line}", end="", flush=True)
            log.write(line); log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{spec.name}/{context} failed with exit {return_code}; see {log_path}")
    output_pattern = spec.output_glob.format_map(values)
    embeddings, native_output, retained, identity_validation = _read_native(
        spec, output_pattern, input_obs)
    if retained < 0.90:
        raise AuditRequired(
            f"AUDIT_REQUIRED: {spec.name} retained only {retained:.1%} of input cells")
    np.save(cells_path, embeddings, allow_pickle=False)
    pooled = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    np.save(context_path, pooled, allow_pickle=False)
    metadata = {
        **identity, **mapping_meta, **input_audit, "display_name": spec.display_name,
        "role": spec.role, "track": spec.track, "official_repo": spec.official_repo,
        "input": str(input_path), "model_input": str(model_input),
        "n_input_cells": int(n_input), "n_output_cells": int(embeddings.shape[0]),
        "retained_fraction": retained, "embedding_dim": int(embeddings.shape[1]),
        "cell_identity_validation": identity_validation,
        "mean_cell_embedding_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "pooled_context_embedding_norm": float(np.linalg.norm(pooled)),
        "embedding_key": spec.embedding_key, "native_output": native_output,
        "cells_sha256": _sha256_file(cells_path),
        "context_sha256": _sha256_file(context_path),
        "command": command, "cwd": str(cwd), "elapsed_seconds": time.monotonic() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[external-frozen] DONE {spec.name}/{context} dim={embeddings.shape[1]} "
          f"elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)
    return metadata


def pool_model(spec: ModelSpec, inputs: dict[str, Path], artifact_dir: str | Path,
               output: str | Path) -> Path:
    artifact_dir, output = Path(artifact_dir), Path(output)
    pooled = {}
    dimensions = set()
    for context in inputs:
        path = artifact_dir / spec.name / context / "context.npy"
        if not path.is_file():
            raise AuditRequired(f"AUDIT_REQUIRED: missing standardized artifact {path}")
        pooled[context] = np.load(path, allow_pickle=False).astype(np.float32)
        dimensions.add(pooled[context].shape)
    if len(dimensions) != 1:
        raise AuditRequired(f"AUDIT_REQUIRED: inconsistent {spec.name} context dimensions {dimensions}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **pooled)
    contexts = list(inputs)
    pairwise = []
    for left_index, left in enumerate(contexts):
        for right in contexts[left_index + 1:]:
            a, b = pooled[left].astype(np.float64), pooled[right].astype(np.float64)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = float(np.dot(a, b) / denom) if denom else 0.0
            pairwise.append({
                "left": left, "right": right, "cosine_similarity": cosine,
                "euclidean_distance": float(np.linalg.norm(a - b)),
            })
    context_qc = {}
    for context in contexts:
        metadata = json.loads(
            (artifact_dir / spec.name / context / "metadata.json").read_text())
        context_qc[context] = {
            key: metadata[key] for key in (
                "n_input_cells", "n_output_cells", "retained_fraction",
                "median_library_size", "fraction_zero", "median_detected_genes",
                "mean_cell_embedding_norm", "pooled_context_embedding_norm",
            )
        }
    output.with_suffix(".qc.json").write_text(json.dumps({
        "model": spec.name, "contexts": context_qc, "pairwise_context_distance": pairwise,
    }, indent=2, sort_keys=True))
    return output
