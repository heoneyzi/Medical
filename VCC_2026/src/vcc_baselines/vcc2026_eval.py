"""Exact VCC 2026 metric evaluation through Arc's public ``cell-eval2``.

This module deliberately contains no metric reimplementation.  It delegates raw
metrics, the generic-response baseline, the five seeded split-half replicate
anchors, and policy scaling to the official ``vcc2026`` preset.  For an external
dataset the resulting scale is *local to that dataset*; it is not an official
Challenge leaderboard score because it does not use the Challenge reference
panel or its server-side anchor bundle.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import anndata as ad
import pandas as pd
import scipy.sparse as sp
import yaml

from .submit import executable


CELL_EVAL2_VERSION = "0.16.0"
PRESET = "vcc2026"
SCORED_METRICS = {
    "pds": "pds_cosine",
    "mse": "expr_mse_unbiased_capped_norm",
    "nmae": "de_wilcoxon_lfc_nmae",
    "fid": "de_wilcoxon_direction_fidelity_yield_raw",
    "reach": "de_wilcoxon_direction_reach_raw",
    "jac": "de_wilcoxon_sig_jaccard",
}


def _sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_target_gene_map(path: str | Path | None) -> dict[str, str] | None:
    """Load an evaluator-only perturbation-label to target-gene map.

    JSON objects are accepted directly.  CSV inputs use ``name`` and
    ``target_gene_name`` (GSE270828 feature README), or the generic
    ``target`` and ``gene`` column pair.  Multiple guides may repeat one
    perturbation, but conflicting target genes are rejected.
    """
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"target-gene map JSON must be an object: {source}")
        mapping = {str(k): str(v) for k, v in raw.items() if pd.notna(v)}
    else:
        frame = pd.read_csv(source)
        pairs = (("name", "target_gene_name"), ("target", "gene"),
                 ("perturbation", "target_gene"))
        try:
            key_col, value_col = next((k, v) for k, v in pairs if {k, v} <= set(frame.columns))
        except StopIteration as exc:
            raise ValueError(
                f"{source} needs name/target_gene_name, target/gene, or "
                "perturbation/target_gene columns"
            ) from exc
        mapping = {}
        for key, group in frame.dropna(subset=[key_col, value_col]).groupby(key_col):
            values = sorted(set(group[value_col].astype(str)))
            if len(values) != 1:
                raise ValueError(f"{source}: {key!r} maps to conflicting genes {values}")
            mapping[str(key)] = values[0]
    if not mapping:
        raise ValueError(f"empty target-gene map: {source}")
    # The published GSE270828 feature table spells positive controls as
    # ``cont_RAB1A`` while the exported Seurat metadata spells them
    # ``cont-RAB1A``.  Add the lossless label alias so evaluation excludes the
    # intended gene instead of silently treating four controls as unresolved.
    for key, value in list(mapping.items()):
        if key.startswith("cont_"):
            alias = "cont-" + key.removeprefix("cont_")
            if alias in mapping and mapping[alias] != value:
                raise ValueError(f"{source}: alias {alias!r} maps to conflicting genes")
            mapping[alias] = value
    return mapping


def _canonical_copy(source: str | Path, destination: str | Path) -> Path:
    """Make the sparse matrix canonical without changing cells, genes or counts."""
    source, destination = Path(source), Path(destination)
    stamp = destination.with_suffix(destination.suffix + ".source.json")
    source_hash = _sha256(source)
    if destination.is_file() and stamp.is_file():
        meta = json.loads(stamp.read_text())
        if (meta.get("source_sha256") == source_hash
                and meta.get("canonical_sha256") == _sha256(destination)):
            return destination

    value = ad.read_h5ad(source)
    if sp.issparse(value.X):
        matrix = value.X.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        value.X = matrix
    destination.parent.mkdir(parents=True, exist_ok=True)
    value.write_h5ad(destination, compression="gzip")
    stamp.write_text(json.dumps({
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "canonical_sha256": _sha256(destination),
        "operation": "csr.sum_duplicates+eliminate_zeros+sort_indices",
        "biological_values_changed": False,
    }, indent=2))
    return destination


def _official_config(path: Path, *, pert_col: str, control: str,
                     target_gene_map: Mapping[str, str] | None) -> dict:
    from cell_eval2.config import EvalConfig
    from cell_eval2.competition import competition_digest

    cfg = EvalConfig.from_preset(PRESET)
    cfg = replace(cfg, pert_col=pert_col, control=control,
                  target_gene_map=dict(target_gene_map) if target_gene_map else None)
    payload = cfg.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return {
        "cell_eval2_version": importlib.metadata.version("cell-eval2"),
        "vcc2026_competition_digest": competition_digest(),
        "config_sha256": _sha256(path),
        "target_gene_map_sha256": _json_digest(target_gene_map) if target_gene_map else None,
    }


def _run(command: list[str]) -> None:
    print("[cell-eval2] $", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def preflight(*, require_gpu: bool = True) -> dict:
    """Verify that the official scorer and, by default, its GPU DE path work."""
    installed = importlib.metadata.version("cell-eval2")
    if installed != CELL_EVAL2_VERSION:
        raise RuntimeError(
            f"cell-eval2 {CELL_EVAL2_VERSION} is required; found {installed}. "
            f"Install `cell-eval2[gpu,gpudge]=={CELL_EVAL2_VERSION}`."
        )
    result = {"cell_eval2_version": installed, "gpu_required": require_gpu}
    if require_gpu:
        import cupy as cp
        import gpudge  # noqa: F401
        import torch
        if cp.cuda.runtime.getDeviceCount() < 1 or not torch.cuda.is_available():
            raise RuntimeError("the official gpudge evaluation path requires a visible CUDA GPU")
        result.update({
            "de_backend": "gpudge",
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
        })
    else:
        result["de_backend"] = "auto"
    return result


def ensure_local_scale(real_h5ad: str | Path, scale_dir: str | Path, *,
                       pert_col: str, control: str,
                       target_gene_map: Mapping[str, str] | None = None,
                       require_gpu: bool = True) -> tuple[Path, dict]:
    """Build/reuse the local baseline and five-split replicate scale."""
    runtime = preflight(require_gpu=require_gpu)
    root = Path(scale_dir)
    real = _canonical_copy(real_h5ad, root / "inputs" / "real.canonical.h5ad")
    ref = ad.read_h5ad(real, backed="r")
    labels = sorted(set(ref.obs[pert_col].astype(str)) - {control})
    genes = set(ref.var_names.astype(str))
    ref.file.close()
    if target_gene_map:
        missing_labels = sorted(set(labels) - set(target_gene_map))
        if missing_labels:
            raise ValueError(
                f"evaluator target-gene map misses {len(missing_labels)} perturbation(s): "
                f"{missing_labels[:10]}"
            )
        mapped_in_panel = sum(target_gene_map[label] in genes for label in labels)
        target_map_audit = {
            "target_gene_map_role": "evaluator_only_target_exclusion",
            "target_labels": len(labels),
            "target_labels_mapped": len(labels),
            "mapped_target_genes_present_in_expression_panel": mapped_in_panel,
        }
    else:
        target_map_audit = {
            "target_gene_map_role": "identity_resolution_by_cell_eval2",
            "target_labels": len(labels),
            "target_labels_mapped": sum(label in genes for label in labels),
            "mapped_target_genes_present_in_expression_panel": sum(label in genes for label in labels),
        }
    config_path = root / "vcc2026.eval.yaml"
    official = _official_config(
        config_path, pert_col=pert_col, control=control,
        target_gene_map=target_gene_map)
    expected = {
        "protocol": "cell-eval2-vcc2026-external-local-anchors-v1",
        "official_challenge_score": False,
        "score_scope": "external_dataset_local_baseline_and_replicate",
        "source_real": str(Path(real_h5ad).resolve()),
        "source_real_sha256": _sha256(real_h5ad),
        "canonical_real_sha256": _sha256(real),
        "preset": PRESET,
        "anchor_splits": 5,
        "anchor_base_seed": 0,
        "runtime_overrides": {"de.backend": "gpudge"} if require_gpu else {},
        **target_map_audit,
        **runtime,
        **official,
    }
    manifest_path = root / "external_scale_manifest.json"
    complete = [
        root / "baseline" / "baseline_agg.csv",
        root / "baseline" / "baseline_meta.json",
        root / "anchor" / "anchor_agg.parquet",
        root / "anchor" / "anchor_meta.json",
    ]
    if manifest_path.is_file():
        observed = json.loads(manifest_path.read_text())
        mismatches = {k: (observed.get(k), v) for k, v in expected.items()
                      if observed.get(k) != v}
        # Schema-only upgrade for scales produced by the first 0.16.0 wrapper:
        # ``de_backend=gpudge`` was already stamped and the artifact sidecars
        # carry the resolved backend; this merely makes the equivalent CLI
        # override explicit in our manifest.
        if set(mismatches) == {"runtime_overrides"} and observed.get("de_backend") == "gpudge":
            observed = expected
            manifest_path.write_text(json.dumps(observed, indent=2, sort_keys=True))
            mismatches = {}
        if mismatches:
            raise RuntimeError(
                f"local scale provenance changed at {root}; use a new output directory. "
                f"Mismatches: {mismatches}"
            )
        if all(path.is_file() for path in complete):
            print(f"[vcc2026 scale] REUSE verified local scale {root}", flush=True)
            return real, observed

    ce2 = executable("cell-eval2") or "cell-eval2"
    cache_real = root / "cache" / "real"
    cache_real.mkdir(parents=True, exist_ok=True)
    backend = ["--set", "de.backend=gpudge"] if require_gpu else []
    print("[vcc2026 scale 1/2] generic-response baseline (score 0)", flush=True)
    _run([
        ce2, "baseline", "-ar", str(real), "--config", str(config_path),
        "--cache-real", str(cache_real), "--cache-pred", str(root / "cache" / "baseline"),
        "--save-pred", str(root / "baseline" / "baseline_pred.h5ad"),
        "-o", str(root / "baseline"), *backend,
    ])
    print("[vcc2026 scale 2/2] five seeded split-half replicate anchors (score 1)", flush=True)
    _run([
        ce2, "run", "-ar", str(real), "--anchor", "--anchor-splits", "5",
        "--anchor-base-seed", "0", "--config", str(config_path),
        "--cache-real", str(cache_real), "-o", str(root / "anchor"), *backend,
    ])
    if not all(path.is_file() for path in complete):
        raise RuntimeError(f"cell-eval2 did not create a complete local scale in {root}")
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True))
    return real, expected


def evaluate(pred_h5ad: str | Path, real_h5ad: str | Path, outdir: str | Path,
             scale_dir: str | Path, *, pert_col: str, control: str,
             target_gene_map_path: str | Path | None = None,
             require_gpu: bool = True) -> dict[str, float]:
    """Return the six locally anchored VCC 2026 component scores and Overall."""
    target_map = load_target_gene_map(target_gene_map_path)
    real, scale_meta = ensure_local_scale(
        real_h5ad, scale_dir, pert_col=pert_col, control=control,
        target_gene_map=target_map, require_gpu=require_gpu)
    out = Path(outdir)
    pred = _canonical_copy(pred_h5ad, out / "pred.canonical.h5ad")
    config_path = Path(scale_dir) / "vcc2026.eval.yaml"
    ce2 = executable("cell-eval2") or "cell-eval2"
    backend = ["--set", "de.backend=gpudge"] if require_gpu else []
    print("[vcc2026 metrics 1/2] raw official six", flush=True)
    _run([
        ce2, "run", "-ap", str(pred), "-ar", str(real),
        "--config", str(config_path),
        "--cache-real", str(Path(scale_dir) / "cache" / "real"),
        "--cache-pred", str(out / "cache" / "pred"),
        "-o", str(out / "run"), *backend,
    ])
    scored_path = out / "scored_local_anchors.csv"
    print("[vcc2026 metrics 2/2] baseline/replicate scaling", flush=True)
    _run([
        ce2, "score", "--user-agg", str(out / "run" / "agg_results.csv"),
        "--user-meta", str(out / "run" / "run_meta.json"),
        "--baseline-agg", str(Path(scale_dir) / "baseline" / "baseline_agg.csv"),
        "--baseline-meta", str(Path(scale_dir) / "baseline" / "baseline_meta.json"),
        "--anchor", str(Path(scale_dir) / "anchor"), "-o", str(scored_path),
    ])

    agg = pd.read_csv(out / "run" / "agg_results.csv")
    mean = agg.loc[agg["statistic"].astype(str).str.lower() == "mean"]
    if len(mean) != 1:
        raise RuntimeError("cell-eval2 aggregate does not contain exactly one mean row")
    scored = pd.read_csv(scored_path).set_index("metric")
    result: dict[str, float] = {}
    for short, canonical in SCORED_METRICS.items():
        if canonical not in scored.index or canonical not in mean.columns:
            raise RuntimeError(f"official vcc2026 metric missing: {canonical}")
        scaled = float(scored.loc[canonical, "from_replicate"])
        raw = float(mean.iloc[0][canonical])
        if not math.isfinite(scaled) or not math.isfinite(raw):
            raise RuntimeError(f"non-finite official metric {canonical}: raw={raw}, score={scaled}")
        result[short] = scaled
        result[f"raw::{canonical}"] = raw
    result["overall"] = float(sum(result[k] for k in SCORED_METRICS) / len(SCORED_METRICS))
    reported = float(scored.loc["avg_score", "from_replicate"])
    if not math.isclose(result["overall"], reported, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"Overall mismatch: recomputed={result['overall']} scorer={reported}")

    run_meta = json.loads((out / "run" / "run_meta.json").read_text())
    audit = {
        **scale_meta,
        "prediction_source": str(Path(pred_h5ad).resolve()),
        "prediction_source_sha256": _sha256(pred_h5ad),
        "prediction_canonical_sha256": _sha256(pred),
        "resolved_de_backend": run_meta.get("resolved_de_backend"),
        "resolved_device": run_meta.get("resolved_device"),
        "six_component_scores": {k: result[k] for k in SCORED_METRICS},
        "overall_unweighted_mean": result["overall"],
    }
    (out / "evaluation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return result
