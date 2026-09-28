"""Audited Replogle 2022 raw-pseudobulk effect-library construction.

The public raw-bulk files contain one mean-expression row per guide population.
We reconstruct population count sums with ``num_cells_filtered``, aggregate
guides by target, and store natural-log CPM fold changes.  No external
perturbation response is consumed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .effect_library import EffectLibrary
from .seeds import stable_seed


OFFICIAL_FILES = {
    "k562": {
        "name": "K562_gwps_raw_bulk_01.h5ad",
        "url": "https://ndownloader.figshare.com/files/35774443",
        "bytes": 374_587_922,
        "md5": "4570b53c9d62ff6df281e622f0350060",
    },
    "rpe1": {
        "name": "rpe1_raw_bulk_01.h5ad",
        "url": "https://ndownloader.figshare.com/files/35775581",
        "bytes": 95_350_546,
        "md5": "74765fa87635467a869ea972356ae0e7",
    },
}


class AuditRequired(RuntimeError):
    """Raised when a data/schema assumption has not been verified."""


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets_from_index(index: Iterable[str]) -> np.ndarray:
    targets: list[str] = []
    malformed: list[str] = []
    for value in map(str, index):
        parts = value.split("_")
        if len(parts) < 4 or not parts[0].isdigit() or not parts[1]:
            malformed.append(value)
            continue
        targets.append(parts[1])
    if malformed:
        raise AuditRequired(
            f"AUDIT_REQUIRED: {len(malformed)} Replogle obs names do not match "
            f"the verified '<row>_<target>_<guide>_<id>' schema: {malformed[:5]}")
    return np.asarray(targets, dtype=str)


def _canonical_gene_mask(data: ad.AnnData, hgnc_path: str | Path | None) -> tuple[np.ndarray, dict]:
    symbols = data.var["gene_name"].astype(str)
    duplicate_symbols = symbols[symbols.duplicated(keep=False)].unique().tolist()
    keep = np.ones(data.n_vars, dtype=bool)
    resolution: dict[str, dict[str, str]] = {}
    if not duplicate_symbols:
        return keep, resolution
    if hgnc_path is None or not Path(hgnc_path).is_file():
        raise AuditRequired(
            f"AUDIT_REQUIRED: duplicated symbols {duplicate_symbols[:10]} require an HGNC table")
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str)
    for symbol in duplicate_symbols:
        approved = hgnc[(hgnc["symbol"] == symbol) & hgnc["ensembl_gene_id"].notna()]
        candidates = set(approved["ensembl_gene_id"].astype(str).str.split(".").str[0])
        positions = np.flatnonzero(symbols.to_numpy() == symbol)
        matches = [pos for pos in positions
                   if str(data.var_names[pos]).split(".")[0] in candidates]
        if len(matches) != 1:
            raise AuditRequired(
                f"AUDIT_REQUIRED: {symbol} duplicate features cannot be uniquely resolved "
                f"to the HGNC-approved Ensembl ID; candidates={sorted(candidates)}")
        keep[positions] = False
        keep[matches[0]] = True
        resolution[symbol] = {
            "kept_ensembl": str(data.var_names[matches[0]]),
            "dropped_ensembl": ";".join(str(data.var_names[pos]) for pos in positions if pos != matches[0]),
            "rule": "HGNC-approved exact Ensembl ID",
        }
    return keep, resolution


def audit_raw_bulk(path: str | Path, hgnc_path: str | Path | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise AuditRequired(f"AUDIT_REQUIRED: missing Replogle file {path}")
    data = ad.read_h5ad(path, backed="r")
    required_obs = {"num_cells_filtered", "core_control"}
    missing = required_obs - set(data.obs.columns)
    if missing:
        data.file.close()
        raise AuditRequired(f"AUDIT_REQUIRED: {path} lacks obs columns {sorted(missing)}")
    if "gene_name" not in data.var.columns:
        data.file.close()
        raise AuditRequired(f"AUDIT_REQUIRED: {path} lacks var['gene_name']")
    targets = _targets_from_index(data.obs_names)
    genes = data.var["gene_name"].astype(str)
    gene_keep, duplicate_resolution = _canonical_gene_mask(data, hgnc_path)
    controls = targets == "non-targeting"
    core = data.obs["core_control"].astype(bool).to_numpy()
    cell_counts = pd.to_numeric(data.obs["num_cells_filtered"], errors="coerce").to_numpy(float)
    invalid_counts = ~np.isfinite(cell_counts) | (cell_counts <= 0)
    used_rows = (~controls) | (controls & core)
    if (invalid_counts & used_rows).any():
        data.file.close()
        raise AuditRequired(
            "AUDIT_REQUIRED: a perturbation or selected core-control row has an invalid "
            "num_cells_filtered value")
    audit = {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "n_rows": int(data.n_obs),
        "n_genes": int(data.n_vars),
        "n_canonical_genes": int(gene_keep.sum()),
        "n_target_labels": int(len(set(targets) - {"non-targeting"})),
        "n_non_targeting_rows": int(controls.sum()),
        "n_core_control_rows": int((controls & core).sum()),
        "n_unselected_control_rows_with_missing_cell_count": int(
            (controls & ~core & invalid_counts).sum()),
        "obs_columns": list(map(str, data.obs.columns)),
        "var_columns": list(map(str, data.var.columns)),
        "matrix": "X",
        "target_parser": "obs_name second underscore-delimited field",
        "control_label": "non-targeting",
        "control_filter": "target=non-targeting AND core_control=True",
        "duplicate_gene_resolution": duplicate_resolution,
    }
    data.file.close()
    if audit["n_core_control_rows"] == 0:
        raise AuditRequired(f"AUDIT_REQUIRED: no verified core controls in {path}")
    return audit


def build_native_lnfc(
    path: str | Path,
    out: str | Path,
    *,
    pseudocount_cpm: float = 1.0,
    lnfc_clip: float = 3.0,
    hgnc_path: str | Path | None = None,
) -> Path:
    """Build/reuse a source-native target-by-gene lnFC cache."""
    path, out = Path(path), Path(out)
    audit = audit_raw_bulk(path, hgnc_path)
    expected = {
        **audit,
        "pseudocount_cpm": float(pseudocount_cpm),
        "lnfc_clip": float(lnfc_clip),
        "lnfc_definition": "ln((pert_CPM+a)/(core_control_CPM+a))",
    }
    sidecar = out.with_suffix(out.suffix + ".manifest.json")
    if out.is_file() and sidecar.is_file() and json.loads(sidecar.read_text()) == expected:
        return out

    data = ad.read_h5ad(path)
    targets = _targets_from_index(data.obs_names)
    gene_keep, _ = _canonical_gene_mask(data, hgnc_path)
    genes = data.var["gene_name"].astype(str).to_numpy()[gene_keep]
    cell_counts = pd.to_numeric(data.obs["num_cells_filtered"], errors="coerce").to_numpy(float)
    controls_all = targets == "non-targeting"
    core_controls = controls_all & data.obs["core_control"].astype(bool).to_numpy()
    used_rows = (~controls_all) | core_controls
    invalid_counts = ~np.isfinite(cell_counts) | (cell_counts <= 0)
    if (invalid_counts & used_rows).any():
        raise AuditRequired(
            "AUDIT_REQUIRED: selected perturbation/core-control rows require finite positive cells")
    # Non-core controls are excluded by the audited control rule, so their
    # missing filtered-cell counts never enter any effect calculation.
    cell_counts[invalid_counts] = 0.0
    matrix = np.asarray(data.X[:, gene_keep], dtype=np.float64)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise AuditRequired("AUDIT_REQUIRED: raw pseudobulk X must be finite and nonnegative")

    controls = core_controls
    ctrl_counts = (matrix[controls] * cell_counts[controls, None]).sum(axis=0)
    ctrl_cpm = 1e6 * ctrl_counts / max(float(ctrl_counts.sum()), 1.0)

    pert_targets = np.asarray(sorted(set(targets) - {"non-targeting"}), dtype=str)
    row_for = {target: i for i, target in enumerate(pert_targets)}
    summed = np.zeros((len(pert_targets), len(genes)), dtype=np.float64)
    n_cells = np.zeros(len(pert_targets), dtype=np.float64)
    for row, target in enumerate(targets):
        if target == "non-targeting":
            continue
        idx = row_for[target]
        summed[idx] += matrix[row] * cell_counts[row]
        n_cells[idx] += cell_counts[row]
    totals = summed.sum(axis=1)
    valid = totals > 0
    pert_targets, summed, n_cells, totals = (
        pert_targets[valid], summed[valid], n_cells[valid], totals[valid])
    pert_cpm = 1e6 * summed / totals[:, None]
    lnfc = np.log((pert_cpm + pseudocount_cpm) /
                  (ctrl_cpm[None, :] + pseudocount_cpm))
    lnfc = np.clip(lnfc, -lnfc_clip, lnfc_clip).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        genes=genes.astype(str),
        targets=pert_targets.astype(str),
        lnfc=lnfc,
        ctrl_cpm=ctrl_cpm.astype(np.float32),
        n_pert_cells=n_cells.astype(np.float32),
    )
    sidecar.write_text(json.dumps(expected, indent=2, sort_keys=True))
    return out


def _index(values: np.ndarray) -> dict[str, int]:
    return {str(value): i for i, value in enumerate(values)}


def internal_cross_source_cv(
    k562_native: str | Path,
    rpe1_native: str | Path,
    out_dir: str | Path,
    *,
    scale_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25),
) -> dict:
    """Select global magnitude using only K562<->RPE1 held-source transfer."""
    with np.load(k562_native, allow_pickle=False) as k, np.load(rpe1_native, allow_pickle=False) as r:
        kt, rt = k["targets"].astype(str), r["targets"].astype(str)
        kg, rg = k["genes"].astype(str), r["genes"].astype(str)
        dual = sorted(set(kt) & set(rt))
        common_genes = sorted(set(kg) & set(rg))
        if not dual or not common_genes:
            raise AuditRequired("AUDIT_REQUIRED: K562/RPE1 have no dual targets/common genes")
        ki_t, ri_t = _index(kt), _index(rt)
        ki_g, ri_g = _index(kg), _index(rg)
        k_mat = k["lnfc"][[ki_t[x] for x in dual]][:, [ki_g[x] for x in common_genes]]
        r_mat = r["lnfc"][[ri_t[x] for x in dual]][:, [ri_g[x] for x in common_genes]]

    rows = []
    for scale in scale_grid:
        directional = []
        for source, truth, direction in ((k_mat, r_mat, "k562_to_rpe1"),
                                         (r_mat, k_mat, "rpe1_to_k562")):
            pred = float(scale) * source
            nmae = float(np.mean(np.abs(pred - truth)) /
                         (np.mean(np.abs(truth)) + 1e-8))
            dot = np.sum(pred * truth, axis=1)
            denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
            cosine = float(np.mean(np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)))
            sign = float(np.mean(np.sign(pred) == np.sign(truth)))
            directional.append(nmae)
            rows.append({"scale": scale, "direction": direction, "nmae": nmae,
                         "cosine": cosine, "sign_agreement": sign,
                         "n_dual_targets": len(dual), "n_common_genes": len(common_genes)})
        rows.append({"scale": scale, "direction": "mean", "nmae": float(np.mean(directional)),
                     "worst_nmae": float(np.max(directional)),
                     "n_dual_targets": len(dual), "n_common_genes": len(common_genes)})
    frame = pd.DataFrame(rows)
    means = frame[frame["direction"] == "mean"].sort_values(["nmae", "worst_nmae", "scale"])
    selected = float(means.iloc[0]["scale"])
    result = {
        "global_scale": selected,
        "selection_metric": "symmetric_cross_source_nmae",
        "n_dual_targets": len(dual),
        "n_common_genes": len(common_genes),
        "metric_source": "replogle_internal_cv",
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "magnitude_sweep.csv", index=False)
    (out_dir / "selected_hyperparameters.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))
    return result


def project_effect_library(
    native: dict[str, str | Path],
    genes: list[str],
    targets: list[str],
    out: str | Path,
    *,
    global_scale: float = 1.0,
) -> tuple[EffectLibrary, pd.DataFrame]:
    """Project exact Replogle targets onto one evaluation gene panel."""
    lib = EffectLibrary(genes)
    coverage = {str(target): {} for target in targets}
    manifests = {}
    for source, filename in native.items():
        filename = Path(filename)
        with np.load(filename, allow_pickle=False) as z:
            src_genes, src_targets = z["genes"].astype(str), z["targets"].astype(str)
            gi, ti = _index(src_genes), _index(src_targets)
            target_rows = [str(target) for target in targets if str(target) in ti]
            projected = np.zeros((len(target_rows), len(genes)), dtype=np.float32)
            present = [(j, gi[gene]) for j, gene in enumerate(genes) if gene in gi]
            if present and target_rows:
                dst, src = zip(*present)
                block = z["lnfc"][[ti[target] for target in target_rows]]
                projected[:, list(dst)] = block[:, list(src)] * float(global_scale)
            ctrl = np.zeros(len(genes), dtype=np.float32)
            if present:
                dst, src = zip(*present)
                ctrl[list(dst)] = z["ctrl_cpm"][list(src)] / 100.0
            lib.control_pb[source] = ctrl
            lib.control_std[source] = np.zeros_like(ctrl)
            lib.pert_genes[source] = target_rows
            lib.relfold[source] = np.exp(projected).astype(np.float32)
            lib._row[source] = {target: i for i, target in enumerate(target_rows)}
            for target in targets:
                coverage[str(target)][source] = str(target) in ti
            manifests[source] = {
                "native_cache": str(filename.resolve()),
                "native_cache_sha256": file_digest(filename),
                "n_native_targets": len(src_targets),
                "n_panel_genes_mapped": len(present),
            }
    out = Path(out); lib.save(out)
    rows = []
    for target, found in coverage.items():
        n = sum(found.values())
        rows.append({"target_gene": target, **{f"in_{s}": v for s, v in found.items()},
                     "n_sources": n,
                     "coverage_class": ("DUAL_SOURCE" if n >= 2 else
                                        f"{next((s.upper() for s, v in found.items() if v), 'MISSING')}_ONLY"
                                        if n == 1 else "MISSING")})
    frame = pd.DataFrame(rows)
    manifest = {
        "protocol": "STRICT_REPLOGLE_ONLY",
        "explicit_perturbation_sources": sorted(native),
        "effect_definition": "natural-log CPM fold change",
        "global_scale": float(global_scale),
        "n_genes": len(genes),
        "n_targets": len(targets),
        "sources": manifests,
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return lib, frame


def write_control_proxy(
    lib: EffectLibrary,
    source: str,
    out: str | Path,
    *,
    n_cells: int = 128,
    library_size: int = 10_000,
    seed: int = 2026,
) -> Path:
    """Create an explicitly labelled integer proxy from control CPM only."""
    mean = np.asarray(lib.control_pb[source], dtype=np.float64)
    if mean.sum() <= 0:
        raise AuditRequired(f"AUDIT_REQUIRED: {source} control has no genes on panel")
    p = mean + 1e-9
    p /= p.sum()
    rng = np.random.default_rng(stable_seed(seed, "replogle-control-proxy", source))
    rows = [sp.csr_matrix(rng.multinomial(library_size, p).astype(np.int32).reshape(1, -1))
            for _ in range(n_cells)]
    matrix = sp.vstack(rows, format="csr")
    obs = pd.DataFrame({"gene": "non-targeting", "context": source},
                       index=[f"{source}_proxy_{i}" for i in range(n_cells)])
    data = ad.AnnData(X=matrix, obs=obs)
    data.var_names = lib.genes
    data.uns["provenance"] = {
        "type": "deterministic_multinomial_control_proxy",
        "explicit_response_used": False,
        "source": source,
        "seed": int(seed),
        "library_size": int(library_size),
    }
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(out, compression="gzip")
    return out
