"""Replogle GWPS effect library (counts space).

For each source (K562, RPE1) and each knocked-down gene we store the *relative
fold* vector (perturbed_mean / control_mean) plus the source control mean & std.
Everything else the methods need is derived from these:

    relfold_j = (mu_pert_j + eps) / (mu_ctrl_j + eps)
    delta_j   = mu_ctrl_j * (relfold_j - 1)            # additive, in counts
    logfc_j   = log(relfold_j)
    z_j       = delta_j / (sigma_ctrl_j + eps)

Transfer to a target context (D1/D2) then reuses the SAME numbers:
    additive : lambda_j = x_ctrl_j(target) + delta_source_j
    logfc    : lambda_j = x_ctrl_j(target) * relfold_source_j
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .config import GwpsCfg
from .io import align_to_genes

EPS = 1.0  # counts-space pseudocount


class EffectLibrary:
    def __init__(self, genes: list[str]):
        self.genes = list(map(str, genes))
        self.control_pb: dict[str, np.ndarray] = {}
        self.control_std: dict[str, np.ndarray] = {}
        self.pert_genes: dict[str, list[str]] = {}
        self.relfold: dict[str, np.ndarray] = {}      # source -> (P, G)
        self._row: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_gwps(cls, cfg: GwpsCfg, genes: list[str], verbose: bool = True) -> "EffectLibrary":
        lib = cls(genes)
        for name, path in cfg.sources.items():
            p = Path(path)
            if not p.exists():
                if verbose:
                    print(f"[effect-lib] WARNING: GWPS source '{name}' not found at {p} — skipping")
                continue
            adata = align_to_genes(ad.read_h5ad(p), genes)
            X = adata.X.astype(np.float32)
            if not cfg.counts_space:                # lognorm -> pseudo-counts
                if sp.issparse(X):
                    X = X.copy()
                    X.data = np.expm1(X.data)
                else:
                    X = np.expm1(X)

            def mean_rows(mask):
                m = X[mask].mean(axis=0)
                return np.asarray(m).ravel().astype(np.float32)

            def std_rows(mask, mean):
                block = X[mask]
                if sp.issparse(block):
                    second = np.asarray(block.multiply(block).mean(axis=0)).ravel()
                else:
                    second = np.square(block).mean(axis=0)
                return np.sqrt(np.maximum(second - np.square(mean), 0)).astype(np.float32)
            if cfg.pert_col not in adata.obs:
                raise KeyError(f"GWPS '{name}' lacks obs['{cfg.pert_col}']; have {list(adata.obs.columns)}")
            pert = adata.obs[cfg.pert_col].astype(str).values
            ctrl_mask = pert == cfg.control_label
            if ctrl_mask.sum() == 0:
                raise ValueError(f"GWPS '{name}' has no controls labelled '{cfg.control_label}'")
            mu_c = mean_rows(ctrl_mask)
            sd_c = std_rows(ctrl_mask, mu_c)
            lib.control_pb[name] = mu_c
            lib.control_std[name] = sd_c
            gene_names, rows = [], []
            for g in np.unique(pert):
                if g == cfg.control_label:
                    continue
                mu_p = mean_rows(pert == g)
                rows.append((mu_p + EPS) / (mu_c + EPS))
                gene_names.append(str(g))
            lib.pert_genes[name] = gene_names
            lib.relfold[name] = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(genes)), np.float32)
            lib._row[name] = {g: i for i, g in enumerate(gene_names)}
            if verbose:
                print(f"[effect-lib] {name}: {len(gene_names)} perturbations, {int(ctrl_mask.sum())} controls")
        return lib

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {"__genes__": np.array(self.genes, dtype=object)}
        for s in self.sources:
            blob[f"ctrl::{s}"] = self.control_pb[s]
            blob[f"std::{s}"] = self.control_std[s]
            blob[f"perts::{s}"] = np.array(self.pert_genes[s], dtype=object)
            blob[f"relfold::{s}"] = self.relfold[s]
        np.savez_compressed(path, **blob)

    @classmethod
    def load(cls, path: str | Path) -> "EffectLibrary":
        z = np.load(path, allow_pickle=True)
        lib = cls(list(z["__genes__"]))
        srcs = sorted({k.split("::", 1)[1] for k in z.files if k.startswith("relfold::")})
        for s in srcs:
            lib.control_pb[s] = z[f"ctrl::{s}"]
            lib.control_std[s] = z[f"std::{s}"]
            lib.pert_genes[s] = list(z[f"perts::{s}"])
            lib.relfold[s] = z[f"relfold::{s}"]
            lib._row[s] = {g: i for i, g in enumerate(lib.pert_genes[s])}
        return lib

    # ------------------------------------------------------------------ #
    @property
    def sources(self) -> list[str]:
        return list(self.relfold.keys())

    def genes_measured(self) -> set[str]:
        out: set[str] = set()
        for s in self.sources:
            out |= set(self.pert_genes[s])
        return out

    def get_relfold(self, source: str, gene: str) -> np.ndarray | None:
        i = self._row.get(source, {}).get(gene)
        return self.relfold[source][i] if i is not None else None

    def get_delta(self, source: str, gene: str) -> np.ndarray | None:
        rf = self.get_relfold(source, gene)
        return self.control_pb[source] * (rf - 1.0) if rf is not None else None

    def mean_relfold(self, gene: str) -> np.ndarray | None:
        vs = [self.get_relfold(s, gene) for s in self.sources]
        vs = [v for v in vs if v is not None]
        return np.mean(vs, axis=0) if vs else None

    def coverage_group(self, gene: str) -> str:
        present = [s for s in self.sources if gene in self._row.get(s, {})]
        if len(present) >= 2:
            return "B_multi_source"
        if present:
            return f"C_{present[0]}_only"
        return "D_missing"
