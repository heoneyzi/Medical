"""Frozen foundation-model adapters (STATE / STACK), precomputed-prediction mode.

Run the model yourself (checkpoints are too large to vendor), save its predicted
AnnData (challenge gene space, obs[target_gene] with a non-targeting group), and
pass it via --fm-pred / VCC_FM_PRED. The adapter converts it into a per-gene
relative-fold effect and plugs into the same generate -> vcc prep pipeline.

STATE  : state tx infer --model-dir <RUN> --checkpoint <CKPT> \
           --adata <CONTROLS>.h5ad --pert-col target_gene --embed-key X_hvg --output <OUT>.h5ad
STACK  : stack-generation --checkpoint <CKPT> --base-adata <REF_CTX>.h5ad \
           --test-adata <TARGET_CONTROLS>.h5ad --genelist <HVG>.pkl \
           --split-column condition --output-dir <OUTDIR>
See docs/STATE_STACK.md (incl. the STATE one-hot -> ESM-2 fix).
"""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np

from ..config import Config
from ..effect_library import EPS
from ..io import align_to_genes, to_dense
from ..methods import Baseline

_CMD = {
    "state": ("state tx infer --model-dir <RUN> --checkpoint <CKPT> --adata <CONTROLS>.h5ad "
              "--pert-col target_gene --embed-key X_hvg --output <OUT>.h5ad  # then --fm-pred <OUT>.h5ad"),
    "stack": ("stack-generation --checkpoint <CKPT> --base-adata <REF_CTX>.h5ad "
              "--test-adata <TARGET_CONTROLS>.h5ad --genelist <HVG>.pkl --split-column condition "
              "--output-dir <OUTDIR>  # then --fm-pred <OUTDIR>/generated.h5ad"),
}


class ExternalFMAdapter(Baseline):
    def __init__(self, name: str, cfg: Config):
        self.name = name
        self._path = os.environ.get("VCC_FM_PRED") or getattr(cfg, "_fm_pred", "")

    def build(self, lib, genes, cfg):
        super().build(lib, genes, cfg)
        self._relfold: dict[str, np.ndarray] = {}
        if self._path and Path(self._path).exists():
            a = align_to_genes(ad.read_h5ad(self._path), genes)
            pcol = cfg.data.pert_col
            if pcol not in a.obs:
                raise KeyError(f"FM prediction '{self._path}' lacks obs['{pcol}']")
            X = to_dense(a.X).astype(np.float32)
            lab = a.obs[pcol].astype(str).values
            cmask = lab == cfg.data.control_label
            ctrl = X[cmask].mean(0) if cmask.any() else X.mean(0)
            for g in np.unique(lab):
                if g == cfg.data.control_label:
                    continue
                self._relfold[str(g)] = (X[lab == g].mean(0) + EPS) / (ctrl + EPS)
            self._available = True
        else:
            self._available = False

    def covered(self, gene):
        return gene in self._relfold

    def predict_effect(self, ctx, gene):
        if not self._available:
            raise RuntimeError(
                f"Frozen '{self.name}' needs an external prediction. Run:\n    {_CMD[self.name]}\n"
                f"then pass --fm-pred <that.h5ad> (or set VCC_FM_PRED). See docs/STATE_STACK.md."
            )
        rf = self._relfold.get(str(gene))
        if rf is None:
            return self._identity()
        return rf.astype(np.float32) if self.effect_kind == "logfc" else \
            (ctx["pb"] * (rf - 1.0)).astype(np.float32)


def build_fm_adapter(name: str, cfg: Config) -> Baseline:
    return ExternalFMAdapter(name, cfg)


def command_template(name: str) -> str:
    return _CMD[name]
