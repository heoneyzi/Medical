"""Frozen (training-free) effect-prediction methods.

Each method returns a per-gene EFFECT for one (context, gene):
  * effect_kind == 'logfc'    -> relative fold vector (generator multiplies)
  * effect_kind == 'additive' -> count-delta vector (generator adds)

`ctx` = {"pb": context control pseudobulk (counts), "weights": {source: w}} where
the weights come from the chosen context representation (raw / pca / frozen embed).

Ladder mapping (strategy doc):
  A0 no_effect | A1 vcc sample (cli)
  D1 gwps_direct+additive | D2 gwps_direct+logfc | D3 gwps_nearest | D4 gwps_weighted
  D6 esm2_knn
  F6/F7 gwps_weighted with similarity_space=embed (STATE-SE / STACK)
  F1/F4/F5 state|stack (external precomputed) ; H1 ensemble (cli)
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .effect_library import EffectLibrary

METHODS = ["no_effect", "global_mean", "replogle_k562", "gwps_direct", "gwps_nearest",
           "gwps_weighted", "esm2_knn", "state", "stack", "ensemble"]


class Baseline:
    name = "base"
    effect_kind = "logfc"

    def build(self, lib: EffectLibrary | None, genes: list[str], cfg: Config) -> None:
        self.lib = lib
        self.genes = genes
        self.G = len(genes)
        self.effect_kind = cfg.predict.effect
        self.cfg = cfg

    def _identity(self) -> np.ndarray:
        return np.ones(self.G, np.float32) if self.effect_kind == "logfc" else np.zeros(self.G, np.float32)

    def predict_effect(self, ctx: dict, gene: str) -> np.ndarray:
        raise NotImplementedError

    def covered(self, gene: str) -> bool:
        return True


class NoEffect(Baseline):
    """A0: relfold=1 / delta=0 -> generator reproduces the control population."""
    name = "no_effect"

    def predict_effect(self, ctx, gene):
        return self._identity()


class GlobalMean(Baseline):
    name = "global_mean"

    def build(self, lib, genes, cfg):
        super().build(lib, genes, cfg)
        rfs = [lib.relfold[s] for s in lib.sources] if lib else []
        self._gm = np.concatenate(rfs).mean(0) if rfs else np.ones(len(genes), np.float32)

    def predict_effect(self, ctx, gene):
        if self.effect_kind == "logfc":
            return self._gm.astype(np.float32)
        # additive from a generic source control
        s = self.lib.sources[0]
        return (self.lib.control_pb[s] * (self._gm - 1.0)).astype(np.float32)


class _GwpsBase(Baseline):
    def build(self, lib, genes, cfg):
        super().build(lib, genes, cfg)
        if lib is None or not lib.sources:
            raise ValueError(f"method '{self.name}' needs a GWPS effect library")
        self._measured = lib.genes_measured()

    def covered(self, gene):
        return gene in self._measured

    def _combine(self, gene, weights: dict[str, float]) -> np.ndarray:
        srcs = [s for s in self.lib.sources if self.lib.get_relfold(s, gene) is not None]
        if not srcs:
            return self._identity()
        w = np.array([weights.get(s, 0.0) for s in srcs], np.float32)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(srcs), np.float32) / len(srcs)
        out = np.zeros(self.G, np.float32)
        for wi, s in zip(w, srcs):
            e = self.lib.get_relfold(s, gene) if self.effect_kind == "logfc" else self.lib.get_delta(s, gene)
            out += wi * e
        return out


class GwpsDirect(_GwpsBase):
    """D1/D2: source-averaged transfer (equal weights), context-agnostic."""
    name = "gwps_direct"

    def predict_effect(self, ctx, gene):
        srcs = self.lib.sources
        return self._combine(gene, {s: 1.0 for s in srcs})


class ReplogleK562(_GwpsBase):
    """B00: exact K562 signature, with explicit no-effect when unavailable."""
    name = "replogle_k562"

    def build(self, lib, genes, cfg):
        super().build(lib, genes, cfg)
        if "k562" not in lib.sources:
            raise ValueError("replogle_k562 requires a source named 'k562'")

    def predict_effect(self, ctx, gene):
        return self._combine(gene, {"k562": 1.0})


class GwpsNearest(_GwpsBase):
    """D3: use only the highest-weight (most similar) source."""
    name = "gwps_nearest"

    def predict_effect(self, ctx, gene):
        w = ctx["weights"]
        best = max(w, key=w.get)
        return self._combine(gene, {best: 1.0})


class GwpsWeighted(_GwpsBase):
    """D4 (raw/pca) or F6/F7 (embed): similarity-weighted source blend."""
    name = "gwps_weighted"

    def predict_effect(self, ctx, gene):
        return self._combine(gene, ctx["weights"])


class Esm2KNN(_GwpsBase):
    """D6: unseen gene -> weighted avg of K nearest MEASURED genes by ESM-2."""
    name = "esm2_knn"

    def build(self, lib, genes, cfg):
        super().build(lib, genes, cfg)
        self.inner = GwpsWeighted(); self.inner.build(lib, genes, cfg)
        from .embeddings import GeneEmbedder
        self.emb = GeneEmbedder(cfg.esm2)
        self.k = cfg.esm2.knn_k
        self._meas = sorted(self._measured)
        self._M = self.emb.embed(self._meas)

    def predict_effect(self, ctx, gene):
        if self.covered(gene):
            return self.inner.predict_effect(ctx, gene)
        q = self.emb.embed([gene])[0]
        qn = q / (np.linalg.norm(q) + 1e-8)
        Mn = self._M / (np.linalg.norm(self._M, axis=1, keepdims=True) + 1e-8)
        sims = Mn @ qn
        idx = np.argsort(-sims)[: self.k]
        a = np.exp((sims[idx] - sims[idx].max()) / max(self.cfg.predict.tau, 1e-6))
        a /= a.sum()
        out = np.zeros(self.G, np.float32)
        for wk, j in zip(a, idx):
            out += wk * self.inner.predict_effect(ctx, self._meas[j])
        return out


def build_method(cfg: Config, lib: EffectLibrary | None, genes: list[str]) -> Baseline:
    name = cfg.predict.method
    table = {"no_effect": NoEffect, "global_mean": GlobalMean,
             "replogle_k562": ReplogleK562, "gwps_direct": GwpsDirect,
             "gwps_nearest": GwpsNearest, "gwps_weighted": GwpsWeighted, "esm2_knn": Esm2KNN}
    if name in table:
        m: Baseline = table[name]()
    elif name in ("state", "stack"):
        from .adapters import build_fm_adapter
        m = build_fm_adapter(name, cfg)
    else:
        raise ValueError(f"Unknown method '{name}'. Choose from {METHODS}")
    m.build(lib, genes, cfg)
    return m
