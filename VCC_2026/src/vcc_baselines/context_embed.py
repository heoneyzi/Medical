"""Context representations for source similarity weighting.

Same GWPS effect + same generator, only the *context representation* changes ->
this isolates the pure value of a frozen FM embedding (F6/F7) over raw/PCA (D4):

    raw   : cosine of log1p pseudobulks              (S0, always available)
    pca   : cosine in PCA space of the pseudobulks   (S1)
    embed : cosine of precomputed frozen embeddings  (S3/S4/S5: STATE-SE, STACK, scGPT...)

`weights(context, sources, tau)` returns softmax-normalised source weights used by
the nearest / weighted / hybrid transfer methods.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class ContextEmbedder:
    def __init__(self, source_pb: dict[str, np.ndarray], context_pb: dict[str, np.ndarray],
                 space: str = "raw", pca_components: int = 50,
                 precomputed: str = ""):
        self.space = space
        self.vec: dict[str, np.ndarray] = {}
        names = list(source_pb) + list(context_pb)
        pbs = {**source_pb, **context_pb}

        if space == "embed":
            if not precomputed or not Path(precomputed).exists():
                print(f"[context] embed space requested but no embeddings at "
                      f"'{precomputed}'; falling back to raw")
                self.space = "raw"
            else:
                z = np.load(precomputed, allow_pickle=True)
                missing = [n for n in names if n not in z.files]
                if missing:
                    print(f"[context] embeddings missing {missing}; falling back to raw")
                    self.space = "raw"
                else:
                    self.vec = {n: z[n].astype(np.float32) for n in names}
                    return

        if self.space == "pca":
            M = np.stack([np.log1p(pbs[n]) for n in names])          # (N, G)
            Mc = M - M.mean(0, keepdims=True)
            k = int(min(pca_components, max(1, min(M.shape) - 1)))
            # SVD-based PCA projection
            U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
            proj = Mc @ Vt[:k].T
            self.vec = {n: proj[i] for i, n in enumerate(names)}
        else:  # raw
            self.vec = {n: np.log1p(pbs[n]).astype(np.float32) for n in names}

    def weights(self, context: str, sources: list[str], tau: float = 0.1) -> dict[str, float]:
        vc = self.vec[context]
        sims = np.array([_cos(vc, self.vec[s]) for s in sources])
        w = np.exp((sims - sims.max()) / max(tau, 1e-6))
        w /= w.sum()
        return dict(zip(sources, w.tolist()))

    def max_cosine(self, context: str, sources: list[str]) -> float:
        """Confidence proxy (D5): how close the context is to its nearest source."""
        vc = self.vec[context]
        return float(np.clip(max(_cos(vc, self.vec[s]) for s in sources), 0.0, 1.0))
