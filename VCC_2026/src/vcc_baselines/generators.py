"""Raw-count single-cell generators (Problem B in the strategy doc).

Given a target context's control cells (raw counts) and a predicted per-gene
effect (relative fold for logfc, or additive delta), emit `n` raw INTEGER-count
cells for one perturbation. 2026 scores in counts space, so this stage is
separate from — and ablated independently of — the effect prediction.

    G0 multinomial : preserves each control cell's library size (recommended)
    G1 poisson     : independent Poisson per gene (library size drifts)
    G2 nbinom      : Poisson + overdispersion (theta)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def pooled_gene_probs(ctrl_X, alpha: float = 1e-3) -> np.ndarray:
    """Pooled control composition with a non-zero prior for unseen genes."""
    if alpha <= 0:
        raise ValueError("alpha must be positive for zero-lock-free generation")
    counts = np.asarray(ctrl_X.sum(axis=0)).ravel().astype(np.float64)
    counts += float(alpha)
    total = float(counts.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("control population has no finite positive counts")
    return counts / total


def generate_pseudobulk_multinomial(
    ctrl_X,
    relfold: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float = 1e-3,
) -> sp.csr_matrix:
    """Generate sparse integer cells from one pooled target-context rate.

    Unlike per-cell multiplication, this permits genes absent in an individual
    sampled control cell to be induced by a perturbation. Row library sizes are
    still copied from the sampled controls.
    """
    p_ctrl = pooled_gene_probs(ctrl_X, alpha=alpha)
    fold = np.asarray(relfold, dtype=np.float64)
    if fold.ndim != 1 or fold.shape[0] != p_ctrl.shape[0]:
        raise ValueError("relfold must be one-dimensional and match the gene panel")
    if not np.isfinite(fold).all() or (fold < 0).any():
        raise ValueError("relfold must be finite and nonnegative")
    p = p_ctrl * fold
    total = float(p.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("perturbed pooled probability has no positive mass")
    p /= total

    libsizes = np.asarray(ctrl_X.sum(axis=1)).ravel().astype(np.int64)
    rows: list[sp.csr_matrix] = []
    for library_size in libsizes:
        if library_size <= 0:
            rows.append(sp.csr_matrix((1, len(p)), dtype=np.int32))
            continue
        counts = rng.multinomial(int(library_size), p).astype(np.int32)
        rows.append(sp.csr_matrix(counts.reshape(1, -1)))
    out = sp.vstack(rows, format="csr")
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _rates(base: np.ndarray, effect: np.ndarray, kind: str) -> np.ndarray:
    """Per-cell, per-gene non-negative rate from control counts + effect."""
    if kind == "logfc":                 # multiplicative
        lam = base * effect[None, :]
    elif kind == "additive":            # shift, clamped at 0
        lam = base + effect[None, :]
    else:
        raise ValueError(f"unknown effect kind '{kind}'")
    return np.clip(lam, 0.0, None)


def generate(
    ctrl_cells: np.ndarray,      # (n, G) sampled control cells, raw counts
    effect: np.ndarray,          # (G,) relfold (logfc) or delta (additive)
    *,
    kind: str = "logfc",
    generator: str = "multinomial",
    rng: np.random.Generator,
    nb_theta: float = 10.0,
    multinomial_alpha: float = 1e-3,
) -> np.ndarray:
    n, G = ctrl_cells.shape
    if generator == "pseudobulk_multinomial":
        if kind != "logfc":
            raise ValueError("pseudobulk_multinomial requires a relative-fold effect")
        return generate_pseudobulk_multinomial(
            ctrl_cells, effect, rng=rng, alpha=multinomial_alpha)
    lam = _rates(ctrl_cells.astype(np.float32), effect.astype(np.float32), kind)

    if generator == "poisson":
        return rng.poisson(lam).astype(np.int32)

    if generator == "nbinom":
        # NB as Gamma-Poisson: shape=theta, scale=lam/theta -> mean lam, var lam+lam^2/theta
        theta = max(nb_theta, 1e-3)
        gam = rng.gamma(shape=theta, scale=np.maximum(lam, 1e-8) / theta)
        return rng.poisson(gam).astype(np.int32)

    if generator == "multinomial":
        out = np.zeros((n, G), dtype=np.int32)
        L = ctrl_cells.sum(axis=1).astype(np.int64)        # preserve library size
        lam64 = lam.astype(np.float64)
        row_sums = lam64.sum(axis=1)
        for i in range(n):
            if row_sums[i] <= 0 or L[i] <= 0:
                continue
            p = lam64[i] / row_sums[i]
            p = np.clip(p, 0, None)
            p[-1] = max(0.0, 1.0 - p[:-1].sum())           # absorb float remainder
            out[i] = rng.multinomial(int(L[i]), p)
        return out

    raise ValueError(f"unknown generator '{generator}'")
