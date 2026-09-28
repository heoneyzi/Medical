"""Core per-position measurements for Experiment 1.

Everything here is backend-agnostic (see :mod:`exp1.backend`) and takes state
matrices of shape ``[T, D]``.

Quantities, in the notation of the protocol
-------------------------------------------
For a state ``h_l(t)`` and the per-position reference direction
``r_hat(t) = h_norm(t) / ||h_norm(t)||``:

    p_l(t) = <h_l(t), r_hat(t)>                 aligned component (signed)
    n_l(t) = ||h_l(t)||                         total magnitude
    q_l(t) = ||h_l(t) - p_l(t) r_hat(t)||       orthogonal remainder
    a_l(t) = p_l(t) / n_l(t)                    the cosine GDTR measured

The whole point of the experiment is that ``a`` is a ratio, and that reporting
it alone cannot distinguish "the aligned component was preserved and the
denominator grew" from "the aligned component was overwritten".

Channel split
-------------
The reference itself is dominated by one global direction ``u`` (reported at 94%
in the companion manuscript).  Writing

    r_hat(t) = alpha(t) * u + beta(t) * v(t),   v(t) ⟂ u, ||v(t)|| = 1

gives an exact decomposition of the cosine into a *carrier* channel and a
*content* channel:

    a_l(t) = alpha(t) * a^u_l(t) + beta(t) * a^v_l(t)

with ``a^u_l = cos(h_l, u)`` and ``a^v_l = <h_l/||h_l||, v(t)>``.  The identity
is exact by construction and is asserted in the self-test; if it ever fails, the
reference or ``u`` was mis-specified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .backend import TINY, ns_for


# --------------------------------------------------------------------------
# basic geometry
# --------------------------------------------------------------------------

def unit(x, axis=-1):
    """Row-normalise ``x`` ([T, D] -> [T, D]) with a floor on the norm."""
    xp = ns_for(x)
    n = xp.sqrt(xp.sum(x * x, axis=axis))
    return x / xp.clip_min(n, TINY)[..., None], n


def pnq(h, ref_unit):
    """Return ``(p, n, q, a)`` for one tap.

    Parameters
    ----------
    h : [T, D] state
    ref_unit : [T, D] per-position unit reference direction
    """
    xp = ns_for(h)
    p = xp.sum(h * ref_unit, axis=-1)
    n = xp.sqrt(xp.sum(h * h, axis=-1))
    q = xp.sqrt(xp.clip_min(n * n - p * p, 0.0))
    a = p / xp.clip_min(n, TINY)
    return p, n, q, a


@dataclass
class ChannelSplit:
    a: object       # cosine to the full reference
    a_u: object     # carrier channel, cos(h, u)
    a_v: object     # content channel, <h_hat, v(t)>
    alpha: object   # reference's own weight on u
    beta: object    # reference's own weight on v


def channel_split(h, ref_unit, u) -> ChannelSplit:
    """Split the cosine into carrier (global ``u``) and content (per-position) parts."""
    xp = ns_for(h)
    alpha = xp.sum(ref_unit * u[None, :], axis=-1)              # [T]
    v_raw = ref_unit - alpha[..., None] * u[None, :]
    beta = xp.sqrt(xp.clip_min(xp.sum(v_raw * v_raw, axis=-1), 0.0))
    v = v_raw / xp.clip_min(beta, TINY)[..., None]

    h_hat, _ = unit(h)
    a_u = xp.sum(h_hat * u[None, :], axis=-1)
    a_v = xp.sum(h_hat * v, axis=-1)
    a = xp.sum(h_hat * ref_unit, axis=-1)
    return ChannelSplit(a=a, a_u=a_u, a_v=a_v, alpha=alpha, beta=beta)


def channel_identity_error(split: ChannelSplit):
    """Max |a - (alpha*a_u + beta*a_v)|.  Must be ~0; used as a hard check."""
    xp = ns_for(split.a)
    recon = split.alpha * split.a_u + split.beta * split.a_v
    return xp.max(xp.abs(split.a - recon), axis=0)


# --------------------------------------------------------------------------
# output-space geometry
# --------------------------------------------------------------------------

def acgt_contrast_basis(E, acgt_idx, tol: float = 1e-6):
    """Orthonormal basis of S_ctr = span{e_A - mean, ..., e_T - mean}.

    Rank is at most 3 because the four rows are centred.  Returned as ``[k, D]``.
    This is the subspace that actually moves *relative* logits; the raw 4-row
    span additionally contains the common direction, which only shifts all four
    logits together and cannot change the prediction.
    """
    xp = ns_for(E)
    rows = xp.take_rows(E, acgt_idx)                       # [4, D]
    centred = rows - xp.mean(rows, axis=0, keepdims=True)  # rank <= 3
    _, s, vh = xp.svd(centred)
    keep = [i for i in range(len(s)) if float(s[i]) > tol * float(s[0]) + TINY]
    return vh[keep, :]


def subspace_energy_fraction(z, basis):
    """Fraction of each row's squared norm that lies in ``span(basis)``."""
    xp = ns_for(z)
    coeff = xp.matmul(z, basis.T if hasattr(basis, "T") else basis.transpose(-1, -2))
    num = xp.sum(coeff * coeff, axis=-1)
    den = xp.clip_min(xp.sum(z * z, axis=-1), TINY)
    return num / den


def log_softmax(logits):
    """Numerically stable log-softmax over the last axis, backend agnostic."""
    xp = ns_for(logits)
    m = xp.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    lse = xp.log(xp.sum(xp.exp(shifted), axis=-1))
    return shifted - lse[..., None]


@dataclass
class LensMetrics:
    kl_to_final: object     # KL(p_l || p_final)   -- the protocol's D_l
    kl_from_final: object   # KL(p_final || p_l)   -- reported alongside
    entropy: object         # H(p_l) in nats
    top1_match: object      # 1.0 where argmax(p_l) == argmax(p_final)
    logp_true: Optional[object]  # log p_l(realised next base), if labels given
    acgt_mass: object       # probability mass on A/C/G/T


def lens_metrics(logits, final_logp, acgt_idx, true_next=None) -> LensMetrics:
    """Logit-lens readouts of one tap against the model's own final distribution.

    ``logits`` are ``[T, V]`` produced by applying the model's final norm and
    unembedding to the tap.  ``final_logp`` is ``[T, V]`` log-probabilities of
    the unmodified model at the same positions.
    """
    xp = ns_for(logits)
    logp = log_softmax(logits)
    p = xp.exp(logp)
    final_p = xp.exp(final_logp)

    kl_to_final = xp.sum(p * (logp - final_logp), axis=-1)
    kl_from_final = xp.sum(final_p * (final_logp - logp), axis=-1)
    entropy = -xp.sum(p * logp, axis=-1)

    am = xp.argmax(logp, axis=-1)
    am_final = xp.argmax(final_logp, axis=-1)
    top1 = (am == am_final)
    # cast bool -> float in a backend-neutral way
    top1_match = top1 * 1.0

    acgt = logp[..., acgt_idx] if not hasattr(logp, "index_select") else logp[:, acgt_idx]
    acgt_mass = xp.sum(xp.exp(acgt), axis=-1)

    logp_true = None
    if true_next is not None:
        # gather along vocab axis
        rows = range(logp.shape[0]) if not hasattr(logp, "gather") else None
        if rows is None:
            logp_true = logp.gather(1, true_next[:, None]).squeeze(1)
        else:
            import numpy as np

            logp_true = logp[np.arange(logp.shape[0]), np.asarray(true_next)]

    return LensMetrics(
        kl_to_final=kl_to_final,
        kl_from_final=kl_from_final,
        entropy=entropy,
        top1_match=top1_match,
        logp_true=logp_true,
        acgt_mass=acgt_mass,
    )


# --------------------------------------------------------------------------
# stage-2 summaries
# --------------------------------------------------------------------------

def early_ratio(v_onset, v_pre, v_final, eps: float = 1e-12):
    """Fraction of the pre->final change already achieved at the onset block.

    ``r = (v_onset - v_pre) / (v_final - v_pre)``, clipped to [-1, 2] so that a
    near-zero denominator cannot produce a spike.  Positions whose denominator is
    below ``eps`` in magnitude are returned as NaN and dropped downstream rather
    than winsorised silently.
    """
    import numpy as np

    v_onset = np.asarray(v_onset, dtype=np.float64)
    v_pre = np.asarray(v_pre, dtype=np.float64)
    v_final = np.asarray(v_final, dtype=np.float64)
    den = v_final - v_pre
    out = np.full_like(den, np.nan)
    ok = np.abs(den) > eps
    out[ok] = (v_onset[ok] - v_pre[ok]) / den[ok]
    return np.clip(out, -1.0, 2.0)


def settle_depth(match_matrix, blocks, persistence: int = 1):
    """First block index whose top-1 matches the final *and* stays matched.

    ``match_matrix`` is ``[T, L]`` of 0/1 over the blocks listed in ``blocks``
    (ascending).  ``persistence`` requires the match to hold for that many
    consecutive taps, which stops a single coincidental agreement from being
    read as a decision.  Positions that never satisfy it get NaN, and the count
    of such positions must be reported (this is the censoring lesson).
    """
    import numpy as np

    m = np.asarray(match_matrix, dtype=np.float64)
    T, L = m.shape
    out = np.full(T, np.nan)
    for j in range(L - persistence + 1):
        window_ok = np.all(m[:, j : j + persistence] > 0.5, axis=1)
        newly = np.isnan(out) & window_ok
        out[newly] = blocks[j]
    return out
