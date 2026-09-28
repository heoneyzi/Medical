"""gDTR scalar pipeline — running min, settling depth, deep-thinking ratio.

Inputs are normalized JSD trajectories `D` of shape [L, T] in [0, 1].

Layer indexing convention: external API uses 1-based layer numbering
(e.g. settling_depth_discrete returns values in {1..L}, with L meaning
"never crossed gamma"). Internally torch tensors are 0-indexed.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

from .constants import GAMMA_DEFAULT, L_DEFAULT, RHO_DEFAULT

log = logging.getLogger(__name__)


def _validate(D: torch.Tensor) -> Tuple[int, int]:
    if D.dim() != 2:
        raise ValueError(f"expected D shape [L, T], got {D.shape}")
    L, T = D.shape
    if L < 1 or T < 1:
        raise ValueError(f"empty D shape {D.shape}")
    if torch.isnan(D).any():
        raise ValueError("D contains NaN")
    return L, T


def running_min(D: torch.Tensor) -> torch.Tensor:
    """Running minimum along the layer axis.

    Args:
        D: tensor [L, T] of non-negative values.

    Returns:
        running_min: tensor [L, T], running_min[l, t] = min_{k <= l} D[k, t].
        Layer 0 is the running-min anchor (== D[0]); subsequent layers
        accumulate via element-wise min.
    """
    _validate(D)
    return torch.cummin(D, dim=0).values


def settling_depth_discrete(
    D: torch.Tensor,
    gamma: float = GAMMA_DEFAULT,
) -> torch.Tensor:
    """Discrete settling depth c(i): smallest 1-based layer where running_min <= gamma.

    If the trajectory never falls below gamma, c(i) = L (saturated).

    Args:
        D: tensor [L, T] of normalized JSD in [0, 1].
        gamma: threshold in [0, 1].

    Returns:
        int64 tensor [T] of values in {1..L}.
    """
    L, T = _validate(D)
    rmin = running_min(D)              # [L, T]
    below = rmin <= gamma              # [L, T]
    # First layer index (1-based) where below=True; if none, L
    # argmax on bool returns first True if any
    any_below = below.any(dim=0)       # [T]
    first_idx = below.float().argmax(dim=0)  # 0-based layer index of first True
    c = torch.where(any_below, first_idx + 1, torch.tensor(L, dtype=first_idx.dtype))
    return c.to(torch.int64)


def settling_depth_tail_strict(
    D: torch.Tensor,
    gamma: float = GAMMA_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Strict tail settling depth.

    c_tail_strict(t) is the first 1-based layer such that all raw distances
    from that layer through the final layer are <= gamma. If no such suffix
    exists, c = L + 1 and the corresponding unresolved flag is True.

    Args:
        D: tensor [L, T] of non-negative values.
        gamma: threshold.

    Returns:
        (c [T] int64, unresolved [T] bool).
    """
    L, _ = _validate(D)
    suffix_max = torch.flip(
        torch.cummax(torch.flip(D, dims=(0,)), dim=0).values,
        dims=(0,),
    )
    ok = suffix_max <= gamma
    any_ok = ok.any(dim=0)
    first_idx = ok.float().argmax(dim=0)
    fallback = torch.full_like(first_idx, L + 1)
    c = torch.where(any_ok, first_idx + 1, fallback)
    return c.to(torch.int64), ~any_ok


def settling_depth_tail_soft(
    D: torch.Tensor,
    gamma: float = GAMMA_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft tail settling depth.

    c_tail_soft(t; rho) is the first 1-based layer where at least `rho`
    fraction of the suffix layers have raw distance <= gamma. If no suffix
    satisfies the occupancy rule, c = L + 1 and unresolved is True.

    Args:
        D: tensor [L, T] of non-negative values.
        gamma: threshold.
        rho: suffix occupancy threshold in [0, 1].

    Returns:
        (c [T] int64, unresolved [T] bool).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    L, _ = _validate(D)
    inside = (D <= gamma).to(torch.float32)
    suffix_count = torch.flip(
        torch.cumsum(torch.flip(inside, dims=(0,)), dim=0),
        dims=(0,),
    )
    suffix_len = torch.arange(L, 0, -1, dtype=torch.float32, device=D.device)[:, None]
    ok = (suffix_count / suffix_len) >= rho
    any_ok = ok.any(dim=0)
    first_idx = ok.float().argmax(dim=0)
    fallback = torch.full_like(first_idx, L + 1)
    c = torch.where(any_ok, first_idx + 1, fallback)
    return c.to(torch.int64), ~any_ok


def settling_depth_interp(
    D: torch.Tensor,
    gamma: float = GAMMA_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Continuous (linearly interpolated) settling depth.

    For each position t, find the boundary layer ell (1-based) such that
    running_min(D)[ell-1, t] > gamma >= running_min(D)[ell, t]. Linearly
    interpolate within the boundary using the *raw* JSD sequence values
    D[ell-1, t] (above) and D[ell, t] (at-or-below):

        c_interp(t) = ell + (D[ell-1, t] - gamma) / (D[ell-1, t] - D[ell, t])

    Edge cases:
      - Already below at layer 1 (D[0, t] <= gamma): c_interp = 1.0
      - Never below (saturated): c_interp = L (saturation flag returned)
      - Exact equality D[ell, t] == gamma: returns ell (boundary is the layer itself)
      - Degenerate denominator: clamp to ell

    Args:
        D: tensor [L, T] in [0, 1].
        gamma: threshold.

    Returns:
        (c_interp [T] float32, saturated [T] bool).
    """
    L, T = _validate(D)
    rmin = running_min(D)              # [L, T]
    below = rmin <= gamma              # [L, T]
    any_below = below.any(dim=0)       # [T]
    first_idx = below.float().argmax(dim=0)  # 0-based; meaningful only if any_below

    c_interp = torch.full((T,), float(L), dtype=torch.float32)
    saturated = ~any_below

    # Vectorize: boundary layer is first_idx (0-based)
    # If first_idx == 0, c_interp = 1.0 (below at layer 1 directly)
    immediate = any_below & (first_idx == 0)
    c_interp[immediate] = 1.0

    # General case: first_idx >= 1, interpolate between (first_idx-1) and first_idx
    interp_mask = any_below & (first_idx >= 1)
    if interp_mask.any():
        idx = first_idx[interp_mask]                         # [n]
        # We use raw running-min values at the bracket layers (so the curve we
        # bisect is monotone non-increasing; otherwise gamma may not actually
        # lie between D[idx-1] and D[idx]).
        rm_above = rmin[idx - 1, torch.where(interp_mask)[0]]  # [n]
        rm_below = rmin[idx,     torch.where(interp_mask)[0]]
        denom = (rm_above - rm_below).clamp(min=1e-12)
        frac = (rm_above - gamma) / denom
        frac = frac.clamp(min=0.0, max=1.0)
        # 1-based boundary cell ell = idx (i.e., from layer idx to layer idx+1
        # in 1-based counting). We want c in [idx, idx+1], so:
        c_interp[interp_mask] = (idx.float() + frac).to(torch.float32)
    return c_interp, saturated


def gdtr(
    c: torch.Tensor,
    rho: float = RHO_DEFAULT,
    L: int = L_DEFAULT,
) -> float:
    """Genomic Deep-Thinking Ratio: fraction of positions with c > rho * L.

    Args:
        c: tensor [T] of settling depths (discrete int or interp float).
        rho: deep-regime threshold in [0, 1].
        L: total number of layers (for normalisation of rho).

    Returns:
        scalar float in [0, 1].
    """
    if c.numel() == 0:
        return float("nan")
    threshold = rho * L
    return float((c.float() > threshold).float().mean().item())


def deep_thinking_mask(
    c: torch.Tensor,
    rho: float = RHO_DEFAULT,
    L: int = L_DEFAULT,
) -> torch.Tensor:
    """Boolean mask of deep-thinking positions (c > rho * L)."""
    threshold = rho * L
    return c.float() > threshold


# Convenience monotonicity diagnostics used by Gate A
def jsd_running_min_monotonic(D: torch.Tensor) -> torch.Tensor:
    """Per-position bool: True iff D[:, t] is itself running-min monotone.

    Definition (design 3.1 M2): "JSD running-min monotonicity" — the JSD curve
    after running-min reduction should be non-increasing across layers. Since
    running-min is monotone non-increasing by construction, the natural
    diagnostic is whether the *raw* JSD curve is itself non-increasing
    (equivalently, running_min(D) == D pointwise).

    Args:
        D: [L, T].

    Returns:
        bool tensor [T].
    """
    rmin = running_min(D)
    return torch.all(rmin == D, dim=0)


def top1_monotonic_after_first_match(
    top1: torch.Tensor,
) -> torch.Tensor:
    """Per-position bool for M1 (Gate A).

    M1 (top-1 monotonicity rate): the position is "monotonic" iff once the
    layer top-1 first matches the final-layer top-1, it never deviates again.
    Positions that never match are marked non-monotonic (since the trajectory
    cannot then satisfy "after-first-match stability").

    Args:
        top1: int tensor [L, T]. Final layer is row L-1 (0-indexed).

    Returns:
        bool tensor [T].
    """
    if top1.dim() != 2:
        raise ValueError(f"top1 expected [L, T], got {top1.shape}")
    L, T = top1.shape
    final = top1[L - 1]                           # [T]
    matches = top1 == final[None, :]              # [L, T] bool
    # per t: find first layer where matches=True; from then on must all be True
    any_match = matches.any(dim=0)                # [T]
    first_match = matches.float().argmax(dim=0)   # [T] 0-based; valid when any_match
    out = torch.zeros(T, dtype=torch.bool)
    if not any_match.any():
        return out
    pos_idx = torch.where(any_match)[0]
    fm = first_match[pos_idx]
    # tail check: all matches[fm:L, t] must be True
    for k, t in zip(fm.tolist(), pos_idx.tolist()):
        out[t] = bool(matches[k:, t].all().item())
    return out
