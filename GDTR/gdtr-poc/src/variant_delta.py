"""Variant Delta-metrics for Gate C.

Computes ref-vs-alt comparisons of layer-wise JSD trajectories at the
variant position (defined as the central position of the input window).
"""
from __future__ import annotations

import logging
from typing import Dict

import torch

from .constants import GAMMA_DEFAULT, L_DEFAULT
from .gdtr import settling_depth_discrete, settling_depth_interp

log = logging.getLogger(__name__)


def compute_delta_metrics(
    D_ref: torch.Tensor,
    D_alt: torch.Tensor,
    gamma: float = GAMMA_DEFAULT,
    L: int = L_DEFAULT,
    variant_position: int | None = None,
) -> Dict[str, float | torch.Tensor]:
    """Compute variant Delta-metrics at a single position.

    Args:
        D_ref: [L, T] normalized JSD trajectory for the reference forward.
        D_alt: [L, T] normalized JSD trajectory for the alternate forward.
        gamma: settling threshold (default 0.5).
        L: number of layers (default 8).
        variant_position: 0-based post-BOS position to read out. If None,
            uses the central position T // 2.

    Returns:
        dict with:
          - "delta_c_discrete": int (c_alt - c_ref at variant position)
          - "delta_c_interp":   float
          - "delta_D":          torch.Tensor [L]
          - "max_abs_delta_D":  float
          - "signed_argmax_delta_D": float (delta_D at argmax|delta_D|)
          - "variant_position": int
    """
    if D_ref.shape != D_alt.shape:
        raise ValueError(f"shape mismatch ref={D_ref.shape} alt={D_alt.shape}")
    L_t, T = D_ref.shape
    if L_t != L:
        log.warning("D layer count %d != L=%d argument; using D shape", L_t, L)
        L = L_t
    pos = T // 2 if variant_position is None else int(variant_position)
    if not (0 <= pos < T):
        raise ValueError(f"variant_position {pos} outside [0, {T})")

    c_ref_d = settling_depth_discrete(D_ref, gamma=gamma)[pos].item()
    c_alt_d = settling_depth_discrete(D_alt, gamma=gamma)[pos].item()
    c_ref_i, _ = settling_depth_interp(D_ref, gamma=gamma)
    c_alt_i, _ = settling_depth_interp(D_alt, gamma=gamma)

    delta_D = D_alt[:, pos] - D_ref[:, pos]              # [L]
    abs_dD = delta_D.abs()
    arg_max = int(abs_dD.argmax().item())
    signed_argmax = float(delta_D[arg_max].item())
    return {
        "delta_c_discrete": int(c_alt_d - c_ref_d),
        "delta_c_interp": float(c_alt_i[pos].item() - c_ref_i[pos].item()),
        "delta_D": delta_D.cpu(),
        "max_abs_delta_D": float(abs_dD.max().item()),
        "signed_argmax_delta_D": signed_argmax,
        "argmax_layer_1based": arg_max + 1,
        "variant_position": pos,
    }
