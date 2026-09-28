"""Per-region q70 calibration for gamma_cos at the penultimate layer.

Phase 1.4 deliverable: gamma_cos[region] = 70th percentile of D_cos at
layer (n_layers - 2) (penultimate block before final).
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np

from .constants_evo2 import N_LAYERS

log = logging.getLogger(__name__)


def compute_q70_per_region(
    D_cos: np.ndarray,
    region_labels: np.ndarray,
    n_layers: int = N_LAYERS,
    quantile: float = 0.70,
    layer_for_calibration: int = None,
) -> Dict[str, float]:
    """Compute per-region quantile of D_cos at the penultimate layer.

    Args:
        D_cos: array [N_seqs, n_layers, T] (or flattened [n_layers, N*T] —
            see usage below).
        region_labels: array [N_seqs] of region strings (one per seq) or
            [N_seqs, T] if positions have heterogeneous regions.
        n_layers: total layers.
        quantile: percentile (0-1).
        layer_for_calibration: which layer to take the quantile at; defaults
            to the penultimate (n_layers - 2).

    Returns:
        dict {region_name: gamma_cos (float)}.
    """
    if layer_for_calibration is None:
        layer_for_calibration = n_layers - 2
    if D_cos.ndim != 3:
        raise ValueError(f"expected D_cos shape [N, L, T], got {D_cos.shape}")
    N, L, T = D_cos.shape
    if L != n_layers:
        raise ValueError(f"D_cos n_layers mismatch: got {L}, expected {n_layers}")
    out: Dict[str, float] = {}

    if region_labels.ndim == 1:
        # one label per seq
        for region in np.unique(region_labels):
            mask = region_labels == region
            vals = D_cos[mask, layer_for_calibration, :].reshape(-1)
            if vals.size == 0:
                out[str(region)] = float("nan")
                continue
            q = float(np.quantile(vals, quantile))
            out[str(region)] = q
    elif region_labels.ndim == 2:
        # per-position labels [N, T]
        for region in np.unique(region_labels):
            sel = region_labels == region
            vals = D_cos[:, layer_for_calibration, :][sel]
            if vals.size == 0:
                out[str(region)] = float("nan")
                continue
            q = float(np.quantile(vals, quantile))
            out[str(region)] = q
    else:
        raise ValueError(f"region_labels must be 1- or 2-D, got {region_labels.shape}")

    return out
