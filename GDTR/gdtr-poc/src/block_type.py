"""Block-type stratification utilities for Evo 2 32-block striped topology."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .constants_evo2 import (
    ATTN_LAYERS, HCS_LAYERS, HCM_LAYERS, HCL_LAYERS, N_LAYERS,
)


def block_type(idx: int) -> str:
    """Return one of {'attn', 'hcs', 'hcm', 'hcl'} for block idx 0..31."""
    if idx in ATTN_LAYERS:
        return "attn"
    if idx in HCS_LAYERS:
        return "hcs"
    if idx in HCM_LAYERS:
        return "hcm"
    if idx in HCL_LAYERS:
        return "hcl"
    raise ValueError(f"unknown block idx {idx}")


def stratify_by_block_type(
    values_per_layer: Sequence[float],
    n_layers: int = N_LAYERS,
) -> Dict[str, List[float]]:
    """Group per-layer scalars by block type.

    Args:
        values_per_layer: length n_layers iterable.
        n_layers: 32.

    Returns:
        dict {'attn': [...], 'hcs': [...], 'hcm': [...], 'hcl': [...]}.
    """
    if len(values_per_layer) != n_layers:
        raise ValueError(f"expected {n_layers} values, got {len(values_per_layer)}")
    out: Dict[str, List[float]] = {"attn": [], "hcs": [], "hcm": [], "hcl": []}
    for i, v in enumerate(values_per_layer):
        out[block_type(i)].append(float(v))
    return out


def block_type_array(n_layers: int = N_LAYERS) -> np.ndarray:
    """Return string array [n_layers] with block type per layer."""
    return np.array([block_type(i) for i in range(n_layers)])
