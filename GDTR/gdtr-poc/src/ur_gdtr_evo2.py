"""UR-gDTR for Evo 2: cosine-distance lens at residual stream.

Defines D_cos(l, t) = 1 - cos_sim(h_l(t), h_{L-1}(t)) where h_{L-1} is
``blocks.31`` (the last block before post-norm). Bypasses lm_head entirely.
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn.functional as F

from .constants_evo2 import BOS_OFFSET, N_LAYERS

log = logging.getLogger(__name__)


@torch.no_grad()
def cosine_lens(
    hidden_states: Dict[str, torch.Tensor],
    n_layers: int = N_LAYERS,
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Per-layer cosine distance to the final block (blocks.{n_layers-1}).

    Args:
        hidden_states: dict {layer_name: [B, T, H]}.
        n_layers: number of blocks (default 32).
        bos_offset: 0 for Evo 2.

    Returns:
        D_cos float32 CPU tensor [n_layers, T_real]; D[n_layers-1, :] = 0.
    """
    final_key = "norm"  # FIX: blocks.31 is no-op; use post-norm output as ref
    if final_key not in hidden_states:
        raise KeyError(f"missing {final_key} in hidden_states")
    h_final = hidden_states[final_key].float()  # [B, T, H]
    h_final = h_final[:, bos_offset:, :]
    h_final_n = F.normalize(h_final, p=2, dim=-1)
    B, T_real, H = h_final.shape

    D = torch.zeros((n_layers, T_real), dtype=torch.float32)
    for ell in range(n_layers):
        key = f"blocks.{ell}"
        if key not in hidden_states:
            raise KeyError(f"missing {key} in hidden_states")
        h_l = hidden_states[key].float()[:, bos_offset:, :]
        h_l_n = F.normalize(h_l, p=2, dim=-1)
        cos = (h_l_n * h_final_n).sum(dim=-1)
        d = (1.0 - cos).clamp(min=0.0).mean(dim=0)
        D[ell] = d.cpu()
    # self-ref no longer at index N-1 (norm is external ref); no force-zero
    return D
