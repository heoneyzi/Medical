"""UR-gDTR: cosine-distance lens (auxiliary signal that bypasses lm_head).

Defines D_cos(i, l) = 1 - cos_sim(h_l(i), h_L(i)).
Used as fallback / cross-check when lm_head projection is suspect.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F

from .constants import BOS_OFFSET

log = logging.getLogger(__name__)


@torch.no_grad()
def cosine_lens(
    hidden_states: Sequence[torch.Tensor],
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Per-layer cosine distance to the final pre-ln_f hidden state.

    We compare against `hidden_states[L]` (the final pre-ln_f block output),
    not the post-ln_f tensor, because we want to characterize the residual
    stream itself without LayerNorm-induced rescaling.

    Args:
        hidden_states: tuple as returned by HyenaDNA forward
            (len = L + 2: embed, L blocks, post-ln_f).
        bos_offset: 1 (slice BOS).

    Returns:
        D_cos: float32 CPU tensor [L, T_real]; D_cos[L-1, :] = 0.
    """
    if len(hidden_states) < 3:
        raise ValueError(f"hidden_states too short: {len(hidden_states)}")
    L = len(hidden_states) - 2
    h_final = hidden_states[L].float()  # [B, T, H], pre-ln_f final block
    h_final = h_final[:, bos_offset:, :]
    h_final_n = F.normalize(h_final, p=2, dim=-1)
    B, T_real, H = h_final.shape

    D = torch.zeros((L, T_real), dtype=torch.float32)
    for ell in range(1, L + 1):
        h_l = hidden_states[ell].float()[:, bos_offset:, :]
        h_l_n = F.normalize(h_l, p=2, dim=-1)
        cos = (h_l_n * h_final_n).sum(dim=-1)         # [B, T_real]
        d = (1.0 - cos).clamp(min=0.0).mean(dim=0)
        D[ell - 1] = d.cpu()
    D[L - 1].zero_()
    return D
