"""Logit lens for HyenaDNA — JSD trajectory and top-1 predictions.

Computes per-layer post-BOS distributions p_l(token | position) by applying
final LayerNorm + lm_head to each intermediate hidden state, masking to the
real vocabulary, and softmaxing. The trajectory metric is Jensen-Shannon
divergence to the final-layer distribution, normalized to [0, 1] by
log(VOCAB_REAL).

Implementation follows Appendix C.3:
- hidden_states[1..L] are pre-ln_f block outputs -> ln_f + lm_head
- hidden_states[L+1] is post-ln_f -> lm_head only (matches out.logits)
- mask logits[..., :VOCAB_REAL] before softmax
"""
from __future__ import annotations

import logging
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BOS_OFFSET, LOG_VOCAB_REAL, VOCAB_REAL

log = logging.getLogger(__name__)


def _layer_logits(
    h: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Module,
    is_final: bool,
    vocab_real: int,
) -> torch.Tensor:
    """Project a hidden state to real-vocab logits.

    Args:
        h: tensor [B, T, H].
        ln_f: final LayerNorm module.
        lm_head: nn.Linear(H -> 16).
        is_final: True if `h` is already post-ln_f (skip ln_f).
        vocab_real: 12 — slice padding tokens off.

    Returns:
        logits tensor [B, T, vocab_real] (matches lm_head precision).
    """
    # ln_f / lm_head weights are bfloat16; intermediate hidden_states are
    # float32 (HyenaDNA upcasts internally), so we cast h to the module dtype
    # before applying. Final logits are then cast to float32 by the caller for
    # numerically stable softmax/JSD.
    if is_final:
        h_cast = h.to(lm_head.weight.dtype)
        out = lm_head(h_cast)
    else:
        h_cast = h.to(ln_f.weight.dtype)
        out = lm_head(ln_f(h_cast).to(lm_head.weight.dtype))
    return out[..., :vocab_real]


@torch.no_grad()
def jsd_lens(
    hidden_states: Sequence[torch.Tensor],
    ln_f: nn.Module,
    lm_head: nn.Module,
    vocab_real: int = VOCAB_REAL,
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Compute layer-wise Jensen-Shannon divergence to the final layer.

    Args:
        hidden_states: tuple of tensors as returned by HyenaDNA forward with
            output_hidden_states=True. Layout (Appendix C.3):
              [0]   embedding output
              [1..L] pre-ln_f block outputs
              [L+1] post-ln_f (lm_head input for final logits)
            For the medium model L=8 so len(hidden_states)=10.
        ln_f: final LayerNorm module.
        lm_head: lm_head Linear (out=16, mask to vocab_real).
        vocab_real: real vocabulary size, default 12.
        bos_offset: number of BOS tokens to slice off the time axis (1).

    Returns:
        D: float32 CPU tensor of shape [L, T_real] where T_real = T - bos_offset.
        Values are normalized JSD in [0, 1]; D[L-1, :] = 0 (self-distance).
    """
    if len(hidden_states) < 3:
        raise ValueError(
            f"hidden_states too short ({len(hidden_states)}); expected L+2"
        )
    # L = number of intermediate blocks. hidden_states layout = embed + L + post-ln_f
    L = len(hidden_states) - 2
    final_idx = len(hidden_states) - 1  # post-ln_f
    h_final = hidden_states[final_idx]
    if h_final.dim() != 3:
        raise ValueError(f"expected hidden state dim 3, got {h_final.shape}")

    device = h_final.device
    log_vocab = float(torch.log(torch.tensor(vocab_real, dtype=torch.float32)).item())

    # Final distribution (use float32 for stable softmax + JSD)
    logits_final = _layer_logits(h_final, ln_f, lm_head, is_final=True, vocab_real=vocab_real)
    logits_final = logits_final.float()
    log_p_final = F.log_softmax(logits_final, dim=-1)
    p_final = log_p_final.exp()  # [B, T, V]

    B, T, _ = p_final.shape
    if bos_offset >= T:
        raise ValueError(f"bos_offset {bos_offset} >= T {T}")
    T_real = T - bos_offset

    D = torch.zeros((L, T_real), dtype=torch.float32)
    eps = 1e-30

    # Slice off BOS once
    p_final_real = p_final[:, bos_offset:, :]              # [B, T_real, V]
    log_p_final_real = log_p_final[:, bos_offset:, :]

    for ell in range(1, L + 1):
        h_l = hidden_states[ell]
        logits_l = _layer_logits(h_l, ln_f, lm_head, is_final=False, vocab_real=vocab_real)
        logits_l = logits_l.float()
        log_p_l = F.log_softmax(logits_l, dim=-1)[:, bos_offset:, :]
        p_l = log_p_l.exp()

        # m = 0.5 (p_l + p_final); JSD = 0.5 (KL(p_l||m) + KL(p_final||m))
        m = 0.5 * (p_l + p_final_real)
        log_m = (m + eps).log()
        kl_l_m = (p_l * (log_p_l - log_m)).sum(dim=-1)            # [B, T_real]
        kl_f_m = (p_final_real * (log_p_final_real - log_m)).sum(dim=-1)
        jsd = 0.5 * (kl_l_m + kl_f_m)                             # nat units
        # Numerical floor at 0
        jsd = jsd.clamp(min=0.0)

        # Average across batch (B is typically 1; keep general)
        D[ell - 1] = jsd.mean(dim=0).cpu() / log_vocab

    # Force exact zero for layer L (final == final)
    D[L - 1].zero_()
    if torch.isnan(D).any():
        raise RuntimeError("JSD lens produced NaN")
    return D


@torch.no_grad()
def top1_predictions(
    hidden_states: Sequence[torch.Tensor],
    ln_f: nn.Module,
    lm_head: nn.Module,
    vocab_real: int = VOCAB_REAL,
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Top-1 token id (over real vocab) per layer per post-BOS position.

    Args:
        hidden_states: see jsd_lens.
        ln_f: final LayerNorm.
        lm_head: lm_head Linear.
        vocab_real: 12.
        bos_offset: 1 (slice off BOS).

    Returns:
        int64 CPU tensor of shape [L, T_real].
    """
    L = len(hidden_states) - 2
    final_idx = len(hidden_states) - 1
    h_final = hidden_states[final_idx]
    T = h_final.shape[1]
    T_real = T - bos_offset
    out = torch.zeros((L, T_real), dtype=torch.int64)

    for ell in range(1, L + 1):
        is_final = (ell == L)
        h_l = hidden_states[final_idx if is_final else ell]
        logits = _layer_logits(h_l, ln_f, lm_head, is_final=is_final, vocab_real=vocab_real)
        # Use float32 for stable argmax
        argmax = logits.float().argmax(dim=-1)        # [B, T]
        out[ell - 1] = argmax[0, bos_offset:].cpu()
    return out
