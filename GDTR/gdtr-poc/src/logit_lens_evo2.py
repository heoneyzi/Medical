"""Vortex-aware logit lens for Evo 2 7B.

Computes per-layer JSD trajectories and top-1 predictions from hidden states
captured by the Evo2 forward hook mechanism. Final-layer output exactly
matches `out.logits` because we apply the same RMSNorm + unembed path.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants_evo2 import BOS_OFFSET, LOG_VOCAB, N_LAYERS, VOCAB_SIZE

log = logging.getLogger(__name__)


def all_layer_names(n_layers: int = N_LAYERS, include_norm: bool = True) -> List[str]:
    """Return ``[blocks.0, ..., blocks.{n_layers-1}, norm]``."""
    names = [f"blocks.{i}" for i in range(n_layers)]
    if include_norm:
        names.append("norm")
    return names


@torch.no_grad()
def extract_hidden_states(
    bundle,
    input_ids: torch.Tensor,
    save_layers: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Forward + hook to extract residual-stream taps.

    Args:
        bundle: Evo2Bundle.
        input_ids: [1, T] int64 tensor on cuda.
        save_layers: dotted submodule paths; defaults to all blocks + norm.

    Returns:
        dict {layer_name: hidden_state [B, T, H]} (bfloat16). The Evo2 hook
        keeps tensors on GPU and detaches them.
    """
    if save_layers is None:
        save_layers = all_layer_names()
    out, embeddings = bundle.model(
        input_ids, return_embeddings=True, layer_names=list(save_layers),
    )
    # Evo2 returns ((logits, None), embeddings_dict) when return_embeddings=True
    # We don't need logits here; caller can request norm-tap to reproduce.
    del out
    # BUG-1 FIX: clone embeddings to break Vortex residual aliasing.
    embeddings = {k: v.detach().clone() for k, v in embeddings.items()}
    return embeddings


@torch.no_grad()
def _layer_logits(
    h: torch.Tensor,
    bundle,
    is_post_norm: bool = False,
) -> torch.Tensor:
    """Project a hidden state to real-vocab logits.

    Applies sh.norm (RMSNorm) then sh.unembed, slicing to VOCAB_SIZE.

    Args:
        h: hidden state [B, T, H].
        bundle: Evo2Bundle.
        is_post_norm: True iff h is already post-RMSNorm (skip norm).

    Returns:
        logits [B, T, VOCAB_SIZE] in bfloat16.
    """
    h_cast = h.to(bundle.embedding_weight.dtype)
    if is_post_norm:
        out = bundle.unembed(h_cast)
    else:
        out = bundle.unembed(bundle.norm(h_cast))
    return out[..., :VOCAB_SIZE]


@torch.no_grad()
def jsd_lens(
    hidden_states: Dict[str, torch.Tensor],
    bundle,
    n_layers: int = N_LAYERS,
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Compute per-layer JSD trajectory normalized by log(VOCAB_SIZE).

    Reference distribution = final logits (computed from ``norm`` tap, which
    matches ``out.logits`` exactly per smoke test).

    Args:
        hidden_states: dict from extract_hidden_states. Must include keys
            ``blocks.{0..n_layers-1}`` and ``norm``.
        bundle: Evo2Bundle.
        n_layers: number of blocks.
        bos_offset: 0 for Evo 2.

    Returns:
        D_jsd float32 CPU tensor [n_layers, T_real]; D[n_layers-1, :] = 0.
    """
    if "norm" not in hidden_states:
        raise KeyError("hidden_states must include 'norm' tap.")
    h_norm = hidden_states["norm"]
    if h_norm.dim() != 3:
        raise ValueError(f"expected [B,T,H] norm tensor, got {h_norm.shape}")
    B, T, _ = h_norm.shape
    if bos_offset >= T:
        raise ValueError(f"bos_offset {bos_offset} >= T {T}")
    T_real = T - bos_offset

    # Final reference distribution
    logits_final = _layer_logits(h_norm, bundle, is_post_norm=True).float()
    log_p_final = F.log_softmax(logits_final, dim=-1)
    p_final = log_p_final.exp()
    p_final_real = p_final[:, bos_offset:, :]
    log_p_final_real = log_p_final[:, bos_offset:, :]

    D = torch.zeros((n_layers, T_real), dtype=torch.float32)
    eps = 1e-30

    for ell in range(n_layers):
        key = f"blocks.{ell}"
        if key not in hidden_states:
            raise KeyError(f"missing {key} in hidden_states")
        h_l = hidden_states[key]
        logits_l = _layer_logits(h_l, bundle, is_post_norm=False).float()
        log_p_l = F.log_softmax(logits_l, dim=-1)[:, bos_offset:, :]
        p_l = log_p_l.exp()
        m = 0.5 * (p_l + p_final_real)
        log_m = (m + eps).log()
        kl_l_m = (p_l * (log_p_l - log_m)).sum(dim=-1)
        kl_f_m = (p_final_real * (log_p_final_real - log_m)).sum(dim=-1)
        jsd = 0.5 * (kl_l_m + kl_f_m)
        jsd = jsd.clamp(min=0.0)
        D[ell] = jsd.mean(dim=0).cpu() / LOG_VOCAB

    # Force exact zero at final layer (ell = n_layers-1) — final block tap is
    # NOT identical to post-norm, but we want D=0 by convention. The final-layer
    # JSD will be tiny (block 31 -> norm is just a residual+RMSNorm).
    # We instead overwrite the last row with zeros if user wants strict.
    # Phase 0 convention: D[L-1] := 0. We follow that.
    D[n_layers - 1].zero_()
    if torch.isnan(D).any():
        raise RuntimeError("JSD lens produced NaN")
    return D


@torch.no_grad()
def top1_predictions(
    hidden_states: Dict[str, torch.Tensor],
    bundle,
    n_layers: int = N_LAYERS,
    bos_offset: int = BOS_OFFSET,
) -> torch.Tensor:
    """Top-1 token id (over full 512 vocab) per layer per position.

    Returns:
        int64 CPU tensor [n_layers, T_real].
    """
    if "norm" not in hidden_states:
        raise KeyError("hidden_states must include 'norm' tap.")
    T = hidden_states["norm"].shape[1]
    T_real = T - bos_offset
    out = torch.zeros((n_layers, T_real), dtype=torch.int64)

    for ell in range(n_layers):
        is_final = (ell == n_layers - 1)
        if is_final:
            h = hidden_states["norm"]
            logits = _layer_logits(h, bundle, is_post_norm=True)
        else:
            h = hidden_states[f"blocks.{ell}"]
            logits = _layer_logits(h, bundle, is_post_norm=False)
        argmax = logits.float().argmax(dim=-1)  # [B, T]
        out[ell] = argmax[0, bos_offset:].cpu()
    return out
