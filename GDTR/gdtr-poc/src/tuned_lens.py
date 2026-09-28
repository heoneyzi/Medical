"""Tuned lens (Belrose et al. 2023) for selected Evo 2 layers.

Trains affine A_l in R^{H x H} (+ bias) so that
    unembed(norm(A_l(h_l))) ~ logits_final
under MSE on logits. Initialized as identity. We CLONE the tied embedding
weight before any forward through the unembed pathway since storage is tied
with the input embedding (mutating it would corrupt the model).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants_evo2 import HIDDEN_SIZE, VOCAB_SIZE

log = logging.getLogger(__name__)


class TunedLensAffine(nn.Module):
    """Per-layer affine y = W h + b. Initialized as identity / zero."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.W = nn.Parameter(torch.eye(hidden_size, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., H] -> [..., H]."""
        return F.linear(h, self.W, self.b)


@dataclass
class TunedLensTrainingResult:
    """Output of training run."""

    layer_idx: int
    epochs: int
    loss_curve: List[float]
    final_loss: float


def _frozen_unembed(
    h: torch.Tensor,
    norm: nn.Module,
    embedding_weight_clone: torch.Tensor,
) -> torch.Tensor:
    """Apply norm + (h @ W_emb.T) for logits, using a CLONED weight (no grad).

    The model's `sh.unembed` is tied to the input embedding storage. To avoid
    mutating model weights during tuned-lens training, we clone the weight
    once and matmul ourselves. The clone is .detach().requires_grad_(False).

    Args:
        h: [B, T, H] in float32 or bf16.
        norm: sh.norm (RMSNorm).
        embedding_weight_clone: [V, H] frozen tensor.

    Returns:
        logits [B, T, V] (V == VOCAB_SIZE since clone is sliced upstream).
    """
    h_cast = h.to(embedding_weight_clone.dtype)
    # BUG-2 FIX: bypass model norm (its scale parameter is an inference tensor
    # since the model loads under inference_mode). Reimplement RMSNorm with a
    # cloned scale that is a plain tensor.
    scale = norm.scale.detach().clone().to(h_cast.dtype)
    eps = norm.eps
    H = h_cast.shape[-1]
    rms = h_cast.norm(2, dim=-1, keepdim=True) * (H ** -0.5) + eps
    h_n = (h_cast / rms) * scale
    return F.linear(h_n, embedding_weight_clone)


def train_tuned_lens(
    hidden_layer: torch.Tensor,
    target_logits: torch.Tensor,
    norm_module: nn.Module,
    embedding_weight: torch.Tensor,
    layer_idx: int,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[TunedLensAffine, TunedLensTrainingResult]:
    """Train an affine A_l so that frozen-unembed(norm(A h_l)) ~ target_logits.

    Args:
        hidden_layer: [N, T, H] (concat across N sanity sequences).
        target_logits: [N, T, VOCAB_SIZE] (final-layer logits).
        norm_module: sh.norm.
        embedding_weight: sh.embedding_layer.weight; will be cloned + sliced.
        layer_idx: 30 or 31 (for logging).
        epochs: number of full passes over the N sequences.
        lr: Adam learning rate.
        batch_size: number of sequences per minibatch.
        seed: torch RNG seed.
        device: target device.
        dtype: lens parameter dtype.

    Returns:
        (lens, result). The lens is on `device`; result.loss_curve is per-step.
    """
    torch.manual_seed(seed)
    if hidden_layer.dim() != 3 or target_logits.dim() != 3:
        raise ValueError("expected [N, T, H] and [N, T, V]")
    N, T, H = hidden_layer.shape
    if target_logits.shape[:2] != (N, T):
        raise ValueError("hidden/target N,T mismatch")

    # Clone + slice the embedding weight (storage tied -> never operate on raw)
    emb_clone = embedding_weight.detach().clone()  # [V_full, H]
    emb_clone = emb_clone[:VOCAB_SIZE, :].contiguous().to(device).requires_grad_(False)

    lens = TunedLensAffine(hidden_size=H, dtype=dtype).to(device)
    opt = torch.optim.Adam(lens.parameters(), lr=lr)

    # BUG-2 FIX: convert inference tensors -> plain tensors so autograd works.
    hidden_layer = hidden_layer.clone().detach().requires_grad_(False)
    target_logits = target_logits.clone().detach().to(device, dtype=torch.float32)
    hidden_layer = hidden_layer.to(device)

    loss_curve: List[float] = []
    n_steps_per_epoch = max(1, math.ceil(N / batch_size))
    for ep in range(epochs):
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + ep))
        ep_loss_sum = 0.0
        n_seen = 0
        for step in range(n_steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            if idx.numel() == 0:
                continue
            h_b = hidden_layer[idx].to(dtype=dtype)  # [b, T, H]
            tgt_b = target_logits[idx]  # [b, T, V]
            y = lens(h_b)  # [b, T, H]
            pred_logits = _frozen_unembed(y, norm_module, emb_clone).float()
            loss = F.mse_loss(pred_logits, tgt_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_curve.append(float(loss.item()))
            ep_loss_sum += float(loss.item()) * idx.numel()
            n_seen += idx.numel()
        log.info("tuned_lens L%d ep %d/%d  mean MSE=%.4e", layer_idx, ep + 1, epochs, ep_loss_sum / max(1, n_seen))

    result = TunedLensTrainingResult(
        layer_idx=layer_idx,
        epochs=epochs,
        loss_curve=loss_curve,
        final_loss=loss_curve[-1] if loss_curve else float("nan"),
    )
    return lens, result


def save_tuned_lens(lens: TunedLensAffine, path: str) -> None:
    """Save lens state_dict to .pt file."""
    torch.save(lens.state_dict(), path)


def load_tuned_lens(path: str, hidden_size: int = HIDDEN_SIZE, device: str = "cuda") -> TunedLensAffine:
    """Load lens checkpoint."""
    lens = TunedLensAffine(hidden_size=hidden_size)
    sd = torch.load(path, map_location=device)
    lens.load_state_dict(sd)
    lens.to(device).eval()
    return lens
