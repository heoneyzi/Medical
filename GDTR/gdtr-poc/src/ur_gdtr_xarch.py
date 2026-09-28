"""UR-gDTR cosine-lens, model-agnostic."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def cosine_lens_xarch(layer_hs, final_ref, bos_offset=0):
    h_final = final_ref.float()[:, bos_offset:, :]
    h_final_n = F.normalize(h_final, p=2, dim=-1)
    L = len(layer_hs)
    T_real = h_final.shape[1]
    D = torch.zeros((L, T_real), dtype=torch.float32)
    for ell in range(L):
        h_l = layer_hs[ell].float()[:, bos_offset:, :]
        h_l_n = F.normalize(h_l, p=2, dim=-1)
        cos = (h_l_n * h_final_n).sum(dim=-1)
        D[ell] = (1.0 - cos).clamp(min=0.0).mean(dim=0).cpu()
    return D

def settling_depth_per_window(D_cos, gamma):
    if isinstance(D_cos, torch.Tensor):
        D_cos = D_cos.numpy()
    D_cos = np.asarray(D_cos, dtype=np.float32)
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.int64)
