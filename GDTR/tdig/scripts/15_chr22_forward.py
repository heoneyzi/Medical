"""PHASE B v2 — chr22 main forward with design v2.

Changes from v1:
  - Persistence-window settling (W=3 default) replaces v1 running-min
  - M4_set inline (Sigma_ref^{-1}-weighted distance, monotone-decrease, no running-min)
  - M5 Option B locked: RMSNormed trajectory for Ref B (numerator+denom consistent)
  - M3 c_geo: 5 alpha/beta cells {(1,0),(0,1),(1,1),(1,0.5),(0.5,1)}
  - M6 D_L2 dropped (consistency check only, computed elsewhere)

Per-window 2-track:
  - all 12,978 windows → Tier 1 settling (15 cells)
  - 100 subset windows → + Tier 2 per-layer scalars (incl M4_set D per layer)
                          + Tier 3 raw h_ell (already on disk for 70; will fill remaining 30)

Loads from population_stats/:
  - gamma_calibration_v2.json
  - sigma_ref_inv_{A,B,C}.npy
  - _geo_pop_stats (v_mean, v_std, kappa_mean, kappa_std)

Target wall on H200 batch=1: ~120 min (extra M4_set computation vs v1).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/gDTR")

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096

# Design v2 α/β cells for M3 c_geo
ALPHA_BETA_CELLS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]


# ─── Settling protocols ─────────────────────────────────────────────────────

def settling_persistence(D: np.ndarray, gamma: float, W: int = 3,
                          max_layer: int | None = None) -> np.ndarray:
    """First ell <= max_layer where D[k] <= gamma for all k in [ell, min(ell+W-1, max_layer)].

    max_layer = L-1 by default. For metrics targeting h_29 (Def 2), set max_layer = L_STAR.
    Lookahead clips at max_layer (a token at the reference is "settled" by definition).
    """
    L, T = D.shape
    if max_layer is None or max_layer >= L:
        max_layer = L - 1
    below = D <= gamma
    rolling = np.zeros((L, T), dtype=bool)
    for ell in range(max_layer + 1):
        end_k = min(ell + W - 1, max_layer)
        rolling[ell] = below[ell:end_k + 1].all(axis=0)
    c = np.full(T, -1, dtype=np.int32)
    any_set = rolling.any(axis=0)
    c[any_set] = np.argmax(rolling, axis=0)[any_set]
    return c


def settling_geo_strict(g: np.ndarray, tau: float) -> np.ndarray:
    """M3 strict: g[k] <= tau for ALL k in [ell, end]."""
    below = g <= tau
    rev_or = np.maximum.accumulate((~below)[::-1], axis=0)[::-1]
    settled = ~rev_or
    c = np.full(g.shape[1], -1, dtype=np.int32)
    any_set = settled.any(axis=0)
    c[any_set] = np.argmax(settled, axis=0)[any_set]
    return c


def settling_monotone_direct(D: np.ndarray, gamma: float) -> np.ndarray:
    """M4_set: monotone-decreasing by construction; direct first-crossing."""
    below = D <= gamma
    c = np.full(D.shape[1], -1, dtype=np.int32)
    any_set = below.any(axis=0)
    c[any_set] = np.argmax(below, axis=0)[any_set]
    return c


# ─── M4_set computation (GPU) ──────────────────────────────────────────────

@torch.no_grad()
def compute_m4_set_gpu(h_blocks: torch.Tensor, h_ref: torch.Tensor,
                       sigma_inv: torch.Tensor) -> torch.Tensor:
    """D_M_set(ell, t) = sqrt((h_l - h_ref)^T Sigma_inv (h_l - h_ref)).

    h_blocks: (L, T, H) fp32 on GPU
    h_ref:    (T, H) on GPU
    sigma_inv:(H, H) on GPU
    Returns:  (L, T) fp32 on GPU
    """
    L, T, H = h_blocks.shape
    diff = h_blocks - h_ref.unsqueeze(0)  # (L, T, H)
    diff_flat = diff.reshape(L * T, H)
    sig_diff = diff_flat @ sigma_inv  # (L*T, H)
    quad = (diff_flat * sig_diff).sum(dim=-1).reshape(L, T)
    return torch.sqrt(torch.clamp(quad, min=0.0))


# ─── Window-level forward ──────────────────────────────────────────────────

@torch.no_grad()
def compute_window_v2(bundle, seq: str, sigma_inv_A, sigma_inv_B, sigma_inv_C):
    """Forward + compute all v2 metric scalars.

    Returns:
        scalars: dict of (L, T) or (T,) numpy arrays
        D_M_set: dict 'A'/'B'/'C' -> (L, T) numpy fp32
        gpu: dict of GPU tensors for Tier 3 dumping
        T: token count
    """
    from src.constants_evo2 import N_LAYERS as L, HIDDEN_SIZE as H
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    input_ids = tokenize(seq, bundle, device="cuda")
    layer_names = all_layer_names()
    hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)

    T = hs["norm"].shape[1]
    h_blocks = torch.stack([hs[f"blocks.{ell}"][0] for ell in range(L)], dim=0).float()  # (L, T, H)
    h_norm = hs["norm"][0].float()  # (T, H)
    h_29 = h_blocks[L_STAR]

    # Batched RMSNorm
    h_flat = h_blocks.reshape(L * T, H).to(bundle.embedding_weight.dtype)
    h_blocks_rms = bundle.norm(h_flat).reshape(L, T, H).float()
    h_29_rms = h_blocks_rms[L_STAR]

    # ─── Cosines (3 refs) ──────────────────────────────────────────────
    cos_A = F.cosine_similarity(h_blocks, h_29.unsqueeze(0).expand(L, -1, -1), dim=-1)
    cos_B = F.cosine_similarity(h_blocks_rms, h_29_rms.unsqueeze(0).expand(L, -1, -1), dim=-1)
    cos_C = F.cosine_similarity(h_blocks, h_norm.unsqueeze(0).expand(L, -1, -1), dim=-1)

    # ─── Magnitudes ────────────────────────────────────────────────────
    norm_h_ell = torch.linalg.vector_norm(h_blocks, dim=-1)  # (L, T)
    norm_h_29 = torch.linalg.vector_norm(h_29, dim=-1)
    norm_rms_h_29 = torch.linalg.vector_norm(h_29_rms, dim=-1)
    norm_h_norm = torch.linalg.vector_norm(h_norm, dim=-1)

    # ─── Steps (raw + RMSNormed for Option B) ──────────────────────────
    deltas_raw = h_blocks[1:] - h_blocks[:-1]   # (L-1, T, H)
    deltas_rms = h_blocks_rms[1:] - h_blocks_rms[:-1]
    step_norm_raw = torch.linalg.vector_norm(deltas_raw, dim=-1)
    step_norm_rms = torch.linalg.vector_norm(deltas_rms, dim=-1)
    step_cos_raw = F.cosine_similarity(deltas_raw[1:], deltas_raw[:-1], dim=-1)  # (L-2, T)

    # ─── Direct residual norms for M5 tau accuracy (no law-of-cosines approximation) ─
    # Especially important for Ref B where ||RMSNorm(h_ell)|| ≈ const breaks the proxy
    res_norm_refA = torch.linalg.vector_norm(h_blocks - h_29.unsqueeze(0), dim=-1)         # (L, T)
    res_norm_refB = torch.linalg.vector_norm(h_blocks_rms - h_29_rms.unsqueeze(0), dim=-1)
    res_norm_refC = torch.linalg.vector_norm(h_blocks - h_norm.unsqueeze(0), dim=-1)

    # ─── M4_set 3 variants ─────────────────────────────────────────────
    D_M_set_A = compute_m4_set_gpu(h_blocks, h_29, sigma_inv_A)
    D_M_set_B = compute_m4_set_gpu(h_blocks_rms, h_29_rms, sigma_inv_B)
    D_M_set_C = compute_m4_set_gpu(h_blocks, h_norm, sigma_inv_C)

    # ─── Output dict ───────────────────────────────────────────────────
    scalars = {
        "cos_refA": cos_A.cpu().numpy().astype(np.float16),
        "cos_refB": cos_B.cpu().numpy().astype(np.float16),
        "cos_refC": cos_C.cpu().numpy().astype(np.float16),
        "norm_h_ell": norm_h_ell.cpu().numpy().astype(np.float32),
        "step_norm_raw": step_norm_raw.cpu().numpy().astype(np.float32),
        "step_norm_rms": step_norm_rms.cpu().numpy().astype(np.float32),
        "step_cos": step_cos_raw.cpu().numpy().astype(np.float16),
        "norm_h_29": norm_h_29.cpu().numpy().astype(np.float32),
        "norm_rms_h_29": norm_rms_h_29.cpu().numpy().astype(np.float32),
        "norm_h_norm": norm_h_norm.cpu().numpy().astype(np.float32),
        # Direct residual norms (avoid law-of-cosines proxy for Ref B)
        "res_norm_refA": res_norm_refA.cpu().numpy().astype(np.float32),
        "res_norm_refB": res_norm_refB.cpu().numpy().astype(np.float32),
        "res_norm_refC": res_norm_refC.cpu().numpy().astype(np.float32),
    }
    D_M_set = {
        "A": D_M_set_A.cpu().numpy().astype(np.float32),
        "B": D_M_set_B.cpu().numpy().astype(np.float32),
        "C": D_M_set_C.cpu().numpy().astype(np.float32),
    }
    gpu = {"h_blocks": h_blocks, "h_norm": h_norm, "h_blocks_rms": h_blocks_rms}
    return scalars, D_M_set, gpu, T


# ─── 15-cell settling ───────────────────────────────────────────────────

def compute_v2_settling(scalars: dict, D_M_set: dict, gamma_v2: dict,
                         persistence_w: int = 3) -> dict:
    """Compute v2 settling depths for all 15 main cells + diagnostic M2 cells."""
    L = N_LAYERS
    T = scalars["norm_h_ell"].shape[1]
    cells = {}

    # ── M1 dir x {A, B, C} ──
    # Ref A/B target h_29 → persistence range clipped at L_STAR (= 29)
    # Ref C targets h_norm → trajectory enters h_norm direction at L30 rotation; use full L
    M1_max_layer = {"refA": L_STAR, "refB": L_STAR, "refC": L - 1}
    for ref in ("refA", "refB", "refC"):
        D_dir = 1.0 - scalars[f"cos_{ref}"].astype(np.float32)
        gamma = gamma_v2[f"D_dir_{ref}_at_28"]["q70"]
        cells[f"M1_dir_{ref}"] = settling_persistence(D_dir, gamma, W=persistence_w,
                                                       max_layer=M1_max_layer[ref])

    # ── M2 mag refA (production) + diagnostic refB/refC ──
    norm_ell = scalars["norm_h_ell"]
    for ref_label, ref_norm_key, gamma_key in [
        ("refA", "norm_h_29", "D_mag_refA_at_28"),
    ]:
        r = norm_ell / (scalars[ref_norm_key][None, :] + 1e-12)
        D_mag = np.abs(r - 1.0)
        gamma = gamma_v2[gamma_key]["q70"]
        # Ref A targets h_29: clip persistence at L_STAR
        cells[f"M2_mag_{ref_label}"] = settling_persistence(D_mag, gamma, W=persistence_w,
                                                             max_layer=L_STAR)
    # Diagnostic M2_mag_refB / refC: just compute and store for completeness
    for ref_label, ref_norm_key in [("refB", "norm_rms_h_29"), ("refC", "norm_h_norm")]:
        r = norm_ell / (scalars[ref_norm_key][None, :] + 1e-12)
        D_mag = np.abs(r - 1.0)
        gamma_loose = float(np.quantile(D_mag.flatten(), 0.99)) if D_mag.size > 0 else 1e6
        max_l = L_STAR if ref_label == "refB" else L - 1
        cells[f"M2_mag_{ref_label}_diag"] = settling_persistence(D_mag, gamma_loose,
                                                                  W=persistence_w, max_layer=max_l)

    # ── M3 c_geo: 5 α/β cells ──
    v_mean = gamma_v2["_geo_pop_stats"]["v_mean"]
    v_std = gamma_v2["_geo_pop_stats"]["v_std"]
    k_mean = gamma_v2["_geo_pop_stats"]["kappa_mean"]
    k_std = gamma_v2["_geo_pop_stats"]["kappa_std"]
    step_n = scalars["step_norm_raw"]  # (L-1, T)
    step_c = scalars["step_cos"].astype(np.float32)  # (L-2, T)
    v = step_n / (norm_ell[: L - 1] + 1e-12)        # (L-1, T)
    kappa = 1.0 - step_c                              # (L-2, T)
    v_use = v[: L - 2]
    v_z = (v_use - v_mean) / (v_std + 1e-12)
    k_z = (kappa - k_mean) / (k_std + 1e-12)
    # M3 c_geo: relax strict "all subsequent" to persistence W=3 for empirical viability
    # (strict version turned out infeasible — standardized g rarely stays below q70 for all
    # remaining layers). W=3 maintains the spirit (no single-dip artifact) without being
    # impossible to satisfy. Strict version still available as settling_geo_strict for ablation.
    L_geo = L - 2  # g defined for ell in 0..L-3
    for (alpha, beta) in ALPHA_BETA_CELLS:
        g = alpha * v_z + beta * k_z
        gamma_key = f"D_geo_a{alpha}_b{beta}_at_26"
        gamma = gamma_v2[gamma_key]["q70"]
        cells[f"M3_geo_a{alpha}_b{beta}"] = settling_persistence(g, gamma, W=persistence_w,
                                                                  max_layer=L_geo - 1)

    # ── M4_set x {A, B, C} ── (monotone, direct first-crossing)
    for ref_label in ("A", "B", "C"):
        D = D_M_set[ref_label]
        gamma = gamma_v2[f"D_Mset_ref{ref_label}_at_28"]["q70"]
        cells[f"M4_set_ref{ref_label}"] = settling_monotone_direct(D, gamma)

    # ── M5 c_τ x {A, B (Option B), C} ──
    # path numerator for Ref A/C: raw steps cumulative
    # path numerator for Ref B: RMSNormed steps cumulative (Option B)
    step_raw = scalars["step_norm_raw"]   # (L-1, T)
    step_rms = scalars["step_norm_rms"]   # (L-1, T)
    path_cum_raw = np.cumsum(step_raw[:L_STAR], axis=0)  # (L*, T)
    path_total_raw = path_cum_raw[L_STAR - 1]
    remaining_raw = np.zeros((L_STAR, T), dtype=np.float32)
    remaining_raw[0] = path_total_raw
    remaining_raw[1:] = path_total_raw[None, :] - path_cum_raw[: L_STAR - 1]
    path_cum_rms = np.cumsum(step_rms[:L_STAR], axis=0)
    path_total_rms = path_cum_rms[L_STAR - 1]
    remaining_rms = np.zeros((L_STAR, T), dtype=np.float32)
    remaining_rms[0] = path_total_rms
    remaining_rms[1:] = path_total_rms[None, :] - path_cum_rms[: L_STAR - 1]

    # Use DIRECT residual norms from compute_window_v2 (no proxy approximation)
    # M5 max_layer = L_STAR - 1 = 28 (tau defined only for ell < L_STAR; tau[L_STAR-1] is boundary)
    tau_max = L_STAR - 1

    # Ref A: raw numerator + raw residual norm
    tau_A = remaining_raw / (scalars["res_norm_refA"][:L_STAR] + 1e-12)
    gamma_A = gamma_v2["tau_refA_at_27"]["q70"]
    cells["M5_tau_refA"] = settling_persistence(tau_A, gamma_A, W=persistence_w, max_layer=tau_max)

    # Ref B Option B: RMSNormed numerator + RMSNormed residual norm (BOTH actual, no proxy)
    tau_B = remaining_rms / (scalars["res_norm_refB"][:L_STAR] + 1e-12)
    gamma_B = gamma_v2["tau_refB_at_27"]["q70"]
    cells["M5_tau_refB"] = settling_persistence(tau_B, gamma_B, W=persistence_w, max_layer=tau_max)

    # Ref C asymmetric: raw numerator + raw vs h_norm residual
    tau_C = remaining_raw / (scalars["res_norm_refC"][:L_STAR] + 1e-12)
    gamma_C = gamma_v2["tau_refC_at_27"]["q70"]
    cells["M5_tau_refC"] = settling_persistence(tau_C, gamma_C, W=persistence_w, max_layer=tau_max)

    return cells


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-tsv", type=Path,
                        default=Path("/path/to/gDTR/data/baselines/chr22_windows.tsv"))
    parser.add_argument("--fasta", type=Path,
                        default=Path("/path/to/gDTR/data/reference/chr22.fa"))
    parser.add_argument("--subset-file", type=Path,
                        default=Path("/path/to/TDiG/data/subset_window_ids.json"))
    parser.add_argument("--gamma-file", type=Path,
                        default=Path("/path/to/TDiG/data/cache/population_stats/gamma_calibration_v2.json"))
    parser.add_argument("--sigma-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/population_stats"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22_v2"))
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--tier3-tokens-per-window", type=int, default=600)
    parser.add_argument("--persistence-w", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading v2 calibration...", flush=True)
    gamma_v2 = json.loads(args.gamma_file.read_text())

    print(f"[setup] loading Sigma_ref_inv x 3...", flush=True)
    sigma_inv_A_np = np.load(args.sigma_dir / "sigma_ref_inv_A.npy")
    sigma_inv_B_np = np.load(args.sigma_dir / "sigma_ref_inv_B.npy")
    sigma_inv_C_np = np.load(args.sigma_dir / "sigma_ref_inv_C.npy")
    sigma_inv_A = torch.from_numpy(sigma_inv_A_np).to("cuda")
    sigma_inv_B = torch.from_numpy(sigma_inv_B_np).to("cuda")
    sigma_inv_C = torch.from_numpy(sigma_inv_C_np).to("cuda")
    print(f"[setup] Sigma loaded to GPU: {sigma_inv_A.shape}", flush=True)

    print(f"[setup] loading windows + FASTA + subset", flush=True)
    df_w = pd.read_csv(args.windows_tsv, sep="\t")
    if args.max_windows > 0:
        df_w = df_w.head(args.max_windows).reset_index(drop=True)
    N = len(df_w)
    print(f"[setup] N={N} windows", flush=True)

    import pysam
    fasta = pysam.FastaFile(str(args.fasta))

    subset = json.loads(args.subset_file.read_text())
    subset_ids = set(subset["chr22"])

    print(f"[setup] loading Evo 2", flush=True)
    from src.model_loader_evo2 import load_evo2
    bundle = load_evo2()
    print(f"[setup] model loaded, variant={bundle.loaded_variant}", flush=True)

    # Outputs
    tier1_path = args.out_dir / "tier1_settling_v2.parquet"
    tier2_path = args.out_dir / "tier2_scalars_subset_v2.h5"
    tier3_path = args.out_dir / "tier3_raw_v2.h5"
    meta_path = args.out_dir / "window_metadata.parquet"

    N_subset = len(subset_ids)
    T_typ = 6000
    T_sub = args.tier3_tokens_per_window
    L = N_LAYERS; H = HIDDEN_SIZE
    subset_local_idx = {wid: i for i, wid in enumerate(sorted(subset_ids))}

    if not tier2_path.exists():
        with h5py.File(tier2_path, "w") as h5:
            h5.create_dataset("cos_refA",     shape=(N_subset, L, T_typ), dtype="float16")
            h5.create_dataset("cos_refB",     shape=(N_subset, L, T_typ), dtype="float16")
            h5.create_dataset("cos_refC",     shape=(N_subset, L, T_typ), dtype="float16")
            h5.create_dataset("norm_h_ell",   shape=(N_subset, L, T_typ), dtype="float32")
            h5.create_dataset("step_norm_raw",shape=(N_subset, L - 1, T_typ), dtype="float32")
            h5.create_dataset("step_norm_rms",shape=(N_subset, L - 1, T_typ), dtype="float32")
            h5.create_dataset("step_cos",     shape=(N_subset, L - 2, T_typ), dtype="float16")
            h5.create_dataset("D_Mset_A",     shape=(N_subset, L, T_typ), dtype="float32")
            h5.create_dataset("D_Mset_B",     shape=(N_subset, L, T_typ), dtype="float32")
            h5.create_dataset("D_Mset_C",     shape=(N_subset, L, T_typ), dtype="float32")
            h5.create_dataset("norm_h_29",      shape=(N_subset, T_typ), dtype="float32")
            h5.create_dataset("norm_rms_h_29",  shape=(N_subset, T_typ), dtype="float32")
            h5.create_dataset("norm_h_norm",    shape=(N_subset, T_typ), dtype="float32")
            h5.create_dataset("window_idx", shape=(N_subset,), dtype="int64")
            h5.create_dataset("done_mask",  shape=(N_subset,), dtype="uint8")
            h5["window_idx"][:] = sorted(subset_ids)
    if not tier3_path.exists():
        with h5py.File(tier3_path, "w") as h5:
            h5.create_dataset("raw_h_ell",           shape=(N_subset, L, T_sub, H), dtype="float32",
                              chunks=(1, L, T_sub, H))
            h5.create_dataset("raw_h_ell_rmsnormed", shape=(N_subset, L, T_sub, H), dtype="float16",
                              chunks=(1, L, T_sub, H))
            h5.create_dataset("raw_h_norm",          shape=(N_subset, T_sub, H), dtype="float16")
            h5.create_dataset("window_idx",  shape=(N_subset,), dtype="int64")
            h5.create_dataset("token_stride", shape=(N_subset,), dtype="int32")
            h5.create_dataset("done_mask",   shape=(N_subset,), dtype="uint8")
            h5["window_idx"][:] = sorted(subset_ids)

    tier1_records = []
    completed_set = set()
    if tier1_path.exists():
        try:
            existing = pd.read_parquet(tier1_path)
            completed_set = set(existing["window_idx"].tolist())
            tier1_records = existing.to_dict(orient="records")
            print(f"[resume] {len(completed_set)} windows already done", flush=True)
        except Exception as e:
            print(f"[resume] {e}", flush=True)

    t_start = time.time()
    processed = 0

    for i, row in df_w.iterrows():
        wid = int(row["window_idx"])
        if wid in completed_set:
            continue
        chrom = row["chrom"]; start = int(row["start"]); end = int(row["end"])

        try:
          with torch.no_grad():
            seq = fasta.fetch(chrom, start, end).upper()
            scalars, D_M_set, gpu_tensors, T = compute_window_v2(
                bundle, seq, sigma_inv_A, sigma_inv_B, sigma_inv_C,
            )
            cells = compute_v2_settling(scalars, D_M_set, gamma_v2, persistence_w=args.persistence_w)

            rec = {"window_idx": wid, "chrom": chrom, "start": start, "end": end, "T": T}
            for cell_id, c_arr in cells.items():
                rec[cell_id] = c_arr[:T].astype(np.int32).tolist()
            tier1_records.append(rec)

            # Subset tier 2 + 3
            if wid in subset_ids:
                li = subset_local_idx[wid]

                def pad(arr, target_T):
                    if arr.shape[-1] >= target_T:
                        return arr[..., :target_T]
                    pads = np.zeros(arr.shape[:-1] + (target_T - arr.shape[-1],), dtype=arr.dtype)
                    return np.concatenate([arr, pads], axis=-1)

                with h5py.File(tier2_path, "a") as h5:
                    h5["cos_refA"][li]      = pad(scalars["cos_refA"], T_typ)
                    h5["cos_refB"][li]      = pad(scalars["cos_refB"], T_typ)
                    h5["cos_refC"][li]      = pad(scalars["cos_refC"], T_typ)
                    h5["norm_h_ell"][li]    = pad(scalars["norm_h_ell"], T_typ)
                    h5["step_norm_raw"][li] = pad(scalars["step_norm_raw"], T_typ)
                    h5["step_norm_rms"][li] = pad(scalars["step_norm_rms"], T_typ)
                    h5["step_cos"][li]      = pad(scalars["step_cos"], T_typ)
                    h5["D_Mset_A"][li]      = pad(D_M_set["A"], T_typ)
                    h5["D_Mset_B"][li]      = pad(D_M_set["B"], T_typ)
                    h5["D_Mset_C"][li]      = pad(D_M_set["C"], T_typ)
                    h5["norm_h_29"][li]     = pad(scalars["norm_h_29"], T_typ)
                    h5["norm_rms_h_29"][li] = pad(scalars["norm_rms_h_29"], T_typ)
                    h5["norm_h_norm"][li]   = pad(scalars["norm_h_norm"], T_typ)
                    h5["done_mask"][li] = 1

                stride = max(1, T // args.tier3_tokens_per_window)
                pos_idx = np.arange(0, T, stride)[: args.tier3_tokens_per_window]
                if len(pos_idx) < args.tier3_tokens_per_window:
                    pos_idx = np.pad(pos_idx,
                                     (0, args.tier3_tokens_per_window - len(pos_idx)),
                                     mode="edge")
                pos_t = torch.as_tensor(pos_idx, dtype=torch.long, device=gpu_tensors["h_blocks"].device)
                raw = gpu_tensors["h_blocks"][:, pos_t, :].detach().cpu().numpy().astype(np.float32)
                raw_rms = gpu_tensors["h_blocks_rms"][:, pos_t, :].detach().cpu().numpy().astype(np.float16)
                raw_norm = gpu_tensors["h_norm"][pos_t, :].detach().cpu().numpy().astype(np.float16)
                with h5py.File(tier3_path, "a") as h5:
                    h5["raw_h_ell"][li] = raw
                    h5["raw_h_ell_rmsnormed"][li] = raw_rms
                    h5["raw_h_norm"][li] = raw_norm
                    h5["token_stride"][li] = stride
                    h5["done_mask"][li] = 1

            del gpu_tensors
            torch.cuda.empty_cache()
            processed += 1

            if processed % args.log_every == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                eta = (N - len(completed_set) - processed) / rate / 60
                gpu_peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [{len(completed_set)+processed}/{N}] rate={rate:.2f} win/s ETA={eta:.1f}min GPU_peak={gpu_peak:.1f}GB", flush=True)

            if processed % args.save_every == 0:
                tmp = tier1_path.with_suffix(".tmp.parquet")
                pd.DataFrame(tier1_records).to_parquet(tmp, index=False)
                tmp.replace(tier1_path)

        except Exception as e:
            print(f"  [ERR window {wid}] {type(e).__name__}: {e}", flush=True)
            raise

    pd.DataFrame(tier1_records).to_parquet(tier1_path, index=False)
    df_w[["window_idx", "chrom", "start", "end", "gc_content", "n_fraction",
          "n_coding_exon", "n_intron", "n_5utr", "n_3utr", "n_splice",
          "n_intergenic"]].to_parquet(meta_path, index=False)

    prov = {
        "script": "15_chr22_forward.py (v2)",
        "host": platform.node(),
        "design_version": "v2",
        "persistence_w": args.persistence_w,
        "n_windows_total": int(N),
        "n_completed_this_run": int(processed),
        "wall_minutes": (time.time() - t_start) / 60,
        "tier1_parquet": str(tier1_path),
        "tier2_h5_subset": str(tier2_path),
        "tier3_h5_subset": str(tier3_path),
        "model_variant": bundle.loaded_variant,
        "v2_changes_applied": [
            "persistence W=3 for M1/M2/M5",
            "M4_set inline (Sigma_ref^{-1}, 3 variants, monotone-decrease, no running-min)",
            "M5 Option B (RMSNormed trajectory for Ref B numerator+denom)",
            "M3 c_geo 5 alpha/beta cells",
            "M2 reported as Residual accumulation magnitude (diagnostic for Ref B/C)",
        ],
    }
    (args.out_dir / "_provenance.json").write_text(json.dumps(prov, indent=2))
    (args.out_dir / "_done").write_text(json.dumps({"ok": True, "n_completed": int(processed)}, indent=2))
    print(f"[done] {args.out_dir}  wall={prov['wall_minutes']:.2f}min")


if __name__ == "__main__":
    main()
