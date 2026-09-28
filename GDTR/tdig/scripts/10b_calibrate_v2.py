"""Phase Pre-V2 calibration — Σ_ref + γ_v2 thresholds for design v2.

SERVER. Runs forward on chr22 sanity sequences (same 100 as 10_population_stats.py),
collects vectors needed for v2 calibration:

  Σ_ref^{(A)} = Cov(h_29 raw)                  → for M4_set Ref A
  Σ_ref^{(B)} = Cov(RMSNorm(h_29))              → for M4_set Ref B
  Σ_ref^{(C)} = Cov(h_norm)                     → for M4_set Ref C
  + Ledoit-Wolf shrinkage, then invert

  γ_v2 at q70 for v2 cells:
    M1 c_dir × {A, B, C}     at ell=28
    M2 c_mag × A             at ell=28
    M3 c_geo at ell=26 across 5 α/β cells
    M4_set × {A, B, C}        at ell=28
    M5 c_τ × {A, B, C} (Option B for B) at ell=27

Outputs (additions to /path/to/TDiG/data/cache/population_stats/):
  sigma_ref_inv_A.npy        (4096, 4096) fp32   ~64 MB
  sigma_ref_inv_B.npy        same
  sigma_ref_inv_C.npy        same
  sigma_ref_meta.json        shrinkage parameters, sample N
  gamma_calibration_v2.json  γ_v2 thresholds for all v2 cells
  _v2_done                   marker

Wall: ~5 min (single forward over 100 sanity seqs + post-processing).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/gDTR")

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096


def ledoit_wolf_shrinkage(sigma_emp: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage estimator. Returns (sigma_shrunk, lambda).

    Uses simple analytic estimator:
        lambda = (||sigma_emp||_F^2 / d - tr(sigma_emp)^2 / d^2) / ((tr(sigma_emp)/d)^2)
    Clip lambda to [0, 1].
    """
    d = sigma_emp.shape[0]
    trace_d = np.trace(sigma_emp) / d
    sigma_norm_sq = np.sum(sigma_emp ** 2)
    # Use simple fixed-point estimate; fallback to fixed lambda if unstable
    lam_est = 1.0 - (trace_d ** 2 * d) / max(sigma_norm_sq, 1e-30)
    lam = float(np.clip(lam_est, 0.05, 0.95))
    sigma_shrunk = (1 - lam) * sigma_emp + lam * trace_d * np.eye(d, dtype=sigma_emp.dtype)
    return sigma_shrunk, lam


def estimate_sigma_ref(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Center, compute empirical Σ, shrinkage, return (sigma_inv, lambda)."""
    print(f"    centering N={samples.shape[0]} samples...", flush=True)
    mu = samples.mean(axis=0)
    centered = samples - mu[None, :]
    print(f"    computing empirical covariance...", flush=True)
    sigma_emp = (centered.T @ centered) / (samples.shape[0] - 1)
    print(f"    Ledoit-Wolf shrinkage...", flush=True)
    sigma_shrunk, lam = ledoit_wolf_shrinkage(sigma_emp.astype(np.float64))
    print(f"    inverting (lambda={lam:.4f})...", flush=True)
    sigma_inv = np.linalg.inv(sigma_shrunk).astype(np.float32)
    return sigma_inv, lam


@torch.no_grad()
def collect_ref_samples(bundle, sequences: list[tuple[str, str]]):
    """Forward sanity sequences, collect h_29 / RMSNorm(h_29) / h_norm pooled samples.

    Also computes raw distance scalars at calibration layers for γ q70.

    Returns:
        samples_dict: 'A'/'B'/'C' -> (N_tokens, H) np.float32
        gamma_data: dict of arrays for γ calibration
    """
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    L = N_LAYERS
    L_STAR_ = L_STAR
    layer_names = all_layer_names()

    samples_h29 = []
    samples_h29_rms = []
    samples_hnorm = []

    # γ calibration distance distributions
    gamma_data = {
        "D_dir_refA_at_28": [], "D_dir_refB_at_28": [], "D_dir_refC_at_28": [],
        "D_mag_refA_at_28": [],
        "geo_v_at_26": [], "geo_kappa_at_26": [],
        # M5 Option B: tau at ell=27 with reference-consistent state
        "tau_refA_at_27": [], "tau_refB_at_27": [], "tau_refC_at_27": [],
        # M4_set raw quadratic values at ell=28; sigma_inv applied AFTER pass
        "raw_h_at_28": [], "raw_h29_per_token": [], "raw_h29_rms_per_token": [], "raw_h_norm_per_token": [],
    }

    print(f"[forward] {len(sequences)} sanity seqs...", flush=True)
    t0 = time.time()
    for i, (sid, seq) in enumerate(sequences):
        input_ids = tokenize(seq, bundle, device="cuda")
        hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)

        T = hs["norm"].shape[1]
        h_blocks = torch.stack([hs[f"blocks.{ell}"][0] for ell in range(L)], dim=0).float()  # (L, T, H)
        h_norm = hs["norm"][0].float()  # (T, H)
        h_29 = h_blocks[L_STAR_]

        # Batched RMSNorm
        h_flat = h_blocks.reshape(L * T, HIDDEN_SIZE).to(bundle.embedding_weight.dtype)
        h_blocks_rms = bundle.norm(h_flat).reshape(L, T, HIDDEN_SIZE).float()
        h_29_rms = h_blocks_rms[L_STAR_]

        # Collect Σ_ref samples
        samples_h29.append(h_29.detach().cpu().numpy().astype(np.float32))
        samples_h29_rms.append(h_29_rms.detach().cpu().numpy().astype(np.float32))
        samples_hnorm.append(h_norm.detach().cpu().numpy().astype(np.float32))

        # γ data at ell=28 (penultimate)
        ell_p = L_STAR_ - 1
        h_ell = h_blocks[ell_p]
        h_ell_rms = h_blocks_rms[ell_p]

        gamma_data["D_dir_refA_at_28"].append((1 - F.cosine_similarity(h_ell, h_29, dim=-1)).cpu().numpy())
        gamma_data["D_dir_refB_at_28"].append((1 - F.cosine_similarity(h_ell_rms, h_29_rms, dim=-1)).cpu().numpy())
        gamma_data["D_dir_refC_at_28"].append((1 - F.cosine_similarity(h_ell, h_norm, dim=-1)).cpu().numpy())

        norm_ell = torch.linalg.vector_norm(h_ell, dim=-1)
        norm_29 = torch.linalg.vector_norm(h_29, dim=-1)
        gamma_data["D_mag_refA_at_28"].append(torch.abs(norm_ell / (norm_29 + 1e-12) - 1).cpu().numpy())

        # geo at ell=26
        h_26 = h_blocks[26]; h_27 = h_blocks[27]
        v_26 = (torch.linalg.vector_norm(h_27 - h_26, dim=-1) /
                (torch.linalg.vector_norm(h_26, dim=-1) + 1e-12))
        kappa_26 = 1 - F.cosine_similarity(h_27 - h_26, h_ell - h_27, dim=-1)
        gamma_data["geo_v_at_26"].append(v_26.cpu().numpy())
        gamma_data["geo_kappa_at_26"].append(kappa_26.cpu().numpy())

        # M5 tau at ell=27 (calibration layer)
        # Option B: under Ref B, both numerator (path) and denominator (h_ell - h_ref) use RMSNormed states
        # Numerator for Ref A/C: raw path from ell=27 to L*=29 = ||h_28-h_27|| + ||h_29-h_28||
        # Numerator for Ref B: RMSNormed path = ||h_28_rms - h_27_rms|| + ||h_29_rms - h_28_rms||
        num_raw_27 = (torch.linalg.vector_norm(h_blocks[28] - h_blocks[27], dim=-1) +
                       torch.linalg.vector_norm(h_blocks[29] - h_blocks[28], dim=-1))
        num_rms_27 = (torch.linalg.vector_norm(h_blocks_rms[28] - h_blocks_rms[27], dim=-1) +
                       torch.linalg.vector_norm(h_blocks_rms[29] - h_blocks_rms[28], dim=-1))
        den_A_27 = torch.linalg.vector_norm(h_blocks[27] - h_29, dim=-1) + 1e-12
        den_B_27 = torch.linalg.vector_norm(h_blocks_rms[27] - h_29_rms, dim=-1) + 1e-12
        den_C_27 = torch.linalg.vector_norm(h_blocks[27] - h_norm, dim=-1) + 1e-12
        gamma_data["tau_refA_at_27"].append((num_raw_27 / den_A_27).cpu().numpy())
        gamma_data["tau_refB_at_27"].append((num_rms_27 / den_B_27).cpu().numpy())  # Option B!
        gamma_data["tau_refC_at_27"].append((num_raw_27 / den_C_27).cpu().numpy())

        # For γ_M_set we need Σ_inv computed first; defer to after Σ estimation
        gamma_data["raw_h_at_28"].append(h_ell.cpu().numpy().astype(np.float32))
        gamma_data["raw_h29_per_token"].append(h_29.cpu().numpy().astype(np.float32))
        gamma_data["raw_h29_rms_per_token"].append(h_29_rms.cpu().numpy().astype(np.float32))
        gamma_data["raw_h_norm_per_token"].append(h_norm.cpu().numpy().astype(np.float32))

        del hs, h_blocks, h_blocks_rms, input_ids
        torch.cuda.empty_cache()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sequences) - i - 1) / rate
            print(f"  [{i+1}/{len(sequences)}] rate={rate:.2f} seq/s eta={eta:.0f}s", flush=True)

    samples_dict = {
        "A": np.concatenate(samples_h29, axis=0),
        "B": np.concatenate(samples_h29_rms, axis=0),
        "C": np.concatenate(samples_hnorm, axis=0),
    }
    print(f"[forward done] {samples_dict['A'].shape[0]} samples per variant, wall={(time.time()-t0):.1f}s", flush=True)
    return samples_dict, gamma_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-fasta", type=Path,
                        default=Path("/path/to/gDTR/data/baselines/phase1_sanity_seqs.fa"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/population_stats"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load fasta
    sequences = []
    with args.sanity_fasta.open() as f:
        cur_id, cur = None, []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur_id: sequences.append((cur_id, "".join(cur)))
                cur_id = line[1:].split()[0]; cur = []
            else:
                cur.append(line)
        if cur_id: sequences.append((cur_id, "".join(cur)))
    print(f"[setup] {len(sequences)} sanity seqs", flush=True)

    print(f"[setup] loading Evo 2", flush=True)
    from src.model_loader_evo2 import load_evo2
    bundle = load_evo2()
    print(f"[setup] model loaded", flush=True)

    samples_dict, gamma_data = collect_ref_samples(bundle, sequences)

    # Estimate Σ_ref × 3
    print(f"\n[Σ_ref] estimating 3 variants...", flush=True)
    sigma_inv = {}
    lambdas = {}
    for variant in ("A", "B", "C"):
        print(f"  Variant {variant}:")
        sig_inv, lam = estimate_sigma_ref(samples_dict[variant])
        sigma_inv[variant] = sig_inv
        lambdas[variant] = lam
        np.save(args.out_dir / f"sigma_ref_inv_{variant}.npy", sig_inv)
        print(f"    saved sigma_ref_inv_{variant}.npy")

    # γ_M_set at ell=28 — apply sigma_inv to (h_28 - h_ref) per variant
    print(f"\n[γ M4_set] computing D_M_set at ell=28...", flush=True)
    M4_dists = {"A": [], "B": [], "C": []}
    for i in range(len(gamma_data["raw_h_at_28"])):
        h28 = gamma_data["raw_h_at_28"][i]  # (T, H)
        h29 = gamma_data["raw_h29_per_token"][i]
        h29_rms = gamma_data["raw_h29_rms_per_token"][i]
        hnorm = gamma_data["raw_h_norm_per_token"][i]
        # M4_set Ref A: (h28 - h29)^T sigma_inv_A (h28 - h29)
        for variant, ref in (("A", h29), ("B", h29_rms), ("C", hnorm)):
            diff = h28 - ref  # (T, H)
            sig_inv = sigma_inv[variant]
            sig_diff = diff @ sig_inv
            quad = np.maximum((diff * sig_diff).sum(axis=-1), 0.0)
            M4_dists[variant].append(np.sqrt(quad))
    M4_dists = {k: np.concatenate(v) for k, v in M4_dists.items()}
    print(f"  D_M_set sample sizes: A={M4_dists['A'].shape}, B={M4_dists['B'].shape}, C={M4_dists['C'].shape}")

    # ─── Compile γ_v2 q70 thresholds ────────────────────────────────────────
    gamma_v2 = {}
    for key in ("D_dir_refA_at_28", "D_dir_refB_at_28", "D_dir_refC_at_28",
                "D_mag_refA_at_28",
                "tau_refA_at_27", "tau_refB_at_27", "tau_refC_at_27"):
        arr = np.concatenate(gamma_data[key])
        arr = arr[np.isfinite(arr)]
        gamma_v2[key] = {
            "q50": float(np.quantile(arr, 0.50)),
            "q70": float(np.quantile(arr, 0.70)),
            "q90": float(np.quantile(arr, 0.90)),
            "n": int(len(arr)),
        }

    # geo: 5 α/β cells at ell=26
    v_all = np.concatenate(gamma_data["geo_v_at_26"])
    k_all = np.concatenate(gamma_data["geo_kappa_at_26"])
    v_z = (v_all - v_all.mean()) / (v_all.std() + 1e-12)
    k_z = (k_all - k_all.mean()) / (k_all.std() + 1e-12)
    alpha_beta_cells = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
    for alpha, beta in alpha_beta_cells:
        g = alpha * v_z + beta * k_z
        gamma_v2[f"D_geo_a{alpha}_b{beta}_at_26"] = {
            "q50": float(np.quantile(g, 0.50)),
            "q70": float(np.quantile(g, 0.70)),
            "q90": float(np.quantile(g, 0.90)),
            "n": int(len(g)),
        }
    gamma_v2["_geo_pop_stats"] = {
        "v_mean": float(v_all.mean()), "v_std": float(v_all.std()),
        "kappa_mean": float(k_all.mean()), "kappa_std": float(k_all.std()),
    }

    # γ_M_set at ell=28
    for variant in ("A", "B", "C"):
        arr = M4_dists[variant]
        gamma_v2[f"D_Mset_ref{variant}_at_28"] = {
            "q50": float(np.quantile(arr, 0.50)),
            "q70": float(np.quantile(arr, 0.70)),
            "q90": float(np.quantile(arr, 0.90)),
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    (args.out_dir / "gamma_calibration_v2.json").write_text(json.dumps(gamma_v2, indent=2))
    print(f"\n[γ_v2] saved gamma_calibration_v2.json")

    # Σ_ref meta
    meta = {
        "script": "10b_calibrate_v2.py",
        "host": platform.node(),
        "n_sanity_seqs": len(sequences),
        "n_samples_per_variant": int(samples_dict["A"].shape[0]),
        "shrinkage_lambdas": lambdas,
        "model_variant": bundle.loaded_variant,
    }
    (args.out_dir / "sigma_ref_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out_dir / "_v2_done").write_text(json.dumps({"ok": True}, indent=2))
    print(f"[done] {args.out_dir}")

    print("\n=== γ_v2 summary (q70) ===")
    for k, v in gamma_v2.items():
        if isinstance(v, dict) and "q70" in v:
            print(f"  {k:40s} q70 = {v['q70']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
