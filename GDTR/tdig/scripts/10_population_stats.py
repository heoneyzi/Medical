"""Population stats + Sigma estimation from chr22 sanity sequences.

SERVER. Forwards 100 chr22 sanity sequences (already in
/path/to/gDTR/data/baselines/phase1_sanity_seqs.fa) through Evo 2 7B and
computes per-layer statistics that calibrate the TDiG metric family.

Outputs to /path/to/TDiG/data/cache/population_stats/:
    per_layer_mean.npy            (32, 4096) fp32
    per_layer_std.npy             (32, 4096) fp32
    sigma_diagonal.npy            (32, 4096) fp32       (= per_layer_std**2)
    sigma_inv_diag.npy            (32, 4096) fp32       (precomputed 1/sigma**2)
    h_norm_mean.npy / h_norm_std.npy
    gamma_calibration.json        14 cell q70 thresholds at penultimate layer
    _provenance.json
    _done

This iteration uses batch=1 sequential forward (matches gDTR-PoC Phase 1.4
exactly for direct comparability of gamma values).
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

# gDTR src on path
sys.path.insert(0, "/path/to/gDTR")


def load_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    cur_id, cur_seq = None, []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            records.append((cur_id, "".join(cur_seq)))
    return records


@torch.no_grad()
def compute_per_layer_stats_and_gamma(
    bundle,
    sequences: list[tuple[str, str]],
    out_dir: Path,
    log_every: int = 10,
):
    """Single pass over sequences, batch=1.

    Accumulates:
        layer_sum  (32, 4096): per-layer per-feature sum  -> mean
        layer_ssq  (32, 4096): per-layer per-feature sum of squares -> var
        norm_sum, norm_ssq for h_norm

    Collects (per-token scalars at ell = L_STAR - 1 = 28 for gamma q70):
        D_dir_refA/B/C, D_mag_refA/C
        D_tau_refA/B/C (tortuosity at ell=28)
        v_at_ell26 + kappa_at_ell26 (for g = z(v)+z(kappa) at ell=26)
    """
    from src.constants_evo2 import N_LAYERS, HIDDEN_SIZE
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    layer_names = all_layer_names()  # ['blocks.0', ..., 'blocks.31', 'norm']
    L, H = N_LAYERS, HIDDEN_SIZE
    L_STAR = 29  # canonical interpretively-distinct tap

    # Welford / Chan's parallel batch-update accumulators (fp64 for stability).
    # State per layer: count (shared), mean (L, H), M2 (L, H).
    # For h_norm: separate count would equal layer-count; reuse.
    layer_mean = np.zeros((L, H), dtype=np.float64)
    layer_M2 = np.zeros((L, H), dtype=np.float64)
    norm_mean = np.zeros(H, dtype=np.float64)
    norm_M2 = np.zeros(H, dtype=np.float64)
    n_tokens = 0  # shared across all layers + h_norm

    # Distance distributions for gamma q70 calibration
    coll: dict[str, list[np.ndarray]] = {
        "D_dir_refA": [], "D_dir_refB": [], "D_dir_refC": [],
        "D_mag_refA": [], "D_mag_refC": [],
        "D_tau_refA": [], "D_tau_refB": [], "D_tau_refC": [],
        "geo_v": [], "geo_kappa": [],
    }

    print(f"[stats] L={L}, H={H}, L*={L_STAR}, n_sequences={len(sequences)}")
    t_total = time.time()

    for i, (seq_id, seq) in enumerate(sequences):
        try:
            input_ids = tokenize(seq, bundle, device="cuda")
            hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)

            T = hs["norm"].shape[1]

            # ─── Chan's parallel Welford update per layer ──────────────────────
            # Compute batch_mean and batch_M2 on GPU (light transfer of [H] only).
            new_count = n_tokens + T
            for ell in range(L):
                h_l = hs[f"blocks.{ell}"][0].float()  # [T, H] on GPU
                batch_mean = h_l.mean(dim=0).cpu().numpy().astype(np.float64)  # (H,)
                # batch_M2 = sum_t (h_l[t] - batch_mean)^2 — computed on GPU
                dev = h_l - h_l.mean(dim=0, keepdim=True)
                batch_M2 = (dev ** 2).sum(dim=0).cpu().numpy().astype(np.float64)
                delta = batch_mean - layer_mean[ell]
                layer_mean[ell] = layer_mean[ell] + delta * (T / new_count)
                layer_M2[ell] = (layer_M2[ell] + batch_M2 +
                                 (delta ** 2) * (n_tokens * T / new_count))

            # Same for h_norm
            h_norm = hs["norm"][0].float()
            batch_mean_n = h_norm.mean(dim=0).cpu().numpy().astype(np.float64)
            dev_n = h_norm - h_norm.mean(dim=0, keepdim=True)
            batch_M2_n = (dev_n ** 2).sum(dim=0).cpu().numpy().astype(np.float64)
            delta_n = batch_mean_n - norm_mean
            norm_mean = norm_mean + delta_n * (T / new_count)
            norm_M2 = norm_M2 + batch_M2_n + (delta_n ** 2) * (n_tokens * T / new_count)

            n_tokens = new_count

            # Reference vectors (L_STAR = 29 is the last interpretively-distinct block)
            h_29 = hs[f"blocks.{L_STAR}"][0].float()  # [T, H] raw
            # RMSNorm(h_29) and RMSNorm(h_ell) for Ref B
            h_29_rms = bundle.norm(h_29.to(bundle.embedding_weight.dtype)).float()

            # Distance scalars at penultimate layer (ell_p = 28 = L_STAR - 1)
            ell_p = L_STAR - 1  # 28
            h_ell = hs[f"blocks.{ell_p}"][0].float()
            h_ell_rms = bundle.norm(h_ell.to(bundle.embedding_weight.dtype)).float()

            # Direction 1 - cos
            D_dirA = (1 - F.cosine_similarity(h_ell, h_29, dim=-1)).cpu().numpy()
            D_dirB = (1 - F.cosine_similarity(h_ell_rms, h_29_rms, dim=-1)).cpu().numpy()
            D_dirC = (1 - F.cosine_similarity(h_ell, h_norm, dim=-1)).cpu().numpy()

            # Magnitude |r - 1|
            norm_ell = torch.linalg.vector_norm(h_ell, dim=-1)  # [T]
            norm_29 = torch.linalg.vector_norm(h_29, dim=-1)
            norm_n = torch.linalg.vector_norm(h_norm, dim=-1)
            D_magA = torch.abs(norm_ell / (norm_29 + 1e-12) - 1).cpu().numpy()
            D_magC = torch.abs(norm_ell / (norm_n + 1e-12) - 1).cpu().numpy()

            # Tortuosity tau — calibrated at ell = 27 (NOT 28, which is trivially 1).
            # τ(27) = (||h_28 - h_27|| + ||h_29 - h_28||) / ||h_27 - h_ref||
            h_27 = hs["blocks.27"][0].float()
            h_27_rms = bundle.norm(h_27.to(bundle.embedding_weight.dtype)).float()
            num_27 = (torch.linalg.vector_norm(h_ell - h_27, dim=-1) +
                      torch.linalg.vector_norm(h_29 - h_ell, dim=-1))  # ||h28-h27|| + ||h29-h28||
            den_A_27 = torch.linalg.vector_norm(h_27 - h_29, dim=-1) + 1e-12
            den_B_27 = torch.linalg.vector_norm(h_27_rms - h_29_rms, dim=-1) + 1e-12
            den_C_27 = torch.linalg.vector_norm(h_27 - h_norm, dim=-1) + 1e-12
            D_tauA = (num_27 / den_A_27).cpu().numpy()
            D_tauB = (num_27 / den_B_27).cpu().numpy()
            D_tauC = (num_27 / den_C_27).cpu().numpy()

            # Geo at ell=26: velocity v_26 = ||h_27 - h_26|| / ||h_26||
            # curvature kappa_26 = 1 - cos(h_27-h_26, h_28-h_27).  h_27 already loaded above.
            h_26 = hs["blocks.26"][0].float()
            h_28 = h_ell  # h_28 same as h_ell since ell_p = 28
            v_26 = (torch.linalg.vector_norm(h_27 - h_26, dim=-1) /
                    (torch.linalg.vector_norm(h_26, dim=-1) + 1e-12))
            d1 = h_27 - h_26
            d2 = h_28 - h_27
            kappa_26 = (1 - F.cosine_similarity(d1, d2, dim=-1))

            coll["D_dir_refA"].append(D_dirA)
            coll["D_dir_refB"].append(D_dirB)
            coll["D_dir_refC"].append(D_dirC)
            coll["D_mag_refA"].append(D_magA)
            coll["D_mag_refC"].append(D_magC)
            coll["D_tau_refA"].append(D_tauA)
            coll["D_tau_refB"].append(D_tauB)
            coll["D_tau_refC"].append(D_tauC)
            coll["geo_v"].append(v_26.cpu().numpy())
            coll["geo_kappa"].append(kappa_26.cpu().numpy())

            del hs, input_ids, h_29, h_ell, h_norm, h_26, h_27, h_28
            del h_29_rms, h_ell_rms, h_27_rms, d1, d2, dev, dev_n
            torch.cuda.empty_cache()

            if (i + 1) % log_every == 0:
                elapsed = time.time() - t_total
                rate = (i + 1) / elapsed
                eta_min = (len(sequences) - i - 1) / rate / 60
                gpu_mb = torch.cuda.max_memory_allocated() / 1e6
                print(f"  [{i+1:3d}/{len(sequences)}] rate={rate:.2f} seq/s eta={eta_min:.1f}min gpu_peak={gpu_mb:.0f}MB", flush=True)

        except Exception as e:
            print(f"  [{i}] ERROR on seq_id={seq_id}: {type(e).__name__}: {e}", flush=True)
            raise

    # ---- Finalize (Welford / Chan's: variance = M2 / count) ----
    layer_var_f64 = layer_M2 / n_tokens
    layer_var = layer_var_f64.astype(np.float32)
    layer_std = np.sqrt(np.maximum(layer_var_f64, 0.0)).astype(np.float32)
    layer_mean_f32 = layer_mean.astype(np.float32)

    norm_var_f64 = norm_M2 / n_tokens
    norm_var = norm_var_f64.astype(np.float32)
    norm_std = np.sqrt(np.maximum(norm_var_f64, 0.0)).astype(np.float32)
    norm_mean_f32 = norm_mean.astype(np.float32)

    sigma_diagonal = layer_var
    sigma_inv_diag = (1.0 / (layer_var_f64 + 1e-12)).astype(np.float32)

    print(f"[stats] forward done in {(time.time()-t_total)/60:.2f} min. n_tokens={n_tokens}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "per_layer_mean.npy", layer_mean_f32)
    np.save(out_dir / "per_layer_std.npy", layer_std)
    np.save(out_dir / "sigma_diagonal.npy", sigma_diagonal)
    np.save(out_dir / "sigma_inv_diag.npy", sigma_inv_diag)
    np.save(out_dir / "h_norm_mean.npy", norm_mean_f32)
    np.save(out_dir / "h_norm_std.npy", norm_std)
    print(f"[stats] saved per-layer mean/std/sigma to {out_dir}")

    # ---- gamma q70 calibration ----
    gamma = {}
    for key in ["D_dir_refA", "D_dir_refB", "D_dir_refC",
                "D_mag_refA", "D_mag_refC",
                "D_tau_refA", "D_tau_refB", "D_tau_refC"]:
        arr = np.concatenate(coll[key])
        # Filter nans/inf for tau (divide-by-tiny edge case)
        arr = arr[np.isfinite(arr)]
        gamma[key] = float(np.quantile(arr, 0.70))
        print(f"  gamma[{key}] = {gamma[key]:.6f}  n={len(arr)}")

    # geo: z(v)+z(kappa), q70
    v_all = np.concatenate(coll["geo_v"])
    k_all = np.concatenate(coll["geo_kappa"])
    v_mean, v_std = float(v_all.mean()), float(v_all.std())
    k_mean, k_std = float(k_all.mean()), float(k_all.std())
    g = (v_all - v_mean) / (v_std + 1e-12) + (k_all - k_mean) / (k_std + 1e-12)
    gamma["D_geo_at_ell26"] = float(np.quantile(g, 0.70))
    gamma["_geo_v_mean"] = v_mean
    gamma["_geo_v_std"] = v_std
    gamma["_geo_kappa_mean"] = k_mean
    gamma["_geo_kappa_std"] = k_std
    print(f"  gamma[D_geo_at_ell26] = {gamma['D_geo_at_ell26']:.6f}")

    gamma["_note"] = "Mahalanobis q70 requires second pass with locked sigma (deferred)."
    (out_dir / "gamma_calibration.json").write_text(json.dumps(gamma, indent=2))
    print(f"[stats] gamma_calibration.json written")

    return n_tokens, gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity-fasta", type=Path,
        default=Path("/path/to/gDTR/data/baselines/phase1_sanity_seqs.fa"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/path/to/TDiG/data/cache/population_stats"),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 = use all")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading Evo 2 ...", flush=True)
    t0 = time.time()
    from src.model_loader_evo2 import load_evo2
    bundle = load_evo2()
    print(f"[setup] model loaded in {time.time()-t0:.1f}s, variant={bundle.loaded_variant}", flush=True)

    print(f"[setup] reading sanity FASTA: {args.sanity_fasta}")
    seqs = load_fasta(args.sanity_fasta)
    if args.max_sequences > 0:
        seqs = seqs[: args.max_sequences]
    print(f"[setup] {len(seqs)} sequences, first 3 lengths: {[len(s) for _, s in seqs[:3]]}")
    if not seqs:
        raise RuntimeError("no sequences loaded")

    n_tokens, gamma = compute_per_layer_stats_and_gamma(
        bundle, seqs, args.out_dir, args.log_every,
    )

    provenance = {
        "script": "10_population_stats.py",
        "host": platform.node(),
        "n_sequences": len(seqs),
        "n_tokens_total": int(n_tokens),
        "sanity_fasta": str(args.sanity_fasta),
        "out_dir": str(args.out_dir),
        "model_variant": bundle.loaded_variant,
        "torch_version": torch.__version__,
        "gamma_summary": {k: v for k, v in gamma.items() if not k.startswith("_")},
    }
    (args.out_dir / "_provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.out_dir / "_done").write_text(json.dumps({"ok": True}, indent=2))
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
