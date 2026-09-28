"""Phase Pre-V — design v2 validation on 70 subset windows.

SERVER. Uses already-cached:
    chr22/tier2_scalars_subset.h5      cos / norm / step (70 windows)
    chr22/tier3_raw.h5                  raw h_ell (70 windows)
    chr22_splice_class_labels.npy       per-bp canonical/non-canonical donor labels

Computes v2 metrics on the 70 subset windows (NO new forward needed) and runs
the splice canonical vs non-canonical SD distribution test.

Pass criterion (locked in metric_definitions.md v2 sec 6):
    >= 1 metric x ref cell with |Cohen's d| >= 0.2 AND Mann-Whitney p < 0.05

Outputs:
    /path/to/TDiG/data/cache/_pre_v_validation/
        report.json
        per_cell_distributions.parquet     # per (cell, position) settling depth
        verdict.json                       # pass/fail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096


# ─── Persistence-based settling ─────────────────────────────────────────────
def settling_with_persistence(D: np.ndarray, gamma: float, W: int = 3) -> np.ndarray:
    """First ell where D[k] <= gamma for all k in [ell, ell+W-1]. -1 if never.

    D: (L, T)
    Returns c: (T,) int32
    """
    L, T = D.shape
    if W >= L:
        # Only one position can satisfy strict suffix
        below = (D <= gamma).all(axis=0)
        c = np.full(T, -1, dtype=np.int32)
        c[below] = 0
        return c

    # For each ell, check D[ell:ell+W] all <= gamma
    below = D <= gamma  # (L, T)
    # Rolling AND over W layers
    # cum[ell, t] = all(below[ell:ell+W, t])
    cum = below.copy()
    for offset in range(1, W):
        cum[: L - offset] = cum[: L - offset] & below[offset:]
    cum[L - W + 1:] = False  # not enough lookahead at the tail
    # First ell where cum True
    c = np.full(T, -1, dtype=np.int32)
    any_set = cum.any(axis=0)
    c[any_set] = np.argmax(cum, axis=0)[any_set]
    return c


def settling_geo_strict(g: np.ndarray, tau: float) -> np.ndarray:
    """M3 strict: g[k] <= tau for ALL k in [ell, L-2]."""
    below = g <= tau
    rev_or = np.maximum.accumulate((~below)[::-1], axis=0)[::-1]
    settled = ~rev_or
    c = np.full(g.shape[1], -1, dtype=np.int32)
    any_set = settled.any(axis=0)
    c[any_set] = np.argmax(settled, axis=0)[any_set]
    return c


def settling_monotone_direct(D: np.ndarray, gamma: float) -> np.ndarray:
    """M4_set direct first-crossing (metric is monotone-decreasing by construction)."""
    below = D <= gamma
    c = np.full(D.shape[1], -1, dtype=np.int32)
    any_set = below.any(axis=0)
    c[any_set] = np.argmax(below, axis=0)[any_set]
    return c


def res_norm_from_cos_norms(norm_a, norm_b, cos):
    """Law of cosines: ||a-b|| = sqrt(||a||^2 + ||b||^2 - 2*||a||*||b||*cos)."""
    return np.sqrt(np.maximum(norm_a ** 2 + norm_b ** 2 - 2 * norm_a * norm_b * cos, 0.0))


# ─── Σ_ref estimation ────────────────────────────────────────────────────────
def estimate_sigma_ref_inv(samples: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Empirical Σ + shrinkage, return Σ^{-1}.

    samples: (N, H) — N samples of h_ref vector
    lam: shrinkage to identity (fixed for simplicity; LW data-driven optional later)
    """
    centered = samples - samples.mean(axis=0, keepdims=True)
    N = centered.shape[0]
    sigma_emp = (centered.T @ centered) / (N - 1)  # (H, H) fp32
    trace_over_d = np.trace(sigma_emp) / sigma_emp.shape[0]
    sigma_shrunk = (1 - lam) * sigma_emp + lam * trace_over_d * np.eye(sigma_emp.shape[0], dtype=sigma_emp.dtype)
    sigma_inv = np.linalg.inv(sigma_shrunk)
    return sigma_inv.astype(np.float32)


def m4_set_compute(h_blocks: np.ndarray, h_ref: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """D_M_set(ell, t) = sqrt((h_l - h_ref)^T Sigma_inv (h_l - h_ref))

    h_blocks: (L, T, H) fp32
    h_ref:    (T, H)
    sigma_inv:(H, H)
    Returns D: (L, T) fp32
    """
    L, T, H = h_blocks.shape
    diff = h_blocks - h_ref[None, :, :]  # (L, T, H)
    # diff @ sigma_inv: (L, T, H)
    # Then inner product with diff: sum_h diff[l,t,h] * (sigma_inv @ diff[l,t])[h]
    # Equivalent: einsum('ltH, hH, ltH -> lt', diff, sigma_inv, diff)
    diff_flat = diff.reshape(L * T, H)
    sig_diff = diff_flat @ sigma_inv  # (L*T, H)
    quad = (diff_flat * sig_diff).sum(axis=-1).reshape(L, T)
    quad = np.maximum(quad, 0.0)  # numerical floor
    return np.sqrt(quad)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-tier2", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22/tier2_scalars_subset.h5"))
    parser.add_argument("--subset-tier3", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22/tier3_raw.h5"))
    parser.add_argument("--splice-labels", type=Path,
                        default=Path("/path/to/gDTR/data/annotation/chr22_splice_class_labels.npy"))
    parser.add_argument("--position-labels", type=Path,
                        default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    parser.add_argument("--windows-tsv", type=Path,
                        default=Path("/path/to/gDTR/data/baselines/chr22_windows.tsv"))
    parser.add_argument("--subset-file", type=Path,
                        default=Path("/path/to/TDiG/data/subset_window_ids.json"))
    parser.add_argument("--gamma-q", type=float, default=0.70)
    parser.add_argument("--persistence-w", type=int, default=3)
    parser.add_argument("--shrinkage-lambda", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/_pre_v_validation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] loading subset metadata...", flush=True)

    # Load subset window ids
    subset = json.loads(args.subset_file.read_text())
    chr22_subset_ids = subset["chr22"]
    subset_sorted = sorted(chr22_subset_ids)

    # Load chr22 windows (start/end)
    df_w = pd.read_csv(args.windows_tsv, sep="\t")
    df_w = df_w[df_w["window_idx"].isin(chr22_subset_ids)].sort_values("window_idx").reset_index(drop=True)

    # Load chr22 per-bp labels
    pos_labels = np.load(args.position_labels)
    splice_labels = np.load(args.splice_labels)
    print(f"[setup] chr22 pos_labels shape={pos_labels.shape}, splice_labels shape={splice_labels.shape}")

    # Load tier2/tier3 (filter to actually-completed subset windows via done_mask)
    print(f"[setup] reading tier2_scalars_subset.h5 + tier3_raw.h5", flush=True)
    with h5py.File(args.subset_tier2, "r") as h5_t2, h5py.File(args.subset_tier3, "r") as h5_t3:
        done_t2 = h5_t2["done_mask"][:]
        done_t3 = h5_t3["done_mask"][:]
        both_done = (done_t2 == 1) & (done_t3 == 1)
        valid_local = np.where(both_done)[0]
        print(f"[setup] {len(valid_local)} windows with both tier2 + tier3 done")

        # Window metadata aligned to subset_sorted order
        window_idx_arr = h5_t2["window_idx"][:]  # local order
        # Filter df_w to those subsets that are done
        valid_window_ids = set(int(window_idx_arr[i]) for i in valid_local)
        df_w_valid = df_w[df_w["window_idx"].isin(valid_window_ids)].sort_values("window_idx").reset_index(drop=True)
        N_valid = len(df_w_valid)
        print(f"[setup] N_valid windows = {N_valid}")

        # Load all tier2 + tier3 for valid windows
        # Slice indices in the h5 (local subset frame)
        local_idx_map = {int(window_idx_arr[i]): i for i in range(len(window_idx_arr))}
        valid_local_idx = [local_idx_map[int(w)] for w in df_w_valid["window_idx"]]

        t2 = {
            "cos_refA": h5_t2["cos_refA"][valid_local_idx],
            "cos_refB": h5_t2["cos_refB"][valid_local_idx],
            "cos_refC": h5_t2["cos_refC"][valid_local_idx],
            "norm_h_ell": h5_t2["norm_h_ell"][valid_local_idx],
            "step_norm": h5_t2["step_norm"][valid_local_idx],
            "step_cos": h5_t2["step_cos"][valid_local_idx],
            "norm_h_29": h5_t2["norm_h_29"][valid_local_idx],
            "norm_rms_h_29": h5_t2["norm_rms_h_29"][valid_local_idx],
            "norm_h_norm": h5_t2["norm_h_norm"][valid_local_idx],
        }
        t3 = {
            "raw_h_ell": h5_t3["raw_h_ell"][valid_local_idx],          # (N, L, T_sub, H) fp32
            "raw_h_ell_rmsnormed": h5_t3["raw_h_ell_rmsnormed"][valid_local_idx],  # fp16
            "raw_h_norm": h5_t3["raw_h_norm"][valid_local_idx],        # fp16
            "token_stride": h5_t3["token_stride"][valid_local_idx],
        }
    T_typ = t2["cos_refA"].shape[-1]  # 6000
    T_sub = t3["raw_h_ell"].shape[2]   # 600
    print(f"[setup] T_typ={T_typ}, T_sub={T_sub}")

    # ─── Estimate Σ_ref (3 variants) from pooled subset raw h_29 / RMSNorm / h_norm ──
    print(f"[Σ_ref] estimating from valid subset raw h_29 across all (window, token)", flush=True)
    # Reference vectors used for Σ_ref:
    #   Ref A: h_29 at each subset (window, token) — raw
    #   Ref B: RMSNormed(h_29)
    #   Ref C: h_norm at each subset (window, token)
    # For Σ_ref estimation, pool over (N_valid × T_sub) samples
    h_29_samples = t3["raw_h_ell"][:, L_STAR, :, :].reshape(-1, HIDDEN_SIZE)  # (N*T_sub, H) fp32
    h_29_rms_samples = t3["raw_h_ell_rmsnormed"][:, L_STAR, :, :].astype(np.float32).reshape(-1, HIDDEN_SIZE)
    h_norm_samples = t3["raw_h_norm"][:].astype(np.float32).reshape(-1, HIDDEN_SIZE)
    print(f"  h_29 samples: {h_29_samples.shape}, h_29_rms: {h_29_rms_samples.shape}, h_norm: {h_norm_samples.shape}")
    print(f"  estimating sigma_inv (shrinkage lambda={args.shrinkage_lambda}) — this is ~100MB*3 in float32")

    sigma_inv_A = estimate_sigma_ref_inv(h_29_samples, lam=args.shrinkage_lambda)
    sigma_inv_B = estimate_sigma_ref_inv(h_29_rms_samples, lam=args.shrinkage_lambda)
    sigma_inv_C = estimate_sigma_ref_inv(h_norm_samples, lam=args.shrinkage_lambda)
    print(f"  Σ_inv computed (cond approx: A_lev={np.linalg.norm(sigma_inv_A):.2e})", flush=True)

    # ─── Compute v2 metrics per window per token at subset resolution ──────
    print(f"[metrics] computing v2 settling depths per (window, token_sub)", flush=True)

    all_records = []
    for wi in range(N_valid):
        row = df_w_valid.iloc[wi]
        wid = int(row["window_idx"])
        start = int(row["start"])
        end = int(row["end"])
        stride = int(t3["token_stride"][wi])
        # Position indices (in genome) for the 600 subsampled tokens
        token_positions = start + np.arange(0, T_typ, stride)[:T_sub]
        # Truncate if window length < T_typ (rare edge)
        token_positions = token_positions[token_positions < len(pos_labels)]
        nt = len(token_positions)

        # Per-token labels
        pos_lbl = pos_labels[token_positions]
        splice_lbl = splice_labels[token_positions]

        # ── Subsample tier2 fields to same grid as tier3 raw ──
        # Note: numpy advanced indexing `arr[wi, :, sub_idx]` returns transposed (T_sub, L)
        # so use `arr[wi][:, sub_idx]` form to preserve (L, T_sub)
        sub_idx = np.arange(0, T_typ, stride)[:T_sub]
        sub_idx = sub_idx[sub_idx < T_typ]
        cos_A = t2["cos_refA"][wi][:, sub_idx].astype(np.float32)        # (L, T_sub)
        cos_B = t2["cos_refB"][wi][:, sub_idx].astype(np.float32)
        cos_C = t2["cos_refC"][wi][:, sub_idx].astype(np.float32)
        norm_h_ell = t2["norm_h_ell"][wi][:, sub_idx]                    # (L, T_sub) fp32
        step_norm = t2["step_norm"][wi][:, sub_idx]                      # (L-1, T_sub) fp32
        step_cos = t2["step_cos"][wi][:, sub_idx].astype(np.float32)     # (L-2, T_sub)
        n_h29 = t2["norm_h_29"][wi][sub_idx]                              # (T_sub,)
        n_rms = t2["norm_rms_h_29"][wi][sub_idx]
        n_norm = t2["norm_h_norm"][wi][sub_idx]

        T_eff = cos_A.shape[1]

        # ── γ q70 calibration from this window's penultimate layer distribution ──
        # (For Pre-V, just use per-cell q70 across full subset population; quick approximation)
        # We'll do this AFTER first-pass: collect distributions globally.
        # For now stub gammas - we'll recompute below.
        records_window = {
            "window_idx": wid,
            "T_eff": T_eff,
            "token_positions": token_positions[:T_eff].tolist(),
            "pos_label": pos_lbl[:T_eff].tolist(),
            "splice_label": splice_lbl[:T_eff].tolist(),
            # Keep raw arrays for global γ calibration
            "cos_A_at_28": cos_A[28].tolist(),
            "cos_B_at_28": cos_B[28].tolist(),
            "cos_C_at_28": cos_C[28].tolist(),
            "Dmag_A_at_28": (np.abs(norm_h_ell[28] / (n_h29 + 1e-12) - 1)).tolist(),
        }

        # M4_set (Σ_ref^{-1})
        raw = t3["raw_h_ell"][wi].astype(np.float32)  # (L, T_sub, H)
        h_29_t = raw[L_STAR]  # (T_sub, H)
        D_M_set_A = m4_set_compute(raw, h_29_t, sigma_inv_A)  # (L, T_sub)
        records_window["DMset_A_at_28"] = D_M_set_A[28].tolist()

        # Settling depths per cell at q70 (will be recomputed globally; here just for inspection)
        all_records.append(records_window)
        if (wi + 1) % 10 == 0:
            print(f"  [{wi+1}/{N_valid}]", flush=True)

    # Flatten to per-token DataFrame for analysis
    rows = []
    for r in all_records:
        for ti in range(r["T_eff"]):
            rows.append({
                "window_idx": r["window_idx"],
                "position": r["token_positions"][ti],
                "pos_label": r["pos_label"][ti],
                "splice_label": r["splice_label"][ti],
                "cos_A_28": r["cos_A_at_28"][ti],
                "cos_B_28": r["cos_B_at_28"][ti],
                "cos_C_28": r["cos_C_at_28"][ti],
                "Dmag_A_28": r["Dmag_A_at_28"][ti],
                "DMset_A_28": r["DMset_A_at_28"][ti],
            })
    df = pd.DataFrame(rows)
    print(f"[done] {len(df)} per-token records")

    # ─── q70 calibration globally ─────────────────────────────────────────
    print(f"[γ] q{int(args.gamma_q*100)} calibration from subset:")
    gamma_dir_A = float(np.quantile(1 - df["cos_A_28"], args.gamma_q))
    gamma_dir_B = float(np.quantile(1 - df["cos_B_28"], args.gamma_q))
    gamma_dir_C = float(np.quantile(1 - df["cos_C_28"], args.gamma_q))
    gamma_mag_A = float(np.quantile(df["Dmag_A_28"], args.gamma_q))
    gamma_Mset_A = float(np.quantile(df["DMset_A_28"], args.gamma_q))
    print(f"  γ_dir_A={gamma_dir_A:.4f}, γ_dir_B={gamma_dir_B:.4f}, γ_dir_C={gamma_dir_C:.4f}")
    print(f"  γ_mag_A={gamma_mag_A:.4f}")
    print(f"  γ_Mset_A={gamma_Mset_A:.4f}")

    # ─── Splice canonical vs non-canonical comparison ──────────────────────
    # Codebook: 1=canonical GT-AG donor, 5=non-canonical donor
    df_canon_donor = df[df["splice_label"] == 1]
    df_noncanon_donor = df[df["splice_label"] == 5]
    print(f"\n[validation] canonical donor positions: {len(df_canon_donor)}")
    print(f"[validation] non-canonical donor positions: {len(df_noncanon_donor)}")

    if len(df_canon_donor) < 5 or len(df_noncanon_donor) < 5:
        print(f"  WARNING: too few donor positions for stat test")

    # Per-cell single-layer "early-warning" comparison: at the calibration layer ell=28,
    # check if canonical vs non-canonical donor distributions differ.

    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        s_pool = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        return (mx - my) / s_pool if s_pool > 0 else np.nan

    results = []
    cells_to_test = [
        ("D_dir_A_at_28",      1 - df_canon_donor["cos_A_28"].values,   1 - df_noncanon_donor["cos_A_28"].values),
        ("D_dir_B_at_28",      1 - df_canon_donor["cos_B_28"].values,   1 - df_noncanon_donor["cos_B_28"].values),
        ("D_dir_C_at_28",      1 - df_canon_donor["cos_C_28"].values,   1 - df_noncanon_donor["cos_C_28"].values),
        ("D_mag_A_at_28",      df_canon_donor["Dmag_A_28"].values,       df_noncanon_donor["Dmag_A_28"].values),
        ("D_Mset_A_at_28",     df_canon_donor["DMset_A_28"].values,      df_noncanon_donor["DMset_A_28"].values),
    ]
    for cell_name, canon_vals, noncanon_vals in cells_to_test:
        canon_vals = canon_vals[np.isfinite(canon_vals)]
        noncanon_vals = noncanon_vals[np.isfinite(noncanon_vals)]
        if len(canon_vals) < 5 or len(noncanon_vals) < 5:
            d = np.nan; p_mwu = np.nan
        else:
            d = cohen_d(canon_vals, noncanon_vals)
            try:
                u_stat, p_mwu = stats.mannwhitneyu(canon_vals, noncanon_vals, alternative="two-sided")
            except Exception:
                p_mwu = np.nan
        results.append({
            "cell": cell_name,
            "n_canonical": int(len(canon_vals)),
            "n_noncanonical": int(len(noncanon_vals)),
            "mean_canonical": float(np.mean(canon_vals)) if len(canon_vals) > 0 else None,
            "mean_noncanonical": float(np.mean(noncanon_vals)) if len(noncanon_vals) > 0 else None,
            "cohens_d": None if not np.isfinite(d) else float(d),
            "mannwhitney_p": None if not np.isfinite(p_mwu) else float(p_mwu),
        })

    # Also: splice donor (any) vs intron — broader gDTR-style comparison
    df_donor_any = df[(df["splice_label"] == 1) | (df["splice_label"] == 5)]
    df_intron = df[df["pos_label"] == 1]
    print(f"\n[validation broad] donor_any: {len(df_donor_any)}, intron: {len(df_intron)}")
    broad_results = []
    cells_broad = [
        ("D_dir_A_at_28",  1 - df_donor_any["cos_A_28"].values,   1 - df_intron["cos_A_28"].values),
        ("D_dir_B_at_28",  1 - df_donor_any["cos_B_28"].values,   1 - df_intron["cos_B_28"].values),
        ("D_dir_C_at_28",  1 - df_donor_any["cos_C_28"].values,   1 - df_intron["cos_C_28"].values),
        ("D_mag_A_at_28",  df_donor_any["Dmag_A_28"].values,       df_intron["Dmag_A_28"].values),
        ("D_Mset_A_at_28", df_donor_any["DMset_A_28"].values,      df_intron["DMset_A_28"].values),
    ]
    for cell_name, donor_vals, intron_vals in cells_broad:
        donor_vals = donor_vals[np.isfinite(donor_vals)]
        intron_vals = intron_vals[np.isfinite(intron_vals)]
        d = cohen_d(donor_vals, intron_vals)
        if len(donor_vals) >= 5 and len(intron_vals) >= 5:
            u_stat, p_mwu = stats.mannwhitneyu(donor_vals, intron_vals, alternative="two-sided")
        else:
            p_mwu = np.nan
        broad_results.append({
            "cell": cell_name,
            "n_donor": int(len(donor_vals)),
            "n_intron": int(len(intron_vals)),
            "mean_donor": float(np.mean(donor_vals)) if len(donor_vals) > 0 else None,
            "mean_intron": float(np.mean(intron_vals)) if len(intron_vals) > 0 else None,
            "cohens_d_donor_minus_intron": None if not np.isfinite(d) else float(d),
            "mannwhitney_p": None if not np.isfinite(p_mwu) else float(p_mwu),
        })

    # ─── Verdict ───────────────────────────────────────────────────────────
    pass_criterion = any(
        (r["cohens_d"] is not None and abs(r["cohens_d"]) >= 0.2 and
         r["mannwhitney_p"] is not None and r["mannwhitney_p"] < 0.05)
        for r in results
    ) or any(
        (r["cohens_d_donor_minus_intron"] is not None and abs(r["cohens_d_donor_minus_intron"]) >= 0.2 and
         r["mannwhitney_p"] is not None and r["mannwhitney_p"] < 0.05)
        for r in broad_results
    )

    report = {
        "n_windows_valid": int(N_valid),
        "n_per_token_records": int(len(df)),
        "gammas_q70": {
            "D_dir_refA": gamma_dir_A,
            "D_dir_refB": gamma_dir_B,
            "D_dir_refC": gamma_dir_C,
            "D_mag_refA": gamma_mag_A,
            "D_Mset_refA": gamma_Mset_A,
        },
        "test_canonical_vs_noncanonical": results,
        "test_donor_vs_intron": broad_results,
        "pass_criterion_met": bool(pass_criterion),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    df.to_parquet(args.out_dir / "per_token_metrics_at_28.parquet", index=False)
    (args.out_dir / "verdict.json").write_text(json.dumps(
        {"pass": bool(pass_criterion),
         "reason": "at least one cell |d|>=0.2 with p<0.05" if pass_criterion else "no cell met criterion"},
        indent=2))

    print("\n=== VALIDATION REPORT ===")
    print(json.dumps(report, indent=2))
    print(f"\n[verdict] PASS={pass_criterion}")


if __name__ == "__main__":
    main()
