"""Phase 1.2 — γ q50/q70/q90 ablation on chr22 v2.

Reuses tier2 scalars (no forward rerun) to recompute 17 settling cells under
γ ∈ {q50, q70, q90} from gamma_calibration_v2.json. Then computes
splice-donor-vs-intron Cohen d for each (cell, γ).

The headline question: does the splice-vs-intron ordering survive γ choice?
If yes → metric is robust to calibration. If no → flag specific cells as
calibration-sensitive in the paper.

Outputs under --out-dir (default: results/gamma_ablation/):
  cell_d_under_gamma.csv      (cell, gamma, donor_mean, intron_mean, cohen_d, donor_n, intron_n)
  cell_d_under_gamma.png      grouped bar / heatmap
  range_under_gamma.csv       (cell, gamma, range_min, range_max, never_settled_pct)
  summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}


# Settling rule helpers — EXACT copy from 15_chr22_forward.py (boundary clip)
def settling_persistence(D, gamma, W=3, max_layer=None):
    """First ell <= max_layer where D[k] <= gamma for all k in [ell, min(ell+W-1, max_layer)].
       Lookahead CLIPS at max_layer (token at reference is "settled" by definition).
       This is the SAME logic as 15_chr22_forward.py — needed to reproduce range [28, 29] for M1."""
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


def settling_monotone_direct(D, gamma, max_layer=None):
    """For monotone-decreasing D: first ell s.t. D[ell] <= gamma. (Matches forward script)"""
    below = D <= gamma
    c = np.full(D.shape[1], -1, dtype=np.int32)
    any_set = below.any(axis=0)
    c[any_set] = np.argmax(below, axis=0)[any_set]
    return c


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def build_context_map(meta_df, wids, pos_labels, n_tokens=6000):
    ctx = np.zeros((len(wids), n_tokens), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(n_tokens), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier2", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--gamma", type=Path,
                   default=Path("/path/to/TDiG/data/cache/population_stats/gamma_calibration_v2.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/gamma_ablation"))
    p.add_argument("--persistence-w", type=int, default=3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print("[load] gamma calibration + meta + tier2 scalars ...")
    gamma_v2 = json.loads(args.gamma.read_text())
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)

    print("[load] tier2 scalars ...")
    with h5py.File(args.tier2, "r") as h5:
        wids = h5["window_idx"][:]
        cos_A = h5["cos_refA"][:].astype(np.float32)
        cos_B = h5["cos_refB"][:].astype(np.float32)
        cos_C = h5["cos_refC"][:].astype(np.float32)
        norm_h_ell = h5["norm_h_ell"][:]
        step_norm_raw = h5["step_norm_raw"][:]
        step_norm_rms = h5["step_norm_rms"][:]
        step_cos = h5["step_cos"][:].astype(np.float32)
        D_Mset_A = h5["D_Mset_A"][:]
        D_Mset_B = h5["D_Mset_B"][:]
        D_Mset_C = h5["D_Mset_C"][:]
        norm_h_29 = h5["norm_h_29"][:]
        norm_rms_h_29 = h5["norm_rms_h_29"][:]
        norm_h_norm = h5["norm_h_norm"][:]
    print(f"[load]   {len(wids)} windows, scalars loaded")

    # Build context map
    ctx_full = build_context_map(meta, wids, pos_labels)  # (N, 6000)
    flat_ctx = ctx_full.flatten()
    donor_mask = (flat_ctx == 5)
    intron_mask = (flat_ctx == 1)
    print(f"[ctx] donor n={int(donor_mask.sum())}, intron n={int(intron_mask.sum())}")

    N_win, L, T = cos_A.shape

    # Per-window settling cell computation per gamma
    quantiles = ("q50", "q70", "q90")
    cell_d_records = []
    range_records = []

    for q in quantiles:
        print(f"\n=== γ = {q} ===")
        cells = {}

        # D_cos = 1 - cos
        # M1 cells
        for ref_label, cos_arr, max_l in [("refA", cos_A, L_STAR),
                                            ("refB", cos_B, L_STAR),
                                            ("refC", cos_C, L - 1)]:
            D_dir = 1 - cos_arr  # (N, L, T)
            gamma = gamma_v2[f"D_dir_{ref_label}_at_28"][q]
            out = np.zeros((N_win, T), dtype=np.int32)
            for i in range(N_win):
                out[i] = settling_persistence(D_dir[i], gamma, args.persistence_w, max_l)
            cells[f"M1_dir_{ref_label}"] = out

        # M2 cells (refA only)
        D_mag_A = np.abs(norm_h_ell / (norm_h_29[:, None, :] + 1e-12) - 1)  # (N, L, T)
        gamma_mag = gamma_v2["D_mag_refA_at_28"][q]
        out = np.zeros((N_win, T), dtype=np.int32)
        for i in range(N_win):
            out[i] = settling_persistence(D_mag_A[i], gamma_mag, args.persistence_w, L_STAR)
        cells["M2_mag_refA"] = out

        # M3 cells (geometry)
        # geo: alpha * v_z + beta * kappa_z at each layer
        # Need v(ell, t) and kappa(ell, t), from cached step_norm_raw / step_cos
        # step_norm_raw[ell, t] = ||h_{ell+1} - h_ell||, for ell in [0..L-2]
        # v(ell, t) = step_norm_raw[ell] / (norm_h_ell[ell] + eps)
        # kappa(ell, t) = 1 - step_cos[ell, t], for ell in [0..L-3]
        v_raw = step_norm_raw / (norm_h_ell[:, :-1, :] + 1e-12)  # (N, L-1, T)
        kappa = 1 - step_cos  # (N, L-2, T)
        # z-score per population (use pop_stats from gamma file)
        v_mean = gamma_v2["_geo_pop_stats"]["v_mean"]
        v_std = gamma_v2["_geo_pop_stats"]["v_std"]
        kappa_mean = gamma_v2["_geo_pop_stats"]["kappa_mean"]
        kappa_std = gamma_v2["_geo_pop_stats"]["kappa_std"]
        v_z = (v_raw - v_mean) / (v_std + 1e-12)
        kappa_z = (kappa - kappa_mean) / (kappa_std + 1e-12)
        # Trim v_z to L-2 (since kappa is L-2)
        v_z_trim = v_z[:, :-1, :]  # (N, L-2, T)
        for alpha, beta in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]:
            g = alpha * v_z_trim + beta * kappa_z  # (N, L-2, T)
            # Pad to (N, L, T) with NaN/inf so settling never crosses on missing layers
            g_full = np.full((N_win, L, T), np.inf, dtype=np.float32)
            g_full[:, :L - 2, :] = g
            gamma_geo = gamma_v2[f"D_geo_a{alpha}_b{beta}_at_26"][q]
            out = np.zeros((N_win, T), dtype=np.int32)
            for i in range(N_win):
                out[i] = settling_persistence(g_full[i], gamma_geo, args.persistence_w, L - 3)
            cells[f"M3_geo_a{alpha}_b{beta}"] = out

        # M4_set cells (monotone direct)
        for ref_label, D_set in [("refA", D_Mset_A), ("refB", D_Mset_B), ("refC", D_Mset_C)]:
            gamma_set = gamma_v2[f"D_Mset_{ref_label.replace('ref', 'ref')}_at_28"][q]
            out = np.zeros((N_win, T), dtype=np.int32)
            for i in range(N_win):
                out[i] = settling_monotone_direct(D_set[i], gamma_set, max_layer=L_STAR)
            cells[f"M4_set_{ref_label}"] = out

        # M5 tau cells — mirror forward script logic exactly
        # ||h_ell - h_ref|| via norms + cosine identity (no tier3 reload)
        def res_norm(norms_ell, norms_ref, cos_):
            sq = norms_ell**2 + norms_ref[:, None, :]**2 - 2 * norms_ell * norms_ref[:, None, :] * cos_
            return np.sqrt(np.maximum(sq, 0.0))

        res_A = res_norm(norm_h_ell, norm_h_29, cos_A)              # (N, L, T)
        res_C = res_norm(norm_h_ell, norm_h_norm, cos_C)            # (N, L, T)

        # path numerator = remaining cumulative step norm from ell to L_STAR-1
        path_cum_raw = np.cumsum(step_norm_raw[:, :L_STAR, :], axis=1)  # (N, L*, T)
        path_total_raw = path_cum_raw[:, L_STAR - 1:L_STAR, :]
        remaining_raw = np.concatenate([path_total_raw,
                                         path_total_raw - path_cum_raw[:, :L_STAR - 1, :]],
                                        axis=1)  # (N, L*, T)
        path_cum_rms = np.cumsum(step_norm_rms[:, :L_STAR, :], axis=1)
        path_total_rms = path_cum_rms[:, L_STAR - 1:L_STAR, :]
        remaining_rms = np.concatenate([path_total_rms,
                                         path_total_rms - path_cum_rms[:, :L_STAR - 1, :]],
                                        axis=1)  # (N, L*, T)

        tau_max = L_STAR - 1  # 28; matches forward script

        # τ_A = raw path / raw res_A
        tau_A = remaining_raw / (res_A[:, :L_STAR, :] + 1e-12)  # (N, L*, T)
        tau_A_full = np.full((N_win, L, T), np.inf, dtype=np.float32)
        tau_A_full[:, :L_STAR, :] = tau_A
        gamma_tau_A = gamma_v2["tau_refA_at_27"][q]
        out = np.zeros((N_win, T), dtype=np.int32)
        for i in range(N_win):
            out[i] = settling_persistence(tau_A_full[i], gamma_tau_A, args.persistence_w, tau_max)
        cells["M5_tau_refA"] = out

        # τ_C = raw path / raw res_C
        tau_C = remaining_raw / (res_C[:, :L_STAR, :] + 1e-12)
        tau_C_full = np.full((N_win, L, T), np.inf, dtype=np.float32)
        tau_C_full[:, :L_STAR, :] = tau_C
        gamma_tau_C = gamma_v2["tau_refC_at_27"][q]
        out = np.zeros((N_win, T), dtype=np.int32)
        for i in range(N_win):
            out[i] = settling_persistence(tau_C_full[i], gamma_tau_C, args.persistence_w, tau_max)
        cells["M5_tau_refC"] = out

        # τ_B = RMS path / RMS res_B — requires norm_rms_h_ell which is NOT cached.
        # Skip with sentinel (would need tier3 reload to recompute).
        cells["M5_tau_refB"] = np.full((N_win, T), -1, dtype=np.int32)

        # Compute Cohen d for each cell at this γ
        for cell_name, cell_arr in cells.items():
            flat = cell_arr.flatten()
            donor_vals = flat[donor_mask]
            intron_vals = flat[intron_mask]
            # Mask out invalid (-1) tokens
            donor_vals = donor_vals[(donor_vals >= 0) & (donor_vals < N_LAYERS + 1)]
            intron_vals = intron_vals[(intron_vals >= 0) & (intron_vals < N_LAYERS + 1)]
            if len(donor_vals) < 30 or len(intron_vals) < 30:
                d = np.nan
            else:
                d = cohens_d(donor_vals, intron_vals)
            cell_d_records.append({
                "cell": cell_name, "gamma": q,
                "donor_n": len(donor_vals), "intron_n": len(intron_vals),
                "donor_mean": float(donor_vals.mean()) if len(donor_vals) > 0 else np.nan,
                "intron_mean": float(intron_vals.mean()) if len(intron_vals) > 0 else np.nan,
                "cohen_d": d,
            })
            # Range stats
            valid = flat[(flat >= 0) & (flat < N_LAYERS + 1)]
            range_records.append({
                "cell": cell_name, "gamma": q,
                "range_min": int(valid.min()) if len(valid) > 0 else -1,
                "range_max": int(valid.max()) if len(valid) > 0 else -1,
                "never_settled_pct": float((flat == -1).mean() * 100),
            })
            print(f"  {cell_name:20s}  d={d:+.3f}  range=[{range_records[-1]['range_min']},{range_records[-1]['range_max']}]  never={range_records[-1]['never_settled_pct']:.1f}%")

    # Save
    cd_df = pd.DataFrame(cell_d_records)
    cd_df.to_csv(args.out_dir / "cell_d_under_gamma.csv", index=False)
    rg_df = pd.DataFrame(range_records)
    rg_df.to_csv(args.out_dir / "range_under_gamma.csv", index=False)
    print(f"\n[done] CSVs saved")

    # Plot: grouped bar chart, 17 cells × 3 γ
    fig, ax = plt.subplots(figsize=(16, 6))
    cells_in_order = cd_df.cell.unique()
    x = np.arange(len(cells_in_order))
    bar_w = 0.25
    colors = {"q50": "#1f77b4", "q70": "#d62728", "q90": "#2ca02c"}
    for i, q in enumerate(quantiles):
        sub = cd_df[cd_df.gamma == q].set_index("cell").reindex(cells_in_order)
        ax.bar(x + (i - 1) * bar_w, sub.cohen_d, bar_w, color=colors[q],
               label=f"γ = {q}", alpha=0.85)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cells_in_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cohen's d (splice_donor − intron)")
    ax.set_title("(1.2) γ ablation — Cohen d under q50/q70/q90 calibration\n"
                 "Sign + ordering preserved across γ = metric is calibration-robust")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"cell_d_under_gamma.{ext}", dpi=150)
    plt.close(fig)
    print(f"[plot] saved")

    # Summary
    summary = {
        "n_cells_computed": len(cells_in_order),
        "robust_cells": [],
        "sensitive_cells": [],
    }
    for cell in cells_in_order:
        ds = cd_df[cd_df.cell == cell].set_index("gamma").cohen_d
        if ds.isna().all():
            continue
        signs = np.sign(ds.dropna().values)
        if len(set(signs)) == 1 and ds.dropna().std() < 0.2:
            summary["robust_cells"].append(cell)
        else:
            summary["sensitive_cells"].append({
                "cell": cell, "ds": ds.to_dict(),
            })
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[summary] robust: {len(summary['robust_cells'])}/{len(cells_in_order)}")
    print(f"          sensitive: {len(summary['sensitive_cells'])}")


if __name__ == "__main__":
    main()
