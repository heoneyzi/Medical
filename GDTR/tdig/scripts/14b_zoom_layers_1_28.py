"""Zoomed visualizations of layers 1-28 (excluding L29-L30 jump).

L29-L30 magnitude blowup obscures the trajectory dynamics in earlier layers
when plotted on the same scale. This script re-plots the relevant quantities
restricted to the L1-L28 range so the actual structure is visible.

Outputs (in figures/zoom_1_28/):
  fig_z_velocity_L1_28.{png,pdf}             per-layer velocity (no L29 jump)
  fig_z_curvature_L1_28.{png,pdf}             per-layer curvature
  fig_z_cumulative_relative_L1_28.{png,pdf}  cumulative relative-velocity in L1-L28
  fig_z_per_layer_velocity_lines.{png,pdf}    per-layer mean velocity as lines per context
  fig_z_tortuosity_L1_28.{png,pdf}            tortuosity τ(ℓ) in L1-L28 only
  fig_z_combined_2x2.{png,pdf}                 4-panel combined summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

L_STAR = 29

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "intron", "coding_exon",
                  "5utr", "3utr", "intergenic"]
CONTEXT_COLORS = {
    "splice_donor": "#d62728", "splice_acceptor": "#ff7f0e",
    "intron": "#1f77b4", "coding_exon": "#2ca02c",
    "5utr": "#9467bd", "3utr": "#8c564b", "intergenic": "#7f7f7f",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier2", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/figures/zoom_1_28"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"

    L_MIN, L_MAX = 1, 28  # layers to zoom

    print("[load] tier2 + metadata + labels")
    with h5py.File(args.tier2, "r") as h5:
        step_norm = h5["step_norm_raw"][:]   # (100, 31, 6000)
        step_cos = h5["step_cos"][:]          # (100, 30, 6000)
        norm_ell = h5["norm_h_ell"][:]        # (100, 32, 6000)
        norm_h_29 = h5["norm_h_29"][:]
        cos_refA = h5["cos_refA"][:]
        wids = h5["window_idx"][:]

    pos_labels = np.load(args.pos_labels)
    meta = pd.read_parquet(args.meta)

    # Per-token context map for 100 subset windows
    print("[ctx] building per-token context map")
    context_per_token = np.zeros((len(wids), 6000), dtype=np.uint8)
    for i, wid in enumerate(wids):
        meta_row = meta[meta["window_idx"] == int(wid)].iloc[0]
        start = int(meta_row["start"])
        positions = np.clip(start + np.arange(6000), 0, len(pos_labels) - 1)
        context_per_token[i] = pos_labels[positions]

    # Derive quantities
    velocity = step_norm.astype(np.float32) / (norm_ell[:, :-1].astype(np.float32) + 1e-12)  # (100, 31, 6000)
    curvature = 1.0 - step_cos.astype(np.float32)  # (100, 30, 6000)

    def per_ctx_mean(arr, n_layers_in_arr):
        """arr: (100, n_layers, 6000) -> (7, n_layers) mean."""
        result = np.full((7, n_layers_in_arr), np.nan)
        for ctx_id in range(7):
            mask = (context_per_token == ctx_id)
            for ell in range(n_layers_in_arr):
                vals = arr[:, ell, :][mask]
                if len(vals) > 0:
                    result[ctx_id, ell] = float(vals.mean())
        return result

    print("[V2 zoom] velocity per layer (L1-L28)")
    vel_per_ctx = per_ctx_mean(velocity, 31)  # full (7, 31)
    # Slice to L1-L28 (velocity index = layer transition ell -> ell+1)
    vel_zoom = vel_per_ctx[:, L_MIN:L_MAX]  # (7, 27)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df_v = pd.DataFrame(vel_zoom, index=[POS_CODEBOOK[i] for i in range(7)],
                        columns=[f"L{l}" for l in range(L_MIN, L_MAX)]).loc[CONTEXT_ORDER]
    sns.heatmap(df_v, cmap="viridis", cbar_kws={"label": "Mean velocity v_ℓ"}, ax=ax,
                annot=False)
    ax.set_title(f"V2 (zoom L{L_MIN}-{L_MAX-1}) velocity v_ℓ per context — LINEAR scale (L29 jump excluded)")
    ax.set_xlabel("Layer transition ℓ → ℓ+1")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_velocity_L1_28.{ext}", dpi=150)
    plt.close(fig)

    print("[V3 zoom] curvature per layer (L1-L28)")
    curv_per_ctx = per_ctx_mean(curvature, 30)  # (7, 30)
    curv_zoom = curv_per_ctx[:, L_MIN:L_MAX]  # (7, 27)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df_c = pd.DataFrame(curv_zoom, index=[POS_CODEBOOK[i] for i in range(7)],
                        columns=[f"L{l}" for l in range(L_MIN, L_MAX)]).loc[CONTEXT_ORDER]
    sns.heatmap(df_c, cmap="magma", cbar_kws={"label": "Mean curvature κ_ℓ"}, ax=ax,
                annot=False)
    ax.set_title(f"V3 (zoom L{L_MIN}-{L_MAX-1}) curvature κ_ℓ — clearer per-layer pattern")
    ax.set_xlabel("Layer ℓ")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_curvature_L1_28.{ext}", dpi=150)
    plt.close(fig)

    print("[V_lines] per-layer velocity AND curvature as line plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: velocity vs layer per context
    ax = axes[0]
    layers_v = range(L_MIN, L_MAX)
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        ax.plot(layers_v, vel_per_ctx[ctx_id, L_MIN:L_MAX],
                label=ctx_name, color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer transition ℓ → ℓ+1")
    ax.set_ylabel("Mean velocity v_ℓ")
    ax.set_title(f"V_lines (zoom L{L_MIN}-{L_MAX-1}) — per-layer velocity per context")
    ax.legend(loc="best", fontsize=8)
    # Right: curvature vs layer per context
    ax = axes[1]
    layers_c = range(L_MIN, L_MAX)
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        ax.plot(layers_c, curv_per_ctx[ctx_id, L_MIN:L_MAX],
                label=ctx_name, color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="s", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean curvature κ_ℓ")
    ax.set_title("Per-layer curvature κ_ℓ per context")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_per_layer_velocity_curvature_lines.{ext}", dpi=150)
    plt.close(fig)

    print("[V4 zoom] cumulative relative velocity L1-L28")
    # cum_v[ell] = sum over k=1..ell of velocity[k]; normalize by cum_v[L_MAX]
    cum_vel = np.cumsum(velocity.astype(np.float64), axis=1)  # (100, 31, 6000)
    # Normalize at L_MAX (so within L1-L28, fraction is meaningful)
    norm_at_LMAX = cum_vel[:, L_MAX - 1:L_MAX, :] + 1e-12
    cum_vel_norm = cum_vel / norm_at_LMAX

    fig, ax = plt.subplots(figsize=(10, 5))
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        if int(mask.sum()) < 50:
            continue
        # Per-context mean curve
        curves = []
        # Subsample tokens per window for speed
        for i in range(len(wids)):
            for t in range(0, 6000, 10):  # every 10th token
                if mask[i, t]:
                    curves.append(cum_vel_norm[i, L_MIN-1:L_MAX, t])
        if curves:
            stacked = np.stack(curves)
            mean_curve = stacked.mean(axis=0)
            ax.plot(range(L_MIN, L_MAX + 1), mean_curve, label=ctx_name,
                    color=CONTEXT_COLORS[ctx_name], lw=1.8)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Cumulative relative velocity (normalized to L28)")
    ax.set_title(f"V4 (zoom L{L_MIN}-{L_MAX-1}) Cumulative Σ v_k / total — within early-mid layers")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_cumulative_relative_L1_28.{ext}", dpi=150)
    plt.close(fig)

    print("[V9 zoom] tortuosity τ(ℓ) for ℓ in L1-L28 per context")
    # Use Ref A: numerator raw cumulative remaining path, denominator ||h_ell - h_29||
    a2 = norm_ell.astype(np.float32) ** 2
    b2 = (norm_h_29.astype(np.float32) ** 2)[:, None, :]
    cross = (norm_ell.astype(np.float32) *
             norm_h_29.astype(np.float32)[:, None, :] *
             cos_refA.astype(np.float32))
    res_A = np.sqrt(np.maximum(a2 + b2 - 2 * cross, 0.0))
    cum_to_29 = np.cumsum(step_norm.astype(np.float32)[:, :L_STAR], axis=1)
    total = cum_to_29[:, -1:]
    remaining = np.concatenate([total, total - cum_to_29[:, :-1]], axis=1)  # (100, L*, 6000)
    tau = remaining / (res_A[:, :L_STAR] + 1e-12)  # (100, L*, 6000)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        mean_tau = np.zeros(L_MAX - L_MIN)
        for j, ell in enumerate(range(L_MIN, L_MAX)):
            vals = tau[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            # robust mean: clip at q99
            if len(vals) > 50:
                clip = np.quantile(vals, 0.99)
                vals = vals[vals < clip]
            mean_tau[j] = vals.mean() if len(vals) > 0 else np.nan
        ax.plot(range(L_MIN, L_MAX), mean_tau, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean τ(ℓ)  (Ref A, q99 clipped)")
    ax.set_title(f"V9 (zoom L{L_MIN}-{L_MAX-1}) tortuosity τ in early-mid layers")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_tortuosity_L1_28.{ext}", dpi=150)
    plt.close(fig)

    # Combined 2x2 panel summary
    print("[combined] 2x2 panel summary")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # (0,0) velocity line plot
    ax = axes[0, 0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        ax.plot(range(L_MIN, L_MAX), vel_per_ctx[ctx_id, L_MIN:L_MAX],
                label=ctx_name, color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean v_ℓ")
    ax.set_title("(a) Velocity v_ℓ per context — pre-L29")
    ax.legend(loc="best", fontsize=8)

    # (0,1) curvature line plot
    ax = axes[0, 1]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        ax.plot(range(L_MIN, L_MAX), curv_per_ctx[ctx_id, L_MIN:L_MAX],
                label=ctx_name, color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="s", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean κ_ℓ")
    ax.set_title("(b) Curvature κ_ℓ per context — pre-L29")
    ax.legend(loc="best", fontsize=8)

    # (1,0) cumulative relative velocity
    ax = axes[1, 0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        if int(mask.sum()) < 50:
            continue
        curves = []
        for i in range(len(wids)):
            for t in range(0, 6000, 10):
                if mask[i, t]:
                    curves.append(cum_vel_norm[i, L_MIN-1:L_MAX, t])
        if curves:
            stacked = np.stack(curves)
            mean_curve = stacked.mean(axis=0)
            ax.plot(range(L_MIN, L_MAX + 1), mean_curve, label=ctx_name,
                    color=CONTEXT_COLORS[ctx_name], lw=1.8)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Cumulative Σv_k / total (norm @ L28)")
    ax.set_title("(c) Cumulative relative velocity (within L1-L28)")
    ax.legend(loc="lower right", fontsize=8)

    # (1,1) tortuosity
    ax = axes[1, 1]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        mean_tau = np.zeros(L_MAX - L_MIN)
        for j, ell in enumerate(range(L_MIN, L_MAX)):
            vals = tau[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 50:
                clip = np.quantile(vals, 0.99)
                vals = vals[vals < clip]
            mean_tau[j] = vals.mean() if len(vals) > 0 else np.nan
        ax.plot(range(L_MIN, L_MAX), mean_tau, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean τ(ℓ)")
    ax.set_title("(d) Tortuosity τ per context")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Zoom L{L_MIN}-{L_MAX-1}: trajectory dynamics excluding L29 magnitude blowup",
                  fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig_z_combined_2x2.{ext}", dpi=150)
    plt.close(fig)

    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
