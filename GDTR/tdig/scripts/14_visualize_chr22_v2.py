"""Visualization suite for chr22 v2 outputs.

Generates the v1-v9 figure family from chr22 v2 data + analysis CSVs.

CPU-only (no GPU needed) — safe to run alongside chr17/variants GPU forwards.

Inputs:
  /path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet  (settling depths)
  /path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5 (100 subset per-layer scalars)
  /path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5            (100 subset raw h_ell)
  /path/to/TDiG/data/cache/_v2_analysis/{per_cell_summary, splice_vs_intron,
    canonical_vs_noncanonical, per_context_distributions}.csv
  /path/to/gDTR/data/annotation/chr22_position_labels.npy

Outputs: /path/to/TDiG/data/cache/_v2_analysis/figures/
  fig_v1_trajectory_pca.{png,pdf}      single-token trajectory in 2D PCA
  fig_v2_velocity_heatmap.{png,pdf}     7 contexts × 31 layers mean velocity
  fig_v3_curvature_heatmap.{png,pdf}    7 contexts × 30 layers mean curvature
  fig_v4_cumulative_path.{png,pdf}       per-context cumulative ||delta h|| / ||h_0||
  fig_v5_2d_signature.{png,pdf}          (c_dir_A, c_mag_A) scatter colored by context
  fig_v7_context_heatmap.{png,pdf}      17 cells × 7 contexts Cohen's d heatmap
  fig_v9_tortuosity_profile.{png,pdf}   per-context mean tau(ell)
  fig_summary_splice_d.{png,pdf}         per-cell splice donor vs intron Cohen's d bar
  fig_summary_canonical_d.{png,pdf}      per-cell canonical vs non-canonical Cohen's d bar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096

# Context codebook
POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "intron", "coding_exon",
                  "5utr", "3utr", "intergenic"]

# Color palette (consistent across figures)
CONTEXT_COLORS = {
    "splice_donor": "#d62728",      # red
    "splice_acceptor": "#ff7f0e",   # orange
    "intron": "#1f77b4",            # blue
    "coding_exon": "#2ca02c",       # green
    "5utr": "#9467bd",              # purple
    "3utr": "#8c564b",              # brown
    "intergenic": "#7f7f7f",        # gray
}


def setup_style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })


def fig_v7_context_heatmap(ctx_csv: Path, out_dir: Path):
    """17 cells × 7 contexts: Cohen's d vs intron baseline."""
    df = pd.read_csv(ctx_csv)
    # Pivot: cell × context, value = mean settling depth
    piv_mean = df.pivot(index="cell", columns="context", values="mean")
    # Compute d vs intron baseline per cell
    intron_means = piv_mean["intron"]
    intron_std = df[df["context"] == "intron"].set_index("cell")["std"]
    d_table = piv_mean.subtract(intron_means, axis=0).divide(intron_std + 1e-9, axis=0)
    d_table = d_table[CONTEXT_ORDER]
    # Drop intron column (always 0)
    d_table = d_table.drop(columns=["intron"])
    # Order rows by absolute max |d| (excluding intron)
    abs_max = d_table.abs().max(axis=1)
    d_table = d_table.loc[abs_max.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(d_table, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
                cbar_kws={"label": "Cohen's d vs intron"},
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Per-cell × per-context Cohen's d vs intron baseline\nchr22 v2 (12,978 windows × 77.9M positions)")
    ax.set_xlabel("Context class")
    ax.set_ylabel("Metric × Reference cell")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v7_context_heatmap.{ext}", dpi=150)
    plt.close(fig)
    print(f"  fig_v7_context_heatmap saved")


def fig_summary_bars(svi_csv: Path, cnc_csv: Path, out_dir: Path):
    """Per-cell Cohen's d bar charts."""
    svi = pd.read_csv(svi_csv).dropna(subset=["cohens_d_donor_minus_intron"])
    svi = svi.sort_values("cohens_d_donor_minus_intron")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in svi["cohens_d_donor_minus_intron"]]
    ax.barh(svi["cell"], svi["cohens_d_donor_minus_intron"], color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Cohen's d (splice donor - intron)")
    ax.set_title("Per-cell discrimination: splice donor (187K) vs intron (41M)\nNegative = donor settles earlier; positive = donor settles later")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_summary_splice_d.{ext}", dpi=150)
    plt.close(fig)

    cnc = pd.read_csv(cnc_csv).dropna(subset=["cohens_d_canon_minus_noncanon"])
    cnc = cnc.sort_values("cohens_d_canon_minus_noncanon")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in cnc["cohens_d_canon_minus_noncanon"]]
    ax.barh(cnc["cell"], cnc["cohens_d_canon_minus_noncanon"], color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Cohen's d (canonical - non-canonical donor)")
    ax.set_title("Per-cell discrimination: canonical GT-AG donor (193K) vs non-canonical (5,808)")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_summary_canonical_d.{ext}", dpi=150)
    plt.close(fig)
    print(f"  fig_summary_splice_d + canonical_d saved")


def fig_v2_v3_v9(tier2_path: Path, tier3_path: Path, pos_labels: np.ndarray,
                  window_meta: pd.DataFrame, out_dir: Path):
    """V2 velocity heatmap, V3 curvature heatmap, V9 tortuosity profile (subset)."""
    with h5py.File(tier2_path, "r") as h5:
        step_norm = h5["step_norm_raw"][:]   # (100, 31, 6000)
        step_cos = h5["step_cos"][:]          # (100, 30, 6000)
        norm_ell = h5["norm_h_ell"][:]        # (100, 32, 6000)
        norm_h_29 = h5["norm_h_29"][:]
        cos_refA = h5["cos_refA"][:]
        wids = h5["window_idx"][:]
    with h5py.File(tier3_path, "r") as h5:
        token_stride = h5["token_stride"][:]

    # Velocity v_ell = step_norm / norm_h_ell[:-1]
    velocity = step_norm.astype(np.float32) / (norm_ell[:, :-1].astype(np.float32) + 1e-12)  # (100, 31, 6000)
    # Curvature kappa_ell = 1 - step_cos
    curvature = 1.0 - step_cos.astype(np.float32)  # (100, 30, 6000)

    # Match each token to its context using window start + position
    print("  building per-token context map...")
    context_per_token = np.zeros((len(wids), 6000), dtype=np.uint8)
    for i, wid in enumerate(wids):
        meta_row = window_meta[window_meta["window_idx"] == wid].iloc[0]
        start = int(meta_row["start"])
        positions = start + np.arange(6000)
        positions = np.clip(positions, 0, len(pos_labels) - 1)
        context_per_token[i] = pos_labels[positions]

    # Aggregate per context class
    def per_context_mean(arr, n_layers, axis_layer=1):
        """arr: (100, n_layers, 6000), return (7, n_layers) mean."""
        result = np.full((7, n_layers), np.nan)
        for ctx_id in range(7):
            mask = (context_per_token == ctx_id)  # (100, 6000)
            for ell in range(n_layers):
                vals = arr[:, ell, :][mask]
                if len(vals) > 0:
                    result[ctx_id, ell] = float(vals.mean())
        return result

    print("  V2 velocity heatmap (log scale)...")
    vel_per_ctx = per_context_mean(velocity, 31)  # (7, 31)
    fig, ax = plt.subplots(figsize=(11, 5))
    df_vel = pd.DataFrame(vel_per_ctx, index=[POS_CODEBOOK[i] for i in range(7)],
                          columns=[f"L{l}" for l in range(31)])
    df_vel = df_vel.loc[CONTEXT_ORDER]
    # Log10 transform — L29 jump is several orders larger than other transitions
    df_vel_log = np.log10(df_vel + 1e-12)
    sns.heatmap(df_vel_log, cmap="viridis",
                cbar_kws={"label": "log10(Mean velocity v_ℓ)"}, ax=ax)
    ax.set_title("V2 Per-layer velocity (log10) by context (chr22 subset, 100 windows)")
    ax.set_xlabel("Layer transition ℓ → ℓ+1")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v2_velocity_heatmap.{ext}", dpi=150)
    plt.close(fig)

    print("  V3 curvature heatmap...")
    curv_per_ctx = per_context_mean(curvature, 30)  # (7, 30)
    fig, ax = plt.subplots(figsize=(11, 5))
    df_curv = pd.DataFrame(curv_per_ctx, index=[POS_CODEBOOK[i] for i in range(7)],
                           columns=[f"L{l}" for l in range(30)])
    df_curv = df_curv.loc[CONTEXT_ORDER]
    sns.heatmap(df_curv, cmap="magma", cbar_kws={"label": "Mean curvature κ_ℓ"}, ax=ax)
    ax.set_title("V3 Per-layer curvature by context (1 - cos(Δh_ℓ, Δh_ℓ₊₁))")
    ax.set_xlabel("Layer ℓ (curvature between Δh_ℓ and Δh_ℓ₊₁)")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v3_curvature_heatmap.{ext}", dpi=150)
    plt.close(fig)

    print("  V4 cumulative path (RMSNormed + relative-magnitude)...")
    # Two panels:
    # LEFT: cumulative velocity v_ell (scale-invariant ratio) — meaningful at intermediate layers
    # RIGHT: raw cumulative ||delta h|| on log scale (shows L29 jump in context)
    cum_vel = np.cumsum(velocity.astype(np.float64), axis=1)  # (100, 31, 6000)
    cum_vel_norm = cum_vel / (cum_vel[:, -1:, :] + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        n_tok = int(mask.sum())
        if n_tok < 50:
            continue
        # Average across tokens in this context (per layer)
        ctx_curves = []
        for i in range(len(wids)):
            for t in range(6000):
                if mask[i, t]:
                    ctx_curves.append(cum_vel_norm[i, :, t])
        if ctx_curves:
            stacked = np.stack(ctx_curves)
            mean_curve = stacked.mean(axis=0)
            ax.plot(range(1, 32), mean_curve, label=ctx_name,
                    color=CONTEXT_COLORS[ctx_name], lw=1.8)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Cumulative relative velocity fraction")
    ax.set_title("V4 (left) Cumulative ||Δh|| / ||h_ell|| fraction\n(removes magnitude growth confound)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)

    # RIGHT: Raw cumulative path, log y-scale (so L29 jump still visible without dominating)
    cum_path_raw = np.cumsum(step_norm.astype(np.float64), axis=1)
    ax = axes[1]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        n_tok = int(mask.sum())
        if n_tok < 50:
            continue
        ctx_curves = []
        for i in range(len(wids)):
            for t in range(6000):
                if mask[i, t]:
                    ctx_curves.append(cum_path_raw[i, :, t])
        if ctx_curves:
            stacked = np.stack(ctx_curves)
            mean_curve = stacked.mean(axis=0)
            ax.plot(range(1, 32), mean_curve, label=ctx_name,
                    color=CONTEXT_COLORS[ctx_name], lw=1.8)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Cumulative ||Δh||  (raw)")
    ax.set_yscale("log")
    ax.set_title("V4 (right) Cumulative raw ||Δh|| (log scale)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v4_cumulative_path.{ext}", dpi=150)
    plt.close(fig)

    print("  V9 tortuosity profile...")
    # tau(ell) = remaining_path / ||h_ell - h_29||
    # Use res_norm from cos+norms (law of cosines): res_norm = sqrt(||h_ell||^2 + ||h_29||^2 - 2 ||h_ell|| ||h_29|| cos)
    a2 = norm_ell.astype(np.float32) ** 2  # (100, 32, 6000)
    b2 = (norm_h_29.astype(np.float32) ** 2)[:, None, :]  # (100, 1, 6000)
    cross = (norm_ell.astype(np.float32) *
             norm_h_29.astype(np.float32)[:, None, :] *
             cos_refA.astype(np.float32))
    res_A = np.sqrt(np.maximum(a2 + b2 - 2 * cross, 0.0))  # (100, 32, 6000)
    # Cumulative path
    cum_to_29 = np.cumsum(step_norm.astype(np.float32)[:, :L_STAR], axis=1)  # (100, L*, 6000)
    total = cum_to_29[:, -1:]
    remaining = np.concatenate([total, total - cum_to_29[:, :-1]], axis=1)  # (100, L*, 6000)
    tau = remaining / (res_A[:, :L_STAR] + 1e-12)  # (100, L*, 6000)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (context_per_token == ctx_id)
        # Mean tau per layer across tokens in this context
        mean_tau = np.zeros(L_STAR)
        for ell in range(L_STAR):
            vals = tau[:, ell, :][mask]
            if len(vals) > 0:
                # Clip for plotting (tortuosity can have outliers)
                vals = vals[np.isfinite(vals)]
                vals = vals[vals < np.quantile(vals, 0.99)]
                mean_tau[ell] = vals.mean() if len(vals) > 0 else np.nan
        ax.plot(range(L_STAR), mean_tau, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean τ(ℓ)  (Ref A, 99%ile clipped)")
    ax.set_title("V9 Tortuosity τ(ℓ) by context  (Ref A: raw path / ||h_ℓ - h_29||)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v9_tortuosity_profile.{ext}", dpi=150)
    plt.close(fig)
    print(f"  V2 + V3 + V4 + V9 saved")


def fig_v5_2d_signature(tier1_path: Path, pos_labels: np.ndarray,
                          window_meta: pd.DataFrame, out_dir: Path):
    """V5 (c_M1_dir_refA, c_M3_geo_a0.0_b1.0) scatter — Def 2 vs Def 1."""
    df = pd.read_parquet(tier1_path)
    cell_x = "M1_dir_refA"
    cell_y = "M3_geo_a0.0_b1.0"
    # Sample 5000 tokens per context for plotting
    print(f"  building per-token records for V5...")
    rng = np.random.default_rng(42)
    samples_by_ctx = {ctx: [] for ctx in CONTEXT_ORDER}
    for _, row in df.iterrows():
        start = int(row["start"]); T = int(row["T"])
        positions = start + np.arange(T)
        positions = positions[positions < len(pos_labels)]
        ctx_arr = pos_labels[positions]
        c1 = np.array(row[cell_x], dtype=np.int32)[: len(positions)]
        c2 = np.array(row[cell_y], dtype=np.int32)[: len(positions)]
        for ctx_id, ctx_name in POS_CODEBOOK.items():
            mask = ctx_arr == ctx_id
            n_here = mask.sum()
            if n_here == 0:
                continue
            x_vals = c1[mask]; y_vals = c2[mask]
            # Filter -1
            valid = (x_vals != -1) & (y_vals != -1)
            x_vals = x_vals[valid]; y_vals = y_vals[valid]
            samples_by_ctx[ctx_name].append(np.column_stack([x_vals, y_vals]))

    # Two-panel V5:
    # LEFT: hexbin 2D histogram (all samples pooled, density visible despite integer cells)
    # RIGHT: per-context KDE of (Δc = c_geo - c_dir) — shows separation directly
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    # Combine all samples
    all_pts = []
    all_ctx = []
    for ctx_name in CONTEXT_ORDER:
        if not samples_by_ctx[ctx_name]:
            continue
        arr = np.concatenate(samples_by_ctx[ctx_name])
        n = min(len(arr), 5000)
        idx = rng.choice(len(arr), size=n, replace=False)
        sub = arr[idx].astype(np.float32)
        # Jitter (uniform in [-0.35, 0.35] to avoid integer pile-up)
        sub += rng.uniform(-0.35, 0.35, size=sub.shape)
        all_pts.append(sub)
        all_ctx.extend([ctx_name] * len(sub))
    all_pts = np.concatenate(all_pts)
    # Hexbin density (combined population — for sanity of integer clustering)
    hb = ax.hexbin(all_pts[:, 0], all_pts[:, 1], gridsize=40, cmap="viridis",
                    bins="log", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(N tokens)")
    ax.set_xlabel("c_dir_refA  (Def 2: settling toward h_29)")
    ax.set_ylabel("c_geo (α=0, β=1, curvature-only)  (Def 1: trajectory stops)")
    ax.set_title("V5 (left) 2D signature density (chr22 sampled, jittered)")

    # RIGHT: KDE of Δc = c_geo - c_dir per context
    ax = axes[1]
    for ctx_name in CONTEXT_ORDER:
        if not samples_by_ctx[ctx_name]:
            continue
        arr = np.concatenate(samples_by_ctx[ctx_name])
        if len(arr) < 50:
            continue
        delta_c = arr[:, 1].astype(np.float32) - arr[:, 0].astype(np.float32)
        sns.kdeplot(delta_c, ax=ax, label=ctx_name,
                    color=CONTEXT_COLORS[ctx_name], lw=2, bw_adjust=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δc = c_geo (Def 1) − c_dir_refA (Def 2)")
    ax.set_ylabel("Density")
    ax.set_title("V5 (right) Per-context Δc distribution\nNegative = trajectory stops before direction settles")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v5_2d_signature.{ext}", dpi=150)
    plt.close(fig)
    print(f"  V5 saved (hexbin + Δc KDE)")


def fig_v1_trajectory_pca(tier3_path: Path, pos_labels: np.ndarray,
                           window_meta: pd.DataFrame, out_dir: Path):
    """V1 trajectory in 2D PCA on RMSNORMED h_ell (eliminates magnitude growth dominance)."""
    from sklearn.decomposition import PCA
    print("  loading RMSNormed h_ell subset for PCA...")
    with h5py.File(tier3_path, "r") as h5:
        wids = h5["window_idx"][:]
        chosen_wins = list(range(min(20, len(wids))))
        token_sub_idx = np.linspace(0, 600 - 1, 50, dtype=int)  # 50 tokens per window
        all_h = []
        for wi in chosen_wins:
            # Use RMSNormed instead of raw — removes magnitude-growth confound
            h_w = h5["raw_h_ell_rmsnormed"][wi, :, token_sub_idx, :]  # (32, 50, 4096) fp16
            all_h.append(h_w.astype(np.float32))
        h_arr = np.stack(all_h, axis=0)  # (20, 32, 50, 4096)
    print(f"  RMSNormed h shape for PCA: {h_arr.shape}")

    # Per-layer mean-center (eliminate layer-systematic effect dominating PC1)
    layer_means = h_arr.mean(axis=(0, 2), keepdims=True)  # (1, 32, 1, 4096)
    h_centered = h_arr - layer_means
    # Reshape for PCA: (n_samples, 4096) where n_samples = 20 * 32 * 50 = 32000
    n_w, n_l, n_t, n_h = h_centered.shape
    flat = h_centered.reshape(-1, n_h)

    print("  fitting PCA(n_components=2)...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(flat)  # (32000, 2)
    explained = pca.explained_variance_ratio_
    print(f"  explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")
    proj = proj.reshape(n_w, n_l, n_t, 2)

    # Get context for each (window, token)
    print("  gathering contexts...")
    ctx_arr = np.zeros((n_w, n_t), dtype=np.uint8)
    for wi_local, wi in enumerate(chosen_wins):
        actual_wid = int(wids[wi])
        meta_row = window_meta[window_meta["window_idx"] == actual_wid].iloc[0]
        start = int(meta_row["start"])
        token_positions = start + token_sub_idx
        token_positions = np.clip(token_positions, 0, len(pos_labels) - 1)
        ctx_arr[wi_local] = pos_labels[token_positions]

    # Plot — single panel showing trajectories of selected tokens
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # LEFT: per-layer scatter (all 32000 points colored by context, layer = alpha gradient)
    ax = axes[0]
    proj_flat = proj.reshape(-1, 2)
    ctx_flat = np.tile(ctx_arr[:, None, :], (1, n_l, 1)).flatten()
    layer_flat = np.tile(np.arange(n_l)[None, :, None], (n_w, 1, n_t)).flatten()
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        mask = ctx_flat == ctx_id
        if mask.sum() == 0:
            continue
        # Plot all layers but with layer-dependent alpha so we see trajectory direction
        alphas = (layer_flat[mask] / (n_l - 1)) * 0.6 + 0.05
        ax.scatter(proj_flat[mask, 0], proj_flat[mask, 1],
                    s=2, c=[CONTEXT_COLORS[ctx_name]] * mask.sum(),
                    alpha=0.15, label=ctx_name)
    ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}% var)")
    ax.set_title("V1 (left) PCA on RMSNormed h_ell (per-layer-centered)\ncolored by context, faint=early layers; eliminates magnitude growth confound")
    ax.legend(loc="upper left", markerscale=3, fontsize=9)

    # RIGHT: a few example trajectories — one token per context, all layers
    ax = axes[1]
    chosen_tokens = []
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        # find first (window, token) with this context
        found = np.argwhere(ctx_arr == ctx_id)
        if len(found) == 0:
            continue
        wi, ti = found[0]
        traj = proj[wi, :, ti, :]  # (32, 2)
        ax.plot(traj[:, 0], traj[:, 1], "-",
                color=CONTEXT_COLORS[ctx_name], lw=1.5, alpha=0.8)
        ax.scatter(traj[:, 0], traj[:, 1], s=15,
                   c=range(n_l), cmap="viridis",
                   edgecolors=CONTEXT_COLORS[ctx_name], linewidths=0.6,
                   label=ctx_name)
        # Mark start (L=0) with circle and end (L=31) with x
        ax.plot(traj[0, 0], traj[0, 1], "o", ms=12, mec="black", mfc="none", mew=1.5)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", ms=15, mec="black", mew=2.0)
        chosen_tokens.append((ctx_name, wi, ti))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("V1 (right) example trajectories\n(○ start L=0, × end L=31; viridis colormap = layer index)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_v1_trajectory_pca.{ext}", dpi=150)
    plt.close(fig)
    print(f"  V1 saved ({len(chosen_tokens)} trajectories)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet"))
    parser.add_argument("--tier2", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5"))
    parser.add_argument("--tier3", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/_v2_analysis"))
    parser.add_argument("--meta", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    parser.add_argument("--pos-labels", type=Path,
                        default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/_v2_analysis/figures"))
    parser.add_argument("--skip-v1", action="store_true", help="skip PCA (needs 0.5+ GB RAM)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"[setup] loading metadata + labels...")
    window_meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)

    print(f"[V7] context heatmap...")
    fig_v7_context_heatmap(args.analysis_dir / "per_context_distributions.csv", args.out_dir)

    print(f"[summary] splice + canonical d bars...")
    fig_summary_bars(args.analysis_dir / "splice_vs_intron.csv",
                      args.analysis_dir / "canonical_vs_noncanonical.csv",
                      args.out_dir)

    print(f"[V2/V3/V4/V9] heatmaps + tortuosity (subset)...")
    fig_v2_v3_v9(args.tier2, args.tier3, pos_labels, window_meta, args.out_dir)

    print(f"[V5] 2D signature scatter (Def 2 vs Def 1)...")
    fig_v5_2d_signature(args.tier1, pos_labels, window_meta, args.out_dir)

    if not args.skip_v1:
        print(f"[V1] trajectory PCA (sampled 20 windows × 32 layers × 50 tokens)...")
        fig_v1_trajectory_pca(args.tier3, pos_labels, window_meta, args.out_dir)
    else:
        print(f"[V1] skipped (--skip-v1)")

    print(f"\n[done] all figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
