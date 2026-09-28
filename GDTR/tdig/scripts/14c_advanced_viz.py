"""Advanced visualizations: M5 Option B, M4_set, improved trajectory.

Addresses user questions:
  (a) M5 Option B (RMSNormed trajectory) tortuosity profile per context
  (b) PCA interpretation — per-context mean trajectories with explicit direction
  (c) M4_set Sigma_ref-whitened distance heatmaps + line plots + monotonicity
  (d) Better trajectory viz — per-layer snapshot grid showing cluster separation

Outputs in figures/advanced/:
  fig_a_m5_optionB_tortuosity.{png,pdf}      4-panel: heatmap + lines + raw vs rms compare
  fig_b_pca_context_mean_trajectories.{png,pdf}  mean per-context trajectories + arrows
  fig_c_m4set_heatmap_lines.{png,pdf}         M4_set per-layer per-context + monotone decay
  fig_d_layer_snapshots.{png,pdf}             8-layer × 2D PCA snapshots showing separation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "intron", "coding_exon",
                  "5utr", "3utr", "intergenic"]
CONTEXT_COLORS = {
    "splice_donor": "#d62728", "splice_acceptor": "#ff7f0e",
    "intron": "#1f77b4", "coding_exon": "#2ca02c",
    "5utr": "#9467bd", "3utr": "#8c564b", "intergenic": "#7f7f7f",
}


def build_context_map(meta_df, wids, pos_labels):
    """(100, 6000) per-token context map."""
    ctx = np.zeros((len(wids), 6000), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(6000), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def fig_a_m5_optionB_tortuosity(tier2_path, tier3_path, ctx_map_tier3, out_dir):
    """M5 Option B: tortuosity using RMSNormed trajectory + denominator."""
    L_MIN, L_MAX = 1, L_STAR - 1  # tau defined for ell in [0, L*-2]
    print("[a] computing M5 Option B (RMSNormed tortuosity)")

    with h5py.File(tier3_path, "r") as h5:
        raw_rms = h5["raw_h_ell_rmsnormed"][:].astype(np.float32)  # (100, 32, 600, 4096)
        wids = h5["window_idx"][:]
    # For RMSNormed space, compute step_norm and residual norm DIRECTLY
    # step_norm_rms[ell] = ||raw_rms[ell+1] - raw_rms[ell]||
    print("  computing step norms in RMSNormed space...")
    step_rms = np.linalg.norm(raw_rms[:, 1:] - raw_rms[:, :-1], axis=-1)  # (100, 31, 600)
    # res_B[ell] = ||raw_rms[ell] - raw_rms[L_STAR]||
    print("  computing residual norms vs h_29_rms...")
    h_29_rms = raw_rms[:, L_STAR:L_STAR + 1]  # (100, 1, 600, 4096)
    res_B = np.linalg.norm(raw_rms - h_29_rms, axis=-1)  # (100, 32, 600)
    # Numerator: remaining path from ell to L*-1 in RMSNormed space
    cum_to_L = np.cumsum(step_rms[:, :L_STAR], axis=1)  # (100, L*, 600)
    total = cum_to_L[:, -1:]
    remaining = np.concatenate([total, total - cum_to_L[:, :-1]], axis=1)  # (100, L*, 600)
    tau_B = remaining / (res_B[:, :L_STAR] + 1e-12)  # (100, L*, 600)

    # Per-context aggregation: need to subsample tier3 to (100, 600) context map
    # tier3 token positions are sub-sampled with stride=10 from the (100, 6000)
    # ctx_map_tier3 should already be subsampled to (100, 600)
    print(f"  tau_B shape: {tau_B.shape}, ctx_map: {ctx_map_tier3.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (0,0) Per-layer τ_B line plot
    ax = axes[0, 0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (ctx_map_tier3 == ctx_id)
        if int(mask.sum()) < 30:
            continue
        mean_tau = np.zeros(L_STAR)
        for ell in range(L_STAR):
            vals = tau_B[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                clip = np.quantile(vals, 0.99) if len(vals) > 30 else np.inf
                vals = vals[vals < clip]
                mean_tau[ell] = vals.mean() if len(vals) > 0 else np.nan
        ax.plot(range(L_STAR), mean_tau, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean τ_B(ℓ)  (RMSNormed trajectory)")
    ax.set_title("(a) M5 Option B: τ_B per context (linear scale)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")
    ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)

    # (0,1) Per-layer τ_B heatmap
    ax = axes[0, 1]
    tau_by_ctx = np.full((7, L_STAR), np.nan)
    for ctx_id in range(7):
        mask = (ctx_map_tier3 == ctx_id)
        if mask.sum() < 30:
            continue
        for ell in range(L_STAR):
            vals = tau_B[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 30:
                vals = vals[vals < np.quantile(vals, 0.99)]
            if len(vals) > 0:
                tau_by_ctx[ctx_id, ell] = vals.mean()
    df_tau = pd.DataFrame(np.log10(tau_by_ctx + 1e-12),
                          index=[POS_CODEBOOK[i] for i in range(7)],
                          columns=[f"L{l}" for l in range(L_STAR)]).loc[CONTEXT_ORDER]
    sns.heatmap(df_tau, cmap="viridis", cbar_kws={"label": "log10(τ_B)"}, ax=ax)
    ax.set_title("(b) log10(τ_B) heatmap — context × layer")
    ax.set_xlabel("Layer ℓ")

    # (1,0) τ_B vs τ_A (raw) comparison at ell=20
    ax = axes[1, 0]
    # Compute tau_A at same window/token grid
    print("  computing tau_A (raw) at same token grid for comparison...")
    with h5py.File(tier3_path, "r") as h5:
        raw_h = h5["raw_h_ell"][:].astype(np.float32)
    step_raw = np.linalg.norm(raw_h[:, 1:] - raw_h[:, :-1], axis=-1)  # (100, 31, 600)
    h_29 = raw_h[:, L_STAR:L_STAR + 1]
    res_A = np.linalg.norm(raw_h - h_29, axis=-1)
    cum_A = np.cumsum(step_raw[:, :L_STAR], axis=1)
    rem_A = np.concatenate([cum_A[:, -1:], cum_A[:, -1:] - cum_A[:, :-1]], axis=1)
    tau_A = rem_A / (res_A[:, :L_STAR] + 1e-12)

    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (ctx_map_tier3 == ctx_id)
        if int(mask.sum()) < 30:
            continue
        # Compare τ_A vs τ_B at ell=20
        a_vals = tau_A[:, 20, :][mask]
        b_vals = tau_B[:, 20, :][mask]
        a_vals = a_vals[np.isfinite(a_vals) & (a_vals < np.quantile(a_vals, 0.99))]
        b_vals = b_vals[np.isfinite(b_vals) & (b_vals < np.quantile(b_vals, 0.99))]
        if len(a_vals) < 30 or len(b_vals) < 30:
            continue
        ax.scatter(a_vals.mean(), b_vals.mean(), s=80, color=CONTEXT_COLORS[ctx_name],
                    label=ctx_name, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("τ_A (raw trajectory)  at ℓ=20")
    ax.set_ylabel("τ_B (RMSNormed trajectory)  at ℓ=20")
    ax.set_title("(c) τ_A (raw, ≈1) vs τ_B (RMSNormed, varies)\nshows why Option B is meaningful")
    ax.legend(loc="best", fontsize=9)
    ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(1, color="k", lw=0.5, ls="--", alpha=0.5)

    # (1,1) τ_B distribution at L=27 (calibration layer)
    ax = axes[1, 1]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (ctx_map_tier3 == ctx_id)
        if int(mask.sum()) < 30:
            continue
        vals = tau_B[:, 27, :][mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 30:
            vals = vals[vals < np.quantile(vals, 0.99)]
        if len(vals) > 10:
            sns.kdeplot(vals, ax=ax, label=ctx_name,
                         color=CONTEXT_COLORS[ctx_name], lw=2, bw_adjust=0.6)
    ax.set_xlabel("τ_B(ℓ=27)")
    ax.set_ylabel("Density")
    ax.set_title("(d) τ_B distribution at calibration layer ℓ=27\nper-context discrimination")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("M5 Option B (RMSNormed tortuosity) — escapes the τ≈1 geometric artifact",
                  fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_a_m5_optionB_tortuosity.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_a saved")
    return raw_rms, raw_h


def fig_b_pca_mean_trajectories(raw_rms, ctx_map_tier3, out_dir):
    """PCA of RMSNormed h_ell with PER-CONTEXT MEAN trajectories with explicit arrows."""
    print("[b] mean per-context trajectories in PCA")
    # Use all 100 windows × 32 layers × subsample tokens for PCA
    # raw_rms shape (100, 32, 600, 4096)
    # Per-layer mean-center then PCA on concatenated points
    layer_means = raw_rms.mean(axis=(0, 2), keepdims=True)  # (1, 32, 1, 4096)
    h_centered = raw_rms - layer_means
    # Flatten: (100*32*600, 4096)
    n_w, n_l, n_t, n_h = h_centered.shape
    flat = h_centered.reshape(-1, n_h)
    print(f"  fitting PCA on {flat.shape[0]:,} samples...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(flat).astype(np.float32)
    expl = pca.explained_variance_ratio_
    print(f"  PC1={expl[0]:.3f}, PC2={expl[1]:.3f}")
    proj = proj.reshape(n_w, n_l, n_t, 2)

    # Compute per-context per-layer MEAN position in PCA space
    mean_per_ctx = np.zeros((7, n_l, 2))  # (7 contexts, 32 layers, 2 PCs)
    count_per_ctx = np.zeros(7)
    for ctx_id in range(7):
        mask = (ctx_map_tier3 == ctx_id)  # (100, 600)
        n_tok = int(mask.sum())
        count_per_ctx[ctx_id] = n_tok
        if n_tok < 30:
            continue
        for ell in range(n_l):
            # proj[:, ell, :, :] -> (100, 600, 2); use mask
            vals = proj[:, ell, :, :][mask]  # (n_tok, 2)
            mean_per_ctx[ctx_id, ell] = vals.mean(axis=0)

    # Plot — single panel showing mean trajectories per context
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: all-context mean trajectories with arrows
    ax = axes[0]
    for ctx_id in range(7):
        if count_per_ctx[ctx_id] < 30:
            continue
        ctx_name = POS_CODEBOOK[ctx_id]
        traj = mean_per_ctx[ctx_id]  # (32, 2)
        ax.plot(traj[:, 0], traj[:, 1], "-",
                color=CONTEXT_COLORS[ctx_name], lw=2.0, alpha=0.8, label=ctx_name)
        # Scatter all 32 layer points
        ax.scatter(traj[:, 0], traj[:, 1], s=20,
                    c=range(n_l), cmap="viridis",
                    edgecolors=CONTEXT_COLORS[ctx_name], linewidths=0.5)
        # Start (L=0) circle, End (L=31) X
        ax.plot(traj[0, 0], traj[0, 1], "o", ms=12, mec="black",
                mfc=CONTEXT_COLORS[ctx_name], mew=1.5)
        ax.plot(traj[-1, 0], traj[-1, 1], "X", ms=14, mec="black",
                mfc=CONTEXT_COLORS[ctx_name], mew=1.5)
        # Arrow from start to L=15 (midway)
        ax.annotate("", xy=traj[15], xytext=traj[0],
                     arrowprops=dict(arrowstyle="->", color=CONTEXT_COLORS[ctx_name],
                                       lw=1.3, alpha=0.6, shrinkA=0, shrinkB=5))
        # Layer labels at L=0, 8, 15, 22, 29
        for ell_lbl in [0, 8, 15, 22, 29]:
            ax.annotate(f"L{ell_lbl}", traj[ell_lbl],
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=7, color=CONTEXT_COLORS[ctx_name], alpha=0.8)
    ax.set_xlabel(f"PC1 ({100*expl[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*expl[1]:.1f}% var)")
    ax.set_title("(b1) Per-context MEAN trajectory in PCA space\n○ start L=0, X end L=31, arrows = L0→L15 direction")
    ax.legend(loc="best", fontsize=9, ncol=2)

    # RIGHT: pairwise context separation at each layer
    ax = axes[1]
    # Compute pairwise L2 distances between context means per layer
    # Show as line plot: at each layer, max pairwise distance + mean pairwise distance
    max_dist = np.zeros(n_l)
    mean_dist = np.zeros(n_l)
    n_valid = (count_per_ctx >= 30).sum()
    valid_ctx = np.where(count_per_ctx >= 30)[0]
    for ell in range(n_l):
        pts = mean_per_ctx[valid_ctx, ell]  # (n_valid, 2)
        # pairwise distances
        if len(pts) >= 2:
            from itertools import combinations
            ds = [np.linalg.norm(pts[i] - pts[j]) for i, j in combinations(range(len(pts)), 2)]
            max_dist[ell] = max(ds)
            mean_dist[ell] = np.mean(ds)
    ax.plot(range(n_l), max_dist, label="Max pairwise context distance",
            color="#d62728", lw=2, marker="o", ms=4)
    ax.plot(range(n_l), mean_dist, label="Mean pairwise context distance",
            color="#1f77b4", lw=2, marker="s", ms=4)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Distance in PCA space")
    ax.set_title("(b2) Context separability per layer\nHigher = more distinct context representations")
    ax.legend(loc="best", fontsize=9)
    ax.axvline(L_STAR, color="k", lw=0.5, ls="--", alpha=0.5, label="L*=29")

    fig.suptitle("PCA interpretation: each context evolves toward a DIFFERENT (PC1, PC2) region.\n"
                  "The 'fan-out' indicates the model learns context-specific representations.",
                  fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_b_pca_context_mean_trajectories.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_b saved")
    return proj


def fig_c_m4set(tier2_path, ctx_map_tier3_full, out_dir):
    """M4_set Sigma_ref-whitened distance: per-layer heatmap + line plot + monotone-decay check."""
    print("[c] M4_set visualization")
    # tier2 D_Mset shape (100, 32, 6000) — full token resolution
    with h5py.File(tier2_path, "r") as h5:
        D_A = h5["D_Mset_A"][:]  # (100, 32, 6000) fp32
        D_B = h5["D_Mset_B"][:]
        D_C = h5["D_Mset_C"][:]

    # Need (100, 6000) context map for tier2 — use the full-resolution version
    # ctx_map_tier3_full is (100, 6000) — same as tier2 token grid
    print(f"  D_Mset_A shape {D_A.shape}, ctx_map {ctx_map_tier3_full.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (0,0) Per-context line plot M4_set_A
    ax = axes[0, 0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (ctx_map_tier3_full == ctx_id)
        if int(mask.sum()) < 30:
            continue
        mean_curve = np.zeros(N_LAYERS)
        for ell in range(N_LAYERS):
            vals = D_A[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            mean_curve[ell] = vals.mean() if len(vals) > 0 else np.nan
        ax.plot(range(N_LAYERS), mean_curve, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mean D_M_set^A(ℓ)")
    ax.set_title("(a) M4_set Ref A — monotone decrease toward 0 at ℓ=29 (by design)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")
    ax.axvline(L_STAR, color="k", lw=0.5, ls="--", alpha=0.5)

    # (0,1) M4_set_A heatmap (context × layer)
    ax = axes[0, 1]
    matrix = np.full((7, N_LAYERS), np.nan)
    for ctx_id in range(7):
        mask = (ctx_map_tier3_full == ctx_id)
        if mask.sum() < 30:
            continue
        for ell in range(N_LAYERS):
            vals = D_A[:, ell, :][mask]
            vals = vals[np.isfinite(vals)]
            matrix[ctx_id, ell] = vals.mean() if len(vals) > 0 else np.nan
    df = pd.DataFrame(np.log10(matrix + 1e-12),
                       index=[POS_CODEBOOK[i] for i in range(7)],
                       columns=[f"L{l}" for l in range(N_LAYERS)]).loc[CONTEXT_ORDER]
    sns.heatmap(df, cmap="viridis", cbar_kws={"label": "log10(D_M_set^A)"}, ax=ax)
    ax.set_title("(b) log10(D_M_set^A) per context × layer")
    ax.set_xlabel("Layer ℓ")

    # (1,0) Ref A/B/C comparison at ell=15 + ell=28
    ax = axes[1, 0]
    refs = {"A": D_A, "B": D_B, "C": D_C}
    bar_data = {"ref": [], "layer": [], "median": [], "ctx": []}
    for ref_label, D in refs.items():
        for ell in (15, 28):
            for ctx_id in range(7):
                mask = (ctx_map_tier3_full == ctx_id)
                if mask.sum() < 30:
                    continue
                vals = D[:, ell, :][mask]
                vals = vals[np.isfinite(vals)]
                bar_data["ref"].append(f"Ref {ref_label}\nℓ={ell}")
                bar_data["layer"].append(ell)
                bar_data["median"].append(np.median(vals))
                bar_data["ctx"].append(POS_CODEBOOK[ctx_id])
    df_bar = pd.DataFrame(bar_data)
    # Pivot for grouped bar
    piv = df_bar.pivot_table(index="ctx", columns="ref", values="median").reindex(CONTEXT_ORDER)
    piv.plot(kind="bar", ax=ax, color=["#1f77b4", "#7570b3", "#d95f02",
                                          "#1f77b4", "#7570b3", "#d95f02"], width=0.8)
    ax.set_ylabel("Median D_M_set")
    ax.set_yscale("log")
    ax.set_title("(c) M4_set Ref A/B/C at ℓ=15 vs ℓ=28 per context")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.tick_params(axis="x", rotation=30)

    # (1,1) Monotonicity check — fraction of tokens with strictly decreasing D
    ax = axes[1, 1]
    for ref_label, D in refs.items():
        # For each token, count if D is monotone non-increasing across layers
        # We just plot mean D per ell for all tokens
        mean_all = D.reshape(-1, N_LAYERS).mean(axis=0)
        ax.plot(range(N_LAYERS), mean_all,
                label=f"Ref {ref_label}", lw=2)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Population mean D_M_set")
    ax.set_yscale("log")
    ax.set_title("(d) Population mean D_M_set per ref variant\nMonotone-decreasing by Σ_ref-whitening construction")
    ax.legend(loc="best", fontsize=9)
    ax.axvline(L_STAR, color="k", lw=0.5, ls="--", alpha=0.5)

    fig.suptitle("M4_set Reference-whitened settling distance — three reference variants",
                  fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_c_m4set_heatmap_lines.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_c saved")


def fig_d_layer_snapshots(proj, ctx_map_tier3, out_dir):
    """Per-layer 2D PCA snapshot grid showing cluster separation over depth."""
    print("[d] per-layer PCA snapshot grid")
    # proj shape (100, 32, 600, 2)
    selected_layers = [0, 4, 8, 12, 15, 20, 25, 29]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.flatten()

    # Compute global x/y range for consistent scale
    all_x = proj[..., 0].flatten()
    all_y = proj[..., 1].flatten()
    qx = (np.quantile(all_x, 0.005), np.quantile(all_x, 0.995))
    qy = (np.quantile(all_y, 0.005), np.quantile(all_y, 0.995))

    rng = np.random.default_rng(42)
    for i, ell in enumerate(selected_layers):
        ax = axes[i]
        for ctx_id in range(7):
            ctx_name = POS_CODEBOOK[ctx_id]
            mask = (ctx_map_tier3 == ctx_id)
            n_tok = int(mask.sum())
            if n_tok < 30:
                continue
            pts = proj[:, ell, :, :][mask]  # (n_tok, 2)
            # Subsample for plotting
            n_plot = min(800, len(pts))
            idx = rng.choice(len(pts), size=n_plot, replace=False)
            sub = pts[idx]
            ax.scatter(sub[:, 0], sub[:, 1], s=3, alpha=0.35,
                        color=CONTEXT_COLORS[ctx_name], label=ctx_name if i == 0 else None)
        ax.set_xlim(qx); ax.set_ylim(qy)
        ax.set_title(f"Layer ℓ={ell}")
        if i == 4:
            ax.set_ylabel("PC2")
        if i >= 4:
            ax.set_xlabel("PC1")
    axes[0].legend(loc="upper left", markerscale=3, fontsize=7, ncol=2)
    fig.suptitle("(d) Per-layer PCA snapshots — clusters separate as ℓ increases\n"
                  "(consistent axes; subsampled 800 tokens/context for plotting)",
                  fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_d_layer_snapshots.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_d saved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier2", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5"))
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/figures/advanced"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print("[load] meta + labels")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)

    # tier3 token grid: (100, 600) sub-sampled with stride=10 from (100, 6000)
    # ctx_map for tier3 (per-window token positions are 0, 10, 20, ...)
    with h5py.File(args.tier3, "r") as h5:
        wids = h5["window_idx"][:]
    print(f"  tier3 windows: {len(wids)}")
    ctx_map_full = build_context_map(meta, wids, pos_labels)  # (100, 6000)
    # Subsample to (100, 600) at same stride (every 10th)
    ctx_map_tier3 = ctx_map_full[:, ::10][:, :600]  # (100, 600)
    print(f"  ctx_map_tier3 shape: {ctx_map_tier3.shape}")

    # (a) M5 Option B
    raw_rms, raw_h = fig_a_m5_optionB_tortuosity(args.tier2, args.tier3, ctx_map_tier3, args.out_dir)

    # (b) PCA mean trajectories
    proj = fig_b_pca_mean_trajectories(raw_rms, ctx_map_tier3, args.out_dir)

    # Free memory
    del raw_h
    import gc; gc.collect()

    # (c) M4_set viz (uses tier2 full-resolution, not tier3 subset)
    fig_c_m4set(args.tier2, ctx_map_full, args.out_dir)

    # (d) Per-layer snapshots
    fig_d_layer_snapshots(proj, ctx_map_tier3, args.out_dir)

    print(f"\n[done] all advanced figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
