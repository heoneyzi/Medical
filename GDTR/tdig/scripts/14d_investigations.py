"""Three investigations:
  A: L=4 bimodality — which tokens go to which cluster?
  B: PC1 semantic meaning — correlate PC1 with nucleotide / GC / context
  C: M4_set re-shrinkage — sklearn LedoitWolf vs custom (λ=0.95 clipped)

Outputs in figures/investigations/:
  fig_A_L4_bimodality.{png,pdf}
  fig_B_PC1_semantic.{png,pdf}
  fig_C_m4set_reshrinkage.{png,pdf}
  inv_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "/path/to/gDTR")

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
NUC_COLORS = {"A": "#d62728", "T": "#1f77b4", "G": "#2ca02c", "C": "#ff7f0e", "N": "#7f7f7f"}


def build_context_map_full(meta_df, wids, pos_labels):
    ctx = np.zeros((len(wids), 6000), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(6000), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def fetch_nucleotides(meta_df, wids, fasta_path: Path, token_stride: np.ndarray):
    """Fetch nucleotides at each subsampled token position.

    Returns: (100, 600) char array.
    """
    import pysam
    fa = pysam.FastaFile(str(fasta_path))
    nuc = np.zeros((len(wids), 600), dtype="<U1")
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        end = int(row["end"])
        seq = fa.fetch(row["chrom"], start, end).upper()
        stride = int(token_stride[i])
        sub_idx = np.arange(0, len(seq), stride)[:600]
        # Pad if needed
        if len(sub_idx) < 600:
            sub_idx = np.pad(sub_idx, (0, 600 - len(sub_idx)), mode="edge")
        for j, p in enumerate(sub_idx):
            nuc[i, j] = seq[p] if p < len(seq) else "N"
    return nuc


def investigation_A_L4_bimodality(proj, ctx_map, nuc_map, raw_rms_L4, out_dir):
    """L=4 PCA shows bimodality — what's the split criterion?"""
    print("[A] L=4 bimodality investigation")
    # L=4 projection: proj shape (100, 32, 600, 2)
    L_TARGET = 4
    pts = proj[:, L_TARGET, :, :].reshape(-1, 2)  # (60000, 2)
    nucs = nuc_map.flatten()                       # (60000,)
    ctxs = ctx_map.flatten()                       # (60000,)

    # K-means with k=2 on the L=4 projection
    valid = (nucs != "N")
    pts_v = pts[valid]; nucs_v = nucs[valid]; ctxs_v = ctxs[valid]
    print(f"  fitting k-means k=2 on {len(pts_v):,} L=4 points...")
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(pts_v)
    cluster = km.labels_

    # Cross-tab cluster vs nucleotide
    print("  cluster x nucleotide contingency table:")
    ct = pd.crosstab(pd.Series(cluster, name="cluster"),
                       pd.Series(nucs_v, name="nucleotide"))
    print(ct)
    ct_pct = ct.div(ct.sum(axis=0), axis=1) * 100  # per-nucleotide cluster distribution
    print("  per-nucleotide % in each cluster:")
    print(ct_pct)

    # Also check purine vs pyrimidine pattern
    purine_mask = np.isin(nucs_v, ["A", "G"])
    print(f"  purine (A/G) cluster 0: {(cluster[purine_mask] == 0).sum() / max(purine_mask.sum(), 1):.3f}")
    print(f"  pyrimidine (C/T) cluster 0: {(cluster[~purine_mask] == 0).sum() / max((~purine_mask).sum(), 1):.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (0,0) L=4 scatter colored by nucleotide
    ax = axes[0, 0]
    rng = np.random.default_rng(42)
    sample_n = min(20000, len(pts_v))
    sidx = rng.choice(len(pts_v), size=sample_n, replace=False)
    for nuc_lbl in ["A", "T", "G", "C"]:
        mask = nucs_v[sidx] == nuc_lbl
        if mask.sum() == 0:
            continue
        ax.scatter(pts_v[sidx][mask, 0], pts_v[sidx][mask, 1], s=3, alpha=0.4,
                    color=NUC_COLORS[nuc_lbl], label=nuc_lbl)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(A1) L=4 PCA colored by NUCLEOTIDE\nbimodality split criterion?")
    ax.legend(loc="best", markerscale=4, fontsize=9)

    # (0,1) Cluster assignment fraction by nucleotide
    ax = axes[0, 1]
    ct_pct.T.plot(kind="bar", stacked=True, ax=ax,
                   color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Fraction in cluster")
    ax.set_xlabel("Nucleotide")
    ax.set_title("(A2) Per-nucleotide cluster assignment\n(k-means k=2 on L=4 PCA)")
    ax.legend(title="Cluster", labels=["cluster 0", "cluster 1"], fontsize=9)
    ax.tick_params(axis="x", rotation=0)

    # (1,0) L=4 scatter colored by CONTEXT
    ax = axes[1, 0]
    sidx2 = rng.choice(len(pts_v), size=sample_n, replace=False)
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        mask = ctxs_v[sidx2] == ctx_id
        if mask.sum() == 0:
            continue
        ax.scatter(pts_v[sidx2][mask, 0], pts_v[sidx2][mask, 1], s=3, alpha=0.4,
                    color=CONTEXT_COLORS[ctx_name], label=ctx_name)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(A3) L=4 PCA colored by CONTEXT (same positions)")
    ax.legend(loc="best", markerscale=4, fontsize=8)

    # (1,1) Mean PC1 by nucleotide + context
    ax = axes[1, 1]
    df_pc1 = pd.DataFrame({
        "PC1": pts_v[:, 0], "PC2": pts_v[:, 1],
        "nucleotide": nucs_v,
        "context": [POS_CODEBOOK[c] for c in ctxs_v],
    })
    means_nuc = df_pc1.groupby("nucleotide")["PC1"].mean().sort_values()
    means_ctx = df_pc1.groupby("context")["PC1"].mean().reindex(CONTEXT_ORDER).dropna()
    x_pos = np.arange(max(len(means_nuc), len(means_ctx)))
    width = 0.4
    ax.bar(x_pos[: len(means_nuc)] - width / 2, means_nuc.values, width=width,
            color=[NUC_COLORS[n] for n in means_nuc.index], label="by nucleotide")
    # On separate axis area
    ax.set_xticks(x_pos[: len(means_nuc)] - width / 2)
    ax.set_xticklabels(means_nuc.index)
    ax.set_ylabel("Mean PC1 at L=4")
    ax.set_title("(A4) Mean PC1 at L=4 by nucleotide")
    ax.axhline(0, color="k", lw=0.5)

    fig.suptitle("Investigation A — L=4 bimodality origin", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_A_L4_bimodality.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_A saved")

    return {
        "L4_nucleotide_cluster_fractions": ct_pct.to_dict(),
        "purine_cluster_0_frac": float((cluster[purine_mask] == 0).sum() / max(purine_mask.sum(), 1)),
        "pyrimidine_cluster_0_frac": float((cluster[~purine_mask] == 0).sum() / max((~purine_mask).sum(), 1)),
        "mean_PC1_per_nucleotide": means_nuc.to_dict(),
        "mean_PC1_per_context_L4": means_ctx.to_dict(),
    }


def investigation_B_PC1_semantic(proj, ctx_map, nuc_map, out_dir):
    """PC1 semantic across layers: nucleotide / context / GC association."""
    print("[B] PC1 semantic meaning across layers")
    n_w, n_l, n_t, _ = proj.shape
    pc1 = proj[..., 0]  # (100, 32, 600)
    nucs = nuc_map  # (100, 600)

    # Per-layer mean PC1 by nucleotide
    nuc_layer_mean = {n: np.full(n_l, np.nan) for n in ["A", "T", "G", "C"]}
    for ell in range(n_l):
        for nuc_lbl in ["A", "T", "G", "C"]:
            mask = nucs == nuc_lbl  # (100, 600)
            vals = pc1[:, ell, :][mask]
            if len(vals) > 30:
                nuc_layer_mean[nuc_lbl][ell] = float(vals.mean())

    # Per-layer mean PC1 by context
    ctx_layer_mean = {c: np.full(n_l, np.nan) for c in POS_CODEBOOK.values()}
    for ell in range(n_l):
        for ctx_id, ctx_name in POS_CODEBOOK.items():
            mask = ctx_map == ctx_id
            vals = pc1[:, ell, :][mask]
            if len(vals) > 30:
                ctx_layer_mean[ctx_name][ell] = float(vals.mean())

    # GC content (local 21-bp window): we don't have raw sequence handy, but
    # purine vs pyrimidine effect at each layer is informative
    purine_layer_mean = np.full(n_l, np.nan)
    pyrim_layer_mean = np.full(n_l, np.nan)
    for ell in range(n_l):
        pur_mask = np.isin(nucs, ["A", "G"])
        pyr_mask = np.isin(nucs, ["C", "T"])
        purine_layer_mean[ell] = pc1[:, ell, :][pur_mask].mean()
        pyrim_layer_mean[ell] = pc1[:, ell, :][pyr_mask].mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) PC1 mean per nucleotide across layers
    ax = axes[0, 0]
    for nuc_lbl in ["A", "T", "G", "C"]:
        ax.plot(range(n_l), nuc_layer_mean[nuc_lbl],
                color=NUC_COLORS[nuc_lbl], lw=2, marker="o", ms=3, label=nuc_lbl)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean PC1")
    ax.set_title("(B1) Mean PC1 per nucleotide across layers")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="best", fontsize=9)

    # (0,1) PC1 purine vs pyrimidine
    ax = axes[0, 1]
    ax.plot(range(n_l), purine_layer_mean, color="#d62728", lw=2.5,
            marker="o", ms=3, label="Purine (A, G)")
    ax.plot(range(n_l), pyrim_layer_mean, color="#1f77b4", lw=2.5,
            marker="s", ms=3, label="Pyrimidine (C, T)")
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean PC1")
    ax.set_title("(B2) Purine vs Pyrimidine split — PC1 axis is largely purine/pyrimidine axis?")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="best", fontsize=9)

    # (1,0) Mean PC1 per context across layers
    ax = axes[1, 0]
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        if not np.any(np.isfinite(ctx_layer_mean[ctx_name])):
            continue
        ax.plot(range(n_l), ctx_layer_mean[ctx_name],
                color=CONTEXT_COLORS[ctx_name], lw=2, marker="o", ms=3, label=ctx_name)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean PC1")
    ax.set_title("(B3) Mean PC1 per CONTEXT across layers")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # (1,1) PC1 separation: nucleotide vs context magnitude per layer
    ax = axes[1, 1]
    nuc_sep = np.full(n_l, np.nan)
    ctx_sep = np.full(n_l, np.nan)
    for ell in range(n_l):
        nuc_means = [nuc_layer_mean[n][ell] for n in ["A", "T", "G", "C"]
                      if np.isfinite(nuc_layer_mean[n][ell])]
        if len(nuc_means) >= 2:
            nuc_sep[ell] = max(nuc_means) - min(nuc_means)
        ctx_means = [v[ell] for v in ctx_layer_mean.values()
                      if np.isfinite(v[ell])]
        if len(ctx_means) >= 2:
            ctx_sep[ell] = max(ctx_means) - min(ctx_means)
    ax.plot(range(n_l), nuc_sep, color="#d62728", lw=2, marker="o", ms=3,
            label="PC1 range across NUCLEOTIDES")
    ax.plot(range(n_l), ctx_sep, color="#1f77b4", lw=2, marker="s", ms=3,
            label="PC1 range across CONTEXTS")
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Max − min of group means")
    ax.set_title("(B4) Which feature does PC1 separate best at each layer?")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Investigation B — PC1 axis semantic meaning across layers", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_B_PC1_semantic.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_B saved")

    return {
        "PC1_per_nuc_by_layer": {n: v.tolist() for n, v in nuc_layer_mean.items()},
        "PC1_per_context_by_layer": {c: v.tolist() for c, v in ctx_layer_mean.items()},
        "PC1_purine_layer_mean": purine_layer_mean.tolist(),
        "PC1_pyrim_layer_mean": pyrim_layer_mean.tolist(),
    }


def investigation_C_m4set_reshrinkage(tier3_path, ctx_map_tier3, out_dir):
    """sklearn LedoitWolf vs custom λ=0.95 clipped."""
    print("[C] M4_set re-shrinkage with sklearn LedoitWolf")
    with h5py.File(tier3_path, "r") as h5:
        # Pool h_29 from all 100 subset windows (raw, fp32)
        raw_h_29 = h5["raw_h_ell"][:, L_STAR, :, :].astype(np.float32)  # (100, 600, 4096)
    samples = raw_h_29.reshape(-1, HIDDEN_SIZE)  # (60000, 4096)
    print(f"  samples shape: {samples.shape}")

    # Center
    samples_c = samples - samples.mean(axis=0, keepdims=True)

    # sklearn LedoitWolf
    print("  fitting sklearn LedoitWolf (may take ~30 sec)...")
    lw = LedoitWolf(store_precision=True, assume_centered=False)
    lw.fit(samples)
    lam_sklearn = float(lw.shrinkage_)
    print(f"  sklearn LedoitWolf shrinkage: lambda = {lam_sklearn:.4f}")
    sigma_inv_lw = lw.precision_.astype(np.float32)

    # Recompute D_M_set using new sigma_inv on subset (use 100 windows)
    print("  recomputing D_M_set on subset with new Sigma_inv...")
    with h5py.File(tier3_path, "r") as h5:
        all_h = h5["raw_h_ell"][:].astype(np.float32)  # (100, 32, 600, 4096)
    diff = all_h - all_h[:, L_STAR:L_STAR + 1, :, :]  # (100, 32, 600, 4096)
    # Batched quadratic: (L*100*600, 4096) @ (4096, 4096) -> (L*100*600, 4096)
    # Memory: 100*32*600*4096*4 = 31 GB already in memory. matmul output same size.
    # Process per layer to manage memory
    D_new = np.zeros((100, N_LAYERS, 600), dtype=np.float32)
    for ell in range(N_LAYERS):
        d_ell = diff[:, ell, :, :].reshape(-1, HIDDEN_SIZE)  # (60000, 4096)
        sig_d = d_ell @ sigma_inv_lw  # (60000, 4096)
        quad = (d_ell * sig_d).sum(axis=-1)
        D_new[:, ell, :] = np.sqrt(np.maximum(quad, 0.0)).reshape(100, 600)
    print(f"  D_new shape: {D_new.shape}, sample values L20: {D_new[:, 20, :].mean():.3f}")

    # Old D_M_set (from tier2 — but tier2 is at full 6000 resolution, subsample to 600)
    with h5py.File("/path/to/TDiG/data/cache/chr22_v2/tier2_scalars_subset_v2.h5", "r") as h5:
        D_old = h5["D_Mset_A"][:, :, ::10][:, :, :600]  # (100, 32, 600) subsampled to match

    # Per-context Cohen's d (donor vs intron) — old vs new
    def cohen_d(x, y):
        if len(x) < 5 or len(y) < 5: return np.nan
        mx, my = x.mean(), y.mean()
        s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) /
                     (len(x) + len(y) - 2))
        return (mx - my) / s if s > 0 else np.nan

    donor_mask = ctx_map_tier3 == 5
    intron_mask = ctx_map_tier3 == 1
    d_old = np.full(N_LAYERS, np.nan)
    d_new = np.full(N_LAYERS, np.nan)
    for ell in range(N_LAYERS):
        d_old[ell] = cohen_d(D_old[:, ell, :][donor_mask], D_old[:, ell, :][intron_mask])
        d_new[ell] = cohen_d(D_new[:, ell, :][donor_mask], D_new[:, ell, :][intron_mask])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) Donor mean per layer — old vs new
    ax = axes[0, 0]
    ax.plot(range(N_LAYERS),
            [D_old[:, ell, :][donor_mask].mean() if donor_mask.sum() > 0 else np.nan
             for ell in range(N_LAYERS)],
            "o-", color="#1f77b4", label="OLD (λ=0.95 clipped) donor", lw=2)
    ax.plot(range(N_LAYERS),
            [D_new[:, ell, :][donor_mask].mean() if donor_mask.sum() > 0 else np.nan
             for ell in range(N_LAYERS)],
            "o-", color="#d62728", label=f"NEW (sklearn LW λ={lam_sklearn:.4f}) donor", lw=2)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean D_M_set")
    ax.set_yscale("log"); ax.set_title("(C1) Splice donor D_M_set: OLD vs NEW shrinkage")
    ax.legend(loc="best", fontsize=9)

    # (0,1) Cohen's d donor vs intron per layer
    ax = axes[0, 1]
    ax.plot(range(N_LAYERS), d_old, "o-", color="#1f77b4",
            label="OLD shrinkage", lw=2)
    ax.plot(range(N_LAYERS), d_new, "o-", color="#d62728",
            label="NEW sklearn LW", lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Cohen's d (donor − intron)")
    ax.set_title("(C2) Per-layer Cohen's d: improvement?")
    ax.legend(loc="best", fontsize=9)

    # (1,0) Per-context mean D_M_set NEW
    ax = axes[1, 0]
    for ctx_name in CONTEXT_ORDER:
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        mask = (ctx_map_tier3 == ctx_id)
        if int(mask.sum()) < 30:
            continue
        mean_curve = np.zeros(N_LAYERS)
        for ell in range(N_LAYERS):
            mean_curve[ell] = D_new[:, ell, :][mask].mean()
        ax.plot(range(N_LAYERS), mean_curve, label=ctx_name,
                color=CONTEXT_COLORS[ctx_name], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean D_M_set (NEW)")
    ax.set_yscale("log")
    ax.set_title("(C3) Per-context D_M_set with sklearn LW shrinkage")
    ax.legend(loc="best", fontsize=8)

    # (1,1) Summary: OLD lambda=0.95 vs NEW
    ax = axes[1, 1]
    # |Cohen's d| at ell=15 + ell=20 for OLD vs NEW
    summary_rows = []
    for ctx_name in CONTEXT_ORDER:
        if ctx_name == "intron":
            continue
        ctx_id = [k for k, v in POS_CODEBOOK.items() if v == ctx_name][0]
        ctx_mask = ctx_map_tier3 == ctx_id
        if int(ctx_mask.sum()) < 30:
            continue
        d_o = cohen_d(D_old[:, 20, :][ctx_mask], D_old[:, 20, :][intron_mask])
        d_n = cohen_d(D_new[:, 20, :][ctx_mask], D_new[:, 20, :][intron_mask])
        summary_rows.append({"ctx": ctx_name, "OLD": abs(d_o) if np.isfinite(d_o) else 0,
                              "NEW": abs(d_n) if np.isfinite(d_n) else 0})
    df_s = pd.DataFrame(summary_rows).set_index("ctx")
    df_s.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"])
    ax.set_ylabel("|Cohen's d| vs intron at ℓ=20")
    ax.set_title("(C4) Per-context discrimination at ℓ=20: OLD vs NEW shrinkage")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)

    fig.suptitle(f"Investigation C — M4_set re-shrinkage (sklearn LW λ={lam_sklearn:.4f})",
                  fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_C_m4set_reshrinkage.{ext}", dpi=150)
    plt.close(fig)
    print("  fig_C saved")

    return {
        "sklearn_LW_lambda": lam_sklearn,
        "OLD_lambda_clipped": 0.95,
        "Cohen_d_per_layer_OLD": [None if not np.isfinite(x) else float(x) for x in d_old],
        "Cohen_d_per_layer_NEW": [None if not np.isfinite(x) else float(x) for x in d_new],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--fasta", type=Path,
                   default=Path("/path/to/gDTR/data/reference/chr22.fa"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/figures/investigations"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print("[load] meta + labels + tier3 stride")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    with h5py.File(args.tier3, "r") as h5:
        wids = h5["window_idx"][:]
        token_stride = h5["token_stride"][:]
        raw_rms = h5["raw_h_ell_rmsnormed"][:].astype(np.float32)  # (100, 32, 600, 4096)

    print("[ctx] building per-token context map (tier3 grid)")
    ctx_map_full = build_context_map_full(meta, wids, pos_labels)  # (100, 6000)
    ctx_map_tier3 = ctx_map_full[:, ::10][:, :600]  # (100, 600)

    print("[nuc] fetching nucleotides from chr22.fa")
    nuc_map = fetch_nucleotides(meta, wids, args.fasta, token_stride)
    print(f"  nuc_map shape: {nuc_map.shape}, unique: {np.unique(nuc_map)}")

    print("[PCA] fitting PCA on RMSNormed h_ell (same as 14c)")
    layer_means = raw_rms.mean(axis=(0, 2), keepdims=True)
    h_centered = raw_rms - layer_means
    flat = h_centered.reshape(-1, HIDDEN_SIZE)
    print(f"  fitting PCA on {flat.shape[0]:,} samples...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(flat).astype(np.float32)
    print(f"  PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    proj = proj.reshape(100, N_LAYERS, 600, 2)

    summary = {}
    # Investigation A
    summary["A_L4_bimodality"] = investigation_A_L4_bimodality(
        proj, ctx_map_tier3, nuc_map, raw_rms[:, 4], args.out_dir,
    )
    # Investigation B
    summary["B_PC1_semantic"] = investigation_B_PC1_semantic(proj, ctx_map_tier3, nuc_map, args.out_dir)

    # Free memory before C
    del raw_rms, h_centered, flat, proj
    import gc; gc.collect()

    # Investigation C
    summary["C_m4set_reshrinkage"] = investigation_C_m4set_reshrinkage(
        args.tier3, ctx_map_tier3, args.out_dir,
    )

    (args.out_dir / "inv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] all investigations saved to {args.out_dir}")


if __name__ == "__main__":
    main()
