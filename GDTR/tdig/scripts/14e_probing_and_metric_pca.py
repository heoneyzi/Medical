"""
Analysis B + D on chr22 v2 data.

B  Per-layer probing classifier: train logistic regression (h_ell → context label)
   for each layer. Tells us how RECOVERABLE each context is at each depth — the
   absolute "context signal magnitude" curve that PCA's 2D peephole only proxies.

D  Metric ↔ PCA correlation: fit the same PCA as fig_b (per-layer-centered, 1.92M
   samples), project per-token per-layer to (PC1, PC2), then Spearman-correlate
   each of the 17 settling-cell values (per token) against PC scores at every layer.
   Tells us whether the PCA peephole has a geometric meaning for our 17 metrics.

Outputs (all under --out-dir, default: results/analysis_BD/):

  B:
    per_layer_auroc.csv         per (layer, context-pair) AUROC + n
    per_layer_auroc.png         curves over depth, splice_donor-vs-intron headline
    per_layer_best_acc.csv      best vs intron AUROC per layer (all 6 contexts)
  D:
    metric_pca_corr.csv         (layer, metric, PC) Spearman r
    metric_pca_corr_heatmap.png 17 metrics × 32 layers heatmap (PC1, PC2 panels)
    pca_explained_variance.json PC1+PC2+...+PC10 cumulative for sanity

Compute budget (CPU on server):
  B: 32 layers × ~5-10s LR on ~16K-20K tokens × 4096 dims = ~5-10 min
  D: 1 PCA fit on 1.92M × 4096 (LR/incremental) + 32×17×2 Spearman = ~10-15 min
  Total: ~15-25 min CPU. No GPU contention with ongoing chr17/variants forward.

Usage on server:
  cd ~/TDiG && python scripts/14e_probing_and_metric_pca.py \
      --tier1   ~/gDTR/results/phase1.6/chr22_cache.h5  # NOT used, see below
      --tier1-parquet ~/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet \
      --tier3   ~/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5 \
      --meta    ~/TDiG/data/cache/chr22_v2/window_metadata.parquet \
      --pos-labels ~/gDTR/data/annotation/chr22_position_labels.npy \
      --out-dir ~/TDiG/data/cache/_v2_analysis/analysis_BD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "intron", "coding_exon",
                  "5utr", "3utr", "intergenic"]

# 17 settling cells in tier1 parquet
CELL_NAMES = [
    "M1_dir_refA", "M1_dir_refB", "M1_dir_refC",
    "M2_mag_refA", "M2_mag_refB_diag", "M2_mag_refC_diag",
    "M3_geo_a0.0_b1.0", "M3_geo_a0.5_b1.0", "M3_geo_a1.0_b1.0",
    "M3_geo_a1.0_b0.5", "M3_geo_a1.0_b0.0",
    "M4_set_refA", "M4_set_refB", "M4_set_refC",
    "M5_tau_refA", "M5_tau_refB", "M5_tau_refC",
]


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------

def build_context_map(meta_df, wids, pos_labels, n_tokens=6000):
    """(n_wins, n_tokens) per-token context label array."""
    ctx = np.zeros((len(wids), n_tokens), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(n_tokens), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def load_tier3(tier3_path):
    """Returns raw_rms (100, 32, 600, 4096) fp32, wids (100,)."""
    with h5py.File(tier3_path, "r") as h5:
        raw_rms = h5["raw_h_ell_rmsnormed"][:].astype(np.float32)
        wids = h5["window_idx"][:]
    return raw_rms, wids


def load_tier1_cells_for_windows(parquet_path, wids, n_tokens_tier3, stride=10):
    """Load tier1 parquet, subset to wids, subsample to tier3 grid (stride=10).
       Returns dict cell_name -> (n_wins, n_tokens_tier3) int32 array of settling layer.
       -1 / N_LAYERS encode "never settled" depending on cell convention; preserved as-is.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["window_idx"].isin(set(int(w) for w in wids))]
    df = df.set_index("window_idx")
    out = {}
    for cell in CELL_NAMES:
        arr = np.zeros((len(wids), n_tokens_tier3), dtype=np.int32)
        for i, wid in enumerate(wids):
            cell_list = df.loc[int(wid), cell]  # python list, length T
            cell_arr = np.asarray(cell_list, dtype=np.int32)
            # Subsample to tier3 grid: every stride-th token, up to n_tokens_tier3
            sub = cell_arr[::stride][:n_tokens_tier3]
            if sub.shape[0] < n_tokens_tier3:
                pad = np.full(n_tokens_tier3 - sub.shape[0], -1, dtype=np.int32)
                sub = np.concatenate([sub, pad])
            arr[i] = sub
        out[cell] = arr
    return out


# --------------------------------------------------------------------------
# B  Per-layer probing
# --------------------------------------------------------------------------

def per_layer_probing(raw_rms, ctx_map_tier3, out_dir, max_n_per_class=8000, seed=42):
    """For each layer, fit LR (h_ell -> context) on balanced subsamples.
       Reports pairwise AUROC for (splice_donor vs intron) plus 1-vs-rest for
       all 6 main contexts, per layer.

       raw_rms : (n_w, 32, n_tok, H)  fp32  rmsnormed
       ctx_map_tier3 : (n_w, n_tok)  uint8  context labels at tier3 grid
    """
    n_w, n_l, n_tok, H = raw_rms.shape
    flat_ctx = ctx_map_tier3.flatten()  # (n_w*n_tok,)
    print(f"[B] probing setup: n_w={n_w} n_layer={n_l} n_tok={n_tok} H={H}")

    pair_records = []
    rng = np.random.default_rng(seed)

    # Headline pair: splice_donor (5) vs intron (1)
    donor_id, intron_id = 5, 1
    donor_mask = (flat_ctx == donor_id)
    intron_mask = (flat_ctx == intron_id)
    n_donor = int(donor_mask.sum())
    n_intron = int(intron_mask.sum())
    n_take = min(n_donor, n_intron, max_n_per_class)
    print(f"[B] donor n={n_donor}, intron n={n_intron}, balanced subsample n={n_take} per class")

    donor_idx = np.where(donor_mask)[0]
    intron_idx = np.where(intron_mask)[0]
    rng.shuffle(donor_idx); rng.shuffle(intron_idx)
    use_donor = donor_idx[:n_take]
    use_intron = intron_idx[:n_take]
    pair_idx = np.concatenate([use_donor, use_intron])
    pair_y = np.concatenate([np.ones(n_take, dtype=int), np.zeros(n_take, dtype=int)])

    # Reshape flat-index back to (window, token)
    flat_w_idx = pair_idx // n_tok
    flat_t_idx = pair_idx % n_tok

    for ell in range(n_l):
        # Gather X for this layer
        X = raw_rms[flat_w_idx, ell, flat_t_idx, :]  # (2*n_take, H)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, pair_y, test_size=0.3, stratify=pair_y, random_state=seed,
        )
        clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auroc = roc_auc_score(y_te, prob)
        pair_records.append({
            "layer": ell, "pair": "splice_donor_vs_intron",
            "n_per_class": n_take, "AUROC": auroc,
        })
        print(f"[B]  L={ell:2d}  splice_donor vs intron  AUROC={auroc:.3f}")

    # All-context-vs-intron AUROC, all 6 main contexts
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        if ctx_name == "intron":
            continue
        pos_mask = (flat_ctx == ctx_id)
        neg_mask = (flat_ctx == intron_id)
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        n_t = min(n_pos, n_neg, max_n_per_class)
        if n_t < 200:
            continue
        pos_idx = np.where(pos_mask)[0]; rng.shuffle(pos_idx)
        neg_idx = np.where(neg_mask)[0]; rng.shuffle(neg_idx)
        pi, ni = pos_idx[:n_t], neg_idx[:n_t]
        pair_i = np.concatenate([pi, ni])
        pair_y_ = np.concatenate([np.ones(n_t), np.zeros(n_t)])
        wi = pair_i // n_tok
        ti = pair_i % n_tok
        for ell in range(n_l):
            X = raw_rms[wi, ell, ti, :]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, pair_y_, test_size=0.3, stratify=pair_y_, random_state=seed,
            )
            clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_te, prob)
            pair_records.append({
                "layer": ell, "pair": f"{ctx_name}_vs_intron",
                "n_per_class": n_t, "AUROC": auroc,
            })
        print(f"[B] done {ctx_name}_vs_intron  L_best={max(pair_records[-n_l:], key=lambda r: r['AUROC'])['layer']}")

    df = pd.DataFrame(pair_records)
    csv_path = out_dir / "per_layer_auroc.csv"
    df.to_csv(csv_path, index=False)
    print(f"[B] saved {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for pair, g in df.groupby("pair"):
        ax.plot(g["layer"], g["AUROC"], marker="o", ms=4, lw=1.6, label=pair)
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4, label=f"L*={L_STAR}")
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Pairwise AUROC")
    ax.set_title("(B) Per-layer probing AUROC — context recoverability vs layer depth\n"
                 "(LR on rmsnormed h_ell, balanced subsamples, 70/30 split, n per class capped)")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"per_layer_auroc.{ext}", dpi=150)
    plt.close(fig)
    print(f"[B] figure saved")
    return df


# --------------------------------------------------------------------------
# D  Metric ↔ PCA correlation
# --------------------------------------------------------------------------

def metric_pca_correlation(raw_rms, cells_dict, out_dir,
                           pca_components=10, seed=42):
    """Fit PCA on per-layer-centered raw_rms (same as fig_b), then Spearman-correlate
       each metric per-token value against PC scores at each layer.

       Per-token metric is one int (settling layer); per-token-per-layer PCA score is
       a continuous value. We correlate per (layer, metric, PC) across all
       (window, token) — n = 100 * 600 = 60,000 per cell.
    """
    n_w, n_l, n_tok, H = raw_rms.shape
    print(f"[D] PCA fit setup: n_samples = {n_w*n_l*n_tok:,} × {H}")

    # Per-layer mean centering (same as fig_b)
    print("[D] per-layer mean centering ...")
    layer_means = raw_rms.mean(axis=(0, 2), keepdims=True)  # (1, n_l, 1, H)
    h_centered = raw_rms - layer_means
    flat = h_centered.reshape(-1, H)

    # PCA fit
    print(f"[D] fitting PCA(n_components={pca_components}) on {flat.shape[0]:,} samples ...")
    if flat.shape[0] * H * 4 > 6e10:  # > 60GB, use IncrementalPCA
        pca = IncrementalPCA(n_components=pca_components, batch_size=100000)
        pca.fit(flat)
    else:
        pca = PCA(n_components=pca_components, random_state=seed)
        pca.fit(flat)
    expl = pca.explained_variance_ratio_
    cumv = np.cumsum(expl)
    print(f"[D] explained variance: PC1..PC{pca_components} = {expl}")
    print(f"[D] cumulative: {cumv}")
    (out_dir / "pca_explained_variance.json").write_text(
        json.dumps({"explained_variance_ratio": expl.tolist(),
                     "cumulative": cumv.tolist()}, indent=2)
    )

    # Project
    print("[D] projecting all samples ...")
    proj = pca.transform(flat).astype(np.float32)
    proj = proj.reshape(n_w, n_l, n_tok, pca_components)

    # Correlations
    records = []
    # Use only PC1, PC2 in heatmap; save first 5 in CSV
    n_pc_corr = min(5, pca_components)
    for cell_name in CELL_NAMES:
        cell_vals = cells_dict[cell_name]  # (n_w, n_tok) int32
        # Mask out invalid: -1 (pad) or values >= N_LAYERS (never-settled convention)
        valid_mask = (cell_vals >= 0) & (cell_vals < N_LAYERS + 1)
        if int(valid_mask.sum()) < 1000:
            print(f"[D]   {cell_name}: too few valid ({int(valid_mask.sum())}), skip")
            continue
        cell_flat = cell_vals[valid_mask].astype(np.float32)  # (n_valid,)
        for ell in range(n_l):
            for k in range(n_pc_corr):
                pc_layer = proj[:, ell, :, k]  # (n_w, n_tok)
                pc_flat = pc_layer[valid_mask]
                r, p = spearmanr(cell_flat, pc_flat)
                records.append({
                    "layer": ell, "metric": cell_name, "PC": k + 1,
                    "spearman_r": r, "p": p, "n": int(valid_mask.sum()),
                })
        print(f"[D]   {cell_name}: corr done (n_valid={int(valid_mask.sum())})")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metric_pca_corr.csv", index=False)
    print(f"[D] saved metric_pca_corr.csv")

    # Heatmap PC1 and PC2
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, pc in zip(axes, (1, 2)):
        sub = df[df["PC"] == pc].pivot(index="metric", columns="layer", values="spearman_r")
        sub = sub.reindex(CELL_NAMES)
        sns.heatmap(sub, ax=ax, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
                    cbar_kws={"label": f"Spearman r (metric vs PC{pc})"})
        ax.set_title(f"(D) Metric vs PC{pc} score, Spearman r per layer")
        ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Settling cell")
        ax.axvline(L_STAR + 0.5, color="k", lw=0.6, alpha=0.5)
    fig.suptitle("(D) 17 settling cells × 32 layers × {PC1, PC2}\n"
                 "Strong |r| at some (metric, layer) cell = that PC at that layer encodes that metric's signal",
                 fontsize=11)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"metric_pca_corr_heatmap.{ext}", dpi=150)
    plt.close(fig)
    print(f"[D] heatmap saved")
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1-parquet", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet"))
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/analysis_BD"))
    p.add_argument("--max-n-per-class", type=int, default=8000,
                   help="cap per-class samples for LR probing (CPU budget)")
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--skip-B", action="store_true")
    p.add_argument("--skip-D", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print("[load] meta + pos labels + tier3 ...")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    raw_rms, wids = load_tier3(args.tier3)
    print(f"[load]   raw_rms shape {raw_rms.shape}, dtype {raw_rms.dtype}")
    n_w, n_l, n_tok, H = raw_rms.shape

    print("[load] context map (tier3 grid, stride 10) ...")
    ctx_full = build_context_map(meta, wids, pos_labels, n_tokens=6000)
    ctx_tier3 = ctx_full[:, ::10][:, :n_tok]
    print(f"[load]   ctx_tier3 shape {ctx_tier3.shape}")

    if not args.skip_B:
        print("\n=== B: per-layer probing ===")
        per_layer_probing(raw_rms, ctx_tier3, args.out_dir,
                          max_n_per_class=args.max_n_per_class)
    else:
        print("[skip] B")

    if not args.skip_D:
        print("\n=== D: metric ↔ PCA correlation ===")
        print("[load] tier1 cells subset to tier3 grid ...")
        cells = load_tier1_cells_for_windows(args.tier1_parquet, wids, n_tok, stride=10)
        for k, v in cells.items():
            print(f"   {k}: shape {v.shape}, range [{v.min()}, {v.max()}]")
        metric_pca_correlation(raw_rms, cells, args.out_dir,
                               pca_components=args.pca_components)
    else:
        print("[skip] D")

    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
