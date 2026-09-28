"""T1 + T2 + T3: Is PC1 biologically meaningful?

Answers three questions in one pass:

  T1  PC-alone probing — train 1-D LR(PC_k[layer, token] -> splice_donor vs intron)
      for each (layer, k in {1..5}). Compare AUROC curve to full 4096-D probing
      from analysis B. If PC1 AUROC tracks full AUROC closely => PC1 IS the context
      axis. If PC1 AUROC plateaus much lower => PC1 captures something else.

  T2  Per-context PC distribution — for each (layer, context, PC), compute mean +
      bootstrap 95% CI on the per-token PC score. If splice_donor and intron means
      are well-separated relative to within-context std at L27 => PC1 distinguishes
      biology directly, not via downstream metrics.

  T3  Position confounder check — Spearman(PC_k score, token_position_in_window)
      per layer per PC. If r(PC1, position) is high, PC1 is really an
      "autoregressive depth" axis and our metric correlations are confounded.

PCA is refit identically to analysis D (per-layer mean centering, n_components=10).
proj (100, 32, 600, 10) is saved to disk this time for any follow-up.

Outputs under --out-dir (default: results/analysis_T123/):
  pca_proj.npy                (100, 32, 600, 10) fp32, ~76 MB
  pca_explained_variance.json (sanity reload — should match analysis D exactly)
  T1_pc_alone_auroc.csv       (layer, PC, pair, AUROC, n_per_class)
  T1_pc_alone_auroc.png       curves + full-D probing overlay
  T2_pc_context_distribution.csv  (layer, PC, context, mean, std, ci_low, ci_high)
  T2_pc_context_distribution.png  (mean PC1 per layer per context, error bars)
  T3_pc_position_correlation.csv  (layer, PC, spearman_r, p)
  T3_pc_position_correlation.png  curves per PC

Runtime: ~10-15 min CPU on server (most is PCA fit + project).
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
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "coding_exon",
                  "5utr", "3utr", "intron", "intergenic"]
CONTEXT_COLORS = {
    "splice_donor": "#d62728", "splice_acceptor": "#ff7f0e",
    "intron": "#1f77b4", "coding_exon": "#2ca02c",
    "5utr": "#9467bd", "3utr": "#8c564b", "intergenic": "#7f7f7f",
}
PCS_TO_REPORT = (1, 2, 3, 4, 5)


def build_context_map(meta_df, wids, pos_labels, n_tokens=6000):
    ctx = np.zeros((len(wids), n_tokens), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(n_tokens), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def fit_pca_and_project(raw_rms, n_components=10, seed=42):
    """Same recipe as analysis D fig_b / analysis B+D."""
    n_w, n_l, n_tok, H = raw_rms.shape
    print(f"[PCA] per-layer mean centering ({n_w*n_l*n_tok:,} samples)...")
    layer_means = raw_rms.mean(axis=(0, 2), keepdims=True)
    h_centered = raw_rms - layer_means
    flat = h_centered.reshape(-1, H)
    print(f"[PCA] fitting PCA(n_components={n_components})...")
    if flat.shape[0] * H * 4 > 6e10:
        pca = IncrementalPCA(n_components=n_components, batch_size=100000)
        pca.fit(flat)
    else:
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(flat)
    expl = pca.explained_variance_ratio_
    print(f"[PCA] explained variance: {expl}")
    print(f"[PCA] cumulative: {np.cumsum(expl)}")
    print("[PCA] projecting ...")
    proj = pca.transform(flat).astype(np.float32)
    proj = proj.reshape(n_w, n_l, n_tok, n_components)
    return proj, expl


# --------------------------------------------------------------------------
# T1: PC-alone probing
# --------------------------------------------------------------------------

def t1_pc_alone_auroc(proj, ctx_map, out_dir, max_n_per_class=8000, seed=42):
    n_w, n_l, n_tok, n_pc = proj.shape
    flat_ctx = ctx_map.flatten()
    donor_id, intron_id = 5, 1
    donor_idx = np.where(flat_ctx == donor_id)[0]
    intron_idx = np.where(flat_ctx == intron_id)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(donor_idx); rng.shuffle(intron_idx)
    n_take = min(len(donor_idx), len(intron_idx), max_n_per_class)
    print(f"[T1] donor n={len(donor_idx)}, intron n={len(intron_idx)}, balanced n={n_take}")
    use_idx = np.concatenate([donor_idx[:n_take], intron_idx[:n_take]])
    y = np.concatenate([np.ones(n_take), np.zeros(n_take)])
    wi = use_idx // n_tok
    ti = use_idx % n_tok

    records = []
    for k in PCS_TO_REPORT:
        for ell in range(n_l):
            X = proj[wi, ell, ti, k - 1].reshape(-1, 1)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed)
            clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_te, prob)
            records.append({
                "layer": ell, "PC": k, "pair": "splice_donor_vs_intron",
                "n_per_class": n_take, "AUROC": auroc,
            })
        print(f"[T1] PC{k} done; L_best={max(records[-n_l:], key=lambda r: r['AUROC'])['layer']}"
              f" max_AUROC={max(r['AUROC'] for r in records[-n_l:]):.3f}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "T1_pc_alone_auroc.csv", index=False)
    print(f"[T1] saved CSV")

    # Plot — overlay PC1-5 curves + full-D from analysis_BD if available
    fig, ax = plt.subplots(figsize=(11, 6.5))
    full_d_path = out_dir.parent / "analysis_BD" / "per_layer_auroc.csv"
    if full_d_path.exists():
        full = pd.read_csv(full_d_path)
        full = full[full.pair == "splice_donor_vs_intron"]
        ax.plot(full.layer, full.AUROC, color="black", lw=2.5, marker="o", ms=4,
                label="full 4096-D (analysis B)", zorder=5)
    for k in PCS_TO_REPORT:
        g = df[df.PC == k]
        ax.plot(g.layer, g.AUROC, marker="o", ms=3.5, lw=1.5,
                label=f"PC{k} alone")
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("AUROC (splice_donor vs intron)")
    ax.set_title("(T1) PC-alone probing AUROC vs full 4096-D probing\n"
                 "PC1 close to full = PC1 IS the context axis. Gap = PC1 captures other variance.")
    ax.legend(loc="lower center", ncol=3, fontsize=9)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"T1_pc_alone_auroc.{ext}", dpi=150)
    plt.close(fig)
    print("[T1] figure saved")


# --------------------------------------------------------------------------
# T2: Per-context PC distribution + bootstrap CI
# --------------------------------------------------------------------------

def t2_pc_context_distribution(proj, ctx_map, out_dir, n_boot=200, seed=42,
                                max_per_ctx=8000):
    n_w, n_l, n_tok, n_pc = proj.shape
    flat_ctx = ctx_map.flatten()
    records = []
    rng = np.random.default_rng(seed)

    for ctx_id in range(7):
        ctx_name = POS_CODEBOOK[ctx_id]
        mask = (flat_ctx == ctx_id)
        n_tot = int(mask.sum())
        if n_tot < 30:
            continue
        idx = np.where(mask)[0]
        if n_tot > max_per_ctx:
            idx = rng.choice(idx, size=max_per_ctx, replace=False)
        wi = idx // n_tok
        ti = idx % n_tok
        for ell in range(n_l):
            for k in PCS_TO_REPORT:
                vals = proj[wi, ell, ti, k - 1]
                m = float(vals.mean())
                s = float(vals.std())
                # bootstrap CI on mean
                boots = np.empty(n_boot)
                for b in range(n_boot):
                    samp = rng.choice(vals, size=len(vals), replace=True)
                    boots[b] = samp.mean()
                ci_low = float(np.quantile(boots, 0.025))
                ci_high = float(np.quantile(boots, 0.975))
                records.append({
                    "layer": ell, "PC": k, "context": ctx_name,
                    "n": len(vals), "mean": m, "std": s,
                    "ci_low": ci_low, "ci_high": ci_high,
                })
        print(f"[T2] {ctx_name} done (n={len(idx)})")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "T2_pc_context_distribution.csv", index=False)
    print("[T2] saved CSV")

    # Compute separability at L_STAR-2 = L27 (probing peak)
    print("\n=== T2 separability snapshot at L=27 (probing peak) ===")
    snap = df[(df.layer == 27)].pivot_table(index="context", columns="PC", values="mean")
    snap = snap.reindex(CONTEXT_ORDER)
    print(snap)

    # 3 panels: PC1, PC2, PC3 mean per layer per context with error bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    for ax, k in zip(axes, (1, 2, 3)):
        for ctx_name in CONTEXT_ORDER:
            g = df[(df.PC == k) & (df.context == ctx_name)].sort_values("layer")
            if g.empty:
                continue
            ax.plot(g.layer, g["mean"], color=CONTEXT_COLORS[ctx_name],
                    marker="o", ms=3, lw=1.5, label=ctx_name)
            ax.fill_between(g.layer, g.ci_low, g.ci_high,
                             color=CONTEXT_COLORS[ctx_name], alpha=0.15)
        ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.axvline(27, color="red", ls=":", lw=0.7, alpha=0.5,
                   label="L=27 probing peak")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("Layer ℓ"); ax.set_ylabel(f"PC{k} mean ± 95% CI")
        ax.set_title(f"(T2) PC{k} per-context mean trajectory")
        if k == 1:
            ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"T2_pc_context_distribution.{ext}", dpi=150)
    plt.close(fig)
    print("[T2] figure saved")


# --------------------------------------------------------------------------
# T3: PC vs token-position-in-window correlation
# --------------------------------------------------------------------------

def t3_pc_position_correlation(proj, out_dir, stride=10):
    n_w, n_l, n_tok, n_pc = proj.shape
    # token positions in the tier3 subsampled grid
    # tier3 uses every stride-th token from the (window, 6000-token) grid
    positions = np.arange(n_tok) * stride  # 0, 10, 20, ..., (n_tok-1)*stride
    # broadcast to (n_w, n_tok) — same positional offset for every window
    pos_flat = np.tile(positions, n_w)  # (n_w * n_tok,)

    records = []
    for k in PCS_TO_REPORT:
        for ell in range(n_l):
            pc_flat = proj[:, ell, :, k - 1].flatten()
            r, p = spearmanr(pc_flat, pos_flat)
            records.append({"layer": ell, "PC": k, "spearman_r": r, "p": p,
                              "n": pc_flat.size})
        max_r = max(abs(rec["spearman_r"]) for rec in records[-n_l:])
        print(f"[T3] PC{k} max |r| with position = {max_r:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "T3_pc_position_correlation.csv", index=False)
    print("[T3] saved CSV")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for k in PCS_TO_REPORT:
        g = df[df.PC == k]
        ax.plot(g.layer, g.spearman_r, marker="o", ms=3.5, lw=1.5,
                label=f"PC{k}")
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.axhspan(-0.3, 0.3, color="green", alpha=0.05,
                label="|r| < 0.3 (no strong position confound)")
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Spearman r (PC score vs token position in window)")
    ax.set_title("(T3) PC ↔ token-position correlation — position confounder check\n"
                 "|r| close to ±1 = PC IS positional axis. |r| near 0 = independent of position.")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(-1.0, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"T3_pc_position_correlation.{ext}", dpi=150)
    plt.close(fig)
    print("[T3] figure saved")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/analysis_T123"))
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--max-n-per-class", type=int, default=8000)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--save-proj", action="store_true", default=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print("[load] meta + pos labels + tier3 ...")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    with h5py.File(args.tier3, "r") as h5:
        raw_rms = h5["raw_h_ell_rmsnormed"][:].astype(np.float32)
        wids = h5["window_idx"][:]
    print(f"[load] raw_rms shape {raw_rms.shape}")
    n_w, n_l, n_tok, H = raw_rms.shape

    print("[load] context map (tier3 grid, stride 10) ...")
    ctx_full = build_context_map(meta, wids, pos_labels, n_tokens=6000)
    ctx_tier3 = ctx_full[:, ::10][:, :n_tok]

    print("\n=== PCA fit ===")
    proj, expl = fit_pca_and_project(raw_rms, n_components=args.pca_components)
    print(f"[PCA] proj shape {proj.shape}, dtype {proj.dtype}")
    (args.out_dir / "pca_explained_variance.json").write_text(json.dumps({
        "explained_variance_ratio": expl.tolist(),
        "cumulative": np.cumsum(expl).tolist(),
    }, indent=2))
    if args.save_proj:
        np.save(args.out_dir / "pca_proj.npy", proj)
        print(f"[PCA] saved pca_proj.npy ({proj.nbytes/1e6:.1f} MB)")

    print("\n=== T1: PC-alone probing ===")
    t1_pc_alone_auroc(proj, ctx_tier3, args.out_dir,
                       max_n_per_class=args.max_n_per_class)

    print("\n=== T2: per-context PC distribution + bootstrap CI ===")
    t2_pc_context_distribution(proj, ctx_tier3, args.out_dir,
                                n_boot=args.n_boot)

    print("\n=== T3: PC vs token-position correlation ===")
    t3_pc_position_correlation(proj, args.out_dir)

    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
