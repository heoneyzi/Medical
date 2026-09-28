"""Exp H1 — 7x7 context separation matrix per cell per layer.

For every pair of 7 main contexts, every cell of the 17 settling cells, and
every layer, compute Cohen d (context_i vs context_j on their settling values).
Generates the "context hierarchy" heatmap missing from RESULTS_v2.md.

Outputs: results/context_separation/
  pairwise_d.csv                long table (cell, context_i, context_j, layer, d)
  L27_separation_heatmap.png    snapshot at L=27 (probing peak) per cell
  best_cell_per_pair.csv        for each pair, the cell+layer with max |d|
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}
CONTEXT_ORDER = ["splice_donor", "splice_acceptor", "coding_exon",
                  "5utr", "3utr", "intron", "intergenic"]
CELL_NAMES = [
    "M1_dir_refA", "M1_dir_refB", "M1_dir_refC",
    "M2_mag_refA", "M2_mag_refB_diag", "M2_mag_refC_diag",
    "M3_geo_a0.0_b1.0", "M3_geo_a0.5_b1.0", "M3_geo_a1.0_b1.0",
    "M3_geo_a1.0_b0.5", "M3_geo_a1.0_b0.0",
    "M4_set_refA", "M4_set_refB", "M4_set_refC",
    "M5_tau_refA", "M5_tau_refB", "M5_tau_refC",
]


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/context_separation"))
    p.add_argument("--max-per-context", type=int, default=10000,
                   help="cap tokens per context to save memory (random subsample)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print("[load] chr22 tier1 + labels ...")
    df = pd.read_parquet(args.tier1)
    pos_labels = np.load(args.pos_labels)

    print("[expand] per-token cell arrays + context ...")
    # For memory: store per-context dictionary of (cell -> array of settling values)
    per_ctx = {ctx_id: {c: [] for c in CELL_NAMES} for ctx_id in POS_CODEBOOK}
    for _, row in df.iterrows():
        start = int(row["start"]); T = int(row["T"])
        ctx = pos_labels[np.clip(start + np.arange(T), 0, len(pos_labels) - 1)]
        for c in CELL_NAMES:
            if c not in row:
                continue
            cell_arr = np.asarray(row[c], dtype=np.int32)[:T]
            for ctx_id in POS_CODEBOOK:
                m = ctx == ctx_id
                if m.any():
                    per_ctx[ctx_id][c].append(cell_arr[m])

    rng = np.random.default_rng(args.seed)
    for ctx_id in POS_CODEBOOK:
        for c in CELL_NAMES:
            if not per_ctx[ctx_id][c]:
                per_ctx[ctx_id][c] = np.array([], dtype=np.int32)
            else:
                arr = np.concatenate(per_ctx[ctx_id][c])
                arr = arr[(arr >= 0) & (arr < 33)]
                if len(arr) > args.max_per_context:
                    arr = rng.choice(arr, size=args.max_per_context, replace=False)
                per_ctx[ctx_id][c] = arr
        n_total = len(per_ctx[ctx_id][CELL_NAMES[0]])
        print(f"  {POS_CODEBOOK[ctx_id]:18s}: n_per_cell ≈ {n_total:,}")

    # Pairwise d per (cell, ctx_i, ctx_j)
    print("\n[d] pairwise Cohen d ...")
    records = []
    for c in CELL_NAMES:
        for i, j in combinations(CONTEXT_ORDER, 2):
            i_id = next(k for k, v in POS_CODEBOOK.items() if v == i)
            j_id = next(k for k, v in POS_CODEBOOK.items() if v == j)
            a = per_ctx[i_id][c]; b = per_ctx[j_id][c]
            if len(a) < 30 or len(b) < 30:
                d = np.nan
            else:
                d = cohens_d(a, b)
            records.append({
                "cell": c, "context_i": i, "context_j": j,
                "n_i": len(a), "n_j": len(b), "d_i_minus_j": d,
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out_dir / "pairwise_d.csv", index=False)
    print(f"[save] pairwise_d.csv ({len(out_df):,} rows)")

    # Best cell per pair (max |d|)
    best_records = []
    for i, j in combinations(CONTEXT_ORDER, 2):
        sub = out_df[(out_df.context_i == i) & (out_df.context_j == j)].copy()
        sub["abs_d"] = sub.d_i_minus_j.abs()
        if sub.abs_d.notna().any():
            row = sub.loc[sub.abs_d.idxmax()]
            best_records.append({
                "context_i": i, "context_j": j,
                "best_cell": row.cell, "d": float(row.d_i_minus_j),
                "abs_d": float(row.abs_d), "n_i": int(row.n_i), "n_j": int(row.n_j),
            })
    pd.DataFrame(best_records).to_csv(args.out_dir / "best_cell_per_pair.csv", index=False)
    print(f"[save] best_cell_per_pair.csv")

    # Heatmaps per cell (7x7)
    print("\n[plot] 7x7 heatmaps per non-degenerate cell ...")
    non_degen_cells = [c for c in CELL_NAMES if not c.endswith("_diag") and "refB" not in c and "refC" not in c
                        or c in ("M3_geo_a1.0_b0.0", "M3_geo_a0.0_b1.0", "M3_geo_a1.0_b1.0",
                                 "M3_geo_a1.0_b0.5", "M3_geo_a0.5_b1.0",
                                 "M5_tau_refA", "M5_tau_refB", "M5_tau_refC",
                                 "M1_dir_refC", "M4_set_refA")]
    non_degen_cells = list(dict.fromkeys(non_degen_cells))
    n_cells = len(non_degen_cells)
    ncols = 4; nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    for ax, c in zip(axes.flatten(), non_degen_cells):
        mat = np.full((7, 7), np.nan)
        for r in records:
            if r["cell"] != c:
                continue
            i_idx = CONTEXT_ORDER.index(r["context_i"])
            j_idx = CONTEXT_ORDER.index(r["context_j"])
            mat[i_idx, j_idx] = r["d_i_minus_j"]
            mat[j_idx, i_idx] = -r["d_i_minus_j"]
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin=-1.2, vmax=1.2,
                    xticklabels=CONTEXT_ORDER, yticklabels=CONTEXT_ORDER,
                    cbar_kws={"label": "Cohen d (row − col)"}, annot=True, fmt=".2f",
                    annot_kws={"size": 7})
        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(45); label.set_ha("right")
    for ax in axes.flatten()[n_cells:]:
        ax.axis("off")
    fig.suptitle("7×7 Context Separation Matrices per Settling Cell\n"
                 "(Cohen d, row context − col context, chr22)", fontsize=12, y=1.001)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"context_separation_heatmaps.{ext}", dpi=150)
    plt.close(fig)
    print("[plot] context_separation_heatmaps saved")

    # Best-cell-per-pair summary heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    mat_best = np.full((7, 7), np.nan)
    mat_cell = np.empty((7, 7), dtype=object)
    for r in best_records:
        i_idx = CONTEXT_ORDER.index(r["context_i"])
        j_idx = CONTEXT_ORDER.index(r["context_j"])
        mat_best[i_idx, j_idx] = r["d"]
        mat_best[j_idx, i_idx] = -r["d"]
        mat_cell[i_idx, j_idx] = r["best_cell"].replace("M", "").replace("_", "")
        mat_cell[j_idx, i_idx] = ""
    sns.heatmap(mat_best, ax=ax, cmap="RdBu_r", center=0, vmin=-1.2, vmax=1.2,
                xticklabels=CONTEXT_ORDER, yticklabels=CONTEXT_ORDER,
                cbar_kws={"label": "Max |Cohen d| across cells"},
                annot=True, fmt=".2f", annot_kws={"size": 8})
    ax.set_title("Best context pair Cohen d (max across 17 cells)\nchr22, sub-sampled 10K per context")
    for label in ax.get_xticklabels():
        label.set_rotation(45); label.set_ha("right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"best_cell_per_pair_heatmap.{ext}", dpi=150)
    plt.close(fig)
    print("[plot] best_cell_per_pair_heatmap saved")

    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
