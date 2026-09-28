"""Phase 2.3 — Window-level bootstrap 95% CIs for chr22 v2 headline d-values.

Reads tier1 settling cells, resamples WINDOWS (not tokens, to respect window
independence assumption) with replacement N times, recomputes Cohen d
splice-vs-intron for each (cell, bootstrap iteration). Reports 95% CI.

This addresses the paper credibility concern: "d=-1.05 is a single point
estimate. What's the noise floor?"

Output: results/bootstrap_chr22_ci/
  bootstrap_d_ci.csv           (cell, mean_d, median_d, ci_low, ci_high, std, n_iter)
  bootstrap_distributions.png  histograms per cell
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/bootstrap_chr22_ci"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print(f"[load] chr22 tier1 + labels ...")
    df = pd.read_parquet(args.tier1)
    pos_labels = np.load(args.pos_labels)
    print(f"[load]   {len(df):,} windows")

    # Pre-expand per-cell per-window flat arrays
    print("[expand] window-indexed token arrays ...")
    window_cells = {c: [] for c in CELL_NAMES}
    window_ctx = []
    for _, row in df.iterrows():
        start = int(row["start"]); T = int(row["T"])
        ctx = pos_labels[np.clip(start + np.arange(T), 0, len(pos_labels) - 1)]
        window_ctx.append(ctx)
        for c in CELL_NAMES:
            if c in row:
                window_cells[c].append(np.asarray(row[c], dtype=np.int32)[:T])
            else:
                window_cells[c].append(np.full(T, -1, dtype=np.int32))
    N_win = len(df)
    print(f"[expand]   N_win={N_win}")

    rng = np.random.default_rng(args.seed)
    records = []
    print(f"\n[bootstrap] {args.n_iter} iterations ...")
    for it in range(args.n_iter):
        sample_idx = rng.choice(N_win, size=N_win, replace=True)
        # Concatenate sampled windows
        ctx_b = np.concatenate([window_ctx[i] for i in sample_idx])
        donor_mask = ctx_b == 5
        intron_mask = ctx_b == 1
        for c in CELL_NAMES:
            cell_b = np.concatenate([window_cells[c][i] for i in sample_idx])
            donor_vals = cell_b[donor_mask]
            intron_vals = cell_b[intron_mask]
            donor_vals = donor_vals[(donor_vals >= 0) & (donor_vals < 33)]
            intron_vals = intron_vals[(intron_vals >= 0) & (intron_vals < 33)]
            d = cohens_d(donor_vals, intron_vals)
            records.append({"iter": it, "cell": c, "d": d})
        if (it + 1) % 25 == 0:
            print(f"  iter {it+1}/{args.n_iter}")

    df_b = pd.DataFrame(records)
    # Summarize
    summary = []
    for c in CELL_NAMES:
        sub = df_b[df_b.cell == c].d.dropna().values
        if len(sub) < 5:
            continue
        summary.append({
            "cell": c, "n_iter": int(len(sub)),
            "mean_d": float(sub.mean()),
            "median_d": float(np.median(sub)),
            "std": float(sub.std()),
            "ci_low": float(np.quantile(sub, 0.025)),
            "ci_high": float(np.quantile(sub, 0.975)),
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.out_dir / "bootstrap_d_ci.csv", index=False)
    print(f"\n[save] bootstrap_d_ci.csv")
    print(sdf.to_string(index=False))

    # Plot distributions, 4 cols × N rows
    cells_with_data = sdf.cell.tolist()
    n_cells = len(cells_with_data)
    ncols = 4; nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    for ax, c in zip(axes.flatten(), cells_with_data):
        sub = df_b[df_b.cell == c].d.dropna()
        ax.hist(sub, bins=30, color="#1f77b4", alpha=0.8)
        row = sdf[sdf.cell == c].iloc[0]
        ax.axvline(row["mean_d"], color="red", lw=1.2)
        ax.axvline(row["ci_low"], color="red", ls="--", lw=0.8)
        ax.axvline(row["ci_high"], color="red", ls="--", lw=0.8)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_title(f"{c}\nd={row['mean_d']:+.3f}  [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
                      fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flatten()[n_cells:]:
        ax.axis("off")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"bootstrap_distributions.{ext}", dpi=150)
    plt.close(fig)
    print("[plot] saved")


if __name__ == "__main__":
    main()
