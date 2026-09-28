"""Phase-V2 analysis on chr22 v2 outputs.

Reads tier1_settling_v2.parquet + chr22_position_labels.npy + chr22_splice_class_labels.npy.
Computes per-cell biological signal: splice donor vs intron, canonical vs non-canonical.

Output: /path/to/TDiG/data/cache/_v2_analysis/
  per_cell_summary.csv          mean / median / std / n per cell
  splice_vs_intron.csv          Cohen's d + p per cell
  canonical_vs_noncanonical.csv same per cell, full chr22 scale
  per_context_distributions.csv 17 cells × 7 contexts heatmap data
  report.json                   summary verdict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 5 or ny < 5:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    return (mx - my) / s if s > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--splice-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_splice_class_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading tier1...")
    df = pd.read_parquet(args.tier1)
    print(f"  rows: {len(df)} (windows)")
    cell_cols = [c for c in df.columns if c not in ("window_idx", "chrom", "start", "end", "T")]
    print(f"  cells: {len(cell_cols)}")

    print(f"[setup] loading per-bp labels (chr22)...")
    pos_labels = np.load(args.pos_labels)  # codebook: 0 intergenic ... 6 splice_acceptor
    splice_labels = np.load(args.splice_labels)
    print(f"  pos_labels: {pos_labels.shape}, splice_labels: {splice_labels.shape}")

    # Codebooks
    pos_codebook = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
                    5: "splice_donor", 6: "splice_acceptor"}
    splice_codebook = {0: "not_splice", 1: "canonical_GT_AG_donor", 2: "canonical_GT_AG_acceptor",
                       3: "canonical_GC_AG_donor", 4: "canonical_GC_AG_acceptor",
                       5: "non_canonical_donor", 6: "non_canonical_acceptor"}

    # ─── Flatten to per-position records ─────────────────────────────────
    # df row = one window, each cell column is a list of T settling depths
    print(f"[flatten] per-position records (this may take 1-2 min)...")
    records = []
    for _, row in df.iterrows():
        start = int(row["start"])
        T = int(row["T"])
        positions = np.arange(start, start + T)
        # filter to within chr22
        valid_mask = positions < len(pos_labels)
        positions_valid = positions[valid_mask]
        n_valid = len(positions_valid)

        rec_window = {
            "window_idx": int(row["window_idx"]),
            "position": positions_valid,
            "pos_label": pos_labels[positions_valid],
            "splice_label": splice_labels[positions_valid],
        }
        for cell in cell_cols:
            v = np.array(row[cell], dtype=np.int32)[:n_valid]
            rec_window[cell] = v
        records.append(rec_window)

    # Concatenate into one big DataFrame (per-position)
    print(f"[flatten] concatenating...")
    big = {
        "window_idx": np.concatenate([r["window_idx"] * np.ones(len(r["position"]), dtype=np.int32) for r in records]),
        "position": np.concatenate([r["position"] for r in records]),
        "pos_label": np.concatenate([r["pos_label"] for r in records]),
        "splice_label": np.concatenate([r["splice_label"] for r in records]),
    }
    for cell in cell_cols:
        big[cell] = np.concatenate([r[cell] for r in records])
    big = pd.DataFrame(big)
    print(f"  per-position records: {len(big):,}")

    # ─── Per-cell summary ────────────────────────────────────────────────
    print(f"[summary] per-cell stats...")
    summary_rows = []
    for cell in cell_cols:
        v = big[cell].values
        valid = v[v != -1]
        n_neg = (v == -1).sum()
        if len(valid) > 0:
            summary_rows.append({
                "cell": cell, "mean": valid.mean(), "median": float(np.median(valid)),
                "std": valid.std(), "n_total": len(v), "n_settled": len(valid),
                "n_never": int(n_neg), "pct_never": 100 * n_neg / len(v),
            })
        else:
            summary_rows.append({"cell": cell, "n_total": len(v), "n_never": int(n_neg),
                                  "pct_never": 100.0})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "per_cell_summary.csv", index=False)
    print(f"  saved per_cell_summary.csv")

    # ─── Splice donor vs intron (per cell) ───────────────────────────────
    print(f"[splice] donor (pos_label=5) vs intron (pos_label=1)...")
    donor_mask = big["pos_label"] == 5
    intron_mask = big["pos_label"] == 1
    print(f"  donor positions: {donor_mask.sum():,}, intron positions: {intron_mask.sum():,}")
    svi_rows = []
    for cell in cell_cols:
        donor_vals = big.loc[donor_mask, cell].values
        intron_vals = big.loc[intron_mask, cell].values
        donor_valid = donor_vals[donor_vals != -1]
        intron_valid = intron_vals[intron_vals != -1]
        d = cohen_d(donor_valid, intron_valid)
        if len(donor_valid) >= 5 and len(intron_valid) >= 5:
            try:
                _, p_mwu = stats.mannwhitneyu(donor_valid, intron_valid, alternative="two-sided")
            except Exception:
                p_mwu = np.nan
        else:
            p_mwu = np.nan
        svi_rows.append({
            "cell": cell, "n_donor": int(len(donor_valid)), "n_intron": int(len(intron_valid)),
            "donor_mean": float(donor_valid.mean()) if len(donor_valid) > 0 else None,
            "intron_mean": float(intron_valid.mean()) if len(intron_valid) > 0 else None,
            "cohens_d_donor_minus_intron": float(d) if np.isfinite(d) else None,
            "mannwhitney_p": float(p_mwu) if np.isfinite(p_mwu) else None,
        })
    svi = pd.DataFrame(svi_rows)
    svi.to_csv(args.out_dir / "splice_vs_intron.csv", index=False)
    print(f"  saved splice_vs_intron.csv")

    # ─── Canonical (1) vs non-canonical (5) donor ─────────────────────────
    print(f"[canon] canonical donor (splice=1) vs non-canonical (splice=5)...")
    canon_mask = big["splice_label"] == 1
    noncanon_mask = big["splice_label"] == 5
    print(f"  canonical: {canon_mask.sum():,}, non-canonical: {noncanon_mask.sum():,}")
    cnc_rows = []
    for cell in cell_cols:
        c_vals = big.loc[canon_mask, cell].values
        nc_vals = big.loc[noncanon_mask, cell].values
        c_valid = c_vals[c_vals != -1]
        nc_valid = nc_vals[nc_vals != -1]
        d = cohen_d(c_valid, nc_valid)
        if len(c_valid) >= 5 and len(nc_valid) >= 5:
            try:
                _, p_mwu = stats.mannwhitneyu(c_valid, nc_valid, alternative="two-sided")
            except Exception:
                p_mwu = np.nan
        else:
            p_mwu = np.nan
        cnc_rows.append({
            "cell": cell, "n_canonical": int(len(c_valid)), "n_noncanonical": int(len(nc_valid)),
            "canon_mean": float(c_valid.mean()) if len(c_valid) > 0 else None,
            "noncanon_mean": float(nc_valid.mean()) if len(nc_valid) > 0 else None,
            "cohens_d_canon_minus_noncanon": float(d) if np.isfinite(d) else None,
            "mannwhitney_p": float(p_mwu) if np.isfinite(p_mwu) else None,
        })
    cnc = pd.DataFrame(cnc_rows)
    cnc.to_csv(args.out_dir / "canonical_vs_noncanonical.csv", index=False)
    print(f"  saved canonical_vs_noncanonical.csv")

    # ─── Per-context distributions (17 cells × 7 contexts) ────────────────
    print(f"[context] per-context distribution (17 × 7 heatmap)...")
    context_rows = []
    for cell in cell_cols:
        for label_id, label_name in pos_codebook.items():
            mask = big["pos_label"] == label_id
            vals = big.loc[mask, cell].values
            valid = vals[vals != -1]
            if len(valid) > 5:
                context_rows.append({
                    "cell": cell, "context": label_name, "context_id": label_id,
                    "n": int(len(valid)), "mean": float(valid.mean()),
                    "median": float(np.median(valid)), "std": float(valid.std()),
                })
            else:
                context_rows.append({
                    "cell": cell, "context": label_name, "context_id": label_id,
                    "n": int(len(valid)), "mean": None, "median": None, "std": None,
                })
    ctx = pd.DataFrame(context_rows)
    ctx.to_csv(args.out_dir / "per_context_distributions.csv", index=False)
    print(f"  saved per_context_distributions.csv")

    # ─── gDTR baseline cross-check ────────────────────────────────────────
    # gDTR Phase 1.6 reported: chr22 splice_donor mean_c=25.57, intron mean_c=27.82
    # → Cohen's d (donor - intron) ≈ -0.43 for cosine settling (Ref C in our framework)
    print(f"[gDTR cross-check]")
    if "M1_dir_refC" in cell_cols:
        d_refC = svi[svi["cell"] == "M1_dir_refC"].iloc[0]
        gdtr_baseline_d = -0.43
        match = abs(d_refC["cohens_d_donor_minus_intron"] - gdtr_baseline_d) < 0.2
        print(f"  gDTR baseline d = {gdtr_baseline_d}")
        print(f"  TDiG M1_dir_refC d = {d_refC['cohens_d_donor_minus_intron']:.3f}")
        print(f"  match within ±0.2: {match}")
    else:
        match = None

    # ─── Report ───────────────────────────────────────────────────────────
    report = {
        "n_windows": int(len(df)),
        "n_per_token_records": int(len(big)),
        "n_cells": len(cell_cols),
        "donor_n": int(donor_mask.sum()),
        "intron_n": int(intron_mask.sum()),
        "canonical_donor_n": int(canon_mask.sum()),
        "noncanonical_donor_n": int(noncanon_mask.sum()),
        "gDTR_baseline_match": match,
        "splice_vs_intron_pass_cells": [r["cell"] for r in svi_rows
                                          if r.get("cohens_d_donor_minus_intron") is not None
                                          and abs(r["cohens_d_donor_minus_intron"]) >= 0.2
                                          and r.get("mannwhitney_p") is not None
                                          and r["mannwhitney_p"] < 1e-3],
        "canon_vs_noncanon_pass_cells": [r["cell"] for r in cnc_rows
                                            if r.get("cohens_d_canon_minus_noncanon") is not None
                                            and abs(r["cohens_d_canon_minus_noncanon"]) >= 0.2
                                            and r.get("mannwhitney_p") is not None
                                            and r["mannwhitney_p"] < 1e-3],
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
