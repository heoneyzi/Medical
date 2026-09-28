"""Phase 1.3 — chr17 replication of chr22 v2 headline findings.

Mirrors `13_analyze_chr22_v2.py` for chr17. Reuses the chr22-frozen γ_v2 thresholds
(no re-calibration on chr17 — that's the whole point of "transferability").

Outputs (under --out-dir, default: results/chr17_replication/):
  per_cell_summary_chr17.csv       17 cells × {mean, median, std, never%}
  splice_vs_intron_chr17.csv       17 cells × {n, Cohen d, p}
  canonical_vs_noncanonical_chr17.csv
  per_context_distributions_chr17.csv
  retention_table.csv              chr22_d vs chr17_d × 17 cells (paper §3.1)
  retention_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}

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
                   default=Path("/path/to/TDiG/data/cache/chr17/tier1_settling_v2.parquet"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr17/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr17_position_labels.npy"))
    p.add_argument("--chr22-svi", type=Path,
                   default=Path("/path/to/TDiG/results/splice_vs_intron.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/chr17_replication"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] chr17 tier1 + labels ...")
    df = pd.read_parquet(args.tier1)
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    print(f"[load]   {len(df):,} windows; pos_labels shape {pos_labels.shape}")

    # Build flat (cell, token) arrays + context labels
    print("[expand] expanding tier1 lists to per-token arrays ...")
    per_cell = {c: [] for c in CELL_NAMES}
    contexts = []
    for _, row in df.iterrows():
        wid = int(row["window_idx"]); start = int(row["start"]); T = int(row["T"])
        # context for each token
        pos_range = np.clip(start + np.arange(T), 0, len(pos_labels) - 1)
        ctx = pos_labels[pos_range]
        contexts.append(ctx)
        for c in CELL_NAMES:
            if c in row:
                cell_arr = np.asarray(row[c], dtype=np.int32)[:T]
                per_cell[c].append(cell_arr)
            else:
                per_cell[c].append(np.full(T, -1, dtype=np.int32))
    all_ctx = np.concatenate(contexts)
    for c in CELL_NAMES:
        per_cell[c] = np.concatenate(per_cell[c])
    print(f"[expand]   total tokens: {len(all_ctx):,}")

    # Per-cell summary
    summary = []
    for c in CELL_NAMES:
        arr = per_cell[c]
        valid = arr[(arr >= 0) & (arr < 33)]
        summary.append({
            "cell": c, "n_valid": int(len(valid)),
            "n_total": int(len(arr)),
            "never_settled_pct": float((arr == -1).mean() * 100),
            "mean": float(valid.mean()) if len(valid) > 0 else np.nan,
            "median": float(np.median(valid)) if len(valid) > 0 else np.nan,
            "std": float(valid.std()) if len(valid) > 0 else np.nan,
        })
    pd.DataFrame(summary).to_csv(args.out_dir / "per_cell_summary_chr17.csv", index=False)
    print(f"[A] saved per_cell_summary_chr17.csv")

    # Splice vs intron
    donor_mask = all_ctx == 5
    intron_mask = all_ctx == 1
    print(f"[B] donor n={int(donor_mask.sum()):,}, intron n={int(intron_mask.sum()):,}")
    svi_records = []
    for c in CELL_NAMES:
        arr = per_cell[c]
        donor_vals = arr[donor_mask]
        intron_vals = arr[intron_mask]
        donor_vals = donor_vals[(donor_vals >= 0) & (donor_vals < 33)]
        intron_vals = intron_vals[(intron_vals >= 0) & (intron_vals < 33)]
        if len(donor_vals) < 30 or len(intron_vals) < 30:
            svi_records.append({"cell": c, "donor_n": len(donor_vals),
                                  "intron_n": len(intron_vals),
                                  "donor_mean": np.nan, "intron_mean": np.nan,
                                  "cohen_d": np.nan, "p": np.nan})
            continue
        d = cohens_d(donor_vals, intron_vals)
        try:
            _, pp = mannwhitneyu(donor_vals, intron_vals, alternative="two-sided")
        except Exception:
            pp = np.nan
        svi_records.append({
            "cell": c, "donor_n": int(len(donor_vals)),
            "intron_n": int(len(intron_vals)),
            "donor_mean": float(donor_vals.mean()),
            "intron_mean": float(intron_vals.mean()),
            "cohen_d": d, "p": float(pp),
        })
    svi_df = pd.DataFrame(svi_records)
    svi_df.to_csv(args.out_dir / "splice_vs_intron_chr17.csv", index=False)
    print(f"[B] saved splice_vs_intron_chr17.csv")

    # Per-context distributions
    ctx_records = []
    for ctx_id, ctx_name in POS_CODEBOOK.items():
        m = all_ctx == ctx_id
        for c in CELL_NAMES:
            arr = per_cell[c][m]
            valid = arr[(arr >= 0) & (arr < 33)]
            if len(valid) < 30:
                continue
            ctx_records.append({
                "context": ctx_name, "cell": c,
                "n": int(len(valid)),
                "mean": float(valid.mean()), "median": float(np.median(valid)),
                "std": float(valid.std()),
            })
    pd.DataFrame(ctx_records).to_csv(args.out_dir / "per_context_distributions_chr17.csv", index=False)
    print(f"[C] saved per_context_distributions_chr17.csv")

    # Retention table (vs chr22)
    if args.chr22_svi.exists():
        chr22 = pd.read_csv(args.chr22_svi)
        # chr22 column name may be 'cohens_d_donor_minus_intron' (from 13_analyze)
        d_col = next((c for c in ("cohen_d", "cohens_d_donor_minus_intron",
                                   "cohens_d", "d") if c in chr22.columns), None)
        cell_col = "cell" if "cell" in chr22.columns else chr22.columns[0]
        ret_records = []
        for r in svi_records:
            c17_d = r["cohen_d"]
            row22 = chr22[chr22[cell_col] == r["cell"]]
            c22_d = float(row22[d_col].iloc[0]) if not row22.empty and d_col else np.nan
            ret_pct = (c17_d / c22_d * 100) if (c22_d and not np.isnan(c22_d) and abs(c22_d) > 1e-6) else np.nan
            ret_records.append({"cell": r["cell"], "chr22_d": c22_d, "chr17_d": c17_d,
                                  "retention_pct": ret_pct,
                                  "sign_preserved": (np.sign(c22_d) == np.sign(c17_d)) if not (np.isnan(c22_d) or np.isnan(c17_d)) else None})
        ret_df = pd.DataFrame(ret_records)
        ret_df.to_csv(args.out_dir / "retention_table.csv", index=False)
        print(f"\n[D] retention table:")
        print(ret_df.to_string(index=False))

        summary_json = {
            "n_cells_with_chr22_comparison": int(ret_df.chr22_d.notna().sum()),
            "median_retention_pct": float(np.nanmedian(ret_df.retention_pct)),
            "sign_preserved_count": int(ret_df.sign_preserved.sum()) if "sign_preserved" in ret_df else None,
            "spearman_rho_d22_d17": None,
        }
        try:
            from scipy.stats import spearmanr
            valid = ret_df.dropna(subset=["chr22_d", "chr17_d"])
            if len(valid) >= 5:
                rho, _ = spearmanr(valid.chr22_d, valid.chr17_d)
                summary_json["spearman_rho_d22_d17"] = float(rho)
        except Exception as e:
            print(f"  spearman failed: {e}")
        (args.out_dir / "retention_summary.json").write_text(json.dumps(summary_json, indent=2))
        print(f"[D] summary: {summary_json}")
    else:
        print(f"[D] chr22 svi file not found at {args.chr22_svi}, skip retention")

    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
