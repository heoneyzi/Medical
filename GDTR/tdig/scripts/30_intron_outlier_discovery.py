"""Exp H2 — Intron-outlier functional element discovery.

Within ~40M chr22 intron tokens (and chr17 ~95M), some tokens have
splice-like settling profiles (low M3_geo curvature settling). These are
candidates for cryptic functional elements (splice silencers, branchpoints,
intronic enhancers, TFBS).

Method:
  1. Compute per-token settling vector across 5 M3_geo cells + M5_tau_refB
  2. For each cell, rank intron tokens by settling depth
  3. Top 0.1% (smallest settling = "settles like splice") = outliers
  4. Distance to nearest splice site (within ±200bp) = enrichment test
  5. Report (chrom, pos) coordinates of top candidates for external lookup

Outputs: results/intron_outlier/
  outliers_by_cell.csv          per cell: top 100 outlier intron positions
  outlier_splice_distance.csv   distance to nearest splice donor/acceptor
  enrichment_summary.json       fraction of outliers near splice sites vs random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}

KEY_CELLS = ["M3_geo_a0.0_b1.0", "M3_geo_a0.5_b1.0", "M3_geo_a1.0_b1.0",
              "M3_geo_a1.0_b0.5", "M5_tau_refB"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier1_settling_v2.parquet"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/intron_outlier"))
    p.add_argument("--top-pct", type=float, default=0.5,
                   help="top N percentile of intron tokens by settling = outliers")
    p.add_argument("--distance-bp", type=int, default=200,
                   help="window around splice site for 'near-splice' enrichment")
    p.add_argument("--chrom-label", default="chr22")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tier1 + meta + labels (chrom={args.chrom_label}) ...")
    df = pd.read_parquet(args.tier1)
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    print(f"  windows: {len(df):,}; pos_labels: {len(pos_labels):,}")

    # Build splice site position index for nearest-neighbor distance
    splice_donor_pos = np.where(pos_labels == 5)[0]
    splice_acc_pos = np.where(pos_labels == 6)[0]
    splice_all_pos = np.sort(np.concatenate([splice_donor_pos, splice_acc_pos]))
    print(f"  splice donor positions: {len(splice_donor_pos):,}, acceptor: {len(splice_acc_pos):,}, total: {len(splice_all_pos):,}")

    # Expand: gather intron tokens with (chrom_pos, cell_value)
    print("[expand] intron tokens per cell ...")
    intron_records = []
    for _, row in df.iterrows():
        start = int(row["start"]); T = int(row["T"])
        positions = np.clip(start + np.arange(T), 0, len(pos_labels) - 1)
        ctx = pos_labels[positions]
        intron_mask = (ctx == 1)
        if not intron_mask.any():
            continue
        intron_pos_chrom = positions[intron_mask]  # absolute chrom positions
        for c in KEY_CELLS:
            if c not in row:
                continue
            cell_arr = np.asarray(row[c], dtype=np.int32)[:T][intron_mask]
            for pos, val in zip(intron_pos_chrom, cell_arr):
                if val >= 0 and val < 33:
                    intron_records.append({"pos": int(pos), "cell": c, "val": int(val)})
    intron_df = pd.DataFrame(intron_records)
    print(f"  total (pos, cell) entries: {len(intron_df):,}")

    # Per-cell top-outlier identification (lowest settling = like splice)
    print(f"\n[outlier] top {args.top_pct}% per cell ...")
    outlier_records = []
    for c in KEY_CELLS:
        sub = intron_df[intron_df.cell == c]
        if len(sub) < 100:
            continue
        threshold = np.percentile(sub.val, args.top_pct)
        outliers = sub[sub.val <= threshold]
        print(f"  {c}: median={sub.val.median():.0f}, threshold={threshold:.0f}, n_outlier={len(outliers):,}")
        for _, r in outliers.iterrows():
            # distance to nearest splice site
            idx = np.searchsorted(splice_all_pos, r.pos)
            cands = []
            if idx > 0:
                cands.append(abs(int(r.pos) - int(splice_all_pos[idx - 1])))
            if idx < len(splice_all_pos):
                cands.append(abs(int(r.pos) - int(splice_all_pos[idx])))
            dist = min(cands) if cands else -1
            outlier_records.append({
                "cell": c, "pos": int(r.pos),
                "settling_val": int(r.val),
                "dist_to_splice": int(dist),
                "near_splice": int(dist <= args.distance_bp),
            })

    out_df = pd.DataFrame(outlier_records)
    out_df.to_csv(args.out_dir / "outliers_by_cell.csv", index=False)
    print(f"[save] outliers_by_cell.csv ({len(out_df):,} rows)")

    # Enrichment: fraction near splice vs random intron baseline
    print("\n[enrich] enrichment vs random intron baseline ...")
    enrich_records = []
    rng = np.random.default_rng(42)
    for c in KEY_CELLS:
        outl = out_df[out_df.cell == c]
        if len(outl) < 10:
            continue
        # Random intron positions matched in count for baseline
        cell_intron = intron_df[intron_df.cell == c]
        n_match = min(len(outl), len(cell_intron))
        rand_pos = rng.choice(cell_intron.pos.values, size=n_match, replace=False)
        rand_near = 0
        for p in rand_pos:
            idx = np.searchsorted(splice_all_pos, p)
            cands = []
            if idx > 0:
                cands.append(abs(int(p) - int(splice_all_pos[idx - 1])))
            if idx < len(splice_all_pos):
                cands.append(abs(int(p) - int(splice_all_pos[idx])))
            dist = min(cands) if cands else 1e9
            if dist <= args.distance_bp:
                rand_near += 1
        outl_near_frac = float(outl.near_splice.mean())
        rand_near_frac = rand_near / n_match
        enrich_records.append({
            "cell": c, "n_outlier": len(outl),
            "outlier_near_splice_frac": outl_near_frac,
            "random_near_splice_frac": rand_near_frac,
            "enrichment_fold": outl_near_frac / max(rand_near_frac, 1e-9),
        })
        print(f"  {c}: outlier {outl_near_frac*100:.1f}% near splice "
              f"vs random {rand_near_frac*100:.1f}% — fold {outl_near_frac / max(rand_near_frac, 1e-9):.2f}×")

    pd.DataFrame(enrich_records).to_csv(args.out_dir / "enrichment_summary.csv", index=False)

    summary = {
        "n_intron_tokens_total": len(intron_df) // len(KEY_CELLS),
        "n_splice_sites": int(len(splice_all_pos)),
        "top_pct": args.top_pct,
        "distance_bp": args.distance_bp,
        "per_cell_enrichment": enrich_records,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
