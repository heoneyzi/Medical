"""Pick 100 representative chr22 windows for Tier 3 raw storage.

Runs on SERVER (needs gDTR chr22_windows.tsv). Selection: stratify by
per-window majority context to ensure all 7 context classes are represented.

Output:
    data/subset_window_ids.json
        {"chr22": [window_idx, ...], "selection_seed": 42, "stratification": {...}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_COLS = [
    "n_coding_exon",
    "n_intron",
    "n_5utr",
    "n_3utr",
    "n_splice",
    "n_intergenic",
]


def select_stratified(df: pd.DataFrame, n_total: int, seed: int) -> dict:
    """Pick windows representative across context classes.

    Strategy:
        For each context column, take the top-N windows by that column's count.
        Combine + dedupe. Boundary picks fill to n_total.
    """
    rng = np.random.default_rng(seed)
    per_context = n_total // (len(CONTEXT_COLS) + 1)  # +1 boundary slack
    picks = set()
    contribution = {}

    for col in CONTEXT_COLS:
        top = df.nlargest(per_context, col)["window_idx"].tolist()
        contribution[col] = top
        picks.update(top)

    # Fill remainder with boundary-rich windows (mix of multiple contexts)
    df_b = df.copy()
    df_b["context_diversity"] = (df_b[CONTEXT_COLS] > 0).sum(axis=1)
    boundary_candidates = df_b.nlargest(per_context * 3, "context_diversity")["window_idx"].tolist()
    contribution["boundary_mix"] = []
    for w in boundary_candidates:
        if len(picks) >= n_total:
            break
        if w not in picks:
            picks.add(w)
            contribution["boundary_mix"].append(w)

    # Random fill if still short
    remaining = df[~df["window_idx"].isin(picks)]["window_idx"].tolist()
    rng.shuffle(remaining)
    contribution["random_fill"] = []
    for w in remaining:
        if len(picks) >= n_total:
            break
        picks.add(w)
        contribution["random_fill"].append(int(w))

    return {
        "selected": sorted(int(w) for w in picks),
        "selection_seed": seed,
        "stratification": {k: [int(x) for x in v] for k, v in contribution.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows-tsv-chr22",
        type=Path,
        default=Path("/path/to/gDTR/data/baselines/chr22_windows.tsv"),
    )
    parser.add_argument(
        "--windows-tsv-chr17",
        type=Path,
        default=Path("/path/to/gDTR/data/baselines/chr17_windows.tsv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/path/to/TDiG/data/subset_window_ids.json"),
    )
    args = parser.parse_args()

    out_obj = {"selection_seed": args.seed, "n_windows": args.n_windows}

    # chr22
    if args.windows_tsv_chr22.exists():
        df22 = pd.read_csv(args.windows_tsv_chr22, sep="\t")
        print(f"loaded {len(df22)} chr22 windows")
        r22 = select_stratified(df22, args.n_windows, args.seed)
        out_obj["chr22"] = r22["selected"]
        out_obj["chr22_stratification"] = r22["stratification"]
        out_obj["chr22_source"] = str(args.windows_tsv_chr22)
        print(f"  picked {len(r22['selected'])} chr22 subset windows")
    else:
        print(f"  chr22 windows file not found: {args.windows_tsv_chr22}")

    # chr17 (use a different seed so chr17 picks are not identical patterns to chr22)
    if args.windows_tsv_chr17.exists():
        df17 = pd.read_csv(args.windows_tsv_chr17, sep="\t")
        print(f"loaded {len(df17)} chr17 windows")
        r17 = select_stratified(df17, args.n_windows, args.seed + 1)
        out_obj["chr17"] = r17["selected"]
        out_obj["chr17_stratification"] = r17["stratification"]
        out_obj["chr17_source"] = str(args.windows_tsv_chr17)
        print(f"  picked {len(r17['selected'])} chr17 subset windows")
    else:
        print(f"  chr17 windows file not found: {args.windows_tsv_chr17}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2))
    print(f"\nwrote subset to {args.out}")


if __name__ == "__main__":
    main()
