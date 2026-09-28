#!/usr/bin/env python3
"""Merge strict cell-eval2/vcc2026 outputs without cross-dataset averaging."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", action="append", required=True, metavar="DATASET=CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    frames = []
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"expected DATASET=CSV, got {item!r}")
        name, path = item.split("=", 1)
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", name)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Metrics from different biological modalities are reported side by side,
    # never averaged into a fictitious cross-dataset leaderboard score.
    merged["cross_dataset_aggregate_allowed"] = False
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    show = [c for c in ["dataset", "method", "pds", "mse", "nmae",
                        "fid", "reach", "jac", "overall"] if c in merged]
    print(merged[show].to_string(index=False))


if __name__ == "__main__":
    main()
