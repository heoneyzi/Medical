#!/usr/bin/env python3
"""Merge coverage-stratified exact scores without cross-dataset aggregation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AXES = ("pds", "mse", "nmae", "fid", "reach", "jac")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--protocol", required=True,
                        choices=["strict-replogle-only", "non-strict-shadow"])
    parser.add_argument("--out-wide", required=True)
    parser.add_argument("--out-long", required=True)
    args = parser.parse_args()
    wide_rows, long_rows = [], []
    for item in args.result:
        dataset, filename = item.split("=", 1)
        frame = pd.read_csv(filename)
        for stratum, group in frame.groupby("stratum", sort=False):
            baseline = group[(group["representation"] == "raw") &
                             (group["method"] == "no_effect") &
                             (group["status"] == "ok")]
            if len(baseline) != 1:
                raise ValueError(f"{dataset}/{stratum}: need one raw/no_effect baseline")
            base = baseline.iloc[0]
            for _, row in group.iterrows():
                record = {**row.to_dict(), "dataset": dataset, "protocol": args.protocol,
                          "metric_implementation": "ArcInstitute/cell-eval2@0.16.0:vcc2026",
                          "official_challenge_score": False,
                          "cross_dataset_aggregate_allowed": False,
                          "cross_stratum_aggregate_allowed": False}
                if row["status"] == "ok":
                    for axis in AXES:
                        delta = float(row[axis]) - float(base[axis])
                        record[f"{axis}_improvement_from_no_effect"] = delta
                        long_rows.append({
                            "dataset": dataset, "stratum": stratum,
                            "representation": row["representation"], "method": row["method"],
                            "metric": axis, "value": float(row[axis]),
                            "no_effect": float(base[axis]),
                            "improvement_from_no_effect": delta, "protocol": args.protocol,
                        })
                wide_rows.append(record)
    wide, long = pd.DataFrame(wide_rows), pd.DataFrame(long_rows)
    Path(args.out_wide).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out_wide, index=False)
    long.to_csv(args.out_long, index=False)
    print(wide[["dataset", "stratum", "representation", "method", "status", "overall"]]
          .to_string(index=False))
    print(f"[coverage-compare] wide={args.out_wide} long={args.out_long}")


if __name__ == "__main__":
    main()
