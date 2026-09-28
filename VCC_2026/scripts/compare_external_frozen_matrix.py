#!/usr/bin/env python3
"""Merge external-FM matrices without inventing a cross-dataset score."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AXES = ("pds", "mse", "nmae", "fid", "reach", "jac")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--protocol", choices=["strict-replogle-only", "non-strict-shadow"], required=True)
    parser.add_argument("--out-wide", required=True)
    parser.add_argument("--out-long", required=True)
    args = parser.parse_args()
    wide_rows, long_rows = [], []
    for item in args.result:
        dataset, filename = item.split("=", 1)
        frame = pd.read_csv(filename)
        baseline = frame[(frame["representation"] == "raw") &
                         (frame["method"] == "no_effect") &
                         (frame["status"] == "ok")]
        if len(baseline) != 1:
            raise ValueError(f"{dataset}: expected one successful raw/no_effect baseline")
        baseline = baseline.iloc[0]
        for _, row in frame.iterrows():
            record = {
                "dataset": dataset, "representation": row.get("representation"),
                "method": row.get("method"), "status": row.get("status"),
                "error": row.get("error", ""), "protocol": args.protocol,
                "metric_implementation": "ArcInstitute/cell-eval2@0.16.0:vcc2026",
                "official_challenge_score": False,
                "score_scope": "external-dataset-local-anchors",
                "cross_dataset_aggregate_allowed": False,
                "explicit_response_source": (
                    "Replogle-2022-only" if args.protocol == "strict-replogle-only"
                    else "same-evaluation-dataset-source-contexts"),
            }
            if row.get("status") == "ok":
                for axis in AXES:
                    column = f"{axis}.mean"
                    if column not in frame.columns:
                        raise ValueError(f"{dataset}: missing {column}")
                    value, base = float(row[column]), float(baseline[column])
                    record[axis] = value
                    record[f"{axis}.improvement_from_no_effect"] = value - base
                    long_rows.append({
                        "dataset": dataset, "representation": row["representation"],
                        "method": row["method"], "metric": axis, "value": value,
                        "no_effect": base, "improvement_from_no_effect": value - base,
                        "protocol": args.protocol,
                    })
                record["overall"] = float(row["overall.mean"])
            wide_rows.append(record)
    wide, long = pd.DataFrame(wide_rows), pd.DataFrame(long_rows)
    Path(args.out_wide).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out_wide, index=False)
    long.to_csv(args.out_long, index=False)
    print(wide[["dataset", "representation", "method", "status", "overall"]].to_string(index=False))
    print(f"[external-compare] wide={args.out_wide} long={args.out_long}")


if __name__ == "__main__":
    main()
