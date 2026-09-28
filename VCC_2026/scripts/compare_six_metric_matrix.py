#!/usr/bin/env python3
"""Merge exact cell-eval2/vcc2026 matrices and compute change from no-effect."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AXES = {
    "pds": True,
    "mse": True,
    "nmae": True,
    "fid": True,
    "reach": True,
    "jac": True,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", action="append", required=True, metavar="DATASET=CSV")
    p.add_argument("--out-wide", required=True)
    p.add_argument("--out-long", required=True)
    args = p.parse_args()

    wide_rows, long_rows = [], []
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"expected DATASET=CSV, got {item!r}")
        dataset, filename = item.split("=", 1)
        frame = pd.read_csv(filename)
        good = frame[frame["status"] == "ok"].copy()
        baseline_rows = good[(good["representation"] == "raw") &
                             (good["method"] == "no_effect")]
        if len(baseline_rows) != 1:
            raise ValueError(f"{dataset}: need exactly one raw/no_effect baseline")
        baseline = baseline_rows.iloc[0]
        missing = [f"{axis}.mean" for axis in AXES if f"{axis}.mean" not in good]
        if missing:
            raise ValueError(f"{dataset}: full six-axis output missing {missing}")

        for _, row in frame.iterrows():
            record = {
                "dataset": dataset, "representation": row.get("representation"),
                "method": row.get("method"), "status": row.get("status"),
                "error": row.get("error", ""),
                "protocol": "cell-eval2-vcc2026-external-local-anchors-v1",
                "metric_implementation": "ArcInstitute/cell-eval2@0.16.0:vcc2026",
                "official_challenge_score": False,
                "score_scope": "external-dataset-local-anchors",
                "strict_complete_zero_shot": False,
            }
            if row.get("status") == "ok":
                for axis, higher_is_better in AXES.items():
                    col = f"{axis}.mean"
                    value, base = float(row[col]), float(baseline[col])
                    delta = value - base
                    improvement = delta if higher_is_better else -delta
                    record[axis] = value
                    record[f"{axis}.delta_from_no_effect"] = delta
                    record[f"{axis}.improvement_from_no_effect"] = improvement
                    long_rows.append({
                        "dataset": dataset, "representation": row["representation"],
                        "method": row["method"], "metric": axis, "value": value,
                        "no_effect": base, "delta_from_no_effect": delta,
                        "improvement_from_no_effect": improvement,
                        "higher_is_better": higher_is_better,
                        "protocol": record["protocol"],
                    })
                if "overall.mean" not in row:
                    raise ValueError(f"{dataset}: missing official unweighted Overall")
                record["overall"] = float(row["overall.mean"])
            wide_rows.append(record)

    wide, long = pd.DataFrame(wide_rows), pd.DataFrame(long_rows)
    out_wide, out_long = Path(args.out_wide), Path(args.out_long)
    out_wide.parent.mkdir(parents=True, exist_ok=True)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_wide, index=False)
    long.to_csv(out_long, index=False)
    show = ["dataset", "representation", "method", "status"] + [
        name for axis in AXES for name in (axis, f"{axis}.improvement_from_no_effect")]
    print(wide[[c for c in show if c in wide]].to_string(index=False))
    print(f"[vcc2026-six] wide={out_wide} long={out_long}")


if __name__ == "__main__":
    main()
