"""P2-INDEL-3 — per-class stats with frameshift indels included (Plan §3.3.3).

Combines SNV stats from variants_features_classed.csv (output of
p2_snv_class_join.py) with frameshift / inframe indel stats from
variants_features_indel.csv (output of p2_indel_forward.py).

Outputs:
    ~/gDTR/results/p2_indel/p2_indel_per_class.csv
    ~/gDTR/results/p2_indel/p2_indel_per_class.json
    ~/gDTR/results/p2_indel/F_argmax_combined.{pdf,png}
    ~/gDTR/results/p2_indel/_status.json
    ~/gDTR/results/p2_indel/_done_stats
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p2_indel_stats")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def median_ci(arr: np.ndarray, n_boot: int = 1000,
              seed: int = 42) -> tuple[float, float]:
    if arr.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    meds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        meds[i] = float(np.median(arr[idx]))
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def load_arrs(csv_path: Path, mc_field: str = "mc_class") -> dict:
    by = defaultdict(lambda: {"argmax_layer": [], "max_abs_dD": []})
    if not csv_path.exists():
        return by
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            cls = row.get(mc_field, "MC_unknown")
            cat = row.get("category", "")
            if cat not in ("P_LP", "B_LB", "VUS"):
                continue
            try:
                al = int(row["argmax_layer"])
                md = float(row["max_abs_dD"])
            except (KeyError, ValueError):
                continue
            by[(cls, cat)]["argmax_layer"].append(al)
            by[(cls, cat)]["max_abs_dD"].append(md)
    return by


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2_indel"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_stats"
    if done_marker.exists() and not args.force:
        log.info("p2_indel_stats: _done_stats exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_indel_stats", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        from scipy.stats import kruskal

        snv_csv = GDTR_ROOT / "results" / "p2" / "variants_features_classed.csv"
        indel_csv = out_dir / "variants_features_indel.csv"
        snv = load_arrs(snv_csv)
        ind = load_arrs(indel_csv)

        combined = defaultdict(lambda: {"argmax_layer": [], "max_abs_dD": []})
        for d in (snv, ind):
            for k, v in d.items():
                combined[k]["argmax_layer"].extend(v["argmax_layer"])
                combined[k]["max_abs_dD"].extend(v["max_abs_dD"])

        n_boot = 50 if args.smoke else 1000

        rows_csv = []
        for (cls, cat), arrs in combined.items():
            al = np.asarray(arrs["argmax_layer"], dtype=np.float64)
            md = np.asarray(arrs["max_abs_dD"], dtype=np.float64)
            if al.size < 1:
                continue
            ci_lo, ci_hi = median_ci(al, n_boot=n_boot, seed=42)
            rows_csv.append({
                "mc_class": cls,
                "category": cat,
                "n": int(al.size),
                "median_argmax_layer": float(np.median(al)),
                "mean_argmax_layer": float(al.mean()),
                "iqr_argmax_layer": f"[{float(np.quantile(al, 0.25)):.2f},"
                                     f"{float(np.quantile(al, 0.75)):.2f}]",
                "median_argmax_ci_lo": ci_lo,
                "median_argmax_ci_hi": ci_hi,
                "median_max_abs_dD": float(np.median(md)) if md.size else None,
                "mean_max_abs_dD": float(md.mean()) if md.size else None,
            })

        # Kruskal–Wallis 5-way on P/LP across the substantive classes
        target = ["missense", "nonsense", "canonical_splice", "synonymous", "frameshift"]
        groups = []
        names = []
        for c in target:
            g = combined.get((c, "P_LP"), {}).get("argmax_layer", [])
            if len(g) >= 2:
                groups.append(np.asarray(g, dtype=np.float64))
                names.append(c)
        kw_payload = {"groups": names, "n_per_group": [int(g.size) for g in groups]}
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                kw_payload["H_statistic"] = float(stat)
                kw_payload["p_value"] = float(p)
            except Exception as e:
                kw_payload["error"] = str(e)
        else:
            kw_payload["error"] = "fewer than 2 groups with sufficient n"

        # Write CSV
        cols = ["mc_class", "category", "n",
                "median_argmax_layer", "mean_argmax_layer", "iqr_argmax_layer",
                "median_argmax_ci_lo", "median_argmax_ci_hi",
                "median_max_abs_dD", "mean_max_abs_dD"]
        csv_path = out_dir / "p2_indel_per_class.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_csv:
                w.writerow({k: r.get(k) for k in cols})
        log.info("wrote %s (%d rows)", csv_path, len(rows_csv))

        out = {
            "config": {"n_boot": n_boot, "seed": 42},
            "per_class": rows_csv,
            "kruskal_wallis_5way": kw_payload,
        }
        (out_dir / "p2_indel_per_class.json").write_text(json.dumps(out, indent=2))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order_classes = ["missense", "nonsense", "canonical_splice",
                             "synonymous", "frameshift", "inframe_indel"]
            cats = ["P_LP", "B_LB"]
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
            for ax, cat in zip(axes, cats):
                box_data = []
                tick_labels = []
                for cls in order_classes:
                    arr = np.asarray(combined.get((cls, cat), {}).get("argmax_layer", []),
                                      dtype=np.float64)
                    if arr.size:
                        box_data.append(arr)
                        tick_labels.append(f"{cls}\n(n={arr.size})")
                if box_data:
                    ax.boxplot(box_data, labels=tick_labels, showfliers=False)
                ax.set_title(f"argmax_layer per class — {cat}")
                ax.set_ylabel("argmax_layer (1-indexed)")
                plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"F_argmax_combined.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        kw_p = kw_payload.get("p_value", float("nan"))
        verdict = "PASS" if (np.isfinite(kw_p) and kw_p < 0.01) else "REFRAME"

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_stats",
            "status": verdict,
            "reason": f"5-way KW p={kw_p!r}",
            "metrics": {"kw_p": kw_p, "n_classes_with_data": len(names)},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p2_indel_stats DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_indel_stats FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_stats",
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 1,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
