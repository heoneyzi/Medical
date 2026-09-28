"""P2-SNV-2 — per-class population statistics for SNVs (Plan §3.3.3).

Reads variants_features_classed.csv (output of p2_snv_class_join.py).
For each (mc_class, category) pair compute:
  - n
  - median / mean / IQR of argmax_layer (1-indexed)
  - median / mean of max_abs_dD
  - median / mean of evo2_delta_loglik
  - bootstrap 95% CI on each median (n_boot=1000, seed=42)

Plus the global Kruskal–Wallis test on P/LP argmax_layer across the four
substantive classes:  missense / nonsense / canonical_splice / synonymous.

Outputs:
    ~/gDTR/results/p2/p2_snv_per_class.csv
    ~/gDTR/results/p2/p2_snv_per_class.json
    ~/gDTR/results/p2/F_argmax_per_class.{pdf,png}
    ~/gDTR/results/p2/_status.json
    ~/gDTR/results/p2/_done_stats
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
    log = logging.getLogger("p2_snv_stats")
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
    lo = float(np.quantile(meds, 0.025))
    hi = float(np.quantile(meds, 0.975))
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="bootstrap with n_boot=50 for fast smoke check")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_stats"
    if done_marker.exists() and not args.force:
        log.info("p2_snv_stats: _done_stats exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_snv_stats", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        from scipy.stats import kruskal

        in_csv = out_dir / "variants_features_classed.csv"
        if not in_csv.exists():
            raise FileNotFoundError(f"missing {in_csv} (run p2_snv_class_join.py first)")

        # Group rows by (mc_class, category)
        by = defaultdict(lambda: {
            "argmax_layer": [], "max_abs_dD": [], "evo2_delta_loglik": [],
        })
        all_classes = set()
        with in_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                cls = row.get("mc_class", "MC_unknown")
                cat = row.get("category", "")
                if cat not in ("P_LP", "B_LB", "VUS"):
                    continue
                try:
                    al = int(row["argmax_layer"])
                    max_d = float(row["max_abs_dD"])
                    dll = float(row["evo2_delta_loglik"])
                except (KeyError, ValueError):
                    continue
                by[(cls, cat)]["argmax_layer"].append(al)
                by[(cls, cat)]["max_abs_dD"].append(max_d)
                by[(cls, cat)]["evo2_delta_loglik"].append(dll)
                all_classes.add(cls)

        n_boot = 50 if args.smoke else 1000

        rows_csv = []
        per_class_json = {}
        for (cls, cat), arrs in by.items():
            al = np.asarray(arrs["argmax_layer"], dtype=np.float64)
            max_d = np.asarray(arrs["max_abs_dD"], dtype=np.float64)
            dll = np.asarray(arrs["evo2_delta_loglik"], dtype=np.float64)
            if al.size < 1:
                continue
            iqr_lo = float(np.quantile(al, 0.25))
            iqr_hi = float(np.quantile(al, 0.75))
            med = float(np.median(al))
            mean = float(al.mean())
            ci_lo, ci_hi = median_ci(al, n_boot=n_boot, seed=42)
            md_med = float(np.median(max_d)) if max_d.size else None
            md_mean = float(max_d.mean()) if max_d.size else None
            dll_med = float(np.median(dll)) if dll.size else None
            row = {
                "mc_class": cls,
                "category": cat,
                "n": int(al.size),
                "median_argmax_layer": med,
                "mean_argmax_layer": mean,
                "iqr_argmax_layer": f"[{iqr_lo:.2f},{iqr_hi:.2f}]",
                "median_argmax_ci_lo": ci_lo,
                "median_argmax_ci_hi": ci_hi,
                "median_max_abs_dD": md_med,
                "mean_max_abs_dD": md_mean,
                "median_delta_loglik": dll_med,
            }
            rows_csv.append(row)
            per_class_json[f"{cls}__{cat}"] = row

        # Kruskal–Wallis 4-way on P/LP argmax_layer
        target_classes = ["missense", "nonsense", "canonical_splice", "synonymous"]
        groups = []
        group_names = []
        for c in target_classes:
            g = by.get((c, "P_LP"), {}).get("argmax_layer", [])
            if len(g) >= 2:
                groups.append(np.asarray(g, dtype=np.float64))
                group_names.append(c)
        kw_payload = {"groups": group_names, "n_per_group": [int(g.size) for g in groups]}
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                kw_payload["H_statistic"] = float(stat)
                kw_payload["p_value"] = float(p)
            except Exception as e:
                kw_payload["error"] = str(e)
        else:
            kw_payload["error"] = "fewer than 2 classes with sufficient n"

        # Verdict per Plan §3.3.4
        plp_classes = [c for c in target_classes
                       if (c, "P_LP") in by]
        n_per = {c: len(by[(c, "P_LP")]["argmax_layer"]) for c in plp_classes}
        any_below_15 = any(n_per.get(c, 0) < 15 for c in target_classes)
        all_above_30 = all(n_per.get(c, 0) >= 30 for c in target_classes)
        kw_p = kw_payload.get("p_value", float("nan"))
        if all_above_30 and np.isfinite(kw_p) and kw_p < 0.01:
            verdict = "PASS"
        elif (not any_below_15) and np.isfinite(kw_p) and kw_p < 0.05:
            verdict = "REFRAME"
        else:
            verdict = "REFRAME" if not any_below_15 else "FAIL"

        # ---- Write CSV ----
        csv_path = out_dir / "p2_snv_per_class.csv"
        cols = ["mc_class", "category", "n",
                "median_argmax_layer", "mean_argmax_layer", "iqr_argmax_layer",
                "median_argmax_ci_lo", "median_argmax_ci_hi",
                "median_max_abs_dD", "mean_max_abs_dD", "median_delta_loglik"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_csv:
                w.writerow({k: r.get(k) for k in cols})
        log.info("wrote %s (%d rows)", csv_path, len(rows_csv))

        out = {
            "config": {"n_boot": n_boot, "seed": 42},
            "per_class": rows_csv,
            "kruskal_wallis": kw_payload,
            "verdict": verdict,
        }
        (out_dir / "p2_snv_per_class.json").write_text(json.dumps(out, indent=2))

        # ---- Figure ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order_classes = ["missense", "nonsense", "canonical_splice",
                             "synonymous", "splice_region", "frameshift"]
            cats = ["P_LP", "B_LB"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            for ax, cat in zip(axes, cats):
                box_data = []
                tick_labels = []
                for cls in order_classes:
                    arr = np.asarray(by.get((cls, cat), {}).get("argmax_layer", []),
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
                fig.savefig(out_dir / f"F_argmax_per_class.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_snv_stats",
            "status": verdict,
            "reason": f"KW p={kw_p!r}; n_per_class_PLP={n_per}",
            "metrics": {"kw_p": kw_p, "n_per_class_PLP": n_per},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p2_snv_stats DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_snv_stats FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_snv_stats",
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
