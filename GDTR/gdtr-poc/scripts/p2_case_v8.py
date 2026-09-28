"""P2-CASE — representative variant traces (Plan §3.3.3).

Replaces the n=3 anecdote in 44_t13_case_studies.py with a per-class
representative-trace illustration. Picks 5 representative variants — one per
class — by selecting the variant whose argmax_layer is closest to the
per-class P/LP median. Ties broken by max_abs_dD closest to the per-class
median.

Plots, for each picked representative:
  - per-layer ΔD_cos trace from the cached SNV / indel features CSV
  - overlaid on the per-class median trace

Outputs:
    ~/gDTR/results/p2_case/case_studies_v8.json
    ~/gDTR/results/p2_case/F5_mechanism_cases_v3.{pdf,png}
    ~/gDTR/results/p2_case/_status.json
    ~/gDTR/results/p2_case/_done
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
N_LAYERS = 32

TARGET_CLASSES = ["missense", "nonsense", "canonical_splice", "synonymous", "frameshift"]


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p2_case_v8")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_vec_cos(row: dict) -> np.ndarray:
    return np.array([float(row.get(f"dD_cos_{l}", "nan")) for l in range(N_LAYERS)],
                    dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2_case"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p2_case_v8: _done exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_case", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        snv_csv = GDTR_ROOT / "results" / "p2" / "variants_features_classed.csv"
        indel_csv = GDTR_ROOT / "results" / "p2_indel" / "variants_features_indel.csv"
        snv_rows = load_rows(snv_csv)
        ind_rows = load_rows(indel_csv)
        all_rows = snv_rows + ind_rows
        log.info("loaded %d SNV + %d indel rows", len(snv_rows), len(ind_rows))

        # Group P/LP rows by mc_class
        by_class: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            if row.get("category") != "P_LP":
                continue
            cls = row.get("mc_class", "MC_unknown")
            try:
                _ = int(row["argmax_layer"])
            except (KeyError, ValueError):
                continue
            by_class[cls].append(row)

        # For each class, compute median argmax_layer and max_abs_dD; pick
        # the variant whose argmax_layer is closest to the median, then break
        # ties by |max_abs_dD - median(max_abs_dD)|.
        picks: list[dict] = []
        per_class_trace: dict[str, np.ndarray] = {}
        for cls in TARGET_CLASSES:
            rows = by_class.get(cls, [])
            if not rows:
                log.warning("no P/LP variants for class %s", cls)
                continue
            al = np.array([int(r["argmax_layer"]) for r in rows], dtype=np.float64)
            md = np.array([float(r["max_abs_dD"]) for r in rows], dtype=np.float64)
            med_al = float(np.median(al))
            med_md = float(np.median(md))

            # Compose a sort key
            d_al = np.abs(al - med_al)
            d_md = np.abs(md - med_md)
            best_idx = int(np.lexsort((d_md, d_al))[0])
            picked = rows[best_idx]

            # Per-class median trace (median over rows)
            mat = np.stack([to_vec_cos(r) for r in rows])
            med_trace = np.median(mat, axis=0)
            per_class_trace[cls] = med_trace

            picks.append({
                "mc_class": cls,
                "n_class": len(rows),
                "median_argmax_layer": med_al,
                "median_max_abs_dD": med_md,
                "picked_variant": {
                    "chrom": picked.get("chrom"),
                    "pos": int(picked.get("pos", 0)),
                    "ref": picked.get("ref"),
                    "alt": picked.get("alt"),
                    "gene": picked.get("gene"),
                    "argmax_layer": int(picked["argmax_layer"]),
                    "max_abs_dD": float(picked["max_abs_dD"]),
                    "dD_cos_per_layer": to_vec_cos(picked).tolist(),
                },
                "median_trace_dD_cos": med_trace.tolist(),
            })
            log.info("class %s: n=%d picked %s:%s>%s argmax=L%d",
                     cls, len(rows), picked.get("chrom"),
                     picked.get("ref"), picked.get("alt"),
                     int(picked["argmax_layer"]))

        out = {
            "config": {"n_layers": N_LAYERS, "target_classes": TARGET_CLASSES},
            "variants": picks,
        }
        out_path = out_dir / "case_studies_v8.json"
        out_path.write_text(json.dumps(out, indent=2))

        # Figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = max(1, len(picks))
            fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), sharey=True)
            if n == 1:
                axes = [axes]
            xs = np.arange(1, N_LAYERS + 1)
            for ax, info in zip(axes, picks):
                cls = info["mc_class"]
                med = np.array(info["median_trace_dD_cos"])
                pick = np.array(info["picked_variant"]["dD_cos_per_layer"])
                ax.plot(xs, med, color="#888888", lw=1.5,
                        label=f"class median (n={info['n_class']})")
                ax.plot(xs, pick, color="#d1495b", lw=1.5,
                        label=f"representative L{info['picked_variant']['argmax_layer']}")
                ax.axhline(0, color="k", lw=0.4)
                ax.set_title(cls)
                ax.set_xlabel("layer")
                ax.legend(fontsize=7, loc="best")
            axes[0].set_ylabel("ΔD_cos (alt − ref)")
            fig.suptitle("Per-class representative ΔD_cos traces")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"F5_mechanism_cases_v3.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_case",
            "status": "PASS" if picks else "FAIL",
            "reason": f"picked {len(picks)} representatives across {len(TARGET_CLASSES)} target classes",
            "metrics": {"n_picks": len(picks)},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0 if picks else 1,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        if picks:
            done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p2_case_v8 DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_case_v8 FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_case",
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
