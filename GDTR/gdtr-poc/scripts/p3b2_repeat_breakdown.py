"""P3B-2 — repeat class breakdown (Plan §3.4.B-2.2).

Reads ~/gDTR/results/phase5/q2_enrichment.json (or 50b_phase5_smoothed
output), extracts all rmsk_* annotation rows, sorts by fold enrichment
descending, outputs a 9-row table with biological interpretation columns.

Optionally re-reads ~/gDTR/data/conservation/rmsk.txt.gz (or wherever the
local copy lives) for repFamily sub-classification (LTR → ERV1 / ERVK / ERVL,
LINE → L1 / L2, SINE → Alu / MIR, etc.).

Outputs:
    ~/gDTR/results/p3b2/p3b2_repeat_breakdown.csv
    ~/gDTR/results/p3b2/p3b2_repeat_breakdown.json
    ~/gDTR/results/p3b2/_status.json
    ~/gDTR/results/p3b2/_done
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()


BIO_INTERP = {
    "rmsk_LTR": "long-terminal-repeat retrotransposon (TE-derived enhancer candidate)",
    "rmsk_LINE": "long interspersed nuclear element retrotransposon",
    "rmsk_DNA": "DNA transposon",
    "rmsk_SINE": "short interspersed nuclear element (Alu, MIR; mammalian-conserved on av.)",
    "rmsk_Simple_repeat": "tandem repeat / microsatellite (low-complexity)",
    "rmsk_Low_complexity": "low-complexity sequence (homopolymer-rich)",
    "rmsk_Satellite": "centromeric / pericentromeric satellite",
    "rmsk_Retroposon": "non-LTR retrotransposon",
    "ENCODE_cCRE": "ENCODE candidate cis-regulatory element (functional)",
    "ENCODE_rDHS": "ENCODE representative DNase hypersensitive site (functional)",
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p3b2")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def find_rmsk() -> Path | None:
    for cand in [
        GDTR_ROOT / "data" / "conservation" / "rmsk.txt.gz",
        GDTR_ROOT / "data" / "conservation" / "rmsk.chr22.txt.gz",
        GDTR_ROOT / "data" / "external" / "rmsk.chr22.txt.gz",
        GDTR_ROOT / "data" / "external" / "rmsk.txt.gz",
    ]:
        if cand.exists():
            return cand
    return None


def repFamily_breakdown(rmsk_path: Path, chrom: str = "chr22",
                        log: logging.Logger | None = None) -> dict:
    """Read rmsk.txt.gz and tally per-class repFamily counts on the chromosome."""
    by_class_family: dict[str, Counter] = defaultdict(Counter)
    n = 0
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13 or parts[5] != chrom:
                continue
            cls = parts[11]
            fam = parts[12]
            by_class_family[cls][fam] += 1
            n += 1
    if log:
        log.info("rmsk: %d records on %s, classes=%d",
                 n, chrom, len(by_class_family))
    return {cls: dict(c.most_common()) for cls, c in by_class_family.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p3b2"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p3b2: _done exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p3b2", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        q2_json = GDTR_ROOT / "results" / "phase5" / "q2_enrichment.json"
        if not q2_json.exists():
            raise FileNotFoundError(f"missing {q2_json}")
        d = json.loads(q2_json.read_text())
        ann = d.get("annotation_enrichment", {})
        n_q2 = None
        n_total_valid = None
        # Pick out generic n_q2 and n_total_valid from any record
        for v in ann.values():
            if "n_q2" in v and "n_total_valid" in v:
                n_q2 = int(v["n_q2"]); n_total_valid = int(v["n_total_valid"])
                break

        # Build rows for every rmsk_* + ENCODE row (sorted by fold desc)
        rows = []
        for name, v in ann.items():
            if not name.startswith("rmsk_") and not name.startswith("ENCODE_"):
                continue
            fold = v.get("fold_enrichment")
            try:
                fold_f = float(fold) if fold is not None else float("nan")
            except (TypeError, ValueError):
                fold_f = float("nan")
            rows.append({
                "class": name,
                "n_overlap_q2": int(v.get("n_overlap_q2", 0)),
                "n_total_chr22": int(v.get("n_overlap_total_valid", 0)),
                "fold": fold_f,
                "hypergeom_p": v.get("hypergeom_pval_oneSided_overrep"),
                "biological_interpretation": BIO_INTERP.get(name, ""),
            })
        rows.sort(key=lambda r: (np.isnan(r["fold"]), -r["fold"] if np.isfinite(r["fold"]) else 0.0))

        log.info("extracted %d enrichment rows from %s", len(rows), q2_json)

        # rmsk repFamily breakdown (optional)
        rmsk_path = find_rmsk()
        rf = None
        if rmsk_path is not None:
            log.info("rmsk file: %s", rmsk_path)
            try:
                rf = repFamily_breakdown(rmsk_path, chrom="chr22", log=log)
            except Exception as e:
                log.warning("repFamily breakdown failed: %s", e)
        else:
            log.info("rmsk file not found locally; skipping repFamily breakdown")

        # Verdict (Plan §3.4.B-2.3)
        rec = {r["class"]: r for r in rows}
        ltr = rec.get("rmsk_LTR", {}).get("fold", float("nan"))
        dna = rec.get("rmsk_DNA", {}).get("fold", float("nan"))
        sine = rec.get("rmsk_SINE", {}).get("fold", float("nan"))
        low = rec.get("rmsk_Low_complexity", {}).get("fold", float("nan"))
        simp = rec.get("rmsk_Simple_repeat", {}).get("fold", float("nan"))

        if (np.isfinite(ltr) and ltr >= 1.30
                and np.isfinite(dna) and dna >= 1.10
                and np.isfinite(sine) and sine <= 1.0):
            verdict = "PASS"
        elif np.isfinite(low) and low >= 1.50 and (
                np.isfinite(ltr) and abs(ltr - 1.0) < 0.10):
            verdict = "REFRAME"  # low-complexity-driven
        else:
            verdict = "REFRAME"

        # Write CSV
        csv_path = out_dir / "p3b2_repeat_breakdown.csv"
        cols = ["class", "n_overlap_q2", "n_total_chr22", "fold",
                "hypergeom_p", "biological_interpretation"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        log.info("wrote %s (%d rows)", csv_path, len(rows))

        out = {
            "config": {"q2_n_q2_positions": n_q2,
                       "q2_n_total_valid_positions": n_total_valid},
            "rows": rows,
            "rmsk_repFamily_chr22": rf,
            "verdict": verdict,
        }
        (out_dir / "p3b2_repeat_breakdown.json").write_text(json.dumps(out, indent=2))

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b2",
            "status": verdict,
            "reason": f"LTR={ltr!r} DNA={dna!r} SINE={sine!r} Low_complexity={low!r}",
            "metrics": {"LTR": ltr, "DNA": dna, "SINE": sine,
                        "Simple_repeat": simp, "Low_complexity": low},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p3b2 DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p3b2 FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b2",
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
