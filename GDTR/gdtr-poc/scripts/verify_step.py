"""Verify one revision_v8 step. Reads _status.json + does schema checks.

Implements Plan §5.2. Each entry in STEPS describes the expected output
directory, required file names, and (optionally) CSV / JSON schema checks.

Exit codes:
    0 — PASS
    1 — FAIL
    2 — MISSING / unknown step
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


# ----------------------------------------------------------------------------
# Step registry — derived from Plan §3.X.7 ("Files affected") for each item.
# ----------------------------------------------------------------------------
GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()


STEPS: dict[str, dict] = {
    # ----- Phase 1a: calibration / validation split -----
    "p1a": {
        "dir": GDTR_ROOT / "results" / "p1a",
        "required": ["calib_val_table.csv", "chr17_q70_recheck.json", "_status.json"],
        "csv_checks": [
            (
                "calib_val_table.csv",
                ["chrom", "context", "n", "mean_c", "median_c",
                 "cohens_d_vs_intron", "mwu_p_vs_intron"],
                7,  # 7 contexts × 2 chromosomes minus intron self = 12 expected; allow >=7
            ),
        ],
        "json_checks": [
            (
                "chr17_q70_recheck.json",
                ["chr17_q70", "chr22_q70", "abs_diff", "pass"],
            ),
        ],
    },
    # ----- Phase 1b: canonical splice labels & comparison -----
    "p1b_label": {
        "dir": GDTR_ROOT / "data" / "annotation",
        "required": [
            "chr17_splice_class_labels.npy",
            "chr22_splice_class_labels.npy",
            "splice_class_codebook.json",
        ],
        "json_checks": [
            ("splice_class_codebook.json", ["codebook", "chr17_counts", "chr22_counts"]),
        ],
    },
    "p1b_compare": {
        "dir": GDTR_ROOT / "results" / "p1b",
        "required": ["splice_canonical_compare.json", "_status.json"],
        "json_checks": [
            (
                "splice_canonical_compare.json",
                ["per_class", "pairwise", "gamma_cos"],
            ),
        ],
    },
    # ----- Phase 2: SNV per-class + indels + case study -----
    "p2_snv_join": {
        "dir": GDTR_ROOT / "results" / "p2",
        "required": ["variants_features_classed.csv", "_status.json"],
        "csv_checks": [
            (
                "variants_features_classed.csv",
                ["chrom", "pos", "ref", "alt", "gene", "category", "stars",
                 "max_abs_dD", "argmax_layer", "mc_class"],
                100,
            ),
        ],
    },
    "p2_snv_stats": {
        "dir": GDTR_ROOT / "results" / "p2",
        "required": ["p2_snv_per_class.csv", "p2_snv_per_class.json", "_status.json"],
        "csv_checks": [
            (
                "p2_snv_per_class.csv",
                ["mc_class", "category", "n",
                 "median_argmax_layer", "iqr_argmax_layer",
                 "median_max_abs_dD"],
                4,
            ),
        ],
        "json_checks": [
            ("p2_snv_per_class.json", ["per_class", "kruskal_wallis", "config"]),
        ],
    },
    "p2_indel_select": {
        "dir": GDTR_ROOT / "results" / "p2_indel",
        "required": ["frameshift_panel.tsv", "_status.json"],
    },
    "p2_indel_forward": {
        "dir": GDTR_ROOT / "results" / "p2_indel",
        "required": ["variants_features_indel.csv", "_status.json"],
        "csv_checks": [
            (
                "variants_features_indel.csv",
                ["chrom", "pos", "ref", "alt", "gene", "category", "stars",
                 "max_abs_dD", "argmax_layer",
                 "mc_class", "indel_len", "frame_class", "frame_position"],
                10,
            ),
        ],
    },
    "p2_indel_stats": {
        "dir": GDTR_ROOT / "results" / "p2_indel",
        "required": ["p2_indel_per_class.csv", "p2_indel_per_class.json", "_status.json"],
    },
    "p2_case": {
        "dir": GDTR_ROOT / "results" / "p2_case",
        "required": ["case_studies_v8.json", "_status.json"],
        "json_checks": [
            ("case_studies_v8.json", ["variants", "config"]),
        ],
    },
    # ----- Phase 3B: Q2 defend -----
    "p3b1": {
        "dir": GDTR_ROOT / "results" / "p3b1",
        "required": ["p3b1_func_pos.json", "_status.json"],
        "json_checks": [
            ("p3b1_func_pos.json", ["per_dataset", "config"]),
        ],
    },
    "p3b2": {
        "dir": GDTR_ROOT / "results" / "p3b2",
        "required": [
            "p3b2_repeat_breakdown.csv",
            "p3b2_repeat_breakdown.json",
            "_status.json",
        ],
        "csv_checks": [
            (
                "p3b2_repeat_breakdown.csv",
                ["class", "n_overlap_q2", "n_total_chr22", "fold",
                 "hypergeom_p", "biological_interpretation"],
                3,
            ),
        ],
    },
    "p3b3_chr17": {
        "dir": GDTR_ROOT / "results" / "p3b3_chr17",
        "required": ["q2_enrichment_chr17.json", "_status.json"],
    },
    "prep_chr1": {
        "dir": GDTR_ROOT / "data" / "reference",
        "required": ["chr1_windows.tsv"],
    },
    "p3b3_chr1_forward": {
        "dir": GDTR_ROOT / "results" / "p3b3_chr1",
        "required": ["chr1_cache.h5", "_status.json"],
    },
    "p3b3_chr1_enrichment": {
        "dir": GDTR_ROOT / "results" / "p3b3_chr1",
        "required": ["q2_enrichment_chr1.json", "_status.json"],
    },
}


def load_status(d: Path):
    p = d / "_status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def check_csv_schema(path: Path, expected_cols, min_rows: int = 1):
    if not path.exists():
        return False, "missing"
    try:
        with path.open() as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not set(expected_cols).issubset(set(cols)):
                missing = sorted(set(expected_cols) - set(cols))
                return False, (
                    f"schema mismatch: missing columns {missing}; "
                    f"have {cols}; want superset of {list(expected_cols)}"
                )
            n = sum(1 for _ in r)
            if n < min_rows:
                return False, f"only {n} rows (need >= {min_rows})"
        return True, f"ok ({n} rows)"
    except Exception as e:
        return False, f"error reading csv: {e}"


def check_json_schema(path: Path, expected_keys):
    if not path.exists():
        return False, "missing"
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return False, f"error parsing json: {e}"
    if not isinstance(d, dict):
        return False, "json root is not an object"
    missing = sorted(k for k in expected_keys if k not in d)
    if missing:
        return False, f"missing top-level keys: {missing}"
    return True, "ok"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: verify_step.py <step_id>", file=sys.stderr)
        sys.exit(2)
    step = sys.argv[1]
    if step not in STEPS:
        print(f"unknown step {step}; known: {sorted(STEPS)}", file=sys.stderr)
        sys.exit(2)
    spec = STEPS[step]
    base = Path(spec["dir"]).expanduser()

    # Required files
    for fname in spec.get("required", []):
        p = base / fname
        if not p.exists():
            print(f"{step}: MISSING {p}")
            sys.exit(2)

    # Status
    s = load_status(base)
    if s is None:
        print(f"{step}: MISSING or unparseable _status.json at {base}")
        sys.exit(2)
    status = s.get("status", "").upper()
    if status not in ("PASS", "REFRAME"):
        print(f"{step}: status={status} reason={s.get('reason')}")
        sys.exit(1)

    # CSV schema
    for fname, cols, min_rows in spec.get("csv_checks", []):
        ok, msg = check_csv_schema(base / fname, cols, min_rows)
        if not ok:
            print(f"{step}/{fname}: {msg}")
            sys.exit(1)

    # JSON schema
    for fname, keys in spec.get("json_checks", []):
        ok, msg = check_json_schema(base / fname, keys)
        if not ok:
            print(f"{step}/{fname}: {msg}")
            sys.exit(1)

    print(f"{step}: PASS ({status}; reason={s.get('reason', '')})")
    sys.exit(0)


if __name__ == "__main__":
    main()
