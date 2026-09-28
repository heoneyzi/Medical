"""P2-SNV-1 — join ClinVar MC field to variants_features.csv (Plan §3.3.3).

Parses ClinVar VCF for the MC INFO field, joins to the cached
phase3_main/variants_features.csv on (chrom, pos, ref, alt). Adds one new
column `mc_class` per variant.

MC field format: `MC=SO:NNNNNNN|name1,SO:NNNNNNN|name2`. When multiple
consequences are listed, choose by priority (Plan):

    canonical_splice  > splice_region  > nonsense > frameshift
    > inframe_indel   > missense       > synonymous
    > 5UTR            > 3UTR           > intron > other

Outputs:
    ~/gDTR/results/p2/variants_features_classed.csv  (78 SNV cols + mc_class)
    ~/gDTR/results/p2/p2_snv_class_join.json         (per-class counts)
    ~/gDTR/results/p2/_status.json
    ~/gDTR/results/p2/_done_join
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()

# SO term → class label.  Priority matters when one variant has multiple terms.
SO_TO_CLASS = {
    # canonical splice (donor / acceptor at boundary)
    "SO:0001574": "canonical_splice",  # splice_donor_variant
    "SO:0001575": "canonical_splice",  # splice_acceptor_variant
    # splice region (broader, ±10bp)
    "SO:0001629": "splice_region",
    # nonsense / stop_gained
    "SO:0001587": "nonsense",
    # frameshift
    "SO:0001589": "frameshift",
    # inframe indel
    "SO:0001821": "inframe_indel",  # inframe_insertion
    "SO:0001822": "inframe_indel",  # inframe_deletion
    # missense
    "SO:0001583": "missense",
    # synonymous
    "SO:0001819": "synonymous",
    # UTRs
    "SO:0001623": "5UTR",
    "SO:0001624": "3UTR",
    # intron
    "SO:0001627": "intron",
}

# Lower index = higher priority.
PRIORITY = [
    "canonical_splice", "splice_region",
    "nonsense", "frameshift",
    "inframe_indel", "missense", "synonymous",
    "5UTR", "3UTR", "intron", "other",
]
PRIORITY_RANK = {c: i for i, c in enumerate(PRIORITY)}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p2_snv_join")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
        else:
            out[kv] = "1"
    return out


def mc_to_class(mc_field: str | None) -> str:
    """Parse `MC=SO:0001587|nonsense,SO:0001619|non_coding_transcript_variant`
    and assign the highest-priority class.
    """
    if not mc_field:
        return "MC_unknown"
    classes_seen = set()
    for item in mc_field.split(","):
        # Each entry is `SO:NNNNNNN|name`
        if "|" in item:
            so_id, _ = item.split("|", 1)
        else:
            so_id = item
        so_id = so_id.strip()
        cls = SO_TO_CLASS.get(so_id)
        if cls is not None:
            classes_seen.add(cls)
    if not classes_seen:
        return "other"
    return min(classes_seen, key=lambda c: PRIORITY_RANK.get(c, 99))


def load_clinvar_mc(vcf_path: Path, log: logging.Logger,
                    smoke: bool = False) -> dict[tuple[str, int, str, str], str]:
    """Parse CLINVAR VCF.gz and return dict {(chrom, pos, ref, alt) -> mc_class}."""
    out: dict[tuple[str, int, str, str], str] = {}
    n = 0
    n_with_mc = 0
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                pos_i = int(pos)
            except ValueError:
                continue
            info = parse_info(parts[7])
            mc = info.get("MC")
            cls = mc_to_class(mc)
            if mc is not None:
                n_with_mc += 1
            chrom_no = chrom.lstrip("chr")
            out[(chrom_no, pos_i, ref, alt)] = cls
            if smoke and len(out) >= 200_000:
                break
            if n % 1_000_000 == 0:
                log.info("  parsed %d ClinVar records (%d with MC)", n, n_with_mc)
    log.info("ClinVar: %d records parsed, %d with MC", n, n_with_mc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="cap ClinVar parse at 200k records for fast smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_join"
    if done_marker.exists() and not args.force:
        log.info("p2_snv_join: _done_join exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_snv_join", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        vcf_path = GDTR_ROOT / "data" / "variants" / "clinvar.vcf.gz"
        snv_csv = GDTR_ROOT / "results" / "phase3_main" / "variants_features.csv"
        if not vcf_path.exists():
            raise FileNotFoundError(f"missing {vcf_path}")
        if not snv_csv.exists():
            raise FileNotFoundError(f"missing {snv_csv}")

        # Load all ClinVar (with MC) once
        log.info("parsing ClinVar VCF for MC field — this may take ~3 min")
        mc_lookup = load_clinvar_mc(vcf_path, log, smoke=args.smoke)

        # Stream join to features.csv
        out_csv = out_dir / "variants_features_classed.csv"
        per_class = Counter()
        per_class_per_cat = {}
        n_total = 0
        n_with_class = 0

        with snv_csv.open() as fin, out_csv.open("w", newline="") as fout:
            reader = csv.DictReader(fin)
            cols = list(reader.fieldnames or [])
            new_cols = cols + ["mc_class"]
            writer = csv.DictWriter(fout, fieldnames=new_cols)
            writer.writeheader()
            for row in reader:
                n_total += 1
                key = (str(row["chrom"]).lstrip("chr"),
                       int(row["pos"]), row["ref"], row["alt"])
                cls = mc_lookup.get(key, "MC_unknown")
                row["mc_class"] = cls
                writer.writerow(row)
                per_class[cls] += 1
                cat = row.get("category", "")
                per_class_per_cat.setdefault(cat, Counter())[cls] += 1
                if cls != "MC_unknown":
                    n_with_class += 1

        log.info("wrote %s (%d rows; %d with class)", out_csv, n_total, n_with_class)
        log.info("per-class counts: %s", dict(per_class.most_common()))

        info_payload = {
            "n_total": n_total,
            "n_with_class": n_with_class,
            "per_class": dict(per_class),
            "per_class_per_category": {
                cat: dict(c) for cat, c in per_class_per_cat.items()
            },
        }
        (out_dir / "p2_snv_class_join.json").write_text(json.dumps(info_payload, indent=2))

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_snv_join",
            "status": "PASS",
            "reason": f"joined {n_with_class}/{n_total} variants with MC class",
            "metrics": info_payload,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p2_snv_join DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_snv_join FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_snv_join",
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
