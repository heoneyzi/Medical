"""P2-INDEL-1 — frameshift panel selection (Plan §3.3.3).

From clinvar_15gene.tsv (output of prep_clinvar_15gene.py), filter to P/LP
indels with stars >= 2 and MC including frameshift_variant or inframe
deletion / insertion. Stratified subsample to ~25 per gene per direction
(frameshift_del / frameshift_ins) with a cap of 600 indels total.

Also picks 600 matched B/LB controls (closest genomic-position B/LB of any
consequence in the same gene).

The MC field needs to come from the ClinVar VCF (it's not in clinvar_15gene.tsv).
We re-parse the VCF here with a position-keyed dict to look up the MC class.

Outputs:
    ~/gDTR/results/p2_indel/frameshift_panel.tsv
    ~/gDTR/results/p2_indel/_status.json
    ~/gDTR/results/p2_indel/_done_select
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p2_indel_select")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


SO_FRAMESHIFT = "SO:0001589"
SO_INFRAME = ("SO:0001821", "SO:0001822")  # ins / del


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
        else:
            out[kv] = "1"
    return out


def parse_mc(mc_field: str | None) -> set[str]:
    if not mc_field:
        return set()
    out = set()
    for item in mc_field.split(","):
        if "|" in item:
            so, _ = item.split("|", 1)
        else:
            so = item
        out.add(so.strip())
    return out


def load_clinvar_indel_mc(vcf_path: Path, log: logging.Logger,
                          smoke: bool = False) -> dict[tuple, set]:
    """Returns {(chrom, pos, ref, alt) -> set of SO terms} restricted to indels."""
    out = {}
    n = 0
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            if len(ref) == 1 and len(alt) == 1:
                continue
            try:
                pos_i = int(pos)
            except ValueError:
                continue
            info = parse_info(parts[7])
            mc = parse_mc(info.get("MC"))
            chrom_no = chrom.lstrip("chr")
            out[(chrom_no, pos_i, ref, alt)] = mc
            if smoke and len(out) >= 50_000:
                break
            if n % 1_000_000 == 0:
                log.info("  parsed %d records (%d indels with MC)", n, len(out))
    log.info("loaded %d indel records from ClinVar VCF", len(out))
    return out


def classify_frame(ref: str, alt: str, so_terms: set[str]) -> tuple[str, int]:
    """Return (frame_class, indel_len).
    frame_class ∈ {true_frameshift_1bp, true_frameshift_2bp, true_frameshift_other,
                   inframe, other_indel}
    indel_len = abs(len(alt) - len(ref))
    """
    indel_len = abs(len(alt) - len(ref))
    has_fs = SO_FRAMESHIFT in so_terms
    has_inf = bool(set(so_terms) & set(SO_INFRAME))
    if has_fs:
        if indel_len == 1:
            return "true_frameshift_1bp", indel_len
        if indel_len == 2:
            return "true_frameshift_2bp", indel_len
        return "true_frameshift_other", indel_len
    if has_inf:
        return "inframe", indel_len
    return "other_indel", indel_len


def load_15gene_tsv(tsv_path: Path) -> list[dict]:
    rows = []
    with tsv_path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            d = dict(zip(header, parts))
            d["pos"] = int(d["pos"])
            try:
                d["stars"] = int(d.get("stars", 0))
            except ValueError:
                d["stars"] = 0
            rows.append(d)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="cap target panel to 50 indels")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-n-indels", type=int, default=600)
    parser.add_argument("--per-gene-per-direction", type=int, default=25)
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2_indel"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_select"
    if done_marker.exists() and not args.force:
        log.info("p2_indel_select: _done_select exists; skipping")
        return

    target_n = 50 if args.smoke else args.target_n_indels

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_indel_select", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        tsv_path = GDTR_ROOT / "data" / "variants" / "clinvar_15gene.tsv"
        vcf_path = GDTR_ROOT / "data" / "variants" / "clinvar.vcf.gz"
        if not tsv_path.exists():
            raise FileNotFoundError(f"missing {tsv_path}")
        if not vcf_path.exists():
            raise FileNotFoundError(f"missing {vcf_path}")

        all_rows = load_15gene_tsv(tsv_path)
        log.info("loaded %d rows from clinvar_15gene.tsv", len(all_rows))
        mc_lookup = load_clinvar_indel_mc(vcf_path, log, smoke=args.smoke)

        # Build candidate indel pool: P/LP, stars >= 2, indel
        candidates: list[dict] = []
        for r in all_rows:
            if r.get("category") != "P_LP":
                continue
            if r.get("stars", 0) < 2:
                continue
            ref = r.get("ref", ""); alt = r.get("alt", "")
            if len(ref) == 1 and len(alt) == 1:
                continue
            chrom_no = str(r.get("chrom", "")).lstrip("chr")
            key = (chrom_no, r["pos"], ref, alt)
            mc = mc_lookup.get(key, set())
            frame_class, indel_len = classify_frame(ref, alt, mc)
            if frame_class.startswith("true_frameshift") or frame_class == "inframe":
                r2 = dict(r)
                r2["mc_class"] = "frameshift" if frame_class.startswith("true_frameshift") else "inframe_indel"
                r2["indel_len"] = indel_len
                r2["frame_class"] = frame_class
                # direction: insertion if len(alt) > len(ref) else deletion
                r2["direction"] = "ins" if len(alt) > len(ref) else "del"
                candidates.append(r2)
        log.info("candidate indels (P/LP, stars>=2, FS/inframe): %d", len(candidates))

        # Stratified pick: per gene, per direction, up to per_gene_per_direction
        rng = np.random.default_rng(42)
        picks: list[dict] = []
        by_gd = defaultdict(list)
        for r in candidates:
            by_gd[(r["gene"], r["direction"])].append(r)
        keys = sorted(by_gd.keys())
        for k in keys:
            arr = by_gd[k]
            if len(arr) > args.per_gene_per_direction:
                idx = rng.choice(len(arr), size=args.per_gene_per_direction, replace=False)
                arr = [arr[i] for i in sorted(idx)]
            picks.extend(arr)
        if len(picks) > target_n:
            idx = rng.choice(len(picks), size=target_n, replace=False)
            picks = [picks[i] for i in sorted(idx)]
        log.info("picked %d P/LP indels", len(picks))

        # Match B/LB controls — same gene, closest genomic-position B/LB of any consequence
        by_gene_blb: dict[str, list[dict]] = defaultdict(list)
        for r in all_rows:
            if r.get("category") == "B_LB" and r.get("stars", 0) >= 1:
                by_gene_blb[r["gene"]].append(r)

        controls: list[dict] = []
        for r in picks:
            pool = by_gene_blb.get(r["gene"], [])
            if not pool:
                continue
            best = min(pool, key=lambda x: abs(int(x["pos"]) - int(r["pos"])))
            ref_b = best.get("ref", ""); alt_b = best.get("alt", "")
            chrom_no = str(best.get("chrom", "")).lstrip("chr")
            key = (chrom_no, best["pos"], ref_b, alt_b)
            mc = mc_lookup.get(key, set())
            indel_len = abs(len(alt_b) - len(ref_b))
            if indel_len == 0:
                frame_class = "snv"
            else:
                fc, _ = classify_frame(ref_b, alt_b, mc)
                frame_class = fc
            ctrl = dict(best)
            ctrl["mc_class"] = "control"
            ctrl["indel_len"] = indel_len
            ctrl["frame_class"] = frame_class
            ctrl["direction"] = "ins" if len(alt_b) > len(ref_b) else (
                "del" if len(alt_b) < len(ref_b) else "snv")
            controls.append(ctrl)

        all_picks = picks + controls

        # Output TSV
        out_tsv = out_dir / "frameshift_panel.tsv"
        cols = ["chrom", "pos", "ref", "alt", "gene", "category", "stars",
                "mc_class", "indel_len", "frame_class", "direction"]
        with out_tsv.open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in all_picks:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
        log.info("wrote %s (%d rows)", out_tsv, len(all_picks))

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_select",
            "status": "PASS",
            "reason": f"selected {len(picks)} P/LP indels + {len(controls)} controls",
            "metrics": {
                "n_plp_indels": len(picks),
                "n_blb_controls": len(controls),
                "per_frame_class": {
                    fc: sum(1 for r in picks if r["frame_class"] == fc)
                    for fc in ("true_frameshift_1bp", "true_frameshift_2bp",
                               "true_frameshift_other", "inframe")
                },
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p2_indel_select DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_indel_select FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_select",
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
