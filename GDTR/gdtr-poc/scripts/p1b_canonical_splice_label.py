"""P1b — canonical / non-canonical splice motif labels (Plan §3.2.3 step P1b-1).

For each canonical transcript on chr17 / chr22 (using the same
`pick_canonical_transcript` logic as 20_phase2_0_prep_chr17.py:58-73),
iterates intron boundaries and reads the 2-bp donor signal (intron-positions
1-2) and the 2-bp acceptor signal (intron-positions -2,-1). Strand-aware:
on `-` strand, take the reverse complement.

Codebook (uint8, separate file from chr*_position_labels.npy per Plan I10):
    0  not_splice
    1  canonical_GT_AG_donor
    2  canonical_GT_AG_acceptor
    3  canonical_GC_AG_donor
    4  canonical_GC_AG_acceptor
    5  non_canonical_donor
    6  non_canonical_acceptor

Outputs:
    ~/gDTR/data/annotation/chr17_splice_class_labels.npy
    ~/gDTR/data/annotation/chr22_splice_class_labels.npy
    ~/gDTR/data/annotation/splice_class_codebook.json
    ~/gDTR/results/p1b/_status.json
    ~/gDTR/results/p1b/_done
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
SPLICE_PAD = 10  # ±10 bp around boundary, matches existing position label codebook

CODEBOOK = {
    "not_splice": 0,
    "canonical_GT_AG_donor": 1,
    "canonical_GT_AG_acceptor": 2,
    "canonical_GC_AG_donor": 3,
    "canonical_GC_AG_acceptor": 4,
    "non_canonical_donor": 5,
    "non_canonical_acceptor": 6,
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p1b_label")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def find_gtf_db() -> Path:
    """Locate the GENCODE GFF DB. The file is named `chr17_chr22.gtf.db` in
    the repo even though older docs reference `chr17_chr13`. We search the
    annotation dir for any *.gtf.db file.
    """
    base = GDTR_ROOT / "data" / "annotation"
    for cand in [
        base / "gencode.v44.chr17_chr22.gtf.db",
        base / "gencode.v44.chr17_chr13.gtf.db",
    ]:
        if cand.exists():
            return cand
    # Glob fallback
    candidates = sorted(base.glob("*.gtf.db"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no *.gtf.db file in {base}")


COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def revcomp(s: str) -> str:
    return "".join(COMP.get(c, "N") for c in reversed(s))


def pick_canonical_transcript(db, gene):
    """Same logic as 20_phase2_0_prep_chr17.py:58-73."""
    transcripts = list(db.children(gene, featuretype="transcript", level=1))
    if not transcripts:
        return None
    mane = [
        t for t in transcripts
        if "tag" in t.attributes and any("MANE_Select" in v for v in t.attributes["tag"])
    ]
    if mane:
        return max(mane, key=lambda t: t.end - t.start)
    coding = [
        t for t in transcripts
        if t.attributes.get("transcript_type", [""])[0] == "protein_coding"
    ]
    pool = coding if coding else transcripts
    return max(pool, key=lambda t: t.end - t.start)


def classify_intron(donor_2bp: str, acceptor_2bp: str) -> str:
    """Classify by the 5'/3' dinucleotide signals.
    `donor_2bp`   = first 2 bp of intron in transcript orientation (5' end)
    `acceptor_2bp` = last 2 bp of intron in transcript orientation (3' end)
    """
    d = donor_2bp.upper()
    a = acceptor_2bp.upper()
    if d == "GT" and a == "AG":
        return "canonical_GT_AG"
    if d == "GC" and a == "AG":
        return "canonical_GC_AG"
    if d == "AT" and a == "AC":
        return "non_canonical_AT_AC"
    return "non_canonical_other"


def build_splice_class_labels(db, fa, chrom: str, chrom_len: int,
                              log: logging.Logger,
                              smoke_n_genes: int | None = None) -> tuple[np.ndarray, dict]:
    """Returns (labels[chrom_len] uint8, counts dict)."""
    labels = np.zeros(chrom_len, dtype=np.uint8)
    counts = {k: 0 for k in CODEBOOK}
    tally = {
        "introns_total": 0,
        "introns_classified": {
            "canonical_GT_AG": 0, "canonical_GC_AG": 0,
            "non_canonical_AT_AC": 0, "non_canonical_other": 0,
        },
        "n_genes": 0, "n_canonical_transcripts": 0,
    }
    genes = list(db.region(seqid=chrom, featuretype="gene"))
    log.info("[%s] %d genes", chrom, len(genes))
    if smoke_n_genes is not None:
        genes = genes[:smoke_n_genes]
        log.info("[%s] smoke: limiting to %d genes", chrom, len(genes))
    tally["n_genes"] = len(genes)

    for gi, gene in enumerate(genes):
        tr = pick_canonical_transcript(db, gene)
        if tr is None:
            continue
        tally["n_canonical_transcripts"] += 1
        exons = sorted(db.children(tr, featuretype="exon", level=1),
                       key=lambda e: (e.start, e.end))
        if len(exons) < 2:
            continue

        strand = tr.strand
        # Transcript orientation: on - strand we walk introns from RIGHT to LEFT.
        if strand == "+":
            intron_pairs = [(exons[i].end, exons[i + 1].start - 1)
                            for i in range(len(exons) - 1)]
        else:
            # On - strand, exon order in genomic coords is reversed for biology
            # but each intron is still bounded by exons[i].end and exons[i+1].start
            # in genomic 0-based exclusive coords. We will RC the donor/acceptor.
            intron_pairs = [(exons[i].end, exons[i + 1].start - 1)
                            for i in range(len(exons) - 1)]

        for (intron_start_0, intron_end_0) in intron_pairs:
            if intron_end_0 <= intron_start_0:
                continue
            intron_len = intron_end_0 - intron_start_0
            if intron_len < 4:
                continue  # need at least 4 bp to read both di-nuc signals
            tally["introns_total"] += 1

            # Genomic dinucleotides (always in + strand orientation)
            # pyfaidx (opened with as_raw=True, sequence_always_upper=True): use indexing.
            try:
                d_genom = str(fa[chrom][intron_start_0:intron_start_0 + 2])
                a_genom = str(fa[chrom][intron_end_0 - 2:intron_end_0])
                if len(d_genom) != 2 or len(a_genom) != 2:
                    continue
            except Exception:
                continue

            if strand == "+":
                donor_2bp = d_genom
                acceptor_2bp = a_genom
            else:
                # Transcript orientation: reverse complement
                donor_2bp = revcomp(a_genom)
                acceptor_2bp = revcomp(d_genom)

            cls = classify_intron(donor_2bp, acceptor_2bp)
            tally["introns_classified"][cls] += 1

            # Map (donor side, acceptor side) → labels around boundaries.
            # Genomic +strand: donor side = intron_start_0 ; acceptor side = intron_end_0
            # Genomic -strand: donor side = intron_end_0 ; acceptor side = intron_start_0
            if strand == "+":
                donor_pos = intron_start_0
                acceptor_pos = intron_end_0
            else:
                donor_pos = intron_end_0
                acceptor_pos = intron_start_0

            d_a = max(0, donor_pos - SPLICE_PAD)
            d_b = min(chrom_len, donor_pos + SPLICE_PAD)
            a_a = max(0, acceptor_pos - SPLICE_PAD)
            a_b = min(chrom_len, acceptor_pos + SPLICE_PAD)

            if cls == "canonical_GT_AG":
                d_code = CODEBOOK["canonical_GT_AG_donor"]
                a_code = CODEBOOK["canonical_GT_AG_acceptor"]
            elif cls == "canonical_GC_AG":
                d_code = CODEBOOK["canonical_GC_AG_donor"]
                a_code = CODEBOOK["canonical_GC_AG_acceptor"]
            else:
                d_code = CODEBOOK["non_canonical_donor"]
                a_code = CODEBOOK["non_canonical_acceptor"]

            labels[d_a:d_b] = d_code
            labels[a_a:a_b] = a_code

        if (gi + 1) % 500 == 0:
            log.info("[%s] processed %d genes ...", chrom, gi + 1)

    for code_name, code_val in CODEBOOK.items():
        counts[code_name] = int((labels == code_val).sum())
    return labels, {"counts_bp": counts, "tally": tally}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="restrict each chromosome to first 50 genes")
    parser.add_argument("--force", action="store_true",
                        help="override _done marker")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = GDTR_ROOT / "data" / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_label"
    if done_marker.exists() and not args.force:
        log.info("p1b_label: _done_label exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p1b_label", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        import gffutils
        from pyfaidx import Fasta

        gtf_db = find_gtf_db()
        log.info("using gffutils db: %s", gtf_db)
        db = gffutils.FeatureDB(str(gtf_db), keep_order=True)

        per_chrom_counts = {}
        smoke_n = 50 if args.smoke else None

        for chrom in ("chr17", "chr22"):
            fa_path = GDTR_ROOT / "data" / "reference" / f"{chrom}.fa"
            if not fa_path.exists():
                raise FileNotFoundError(f"missing {fa_path}")
            fa = Fasta(str(fa_path), as_raw=True, sequence_always_upper=True)
            chrom_len = len(fa[chrom])
            log.info("[%s] reference len=%d", chrom, chrom_len)

            labels, stats = build_splice_class_labels(db, fa, chrom, chrom_len,
                                                       log, smoke_n_genes=smoke_n)
            out_path = ann_dir / f"{chrom}_splice_class_labels.npy"
            np.save(out_path, labels)
            log.info("[%s] wrote %s (%.1f MB)", chrom, out_path,
                     out_path.stat().st_size / 1e6)
            per_chrom_counts[chrom] = stats

        # Sidecar JSON
        codebook_payload = {
            "codebook": CODEBOOK,
            "splice_pad": SPLICE_PAD,
            "chr17_counts": per_chrom_counts.get("chr17", {}),
            "chr22_counts": per_chrom_counts.get("chr22", {}),
        }
        cb_path = ann_dir / "splice_class_codebook.json"
        cb_path.write_text(json.dumps(codebook_payload, indent=2))
        log.info("wrote %s", cb_path)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1b_label",
            "status": "PASS",
            "reason": "labels written for chr17 and chr22",
            "metrics": per_chrom_counts,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p1b_label DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p1b_label FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1b_label",
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
