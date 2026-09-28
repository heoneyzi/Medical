"""Build per-chromosome position labels from GENCODE v44 GTF (v2 with UTR fix).

Codebook (matches paper's data_ref/chr{17,22}_position_labels.npy):
  0 = intergenic (default)
  1 = intron
  2 = coding_exon (CDS)
  3 = 5'UTR
  4 = 3'UTR
  5 = splice_donor (2 bp downstream of exon end, on intron side)
  6 = splice_acceptor (2 bp upstream of exon start, on intron side)

v2 vs v1: Handle GENCODE's UTR feature naming ambiguity — accept BOTH:
  (a) "five_prime_UTR" / "three_prime_UTR" (explicit)
  (b) "UTR" (implicit — 5' vs 3' inferred from position relative to CDS on strand)

Priority when overlapping (later assignment wins):
  intron < CDS < 5'UTR < 3'UTR < splice_donor < splice_acceptor
"""
from __future__ import annotations
import argparse, time, gzip
from pathlib import Path
from collections import defaultdict

import numpy as np

TDIG = Path("/path/to/TDiG")
HG38 = TDIG / "arch_compare"
GTF = TDIG / "arch_compare/refs/gencode.v44.annotation.gtf.gz"


def parse_gtf_chr(gtf_path: Path, chrom: str):
    """Yield (feature, start, end, strand, transcript_id) for chrom."""
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            fs = line.rstrip("\n").split("\t")
            if len(fs) < 9: continue
            if fs[0] != chrom: continue
            feat = fs[2]
            if feat not in ("exon", "CDS", "UTR", "five_prime_UTR", "three_prime_UTR"):
                continue
            start, end = int(fs[3]) - 1, int(fs[4])  # 0-based half-open
            strand = fs[6]
            attrs = fs[8]
            tid = None
            for kv in attrs.split(";"):
                kv = kv.strip()
                if kv.startswith('transcript_id "'):
                    tid = kv.split('"')[1]
                    break
            yield feat, start, end, strand, tid


def classify_utr(utr_ranges: list[tuple[int, int, str]],
                 cds_per_tx: dict[str, list[tuple[int, int]]],
                 utr_per_tx: dict[str, list[tuple[int, int, str]]]) -> tuple[list, list]:
    """For each UTR span, classify as 5' or 3' by strand + position vs CDS.
    5' UTR = UTR upstream of the CDS start (in transcript direction).
    3' UTR = UTR downstream of the CDS end (in transcript direction).
    """
    utr5, utr3 = [], []
    for tid, spans in utr_per_tx.items():
        cds = cds_per_tx.get(tid, [])
        if not cds:
            continue
        cds_start = min(c[0] for c in cds)
        cds_end = max(c[1] for c in cds)
        strand = spans[0][2]
        for s, e, _ in spans:
            if strand == "+":
                if e <= cds_start:
                    utr5.append((s, e))
                elif s >= cds_end:
                    utr3.append((s, e))
                # overlapping the CDS — split? mark neither (already covered by CDS priority)
            else:  # "-"
                # For - strand, transcript direction is reversed
                if s >= cds_end:
                    utr5.append((s, e))
                elif e <= cds_start:
                    utr3.append((s, e))
    return utr5, utr3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else TDIG / f"paper/data_local/{args.chrom}_position_labels.npy"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading hg38 to get {args.chrom} length …")
    fa_path = HG38 / f"{args.chrom}.fa.gz"
    chrom_len = 0
    with gzip.open(fa_path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                chrom_len += len(line.strip())
    print(f"[{time.strftime('%H:%M:%S')}]   {args.chrom} length: {chrom_len:,}")

    print(f"[{time.strftime('%H:%M:%S')}] parsing GENCODE GTF for {args.chrom} …")
    tx_exons: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    cds_ranges: list[tuple[int, int]] = []
    cds_per_tx: dict[str, list[tuple[int, int]]] = defaultdict(list)
    utr_explicit_5: list[tuple[int, int]] = []
    utr_explicit_3: list[tuple[int, int]] = []
    utr_ambiguous_per_tx: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    feat_counts = defaultdict(int)

    for feat, s, e, strand, tid in parse_gtf_chr(GTF, args.chrom):
        feat_counts[feat] += 1
        if feat == "exon" and tid:
            tx_exons[tid].append((s, e, strand))
        elif feat == "CDS":
            cds_ranges.append((s, e))
            if tid: cds_per_tx[tid].append((s, e))
        elif feat == "five_prime_UTR":
            utr_explicit_5.append((s, e))
        elif feat == "three_prime_UTR":
            utr_explicit_3.append((s, e))
        elif feat == "UTR" and tid:
            utr_ambiguous_per_tx[tid].append((s, e, strand))

    print(f"[{time.strftime('%H:%M:%S')}]   feature counts: {dict(feat_counts)}")
    print(f"[{time.strftime('%H:%M:%S')}]   transcripts with exons: {len(tx_exons)}")

    # Classify ambiguous "UTR" features into 5' or 3' by position vs CDS
    utr_5_from_amb, utr_3_from_amb = classify_utr([], cds_per_tx, utr_ambiguous_per_tx)
    print(f"[{time.strftime('%H:%M:%S')}]   UTR classified from ambiguous: 5'={len(utr_5_from_amb)} 3'={len(utr_3_from_amb)}")

    utr5_all = utr_explicit_5 + utr_5_from_amb
    utr3_all = utr_explicit_3 + utr_3_from_amb
    print(f"[{time.strftime('%H:%M:%S')}]   total 5'UTR: {len(utr5_all)}, 3'UTR: {len(utr3_all)}")

    labels = np.zeros(chrom_len, dtype=np.uint8)

    # 1) intron (from introns between consecutive exons)
    intron_pos = []
    donor_pos = []
    acceptor_pos = []
    for tid, exs in tx_exons.items():
        exs = sorted(exs, key=lambda x: x[0])
        strand = exs[0][2]
        for (s1, e1, _), (s2, e2, _) in zip(exs[:-1], exs[1:]):
            if s2 > e1:
                intron_pos.append((e1, s2))
                if strand == "+":
                    donor_pos.append((e1, min(e1 + 2, s2)))
                    acceptor_pos.append((max(s2 - 2, e1), s2))
                else:
                    donor_pos.append((max(s2 - 2, e1), s2))
                    acceptor_pos.append((e1, min(e1 + 2, s2)))

    for s, e in intron_pos:
        labels[s:e] = 1
    for s, e in cds_ranges:
        labels[s:e] = 2
    for s, e in utr5_all:
        labels[s:e] = 3
    for s, e in utr3_all:
        labels[s:e] = 4
    for s, e in donor_pos:
        labels[s:e] = 5
    for s, e in acceptor_pos:
        labels[s:e] = 6

    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"[{time.strftime('%H:%M:%S')}] label distribution: {counts}")
    print(f"[{time.strftime('%H:%M:%S')}] writing {out} …")
    np.save(out, labels)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
