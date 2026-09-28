"""Per-base annotation tracks.

Builds, for one chromosome, an int8 context track plus the covariate tracks the
adjusted analysis needs.  Heavy dependencies (pyfaidx / pyBigWig) are imported
lazily so that the analysis and self-test paths run without them.

Context precedence (first match wins), chosen to match the published panel:

    splice_donor > splice_acceptor > coding_exon > five_utr > three_utr
    > intron > intergenic

Splice sites are the two intronic bases adjacent to an exon boundary, assigned
strand-aware: donor is the 5' end of the intron on the transcript's strand.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONTEXT_CODES = {
    "unassigned": 0,
    "intergenic": 1,
    "intron": 2,
    "coding_exon": 3,
    "five_utr": 4,
    "three_utr": 5,
    "splice_donor": 6,
    "splice_acceptor": 7,
}
CODE_TO_CONTEXT = {v: k for k, v in CONTEXT_CODES.items()}


@dataclass
class ChromTracks:
    chrom: str
    length: int
    context: np.ndarray        # int8 codes
    repeat: np.ndarray         # uint8, 1 where soft-masked (lowercase)
    gc: np.ndarray             # float32, GC fraction in +-50 bp
    phylop: np.ndarray         # float32, NaN where unavailable
    boundary_dist: np.ndarray  # int32, distance to nearest annotation boundary
    feature_len: np.ndarray    # int32, length of the feature the base sits in
    sequence: Optional[np.ndarray] = None  # uint8 ASCII, upper-cased


def _iter_gtf(gtf_path: str, chrom: str) -> Iterable[Tuple[str, int, int, str, str, str]]:
    import gzip

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[0] != chrom:
                continue
            feat, start, end, strand, attrs = f[2], int(f[3]) - 1, int(f[4]), f[6], f[8]
            tid = ""
            for kv in attrs.split(";"):
                kv = kv.strip()
                if kv.startswith("transcript_id"):
                    tid = kv.split('"')[1] if '"' in kv else kv.split(" ")[-1]
                    break
            yield feat, start, end, strand, tid, attrs


def build_context_track(gtf_path: str, chrom: str, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (context codes, boundary distance, feature length)."""
    ctx = np.full(length, CONTEXT_CODES["intergenic"], dtype=np.int8)
    feat_len = np.zeros(length, dtype=np.int32)
    boundaries: List[int] = []

    exons_by_tx: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    strand_by_tx: Dict[str, str] = {}

    gene_spans: List[Tuple[int, int]] = []
    cds: List[Tuple[int, int]] = []
    utr5: List[Tuple[int, int]] = []
    utr3: List[Tuple[int, int]] = []

    for feat, s, e, strand, tid, _ in _iter_gtf(gtf_path, chrom):
        s = max(0, s)
        e = min(length, e)
        if e <= s:
            continue
        if feat == "gene":
            gene_spans.append((s, e))
            boundaries.extend((s, e))
        elif feat == "exon":
            exons_by_tx[tid].append((s, e))
            strand_by_tx[tid] = strand
            boundaries.extend((s, e))
        elif feat == "CDS":
            cds.append((s, e))
        elif feat in ("five_prime_utr", "5UTR", "five_prime_UTR"):
            utr5.append((s, e))
        elif feat in ("three_prime_utr", "3UTR", "three_prime_UTR"):
            utr3.append((s, e))

    # genes -> intron by default, then paint exonic classes over it
    for s, e in gene_spans:
        ctx[s:e] = CONTEXT_CODES["intron"]
        feat_len[s:e] = np.maximum(feat_len[s:e], e - s)
    for s, e in utr5:
        ctx[s:e] = CONTEXT_CODES["five_utr"]
        feat_len[s:e] = e - s
    for s, e in utr3:
        ctx[s:e] = CONTEXT_CODES["three_utr"]
        feat_len[s:e] = e - s
    for s, e in cds:
        ctx[s:e] = CONTEXT_CODES["coding_exon"]
        feat_len[s:e] = e - s

    # splice sites: two intronic bases flanking each internal exon boundary
    for tid, exons in exons_by_tx.items():
        if len(exons) < 2:
            continue
        exons = sorted(exons)
        strand = strand_by_tx.get(tid, "+")
        for (s1, e1), (s2, _e2) in zip(exons[:-1], exons[1:]):
            intron_s, intron_e = e1, s2
            if intron_e - intron_s < 4:
                continue
            left = slice(intron_s, intron_s + 2)     # 5' end of the intron
            right = slice(intron_e - 2, intron_e)    # 3' end of the intron
            if strand == "+":
                ctx[left] = CONTEXT_CODES["splice_donor"]
                ctx[right] = CONTEXT_CODES["splice_acceptor"]
            else:
                ctx[left] = CONTEXT_CODES["splice_acceptor"]
                ctx[right] = CONTEXT_CODES["splice_donor"]
            boundaries.extend((intron_s, intron_e))

    bd = _distance_to_boundaries(np.asarray(sorted(set(boundaries)), dtype=np.int64), length)
    return ctx, bd, feat_len


def _distance_to_boundaries(bnd: np.ndarray, length: int) -> np.ndarray:
    if bnd.size == 0:
        return np.full(length, np.iinfo(np.int32).max, dtype=np.int32)
    pos = np.arange(length, dtype=np.int64)
    idx = np.searchsorted(bnd, pos)
    left = np.where(idx > 0, pos - bnd[np.clip(idx - 1, 0, len(bnd) - 1)], np.iinfo(np.int32).max)
    right = np.where(idx < len(bnd), bnd[np.clip(idx, 0, len(bnd) - 1)] - pos, np.iinfo(np.int32).max)
    return np.minimum(left, right).astype(np.int32)


def gc_track(seq_upper: np.ndarray, half_window: int = 50) -> np.ndarray:
    is_gc = ((seq_upper == ord("G")) | (seq_upper == ord("C"))).astype(np.float32)
    valid = (seq_upper != ord("N")).astype(np.float32)
    k = 2 * half_window + 1
    csum = np.concatenate([[0.0], np.cumsum(is_gc, dtype=np.float64)])
    cval = np.concatenate([[0.0], np.cumsum(valid, dtype=np.float64)])
    n = len(seq_upper)
    lo = np.clip(np.arange(n) - half_window, 0, n)
    hi = np.clip(np.arange(n) + half_window + 1, 0, n)
    num = csum[hi] - csum[lo]
    den = np.maximum(cval[hi] - cval[lo], 1.0)
    return (num / den).astype(np.float32)


def load_chrom_tracks(
    chrom: str,
    fasta: str,
    gtf: str,
    phylop_bw: Optional[str] = None,
    keep_sequence: bool = True,
) -> ChromTracks:
    from pyfaidx import Fasta

    fa = Fasta(fasta, as_raw=True, sequence_always_upper=False)
    raw = str(fa[chrom][:])
    arr = np.frombuffer(raw.encode(), dtype=np.uint8)
    repeat = ((arr >= ord("a")) & (arr <= ord("z"))).astype(np.uint8)
    upper = np.where(repeat.astype(bool), arr - 32, arr).astype(np.uint8)
    length = len(upper)

    ctx, bd, flen = build_context_track(gtf, chrom, length)
    gc = gc_track(upper)

    phylop = np.full(length, np.nan, dtype=np.float32)
    if phylop_bw:
        try:
            import pyBigWig

            bw = pyBigWig.open(phylop_bw)
            vals = bw.values(chrom, 0, length, numpy=True)
            phylop = np.asarray(vals, dtype=np.float32)
            bw.close()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[labels] phyloP unavailable for {chrom}: {exc}")

    return ChromTracks(
        chrom=chrom,
        length=length,
        context=ctx,
        repeat=repeat,
        gc=gc,
        phylop=phylop,
        boundary_dist=bd,
        feature_len=flen,
        sequence=upper if keep_sequence else None,
    )
