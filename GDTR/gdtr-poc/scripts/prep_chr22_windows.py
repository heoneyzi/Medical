"""Phase 1.6 prep: chr22 windows + per-position annotation + per-window context counts.

Step 2a: 6kb sliding windows, stride 3kb, drop windows with N-fraction > 0.01.
Step 2b: per-position uint8 label (intergenic/intron/exon/UTR/splice).
Step 2c: per-window counts in the central 3kb (anchor-only) for stratification.

Outputs:
    /path/to/gDTR/data/baselines/chr22_windows.tsv
    /path/to/gDTR/data/annotation/chr22_position_labels.npy
    /path/to/gDTR/data/annotation/chr22_position_labels_codebook.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import gffutils
import numpy as np
from pyfaidx import Fasta

REF = "/path/to/gDTR/data/reference/chr22.fa"
GTF_DB = "/path/to/gDTR/data/annotation/gencode.v44.chr17_chr22.gtf.db"

OUT_TSV = Path("/path/to/gDTR/data/baselines/chr22_windows.tsv")
OUT_LABELS = Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy")
OUT_CODEBOOK = Path("/path/to/gDTR/data/annotation/chr22_position_labels_codebook.json")

WINDOW = 6000
STRIDE = 3000
N_FRAC_MAX = 0.01
SPLICE_PAD = 10  # +/- 10 bp around donor / acceptor

CODEBOOK = {
    "intergenic": 0,  # default
    "intron": 1,
    "coding_exon": 2,
    "5utr": 3,
    "3utr": 4,
    "splice_donor": 5,
    "splice_acceptor": 6,
    "repeat": 7,
}
# Precedence (later overwrites earlier so highest-precedence wins):
# intergenic -> intron -> 5utr/3utr -> coding_exon -> splice (donor/acceptor)
# We apply features layer-by-layer in this order.


def pick_canonical_transcript(db: "gffutils.FeatureDB", gene) -> "gffutils.Feature | None":
    """Return MANE Select transcript if available else longest coding else longest."""
    transcripts = list(db.children(gene, featuretype="transcript", level=1))
    if not transcripts:
        return None
    # MANE Select: tag includes 'MANE_Select'
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


def build_chr22_labels(db, chrom_len: int) -> np.ndarray:
    """One label per base on chr22 — uint8 per CODEBOOK."""
    labels = np.zeros(chrom_len, dtype=np.uint8)  # all intergenic to start

    print("[chr22] iterating GENCODE chr22 genes...", flush=True)
    genes = list(db.region(seqid="chr22", featuretype="gene"))
    print(f"[chr22] {len(genes)} chr22 gene records", flush=True)

    n_used = 0
    for gene in genes:
        tr = pick_canonical_transcript(db, gene)
        if tr is None:
            continue
        n_used += 1
        # Children of this transcript
        exons = sorted(db.children(tr, featuretype="exon", level=1),
                       key=lambda e: (e.start, e.end))
        if not exons:
            continue

        tr_start_0 = tr.start - 1  # gffutils is 1-based inclusive
        tr_end_0 = tr.end          # exclusive in numpy slicing
        # Layer 1: paint whole transcript span as intron (default)
        labels[tr_start_0:tr_end_0] = np.maximum(
            labels[tr_start_0:tr_end_0], CODEBOOK["intron"]
        )
        # But we want intron to be a hard label, not max — we want these features
        # to STACK with precedence. Since exon (2) > intron (1), and 5/3 utr (3,4) >
        # exon, and splice (5,6) > UTR — we can simply assign by precedence layer.
        # Restart cleanly:
        labels[tr_start_0:tr_end_0] = CODEBOOK["intron"]

        # Layer 2: paint exons (default coding_exon, then UTR overrides)
        for ex in exons:
            s, e = ex.start - 1, ex.end
            labels[s:e] = CODEBOOK["coding_exon"]

        # Layer 3: UTRs (GENCODE v44 GTF only has "UTR"; split via CDS bounds + strand)
        cds_list = list(db.children(tr, featuretype="CDS", level=1))
        if cds_list:
            cds_min = min(c.start for c in cds_list)
            cds_max = max(c.end for c in cds_list)
            for utr in db.children(tr, featuretype="UTR", level=1):
                u_s, u_e = utr.start - 1, utr.end
                if tr.strand == "+":
                    if utr.end <= cds_min:
                        labels[u_s:u_e] = CODEBOOK["5utr"]
                    elif utr.start >= cds_max:
                        labels[u_s:u_e] = CODEBOOK["3utr"]
                else:
                    if utr.start >= cds_max:
                        labels[u_s:u_e] = CODEBOOK["5utr"]
                    elif utr.end <= cds_min:
                        labels[u_s:u_e] = CODEBOOK["3utr"]

        # Layer 4: splice sites at exon-intron boundaries (intron side, +/-10)
        # For each adjacent exon pair, the intron sits between them.
        # Forward strand convention: donor is at intron 5' end (right after exon),
        # acceptor at intron 3' end (right before next exon).
        if len(exons) >= 2:
            strand = tr.strand
            for i in range(len(exons) - 1):
                # Intron between exons[i] and exons[i+1]
                intron_start_0 = exons[i].end           # 0-based start of intron
                intron_end_0 = exons[i + 1].start - 1   # 0-based exclusive end of intron
                if intron_end_0 <= intron_start_0:
                    continue
                if strand == "+":
                    donor_a = max(0, intron_start_0 - SPLICE_PAD)
                    donor_b = min(chrom_len, intron_start_0 + SPLICE_PAD)
                    acc_a = max(0, intron_end_0 - SPLICE_PAD)
                    acc_b = min(chrom_len, intron_end_0 + SPLICE_PAD)
                else:
                    # On - strand, what was the "intron 5' end" in genome coords
                    # is actually the acceptor for the gene; flip labels.
                    donor_a = max(0, intron_end_0 - SPLICE_PAD)
                    donor_b = min(chrom_len, intron_end_0 + SPLICE_PAD)
                    acc_a = max(0, intron_start_0 - SPLICE_PAD)
                    acc_b = min(chrom_len, intron_start_0 + SPLICE_PAD)
                labels[donor_a:donor_b] = CODEBOOK["splice_donor"]
                labels[acc_a:acc_b] = CODEBOOK["splice_acceptor"]

    print(f"[chr22] painted {n_used} canonical transcripts", flush=True)
    return labels


def main() -> None:
    t0 = time.time()
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)

    print("[chr22] loading reference...", flush=True)
    fa = Fasta(REF, as_raw=True, sequence_always_upper=True)
    seq = str(fa["chr22"][:])
    chrom_len = len(seq)
    print(f"[chr22] len={chrom_len:,}", flush=True)

    # ---- Step 2b: per-position labels ----
    print("[chr22] opening gffutils db...", flush=True)
    db = gffutils.FeatureDB(GTF_DB, keep_order=True)
    labels = build_chr22_labels(db, chrom_len)

    # Write codebook + labels (TODO: repeat masker not available -> repeat==intergenic)
    codebook_full = dict(CODEBOOK)
    codebook_full["_TODO"] = (
        "RepeatMasker BED not available locally; repeat positions "
        "currently labelled as intergenic. Add repeat layer when BED is loaded."
    )
    with open(OUT_CODEBOOK, "w") as f:
        json.dump(codebook_full, f, indent=2)
    np.save(OUT_LABELS, labels)
    print(f"[chr22] wrote labels {OUT_LABELS}  size={OUT_LABELS.stat().st_size/1e6:.1f}MB", flush=True)

    label_counts = {k: int((labels == v).sum()) for k, v in CODEBOOK.items() if k != "repeat"}
    print(f"[chr22] label distribution (bp): {label_counts}", flush=True)

    # ---- Step 2a: 6kb / 3kb sliding windows with N-filter ----
    print("[chr22] enumerating sliding windows...", flush=True)
    rows = []
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    N_BYTE = ord("N")
    G_BYTE = ord("G")
    C_BYTE = ord("C")

    # Pre-compute cumulative counts of N, G, C for O(1) per-window lookup
    cum_n = np.concatenate(([0], np.cumsum((seq_arr == N_BYTE).astype(np.int64))))
    cum_g = np.concatenate(([0], np.cumsum((seq_arr == G_BYTE).astype(np.int64))))
    cum_c = np.concatenate(([0], np.cumsum((seq_arr == C_BYTE).astype(np.int64))))

    def cnt(cum: np.ndarray, a: int, b: int) -> int:
        return int(cum[b] - cum[a])

    # Pre-compute label-count cumulative per code (8 codes)
    label_cum = np.zeros((8, chrom_len + 1), dtype=np.int64)
    for code in range(8):
        label_cum[code, 1:] = np.cumsum((labels == code).astype(np.int64))

    def label_cnt(code: int, a: int, b: int) -> int:
        return int(label_cum[code, b] - label_cum[code, a])

    win_idx = 0
    skipped = 0
    for start in range(0, chrom_len - WINDOW + 1, STRIDE):
        end = start + WINDOW
        n_count = cnt(cum_n, start, end)
        n_frac = n_count / WINDOW
        if n_frac > N_FRAC_MAX:
            skipped += 1
            continue
        gc = (cnt(cum_g, start, end) + cnt(cum_c, start, end))
        denom = WINDOW - n_count
        gc_frac = gc / denom if denom > 0 else 0.0
        center = start + WINDOW // 2
        # Center 3 kb (anchor): [start + 1500, start + 4500)
        c_a = start + (WINDOW - STRIDE) // 2
        c_b = c_a + STRIDE
        n_coding_exon = label_cnt(CODEBOOK["coding_exon"], c_a, c_b)
        n_intron = label_cnt(CODEBOOK["intron"], c_a, c_b)
        n_5utr = label_cnt(CODEBOOK["5utr"], c_a, c_b)
        n_3utr = label_cnt(CODEBOOK["3utr"], c_a, c_b)
        n_splice = (
            label_cnt(CODEBOOK["splice_donor"], c_a, c_b)
            + label_cnt(CODEBOOK["splice_acceptor"], c_a, c_b)
        )
        n_intergenic = label_cnt(CODEBOOK["intergenic"], c_a, c_b)
        rows.append((
            win_idx, "chr22", start, end,
            round(gc_frac, 6), round(n_frac, 6), center,
            n_coding_exon, n_intron, n_5utr, n_3utr, n_splice, n_intergenic,
        ))
        win_idx += 1

    print(f"[chr22] kept {len(rows):,} windows, skipped {skipped} for N>{N_FRAC_MAX}", flush=True)

    # Write TSV
    cols = ["window_idx", "chrom", "start", "end", "gc_content", "n_fraction", "center_pos",
            "n_coding_exon", "n_intron", "n_5utr", "n_3utr", "n_splice", "n_intergenic"]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"[chr22] wrote {OUT_TSV}", flush=True)

    # Summarize per-context "rich" windows (>= 50% of central 3kb)
    rich_thr = STRIDE // 2
    counters = defaultdict(int)
    for r in rows:
        (_, _, _, _, _, _, _, ne, ni, n5, n3, ns, nig) = r
        if ne >= rich_thr:
            counters["exon_rich"] += 1
        if ni >= rich_thr:
            counters["intron_rich"] += 1
        if (n5 + n3) >= rich_thr:
            counters["utr_rich"] += 1
        if nig >= rich_thr:
            counters["intergenic_rich"] += 1
        if ns > 0:
            counters["any_splice"] += 1
    print(f"[chr22] context-rich windows: {dict(counters)}", flush=True)

    print(f"[chr22] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
