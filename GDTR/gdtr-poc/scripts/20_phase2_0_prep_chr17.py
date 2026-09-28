"""Phase 2.0 prep: chr17 windows + per-position annotation + per-window context counts
+ gene-class JSON (cancer_driver vs other for chr17 protein-coding).

Adapted from scripts/prep_chr22_windows.py.

Outputs:
    /path/to/gDTR/data/baselines/chr17_windows.tsv
    /path/to/gDTR/data/annotation/chr17_position_labels.npy
    /path/to/gDTR/data/annotation/chr17_position_labels.json (sidecar)
    /path/to/gDTR/data/baselines/chr17_gene_class.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import gffutils
import numpy as np
from pyfaidx import Fasta

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase2.0"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.0"
LOG = ru.setup_logging(PHASE)

REF = "/path/to/gDTR/data/reference/chr17.fa"
GTF_DB = "/path/to/gDTR/data/annotation/gencode.v44.chr17_chr22.gtf.db"
CHROM = "chr17"

OUT_TSV = Path("/path/to/gDTR/data/baselines/chr17_windows.tsv")
OUT_LABELS = Path("/path/to/gDTR/data/annotation/chr17_position_labels.npy")
OUT_LABELS_JSON = Path("/path/to/gDTR/data/annotation/chr17_position_labels.json")
OUT_GENE_CLASS = Path("/path/to/gDTR/data/baselines/chr17_gene_class.json")

WINDOW = 6000
STRIDE = 3000
N_FRAC_MAX = 0.01
SPLICE_PAD = 10

CODEBOOK = {
    "intergenic": 0,
    "intron": 1,
    "coding_exon": 2,
    "5utr": 3,
    "3utr": 4,
    "splice_donor": 5,
    "splice_acceptor": 6,
    "repeat": 7,
}

CHR17_CANCER_DRIVERS = {"TP53", "BRCA1", "ATM"}


def pick_canonical_transcript(db: "gffutils.FeatureDB", gene) -> "gffutils.Feature | None":
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


def build_chr17_labels(db, chrom_len: int) -> np.ndarray:
    labels = np.zeros(chrom_len, dtype=np.uint8)
    LOG.info("[chr17] iterating GENCODE chr17 genes...")
    genes = list(db.region(seqid=CHROM, featuretype="gene"))
    LOG.info("[chr17] %d chr17 gene records", len(genes))

    n_used = 0
    for gene in genes:
        tr = pick_canonical_transcript(db, gene)
        if tr is None:
            continue
        n_used += 1
        exons = sorted(db.children(tr, featuretype="exon", level=1),
                       key=lambda e: (e.start, e.end))
        if not exons:
            continue

        tr_start_0 = tr.start - 1
        tr_end_0 = tr.end
        # Layer 1 (intron) - hard assign
        labels[tr_start_0:tr_end_0] = CODEBOOK["intron"]

        # Layer 2: exons
        for ex in exons:
            s, e = ex.start - 1, ex.end
            labels[s:e] = CODEBOOK["coding_exon"]

        # Layer 3: UTRs (split by CDS bounds + strand) — Phase 1 fix
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

        # Layer 4: splice donor/acceptor ±10 bp
        if len(exons) >= 2:
            strand = tr.strand
            for i in range(len(exons) - 1):
                intron_start_0 = exons[i].end
                intron_end_0 = exons[i + 1].start - 1
                if intron_end_0 <= intron_start_0:
                    continue
                if strand == "+":
                    donor_a = max(0, intron_start_0 - SPLICE_PAD)
                    donor_b = min(chrom_len, intron_start_0 + SPLICE_PAD)
                    acc_a = max(0, intron_end_0 - SPLICE_PAD)
                    acc_b = min(chrom_len, intron_end_0 + SPLICE_PAD)
                else:
                    donor_a = max(0, intron_end_0 - SPLICE_PAD)
                    donor_b = min(chrom_len, intron_end_0 + SPLICE_PAD)
                    acc_a = max(0, intron_start_0 - SPLICE_PAD)
                    acc_b = min(chrom_len, intron_start_0 + SPLICE_PAD)
                labels[donor_a:donor_b] = CODEBOOK["splice_donor"]
                labels[acc_a:acc_b] = CODEBOOK["splice_acceptor"]

    LOG.info("[chr17] painted %d canonical transcripts", n_used)
    return labels


def build_gene_class(db) -> dict:
    """Build gene class JSON for chr17 protein-coding genes."""
    cancer_driver = []
    other = []
    found_drivers = set()
    for g in db.region(seqid=CHROM, featuretype="gene"):
        gtype = g.attributes.get("gene_type", [None])[0]
        if gtype != "protein_coding":
            continue
        gname = g.attributes.get("gene_name", [g.id])[0]
        rec = {
            "gene_id": g.id,
            "gene_name": gname,
            "start": int(g.start) - 1,
            "end": int(g.end),
            "strand": g.strand,
        }
        if gname in CHR17_CANCER_DRIVERS:
            cancer_driver.append(rec)
            found_drivers.add(gname)
        else:
            other.append(rec)
    LOG.info("[chr17] gene class: cancer_driver=%d (found %s), other=%d",
             len(cancer_driver), sorted(found_drivers), len(other))
    return {
        "chrom": CHROM,
        "cancer_driver_target_set": sorted(CHR17_CANCER_DRIVERS),
        "cancer_driver_found": sorted(found_drivers),
        "cancer_driver": cancer_driver,
        "other": other,
    }


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="prep_chr17"):
        t0 = time.time()
        OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
        OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)

        LOG.info("[chr17] loading reference...")
        fa = Fasta(REF, as_raw=True, sequence_always_upper=True)
        seq = str(fa[CHROM][:])
        chrom_len = len(seq)
        LOG.info("[chr17] len=%s", f"{chrom_len:,}")

        LOG.info("[chr17] opening gffutils db...")
        db = gffutils.FeatureDB(GTF_DB, keep_order=True)
        labels = build_chr17_labels(db, chrom_len)

        # Sidecar JSON metadata
        sidecar = {
            "chrom": CHROM,
            "chrom_len": chrom_len,
            "codebook": CODEBOOK,
            "label_counts_bp": {k: int((labels == v).sum())
                                for k, v in CODEBOOK.items() if k != "repeat"},
            "splice_pad": SPLICE_PAD,
            "_TODO": ("RepeatMasker BED not available locally; repeat positions "
                      "currently labelled as intergenic. Add repeat layer when BED is loaded."),
        }
        OUT_LABELS_JSON.write_text(json.dumps(sidecar, indent=2))
        np.save(OUT_LABELS, labels)
        LOG.info("[chr17] wrote labels %s size=%.1fMB",
                 OUT_LABELS, OUT_LABELS.stat().st_size / 1e6)
        LOG.info("[chr17] label distribution (bp): %s", sidecar["label_counts_bp"])

        # Gene-class JSON (uses same db handle)
        gene_class = build_gene_class(db)
        OUT_GENE_CLASS.write_text(json.dumps(gene_class, indent=2))
        LOG.info("[chr17] wrote gene class -> %s", OUT_GENE_CLASS)

        # ---- Sliding windows ----
        LOG.info("[chr17] enumerating sliding windows...")
        rows = []
        seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        N_BYTE = ord("N"); G_BYTE = ord("G"); C_BYTE = ord("C")
        cum_n = np.concatenate(([0], np.cumsum((seq_arr == N_BYTE).astype(np.int64))))
        cum_g = np.concatenate(([0], np.cumsum((seq_arr == G_BYTE).astype(np.int64))))
        cum_c = np.concatenate(([0], np.cumsum((seq_arr == C_BYTE).astype(np.int64))))

        def cnt(cum, a, b):
            return int(cum[b] - cum[a])

        label_cum = np.zeros((8, chrom_len + 1), dtype=np.int64)
        for code in range(8):
            label_cum[code, 1:] = np.cumsum((labels == code).astype(np.int64))

        def label_cnt(code, a, b):
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
            gc = cnt(cum_g, start, end) + cnt(cum_c, start, end)
            denom = WINDOW - n_count
            gc_frac = gc / denom if denom > 0 else 0.0
            center = start + WINDOW // 2
            c_a = start + (WINDOW - STRIDE) // 2
            c_b = c_a + STRIDE
            n_coding_exon = label_cnt(CODEBOOK["coding_exon"], c_a, c_b)
            n_intron = label_cnt(CODEBOOK["intron"], c_a, c_b)
            n_5utr = label_cnt(CODEBOOK["5utr"], c_a, c_b)
            n_3utr = label_cnt(CODEBOOK["3utr"], c_a, c_b)
            n_splice = (label_cnt(CODEBOOK["splice_donor"], c_a, c_b)
                        + label_cnt(CODEBOOK["splice_acceptor"], c_a, c_b))
            n_intergenic = label_cnt(CODEBOOK["intergenic"], c_a, c_b)
            rows.append((
                win_idx, CHROM, start, end,
                round(gc_frac, 6), round(n_frac, 6), center,
                n_coding_exon, n_intron, n_5utr, n_3utr, n_splice, n_intergenic,
            ))
            win_idx += 1
        LOG.info("[chr17] kept %d windows, skipped %d for N>%.2f",
                 len(rows), skipped, N_FRAC_MAX)

        cols = ["window_idx", "chrom", "start", "end", "gc_content", "n_fraction", "center_pos",
                "n_coding_exon", "n_intron", "n_5utr", "n_3utr", "n_splice", "n_intergenic"]
        with open(OUT_TSV, "w") as f:
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(str(v) for v in row) + "\n")
        LOG.info("[chr17] wrote %s", OUT_TSV)

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
        LOG.info("[chr17] context-rich windows: %s", dict(counters))
        LOG.info("[chr17] done in %.1fs", time.time() - t0)

        ru.write_done(PHASE, PHASE_OUT_DIR,
                      {"n_windows": len(rows), "chrom_len": chrom_len,
                       "n_cancer_driver_genes": len(gene_class["cancer_driver"]),
                       "n_other_protein_coding": len(gene_class["other"])},
                      step_name="prep_chr17")


if __name__ == "__main__":
    main()
