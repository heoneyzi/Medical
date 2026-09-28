"""Build a splice-strength perturbation dataset for gDTR causal-intervention analysis.

Samples splice donor (label=5) and acceptor (label=6) sites from chr22 position
labels, extracts GRCh38 windows centered on each site, and creates 4 variant
sequences per site by systematic mutation of the core consensus motif:

  canonical_strong  → 9-mer (donor) / 23-mer (acceptor) replaced with best consensus
  wild_type         → original sequence, no change
  weakened          → GT/AG preserved; surrounding context mutated toward null
  ablated           → invariant dinucleotide destroyed (GT→GC, AG→AC)

Strand is determined from gffutils GTF; minus-strand sites are handled by
applying mutations in plus-strand coordinates (RC mapping).

Donor PWM uses Shapiro-Senapathy 9-mer (positions −3 to +6 relative to 5'ss).
Acceptor PWM uses a simplified 23-mer (positions −20 to +3 relative to 3'ss).

Output:
  results/splice_strength_dataset/
    sequences.fa   — FASTA, one entry per variant
    metadata.csv   — seq_id, splice_type, site_pos, strand, variant_class,
                     pwm_score, n_mutations, window_start, window_end
    summary.json   — config and global statistics

Usage (server):
  PYTHONPATH=. python scripts/make_splice_strength_dataset.py

Custom paths (optional env overrides):
  FASTA_PATH=/path/to/chr22.fa
  LABELS_PATH=/path/to/chr22_position_labels.npy
  GTF_DB_PATH=/path/to/gencode.v44.chr17_chr22.gtf.db
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Paths (override via env vars on the server)
# ---------------------------------------------------------------------------
GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR"))

FASTA_PATH  = Path(os.environ.get("FASTA_PATH",
                   str(GDTR_ROOT / "data" / "reference" / "chr22.fa")))
LABELS_PATH = Path(os.environ.get("LABELS_PATH",
                   str(GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy")))
GTF_DB_PATH = Path(os.environ.get("GTF_DB_PATH",
                   str(GDTR_ROOT / "data" / "annotation" /
                       "gencode.v44.chr17_chr22.gtf.db")))

OUT_DIR = GDTR_ROOT / "results" / "splice_strength_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("splice_strength_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROM = "chr22"
WINDOW_HALF = 2048          # bp on each side of the splice-site center
N_SITES     = 200           # sites to sample per splice type
SEED        = 42

LABEL_DONOR    = 5          # splice_donor in position_labels.npy
LABEL_ACCEPTOR = 6          # splice_acceptor

# Variant classes ordered from strongest to weakest splicing
VARIANT_CLASSES: list[str] = ["canonical_strong", "wild_type", "weakened", "ablated"]

# ---------------------------------------------------------------------------
# Donor 9-mer PWM
# Shapiro & Senapathy (1987) / Rogan et al. (1998), Human 5'ss
# Positions 0..8 correspond to −3,−2,−1,+1,+2,+3,+4,+5,+6
# Pseudocount 0.001 prevents log(0) for near-invariant positions.
# ---------------------------------------------------------------------------
_EPS = 0.001
DONOR_PWM: dict[str, list[float]] = {
    "A": [0.320, 0.630, 0.090, _EPS,  _EPS,  0.590, 0.720, 0.070, 0.170],
    "C": [0.360, 0.100, 0.030, _EPS,  _EPS,  0.030, 0.060, 0.020, 0.100],
    "G": [0.180, 0.140, 0.780, 0.990, _EPS,  0.140, 0.120, 0.800, 0.150],
    "T": [0.140, 0.130, 0.100, _EPS,  0.999, 0.240, 0.100, 0.110, 0.580],
}
DONOR_PWM_LEN = 9          # exon positions: [0,1,2] / intron positions: [3..8]
DONOR_EXON_END = 3          # 9-mer index up-to (exclusive) that is exonic
# +1 of 5'ss is at index 3 in the 9-mer; center of label is the G at +1

# Strongest possible donor 9-mer (argmax at each position)
DONOR_CANONICAL = "CAGGTAAGT"
# Weakened intronic tail (+3..+6) — keeps GT, disrupts AAGT consensus
DONOR_WEAK_INTRON = "CCCC"     # replaces positions +3,+4,+5,+6

# ---------------------------------------------------------------------------
# Acceptor 23-mer PWM (simplified)
# Positions 0..22 correspond to −20,−19,...,−1,+1,+2,+3
# Key features: pyrimidine-rich PPT (pos 0..16 = −20..−4),
#               invariant A at pos 17 (−3), C at 18 (−2 of pyrimidine context),
#               A at 19 (−2), G at 20 (−1), then +1..+3.
# Reference: Senapathy 1987; Shapiro 1987; simplified from MaxEntScan 3ss model.
# ---------------------------------------------------------------------------
#   idx:  0     1     2     3     4     5     6     7     8     9
#   pos: -20   -19   -18   -17   -16   -15   -14   -13   -12   -11
#   idx: 10    11    12    13    14    15    16    17    18    19    20    21    22
#   pos: -10   -9    -8    -7    -6    -5    -4    -3    -2    -1    +1    +2    +3
_PPT_T  = 0.65   # fraction T at PPT positions (simplified)
_PPT_C  = 0.25   # fraction C at PPT positions
_PPT_A  = 0.06
_PPT_G  = 0.04
ACCEPTOR_PWM: dict[str, list[float]] = {
    "A": ([_PPT_A] * 17 + [0.050, 0.950, _EPS,  0.250, 0.250, 0.250]),
    "C": ([_PPT_C] * 17 + [0.050, 0.020, _EPS,  0.250, 0.250, 0.250]),
    "G": ([_PPT_G] * 17 + [0.050, _EPS,  0.990, 0.250, 0.250, 0.250]),
    "T": ([_PPT_T] * 17 + [0.850, 0.030, _EPS,  0.250, 0.250, 0.250]),
}
ACCEPTOR_PWM_LEN = 23
# −1 of 3'ss (the G of AG) is at index 20 in the 23-mer; center of label = G at −1
ACCEPTOR_AG_IDX = (19, 20)   # indices of A and G in the 23-mer

# Strongest possible acceptor 23-mer: full-T PPT (−20..−3) + AG + GGG(+1..+3)
ACCEPTOR_CANONICAL = "T" * 17 + "TAGG" + "GGG"  # 17+4+3 = 24 → trim to 23
ACCEPTOR_CANONICAL = ACCEPTOR_CANONICAL[:ACCEPTOR_PWM_LEN]

# ---------------------------------------------------------------------------
# Utility: reverse complement
# ---------------------------------------------------------------------------
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def pwm_score(seq: str, pwm: dict[str, list[float]], length: int) -> float:
    """Log-sum PWM score (information content per position, base 2)."""
    if len(seq) != length:
        return float("nan")
    score = 0.0
    for i, nt in enumerate(seq.upper()):
        if nt not in pwm:
            return float("nan")
        f = max(pwm[nt][i], 1e-9)
        score += float(np.log2(f / 0.25))
    return score


# ---------------------------------------------------------------------------
# FASTA reader  (uses pysam on the server; falls back to a slow plain reader)
# ---------------------------------------------------------------------------
def _open_fasta(path: Path):
    try:
        import pysam
        return pysam.FastaFile(str(path))
    except ImportError:
        LOG.warning("pysam not found; falling back to plain FASTA reader (slow)")
        return None


def fetch_seq(fasta, chrom: str, start: int, end: int, fasta_path: Path) -> str:
    """Fetch [start, end) 0-based from FASTA.  Returns upper-case DNA or 'N'*len."""
    length = end - start
    if fasta is not None:
        try:
            return fasta.fetch(chrom, start, end).upper()
        except Exception as e:
            LOG.warning("pysam fetch failed (%s); returning N*%d", e, length)
            return "N" * length
    # Plain fallback: load whole sequence from memory (slow, avoidable)
    if not hasattr(fetch_seq, "_cache"):
        fetch_seq._cache = {}
    if chrom not in fetch_seq._cache:
        LOG.info("loading %s into memory (plain reader) …", chrom)
        seq = []
        with open(fasta_path) as fh:
            inside = False
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    inside = line[1:].split()[0] == chrom
                elif inside:
                    seq.append(line)
        fetch_seq._cache[chrom] = "".join(seq).upper()
    full = fetch_seq._cache[chrom]
    return full[start:end]


# ---------------------------------------------------------------------------
# Strand lookup from gffutils GTF
# ---------------------------------------------------------------------------
def build_strand_map(labels: np.ndarray, gtf_db_path: Path) -> dict[int, str]:
    """Return {position: '+'/'-'} for all labeled donor/acceptor positions."""
    try:
        import gffutils
        db = gffutils.FeatureDB(str(gtf_db_path))
    except Exception as e:
        LOG.warning("gffutils not available (%s); strand will be inferred from sequence.", e)
        return {}

    strand_map: dict[int, str] = {}
    donor_pos    = np.flatnonzero(labels == LABEL_DONOR)
    acceptor_pos = np.flatnonzero(labels == LABEL_ACCEPTOR)
    all_pos = set(int(p) for p in np.concatenate([donor_pos, acceptor_pos]))

    LOG.info("querying GTF for strand of %d splice positions …", len(all_pos))
    for feat in db.features_of_type(["exon", "transcript"]):
        if feat.seqid != CHROM:
            continue
        s = int(feat.start) - 1   # GTF 1-based → 0-based
        e = int(feat.end)
        strand = feat.strand
        # The donor is at the exon end; acceptor is at the exon start
        if strand == "+":
            if e in all_pos:           # donor: last exon base + 1 ≈ intron start
                strand_map[e] = "+"
            if (s - 1) in all_pos:     # acceptor: first intron base on + strand
                strand_map[s - 1] = "+"
        else:
            if (s - 1) in all_pos:
                strand_map[s - 1] = "-"
            if e in all_pos:
                strand_map[e] = "-"
    LOG.info("strand resolved for %d / %d positions", len(strand_map), len(all_pos))
    return strand_map


def infer_strand(seq_at_center: str, site_type: Literal["donor", "acceptor"]) -> str:
    """Heuristic strand detection from the dinucleotide at the splice center."""
    if site_type == "donor":
        if seq_at_center[:2].upper() == "GT":
            return "+"
        if seq_at_center[-2:].upper() == "AC":   # RC of GT on minus strand
            return "-"
    else:
        if seq_at_center[-2:].upper() == "AG":
            return "+"
        if seq_at_center[:2].upper() == "CT":    # RC of AG on minus strand
            return "-"
    return "?"


# ---------------------------------------------------------------------------
# Variant generation — donor
# ---------------------------------------------------------------------------
def make_donor_variants(
    ref_window: str,
    center_in_window: int,
    strand: str,
) -> list[tuple[str, str, str, int]]:
    """Return list of (variant_class, mutated_window, motif_seq, n_mutations).

    The 9-mer occupies [center_in_window-3 : center_in_window+6] in plus-strand
    coordinates.  For minus-strand sites the 9-mer is RC-mapped.
    """
    results = []
    w = list(ref_window)
    motif_start = center_in_window - 3
    motif_end   = center_in_window + 6   # exclusive

    if motif_start < 0 or motif_end > len(w):
        LOG.debug("donor motif out of window bounds; skipping")
        return []

    ref_motif_plus = ref_window[motif_start:motif_end]    # 9-mer, plus strand

    if strand == "+":
        ref_motif = ref_motif_plus
    else:
        ref_motif = revcomp(ref_motif_plus)               # transcript orientation

    def _apply(new_motif_transcript: str) -> tuple[list[str], int]:
        """Convert transcript-orientation 9-mer back to plus-strand coords and patch."""
        if strand == "+":
            new_plus = new_motif_transcript
        else:
            new_plus = revcomp(new_motif_transcript)
        mut = list(ref_window)
        mut[motif_start:motif_end] = list(new_plus)
        n_mut = sum(a != b for a, b in zip(ref_motif_plus, new_plus))
        return mut, n_mut

    # 1. canonical_strong — replace full 9-mer with CAGGTAAGT
    mut_w, n_mut = _apply(DONOR_CANONICAL)
    results.append(("canonical_strong", "".join(mut_w), DONOR_CANONICAL, n_mut))

    # 2. wild_type — no change
    results.append(("wild_type", ref_window, ref_motif, 0))

    # 3. weakened — keep GT at +1+2; replace +3..+6 with CCCC
    #    ref_motif indices: 0=-3, 1=-2, 2=-1, 3=+1, 4=+2, 5=+3, 6=+4, 7=+5, 8=+6
    weak_motif = ref_motif[:5] + DONOR_WEAK_INTRON
    mut_w, n_mut = _apply(weak_motif)
    results.append(("weakened", "".join(mut_w), weak_motif, n_mut))

    # 4. ablated — GT → GC (disrupt invariant T, most common donor-null mutation)
    #    index 4 (+2) T→C
    ablated_motif = ref_motif[:4] + "C" + ref_motif[5:]
    mut_w, n_mut = _apply(ablated_motif)
    results.append(("ablated", "".join(mut_w), ablated_motif, n_mut))

    return results


# ---------------------------------------------------------------------------
# Variant generation — acceptor
# ---------------------------------------------------------------------------
def make_acceptor_variants(
    ref_window: str,
    center_in_window: int,
    strand: str,
) -> list[tuple[str, str, str, int]]:
    """Return list of (variant_class, mutated_window, motif_seq, n_mutations).

    The 23-mer spans [center_in_window-20 : center_in_window+3] in plus-strand
    coords (center = position of G at −1, i.e. the last intron base).
    """
    results = []
    motif_start = center_in_window - 20
    motif_end   = center_in_window + 3    # exclusive (positions −20..+2 → 23 nt)

    if motif_start < 0 or motif_end > len(ref_window):
        LOG.debug("acceptor motif out of window bounds; skipping")
        return []

    ref_motif_plus = ref_window[motif_start:motif_end]

    if strand == "+":
        ref_motif = ref_motif_plus
    else:
        ref_motif = revcomp(ref_motif_plus)

    def _apply(new_motif_transcript: str) -> tuple[list[str], int]:
        if strand == "+":
            new_plus = new_motif_transcript
        else:
            new_plus = revcomp(new_motif_transcript)
        mut = list(ref_window)
        mut[motif_start:motif_end] = list(new_plus)
        n_mut = sum(a != b for a, b in zip(ref_motif_plus, new_plus))
        return mut, n_mut

    # 1. canonical_strong — all-T PPT + TAGG + GGG
    mut_w, n_mut = _apply(ACCEPTOR_CANONICAL)
    results.append(("canonical_strong", "".join(mut_w), ACCEPTOR_CANONICAL, n_mut))

    # 2. wild_type
    results.append(("wild_type", ref_window, ref_motif, 0))

    # 3. weakened — replace all pyrimidines in PPT (indices 0..16) with A
    ppt_region = ref_motif[:17]
    ppt_weak   = "".join("A" if nt in "CT" else nt for nt in ppt_region)
    weak_motif = ppt_weak + ref_motif[17:]   # keep −3..−1, AG, +1..+3
    mut_w, n_mut = _apply(weak_motif)
    results.append(("weakened", "".join(mut_w), weak_motif, n_mut))

    # 4. ablated — AG → AC at indices 19,20 (positions −2,−1)
    ablated_motif = ref_motif[:20] + "C" + ref_motif[21:]  # G(−1) → C
    mut_w, n_mut = _apply(ablated_motif)
    results.append(("ablated", "".join(mut_w), ablated_motif, n_mut))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    LOG.info("loading position labels from %s", LABELS_PATH)
    labels     = np.load(LABELS_PATH)
    chrom_len  = int(labels.shape[0])
    LOG.info("chrom_len=%d", chrom_len)

    donor_positions    = np.flatnonzero(labels == LABEL_DONOR)
    acceptor_positions = np.flatnonzero(labels == LABEL_ACCEPTOR)
    LOG.info("donor sites=%d, acceptor sites=%d",
             donor_positions.size, acceptor_positions.size)

    # Strand map from GTF
    strand_map = build_strand_map(labels, GTF_DB_PATH)

    fasta = _open_fasta(FASTA_PATH)
    LOG.info("FASTA opened: %s", FASTA_PATH)

    records_fa:  list[str]  = []
    rows_csv:    list[dict] = []

    for site_type, positions in (("donor", donor_positions),
                                  ("acceptor", acceptor_positions)):
        # Exclude sites too close to chromosome ends
        valid = positions[
            (positions >= WINDOW_HALF) &
            (positions < chrom_len - WINDOW_HALF)
        ]
        if valid.size == 0:
            LOG.warning("no valid %s sites after boundary filtering", site_type)
            continue

        n_sample = min(N_SITES, valid.size)
        sampled  = rng.choice(valid, n_sample, replace=False)
        sampled.sort()
        LOG.info("sampled %d %s sites", n_sample, site_type)

        pwm = DONOR_PWM if site_type == "donor" else ACCEPTOR_PWM
        pwm_len = DONOR_PWM_LEN if site_type == "donor" else ACCEPTOR_PWM_LEN

        for site_pos in sampled:
            win_start = int(site_pos) - WINDOW_HALF
            win_end   = int(site_pos) + WINDOW_HALF
            center_in_window = WINDOW_HALF          # site_pos is center of window

            ref_window = fetch_seq(fasta, CHROM, win_start, win_end, FASTA_PATH)
            if len(ref_window) != 2 * WINDOW_HALF:
                LOG.debug("incomplete window at %d; skipping", site_pos)
                continue
            if "N" in ref_window:
                n_count = ref_window.count("N")
                if n_count > 50:
                    LOG.debug("too many Ns (%d) at %d; skipping", n_count, site_pos)
                    continue

            # Determine strand
            strand = strand_map.get(int(site_pos), None)
            if strand is None:
                # Heuristic: look at the dinucleotide at the center
                if site_type == "donor":
                    dinuc = ref_window[center_in_window:center_in_window + 2]
                else:
                    dinuc = ref_window[center_in_window - 1:center_in_window + 1]
                strand = infer_strand(dinuc, site_type)
                if strand == "?":
                    LOG.debug("strand unknown at %d %s; skipping", site_pos, site_type)
                    continue

            # Generate variant sequences
            if site_type == "donor":
                variants = make_donor_variants(ref_window, center_in_window, strand)
            else:
                variants = make_acceptor_variants(ref_window, center_in_window, strand)

            if not variants:
                continue

            for variant_class, mut_window, motif_seq, n_mut in variants:
                score = pwm_score(motif_seq, pwm, pwm_len)
                seq_id = (f"{CHROM}_{site_pos}_{site_type[0]}_{strand}_"
                          f"{variant_class}")
                records_fa.append(f">{seq_id}\n{mut_window}")
                rows_csv.append({
                    "seq_id":         seq_id,
                    "splice_type":    site_type,
                    "chrom":          CHROM,
                    "site_pos":       int(site_pos),
                    "strand":         strand,
                    "variant_class":  variant_class,
                    "pwm_score":      f"{score:.4f}" if np.isfinite(score) else "nan",
                    "n_mutations":    n_mut,
                    "motif_seq":      motif_seq,
                    "window_start":   win_start,
                    "window_end":     win_end,
                })

    # Write outputs
    fa_path  = OUT_DIR / "sequences.fa"
    csv_path = OUT_DIR / "metadata.csv"
    json_path = OUT_DIR / "summary.json"

    fa_path.write_text("\n".join(records_fa) + "\n")
    LOG.info("wrote %d sequences to %s", len(records_fa), fa_path)

    fieldnames = ["seq_id", "splice_type", "chrom", "site_pos", "strand",
                  "variant_class", "pwm_score", "n_mutations", "motif_seq",
                  "window_start", "window_end"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_csv)
    LOG.info("wrote %d rows to %s", len(rows_csv), csv_path)

    # Per-class PWM score statistics
    from collections import defaultdict
    score_by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows_csv:
        try:
            score_by_class[row["variant_class"]].append(float(row["pwm_score"]))
        except ValueError:
            pass

    class_stats = {}
    for vc, scores in score_by_class.items():
        arr = np.array(scores)
        class_stats[vc] = {
            "n": len(scores),
            "mean_pwm": float(arr.mean()),
            "median_pwm": float(np.median(arr)),
            "std_pwm": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        }

    summary = {
        "chrom": CHROM,
        "seed": SEED,
        "n_sites_per_type": N_SITES,
        "window_half_bp": WINDOW_HALF,
        "window_total_bp": 2 * WINDOW_HALF,
        "variant_classes": VARIANT_CLASSES,
        "n_sequences_total": len(records_fa),
        "n_rows_csv": len(rows_csv),
        "pwm_stats_by_variant_class": class_stats,
        "paths": {
            "fasta":       str(FASTA_PATH),
            "labels":      str(LABELS_PATH),
            "gtf_db":      str(GTF_DB_PATH),
            "out_fa":      str(fa_path),
            "out_csv":     str(csv_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote summary to %s", json_path)
    LOG.info("done. %d sequences across %d variant classes.",
             len(records_fa), len(VARIANT_CLASSES))


if __name__ == "__main__":
    main()
