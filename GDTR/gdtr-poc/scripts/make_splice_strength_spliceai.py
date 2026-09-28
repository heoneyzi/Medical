"""Build a SpliceAI-calibrated splice-strength perturbation dataset for chr22.

For each high-confidence splice donor/acceptor (SpliceAI raw probability ≥ threshold),
generates 10 sequence variants at 0.1-decrement splice-strength levels relative to the
reference score:

  level 1.0  →  wild-type (no mutations)
  level 0.9  →  1 SNV applied; score ≈ 0.9 × ref_score
  level 0.8  →  cumulative SNVs; score ≈ 0.8 × ref_score
  ...
  level 0.1  →  cumulative SNVs; score ≈ 0.1 × ref_score

Each level is found by a greedy single-SNV search within the core splice motif
(9-mer for donors, 23-mer for acceptors).  At each step the SNV that minimises
|achieved - target| is selected; it is applied cumulatively so subsequent levels
build on prior mutations.

SpliceAI models are loaded from the installed `spliceai` package (TensorFlow/Keras).
Install: pip install spliceai tensorflow-cpu   # or tensorflow-gpu on the server

The scoring window is 15001 bp (odd, so the center is always at index 7500 on both
strands).  FASTA output is the central 4096 bp sub-window (OUTPUT_WIN), matching
the HyenaDNA-medium window convention from the rest of the pipeline.

For minus-strand sites the scoring sequence is reverse-complemented; greedy mutations
are found in transcript orientation then mapped back to plus-strand coordinates.

Output:
  results/splice_strength_spliceai/
    sequences.fa   — FASTA (OUTPUT_WIN bp per entry)
    metadata.csv   — seq_id, splice_type, chrom, site_pos, strand,
                     target_level, achieved_score, ref_score,
                     applied_mutations, n_mutations, window_start, window_end
    summary.json   — config and per-level score statistics

Usage (server, GPU):
  PYTHONPATH=. python scripts/make_splice_strength_spliceai.py

Env-var overrides:
  GDTR_ROOT, FASTA_PATH, LABELS_PATH, GTF_DB_PATH
  N_SITES=200  HIGH_SCORE_THRESHOLD=0.5  SCORE_TOL=0.08
"""
from __future__ import annotations

import csv
import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GDTR_ROOT   = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR"))
FASTA_PATH  = Path(os.environ.get(
    "FASTA_PATH", str(GDTR_ROOT / "data" / "reference" / "chr22.fa")))
LABELS_PATH = Path(os.environ.get(
    "LABELS_PATH", str(GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy")))
GTF_DB_PATH = Path(os.environ.get(
    "GTF_DB_PATH", str(GDTR_ROOT / "data" / "annotation" /
                       "gencode.v44.chr17_chr22.gtf.db")))

OUT_DIR = GDTR_ROOT / "results" / "splice_strength_spliceai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("splice_strength_spliceai")

# ---------------------------------------------------------------------------
# Run-time config (override via env vars)
# ---------------------------------------------------------------------------
CHROM    = "chr22"
SEED     = 42
N_SITES  = int(os.environ.get("N_SITES", "200"))

# SpliceAI scoring window — must be odd so center index is exact
SCORE_WIN = 15001
SCORE_CTR = SCORE_WIN // 2        # = 7500

# FASTA output window (HyenaDNA convention, centered on site)
OUTPUT_WIN = 4096
OUTPUT_HALF = OUTPUT_WIN // 2

# Splice-strength levels: 1.0 (wild-type) → 0.1 in steps of 0.1
N_LEVELS = 10
LEVELS   = np.round(np.linspace(1.0, 0.1, N_LEVELS), 2)   # [1.0, 0.9, ..., 0.1]

HIGH_SCORE_THRESHOLD = float(os.environ.get("HIGH_SCORE_THRESHOLD", "0.5"))
SCORE_TOL = float(os.environ.get("SCORE_TOL", "0.08"))

LABEL_DONOR    = 5
LABEL_ACCEPTOR = 6

# Motif search ranges (relative to site center, 0-based offset)
# Donor:    9-mer positions −3 … +6   → 27 SNV candidates
# Acceptor: 23-mer positions −20 … +3 → 69 SNV candidates
DONOR_SEARCH    = list(range(-3, 7))
ACCEPTOR_SEARCH = list(range(-20, 4))

NUCLEOTIDES = "ACGT"

# ---------------------------------------------------------------------------
# Reverse complement
# ---------------------------------------------------------------------------
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ---------------------------------------------------------------------------
# FASTA access via pyfaidx
# ---------------------------------------------------------------------------
def open_fasta(path: Path):
    from pyfaidx import Fasta
    return Fasta(str(path), sequence_always_upper=True)


def fetch_seq(fa, chrom: str, start: int, end: int) -> str:
    """0-based [start, end) fetch; returns empty string on out-of-bounds."""
    try:
        return str(fa[chrom][start:end])
    except Exception as e:
        LOG.debug("fetch failed (%s, %d, %d): %s", chrom, start, end, e)
        return ""


# ---------------------------------------------------------------------------
# Strand lookup from gffutils GTF
# ---------------------------------------------------------------------------
def build_strand_map(labels: np.ndarray, gtf_db_path: Path) -> dict[int, str]:
    try:
        import gffutils
        db = gffutils.FeatureDB(str(gtf_db_path))
    except Exception as e:
        LOG.warning("gffutils unavailable (%s); strand inferred from dinucleotide.", e)
        return {}

    target_pos = set(int(p) for p in np.flatnonzero(
        (labels == LABEL_DONOR) | (labels == LABEL_ACCEPTOR)))
    strand_map: dict[int, str] = {}
    LOG.info("querying GTF for strand of %d sites …", len(target_pos))

    for feat in db.features_of_type(["exon", "transcript"]):
        if feat.seqid != CHROM:
            continue
        s = int(feat.start) - 1   # GTF 1-based → 0-based
        e = int(feat.end)
        strand = feat.strand
        if strand == "+":
            if e in target_pos:
                strand_map[e] = "+"
            if (s - 1) in target_pos:
                strand_map[s - 1] = "+"
        else:
            if (s - 1) in target_pos:
                strand_map[s - 1] = "-"
            if e in target_pos:
                strand_map[e] = "-"

    LOG.info("strand resolved for %d / %d positions", len(strand_map), len(target_pos))
    return strand_map


def infer_strand_from_seq(
    large_window: str,
    site_type: Literal["donor", "acceptor"],
) -> str:
    """Heuristic strand detection from the dinucleotide at the scoring center."""
    ctr = SCORE_CTR
    if site_type == "donor":
        if large_window[ctr:ctr + 2].upper() == "GT":
            return "+"
        if large_window[ctr - 1:ctr + 1].upper() == "AC":   # RC of GT
            return "-"
    else:
        if large_window[ctr - 1:ctr + 1].upper() == "AG":
            return "+"
        if large_window[ctr:ctr + 2].upper() == "CT":       # RC of AG
            return "-"
    return "?"


# ---------------------------------------------------------------------------
# SpliceAI model loading and batch scoring
# ---------------------------------------------------------------------------
def load_spliceai_models() -> list:
    """Load 4-model SpliceAI ensemble from the installed spliceai package."""
    try:
        from pkg_resources import resource_filename
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except ImportError as exc:
        raise ImportError(
            "SpliceAI requires TensorFlow.  Install with:\n"
            "  pip install spliceai tensorflow-cpu   # CPU\n"
            "  pip install spliceai tensorflow        # GPU"
        ) from exc

    models = []
    for i in range(1, 5):
        try:
            path = resource_filename("spliceai", f"models/spliceai{i}.h5")
            LOG.info("loading SpliceAI model %d from %s", i, path)
            m = tf.keras.models.load_model(path, compile=False)
            models.append(m)
        except Exception as exc:
            LOG.warning("could not load spliceai%d.h5: %s", i, exc)

    if not models:
        raise RuntimeError(
            "No SpliceAI models found.  Install the spliceai package:\n"
            "  pip install spliceai"
        )
    LOG.info("loaded %d SpliceAI model(s)", len(models))
    return models


def _one_hot_batch(seqs: list[str]) -> "np.ndarray":
    """Convert a list of equal-length DNA strings to (N, L, 4) float32 array."""
    nt_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    N, L = len(seqs), len(seqs[0])
    X = np.zeros((N, L, 4), dtype=np.float32)
    for b, seq in enumerate(seqs):
        for i, nt in enumerate(seq.upper()):
            j = nt_idx.get(nt)
            if j is not None:
                X[b, i, j] = 1.0
    return X


def score_batch(
    seqs: list[str],
    models: list,
    center: int,
    splice_type: Literal["donor", "acceptor"],
    batch_size: int = 32,
) -> np.ndarray:
    """Return ensemble-mean SpliceAI probability at `center` for each sequence.

    splice_type 'donor'    → output channel 1
    splice_type 'acceptor' → output channel 2
    """
    ch = 1 if splice_type == "donor" else 2
    N = len(seqs)
    probs = np.zeros(N, dtype=np.float64)

    for start in range(0, N, batch_size):
        batch = seqs[start:start + batch_size]
        X = _one_hot_batch(batch)                          # (bs, L, 4)
        batch_probs = np.zeros(len(batch), dtype=np.float64)
        for model in models:
            y = model.predict(X, verbose=0)                # (bs, L, 3)
            batch_probs += y[:, center, ch]
        probs[start:start + len(batch)] = batch_probs / len(models)

    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Greedy SNV search
# ---------------------------------------------------------------------------
def _generate_snv_candidates(
    current_seq: str,
    center: int,
    search_offsets: list[int],
) -> tuple[list[str], list[tuple[int, str, str]]]:
    """All single-nucleotide variants within search_offsets of center.

    Returns (candidate_seqs, [(abs_pos, ref_nt, alt_nt), ...]).
    """
    cands: list[tuple[int, str, str]] = []
    seqs: list[str] = []
    L = len(current_seq)

    for off in search_offsets:
        pos = center + off
        if pos < 0 or pos >= L:
            continue
        ref_nt = current_seq[pos].upper()
        if ref_nt not in NUCLEOTIDES:
            continue
        for alt_nt in NUCLEOTIDES:
            if alt_nt == ref_nt:
                continue
            mutated = current_seq[:pos] + alt_nt + current_seq[pos + 1:]
            seqs.append(mutated)
            cands.append((pos, ref_nt, alt_nt))

    return seqs, cands


def greedy_step(
    current_seq: str,
    models: list,
    score_center: int,
    splice_type: Literal["donor", "acceptor"],
    target_score: float,
    search_offsets: list[int],
) -> tuple[str, float, tuple[int, str, str] | None]:
    """Apply the single SNV that brings the score closest to target_score.

    Returns (new_seq, achieved_score, best_mutation_or_None).
    If no candidates exist, returns the current sequence unchanged.
    """
    cand_seqs, cand_muts = _generate_snv_candidates(
        current_seq, score_center, search_offsets)

    if not cand_seqs:
        LOG.debug("no SNV candidates; returning unchanged")
        curr_score = score_batch([current_seq], models, score_center, splice_type)[0]
        return current_seq, float(curr_score), None

    scores = score_batch(cand_seqs, models, score_center, splice_type)

    # Select the candidate whose achieved score is closest to target
    distances = np.abs(scores.astype(np.float64) - target_score)
    best_idx = int(np.argmin(distances))

    return cand_seqs[best_idx], float(scores[best_idx]), cand_muts[best_idx]


# ---------------------------------------------------------------------------
# Mutation coordinate conversion (transcript ↔ plus-strand)
# ---------------------------------------------------------------------------
def transcript_pos_to_plus(pos_in_rc: int, win_len: int) -> int:
    """Flip a 0-based index in the RC window back to plus-strand coordinate."""
    return win_len - 1 - pos_in_rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    LOG.info("loading labels from %s", LABELS_PATH)
    labels    = np.load(LABELS_PATH)
    chrom_len = int(labels.shape[0])
    LOG.info("chrom_len = %d", chrom_len)

    strand_map = build_strand_map(labels, GTF_DB_PATH)

    LOG.info("opening FASTA %s", FASTA_PATH)
    fa = open_fasta(FASTA_PATH)

    LOG.info("loading SpliceAI models …")
    models = load_spliceai_models()

    records_fa:  list[str]  = []
    rows_csv:    list[dict] = []

    donor_pos    = np.flatnonzero(labels == LABEL_DONOR)
    acceptor_pos = np.flatnonzero(labels == LABEL_ACCEPTOR)
    LOG.info("total donor=%d, acceptor=%d", donor_pos.size, acceptor_pos.size)

    for site_type, positions in (("donor",    donor_pos),
                                  ("acceptor", acceptor_pos)):
        search_offsets = DONOR_SEARCH if site_type == "donor" else ACCEPTOR_SEARCH

        # Remove sites too close to chromosome boundaries
        margin = SCORE_WIN // 2 + 1
        valid  = positions[(positions >= margin) & (positions < chrom_len - margin)]
        if valid.size == 0:
            LOG.warning("no valid %s sites after boundary filter", site_type)
            continue

        n_sample = min(N_SITES, valid.size)
        sampled  = rng.choice(valid, n_sample, replace=False)
        sampled.sort()
        LOG.info("sampled %d %s sites", n_sample, site_type)

        n_kept = 0
        for idx_s, site_pos in enumerate(sampled):
            site_pos = int(site_pos)
            win_start = site_pos - SCORE_CTR
            win_end   = win_start + SCORE_WIN

            large_window = fetch_seq(fa, CHROM, win_start, win_end)
            if len(large_window) != SCORE_WIN:
                continue
            if large_window.count("N") > 100:
                continue

            # ---- strand ----
            strand = strand_map.get(site_pos)
            if strand is None:
                strand = infer_strand_from_seq(large_window, site_type)
            if strand == "?":
                continue

            # Scoring sequence in transcript orientation
            if strand == "+":
                score_seq    = large_window
                score_center = SCORE_CTR
            else:
                score_seq    = revcomp(large_window)
                # In the RC, the original position SCORE_CTR maps to:
                # rc_center = SCORE_WIN - 1 - SCORE_CTR = 15001 - 1 - 7500 = 7500
                score_center = SCORE_WIN - 1 - SCORE_CTR   # = 7500 for SCORE_WIN=15001

            # ---- reference score ----
            ref_score = float(
                score_batch([score_seq], models, score_center, site_type)[0])

            if ref_score < HIGH_SCORE_THRESHOLD:
                LOG.debug("site %d %s ref_score=%.3f < threshold; skip",
                          site_pos, site_type, ref_score)
                continue

            n_kept += 1
            LOG.info("[%s %d/%d] pos=%d strand=%s ref_score=%.3f",
                     site_type, idx_s + 1, n_sample, site_pos, strand, ref_score)

            # Target scores: proportional to ref_score
            # level 1.0 → ref_score; level 0.9 → 0.9 * ref_score; …
            target_scores = (LEVELS * ref_score).tolist()

            # ---- greedy search for each level ----
            current_score_seq   = score_seq              # in transcript orientation
            current_achieved    = ref_score
            cumulative_muts_rc: list[tuple[int, str, str]] = []   # in transcript coords

            for level_idx, (level, target) in enumerate(
                    zip(LEVELS.tolist(), target_scores)):

                if level_idx == 0:
                    # Level 1.0 = wild-type, no mutations
                    achieved_score = ref_score
                    mut_str = ""
                else:
                    new_seq, achieved_score, best_mut = greedy_step(
                        current_score_seq, models, score_center,
                        site_type, target, search_offsets,
                    )
                    if best_mut is not None:
                        current_score_seq = new_seq
                        cumulative_muts_rc.append(best_mut)

                # Build plus-strand output window
                if strand == "+":
                    plus_window_full = current_score_seq
                    plus_muts = list(cumulative_muts_rc)
                else:
                    plus_window_full = revcomp(current_score_seq)
                    # Convert RC-coordinate mutations back to plus-strand coords
                    plus_muts = [
                        (transcript_pos_to_plus(p, SCORE_WIN),
                         revcomp(ref_nt), revcomp(alt_nt))
                        for p, ref_nt, alt_nt in cumulative_muts_rc
                    ]

                # Slice OUTPUT_WIN around the site center in plus-strand window
                out_start = SCORE_CTR - OUTPUT_HALF
                out_end   = SCORE_CTR + OUTPUT_HALF
                out_seq   = plus_window_full[out_start:out_end]

                if len(out_seq) != OUTPUT_WIN:
                    continue

                # Format applied mutations as "refPOSalt,..." (plus-strand coords,
                # position relative to window_start for readability)
                abs_muts_str = ",".join(
                    f"{ref}{win_start + p}{alt}"
                    for p, ref, alt in plus_muts
                ) if plus_muts else "."

                level_str = f"{level:.1f}"
                seq_id = (f"{CHROM}_{site_pos}_{site_type[0]}{strand}_"
                          f"level{level_str.replace('.', '')}")

                records_fa.append(f">{seq_id}\n{out_seq}")
                rows_csv.append({
                    "seq_id":             seq_id,
                    "splice_type":        site_type,
                    "chrom":              CHROM,
                    "site_pos":           site_pos,
                    "strand":             strand,
                    "target_level":       level_str,
                    "target_score":       f"{target:.4f}",
                    "achieved_score":     f"{achieved_score:.4f}",
                    "ref_score":          f"{ref_score:.4f}",
                    "applied_mutations":  abs_muts_str,
                    "n_mutations":        len(plus_muts),
                    "window_start":       win_start + out_start,
                    "window_end":         win_start + out_end,
                })

        LOG.info("%s: kept %d / %d sampled sites (score ≥ %.2f)",
                 site_type, n_kept, n_sample, HIGH_SCORE_THRESHOLD)

    # ---- write outputs ----
    fa_path   = OUT_DIR / "sequences.fa"
    csv_path  = OUT_DIR / "metadata.csv"
    json_path = OUT_DIR / "summary.json"

    fa_path.write_text("\n".join(records_fa) + "\n")
    LOG.info("wrote %d sequences → %s", len(records_fa), fa_path)

    fieldnames = [
        "seq_id", "splice_type", "chrom", "site_pos", "strand",
        "target_level", "target_score", "achieved_score", "ref_score",
        "applied_mutations", "n_mutations", "window_start", "window_end",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_csv)
    LOG.info("wrote %d rows → %s", len(rows_csv), csv_path)

    # ---- per-level statistics ----
    level_stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "achieved_scores": []})
    for row in rows_csv:
        lv = row["target_level"]
        try:
            level_stats[lv]["achieved_scores"].append(float(row["achieved_score"]))
            level_stats[lv]["n"] += 1
        except ValueError:
            pass

    level_summary = {}
    for lv, d in sorted(level_stats.items()):
        arr = np.array(d["achieved_scores"])
        level_summary[lv] = {
            "n":            d["n"],
            "mean_achieved": float(arr.mean()) if arr.size else None,
            "std_achieved":  float(arr.std(ddof=1)) if arr.size > 1 else None,
            "median_achieved": float(np.median(arr)) if arr.size else None,
        }

    summary = {
        "chrom":                  CHROM,
        "seed":                   SEED,
        "n_sites_per_type":       N_SITES,
        "high_score_threshold":   HIGH_SCORE_THRESHOLD,
        "score_tolerance":        SCORE_TOL,
        "score_window_bp":        SCORE_WIN,
        "output_window_bp":       OUTPUT_WIN,
        "n_levels":               N_LEVELS,
        "levels":                 LEVELS.tolist(),
        "n_sequences_total":      len(records_fa),
        "per_level_stats":        level_summary,
        "paths": {
            "fasta":    str(FASTA_PATH),
            "labels":   str(LABELS_PATH),
            "gtf_db":   str(GTF_DB_PATH),
            "out_fa":   str(fa_path),
            "out_csv":  str(csv_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote summary → %s", json_path)
    LOG.info("done. total %d sequences across %d levels.", len(records_fa), N_LEVELS)


if __name__ == "__main__":
    main()
