"""Control-sequence generators for Gate A baselines.

Implements:
- dinuc_shuffle: Altschul-Erickson dinucleotide-preserving shuffle (uShuffle k=2)
- gc_match_random: i.i.d. random sequences with target GC content
- extract_intergenic_chr17: pull intergenic 6 kb windows from chr17 reference
  (chr22 not yet downloaded — Phase 0 sanity uses chr17 intergenic regions
  as documented; this is acceptable for log-rank baselines, see docstring).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ----- Dinucleotide-preserving shuffle (Altschul-Erickson) -----
def dinuc_shuffle(seq: str, n_shuffles: int = 1, seed: Optional[int] = None) -> List[str]:
    """Generate uShuffle (Altschul & Erickson 1985) dinucleotide-preserving shuffles.

    Algorithm: build an Eulerian multigraph over the alphabet where each
    dinucleotide of `seq` becomes a directed edge. Random Eulerian path from
    `seq[0]` to `seq[-1]` produces a sequence with identical dinucleotide
    composition. Implementation iterates the standard algorithm:
      1. for each node, remove a random edge to be the "last out-edge"
      2. assert a spanning arborescence rooted at the destination — the
         remaining (n_v-1) edges form a tree; if not, re-pick.
      3. shuffle the rest of out-edges, append the saved last-edge.
      4. walk Eulerian path starting from seq[0].

    Notes:
        Pure Python; slow for very long sequences but adequate for 100 x 6 kb.

    Args:
        seq: input string (ACGT...).
        n_shuffles: number of shuffles.
        seed: RNG seed (None -> non-deterministic).

    Returns:
        list of n_shuffles strings, each same length as `seq`.
    """
    if len(seq) < 2:
        return [seq] * n_shuffles
    rng = random.Random(seed)
    chars = sorted(set(seq))
    out: List[str] = []
    for _ in range(n_shuffles):
        out.append(_dinuc_shuffle_one(seq, chars, rng))
    return out


def _dinuc_shuffle_one(seq: str, chars: List[str], rng: random.Random) -> str:
    n = len(seq)
    start, end = seq[0], seq[-1]
    # adjacency: dict char -> list of next chars (ordered list of out-edges)
    adj: dict[str, List[str]] = {c: [] for c in chars}
    for i in range(n - 1):
        adj[seq[i]].append(seq[i + 1])

    # Try up to 50 rejections to find a valid Eulerian path
    for _attempt in range(50):
        # 1. Pick a random "last out-edge" for each node != end (target of Eulerian path)
        last_edge: dict[str, Optional[str]] = {c: None for c in chars}
        rest: dict[str, List[str]] = {c: list(adj[c]) for c in chars}
        ok = True
        for c in chars:
            if c == end:
                continue
            if not rest[c]:
                # No outgoing edge but we still need one to reach `end`.
                # Only OK if c is unreachable (zero in-edges + zero out-edges)
                # In a real walk on seq[0..n-1] this can't occur for c == start
                # or any internal node. Treat as failure.
                ok = False
                break
            idx = rng.randrange(len(rest[c]))
            last_edge[c] = rest[c].pop(idx)

        if not ok:
            continue

        # 2. Verify: the chosen last_edges form a spanning arborescence rooted at `end`.
        # Build the directed graph (parent -> last_edge[parent]) and walk from each
        # node; all should reach `end`.
        valid = True
        for c in chars:
            if c == end:
                continue
            if last_edge[c] is None:
                valid = False
                break
            cur = c
            visited = set()
            while cur != end:
                if cur in visited or last_edge[cur] is None:
                    valid = False
                    break
                visited.add(cur)
                cur = last_edge[cur]
            if not valid:
                break

        if not valid:
            continue

        # 3. Shuffle the "rest" edges per node, then append last_edge.
        for c in chars:
            rng.shuffle(rest[c])
            if last_edge[c] is not None:
                rest[c].append(last_edge[c])

        # 4. Walk Eulerian path from start using rest as edge stacks (FIFO via index).
        ptr = {c: 0 for c in chars}
        out_chars = [start]
        cur = start
        for _ in range(n - 1):
            nxt = rest[cur][ptr[cur]]
            ptr[cur] += 1
            out_chars.append(nxt)
            cur = nxt
        return "".join(out_chars)

    raise RuntimeError("dinuc_shuffle failed to find a valid Eulerian path after 50 attempts")


# ----- GC-matched random sequences -----
def gc_match_random(
    target_gc: float,
    length: int,
    n_seqs: int,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate i.i.d. random sequences with the requested GC fraction.

    Args:
        target_gc: GC fraction in [0, 1] (probability of G or C).
        length: sequence length in bp.
        n_seqs: number of sequences.
        seed: RNG seed.

    Returns:
        list[str] of length n_seqs, each `length` characters in {A,C,G,T}.
    """
    if not (0.0 <= target_gc <= 1.0):
        raise ValueError(f"target_gc out of range: {target_gc}")
    rng = np.random.default_rng(seed)
    # P(G) = P(C) = target_gc/2; P(A) = P(T) = (1-target_gc)/2
    probs = np.array([
        (1 - target_gc) / 2.0,   # A
        target_gc / 2.0,         # C
        target_gc / 2.0,         # G
        (1 - target_gc) / 2.0,   # T
    ])
    alphabet = np.array(["A", "C", "G", "T"])
    out: List[str] = []
    for _ in range(n_seqs):
        idx = rng.choice(4, size=length, p=probs)
        out.append("".join(alphabet[idx].tolist()))
    return out


# ----- Intergenic window sampler -----
def extract_intergenic_chr17(
    fasta_path: str | Path,
    length: int = 6000,
    n: int = 100,
    seed: int = 42,
    avoid_n_threshold: float = 0.01,
    chrom: str = "chr17",
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """Sample intergenic 6 kb windows from chr17.

    Phase 0 limitation: chr22 is not yet downloaded; we fall back to chr17
    (the same reference that holds TP53/BRCA1 of interest). Without a
    RepeatMasker BED we instead avoid (a) windows containing any unresolved
    'N' bases above `avoid_n_threshold` and (b) the TP53/BRCA1 gene bodies
    (hard-coded GENCODE v44 coordinates) so the sanity baseline does not
    overlap signal regions.

    Args:
        fasta_path: path to chr17.fa (indexed via .fai).
        length: window size, default 6000 bp.
        n: number of windows.
        seed: RNG seed (default 42).
        avoid_n_threshold: max fraction of N tolerated in a window.
        chrom: chromosome name (must match fasta header).

    Returns:
        (seqs, coords) where seqs is list[str] (uppercase ACGT-only after
        N-fraction check) and coords is list of (chrom, start, end) tuples
        (0-based, half-open).
    """
    from pyfaidx import Fasta

    fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    chrom_len = len(fa[chrom])

    # GENCODE v44 canonical TP53 and BRCA1 GRCh38 coordinates plus
    # 50 kb pad to leave room for the variant pilot context windows.
    avoid: List[Tuple[int, int]] = [
        (7_618_402, 7_737_550),     # TP53 +/- 50 kb pad
        (43_044_295, 43_175_000),   # BRCA1 +/- 50 kb pad
    ]

    rng = np.random.default_rng(seed)
    out_seqs: List[str] = []
    out_coords: List[Tuple[str, int, int]] = []

    # Restrict sampling to mid-chromosome (avoid centromere region of chr17:
    # ~22.2-25.7 Mb) to reduce N-fraction rejections.
    valid_ranges = [
        (1_000_000, 22_000_000),
        (26_000_000, chrom_len - length - 1_000_000),
    ]

    n_attempts = 0
    max_attempts = n * 200
    while len(out_seqs) < n and n_attempts < max_attempts:
        n_attempts += 1
        rng_idx = int(rng.integers(0, len(valid_ranges)))
        lo, hi = valid_ranges[rng_idx]
        start = int(rng.integers(lo, hi))
        end = start + length
        # Reject overlap with avoidance regions
        if any(not (end <= a or start >= b) for (a, b) in avoid):
            continue
        seq = str(fa[chrom][start:end])
        if len(seq) != length:
            continue
        n_frac = seq.count("N") / length
        if n_frac > avoid_n_threshold:
            continue
        # Force ACGT only (replace any residual ambiguity with random ACGT)
        if any(c not in "ACGT" for c in seq):
            seq = "".join(c if c in "ACGT" else rng.choice(["A", "C", "G", "T"]) for c in seq)
        out_seqs.append(seq)
        out_coords.append((chrom, start, end))

    if len(out_seqs) < n:
        log.warning("only sampled %d/%d intergenic windows (max attempts hit)",
                    len(out_seqs), n)
    return out_seqs, out_coords


def gc_content(seq: str) -> float:
    """GC fraction of a sequence (ignores Ns)."""
    if not seq:
        return float("nan")
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    n_acgt = sum(s.count(c) for c in "ACGT")
    return gc / n_acgt if n_acgt > 0 else float("nan")
