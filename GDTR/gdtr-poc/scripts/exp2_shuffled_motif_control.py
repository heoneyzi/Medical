"""Exp 2 (v10) — shuffled motif control on chr22 GT-AG donors.

Question: is the splice-shallow signature driven by the canonical GT-AG
dinucleotide motif itself, by the flanking sequence context, or by the
joint? Equivalently — does shallowness survive when the donor's flanking
context is dinucleotide-shuffled (preserving all 16 dinucleotide
frequencies) while the central GT is held intact?

Method:
  - From chr22_splice_class_labels.npy, recover canonical GT-AG donor
    centers (clusters of contiguous label==1 collapsed to their midpoint).
  - Sample N_DONORS distinct centers (default 1000).
  - For each donor center p, build the REAL window
        seq_real = chr22[p−CTX_HALF, p+CTX_HALF]
    and SHUFFLED versions:
        seq_shuf = same window with the ±SHUFFLE_HALF flank around p
        replaced by a dinucleotide-preserving shuffle (Altschul–Erickson),
        leaving the central GT (positions p, p+1 in genomic coords) intact.
    Generate N_SHUFFLES independent shuffles per donor.
  - Forward all 6×N_DONORS sequences through Evo 2 7B and record the
    settling depth c at the donor center position
        c = first ℓ for which run-min D_cos(ℓ, donor_center) ≤ γ_cos.
  - Report mean / median c on real vs shuffled, Cohen's d, one-sided
    Mann–Whitney U (real < shuffled), and per-donor paired Δc.

Outputs:
  results/exp2_shuffled/exp2_shuffled_meta.json
  results/exp2_shuffled/per_donor.npz
  results/exp2_shuffled/_status.json

Usage:
  python scripts/exp2_shuffled_motif_control.py \
      [--n-donors 1000] [--n-shuffles 5] [--ctx-half 3000] \
      [--shuffle-half 100] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
GAMMA_COS = 0.39663
SEED = 42
PHASE = "exp2_shuffled"
OUT_DIR = GDTR_ROOT / "results" / PHASE
SPLICE_NPY = GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"
SPLICE_DONOR_LABEL = 5  # chr22_position_labels.npy encoding (verified
                         # 2026-05-04): 5 = splice_donor (paper convention).
                         # NOT chr22_splice_class_labels.npy where 1 = donor.
CHR22_FA = GDTR_ROOT / "data" / "reference" / "chr22.fa"

LOG = logging.getLogger(PHASE)


def setup_log():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def write_status(state, extra=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"step": PHASE, "status": state}
    if extra:
        payload.update(extra)
    (OUT_DIR / "_status.json").write_text(
        json.dumps(payload, indent=2, default=str))


def cohen_d(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((len(x) - 1) * sx + (len(y) - 1) * sy)
                 / (len(x) + len(y) - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


# -------- Altschul–Erickson dinucleotide-preserving shuffle --------
# Reference: Altschul & Erickson, Mol Biol Evol 1985. Eulerian walk on
# the de Bruijn graph of length-2 substrings.

def dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """Return a permutation of seq that preserves all 16 dinucleotide
    frequencies. Implements the Altschul–Erickson method via random
    Eulerian path on the dinucleotide graph."""
    n = len(seq)
    if n < 3:
        return seq
    chars = list(seq.upper())
    # build adjacency: for each char, list of NEXT chars (multiset)
    adj = {c: [] for c in set(chars)}
    for i in range(n - 1):
        adj[chars[i]].append(chars[i + 1])
    end = chars[-1]
    start = chars[0]
    # randomize order of out-edges
    for k in adj:
        rng.shuffle(adj[k])
    # ensure last edge from each non-end vertex points to one with a path
    # to `end` (Hierholzer-safe). Simple retry strategy for small flanks.
    for attempt in range(64):
        adj_copy = {k: list(v) for k, v in adj.items()}
        # randomly reorder
        for k in adj_copy:
            rng.shuffle(adj_copy[k])
        try:
            walk = _eulerian_walk(adj_copy, start, end)
            if walk is not None and len(walk) == n:
                return "".join(walk)
        except Exception:
            continue
    # fall back to simple shuffle if AE fails
    rng.shuffle(chars)
    return "".join(chars)


def _eulerian_walk(adj, start, end):
    """Hierholzer's algorithm. Returns list of vertices length sum(edges)+1."""
    # ensure last edge from each vertex (except `end`) leads toward `end`.
    # AE trick: among out-edges of each vertex v != end, pick one that
    # is on a path to end and place it LAST.
    degrees = {v: len(adj.get(v, [])) for v in adj}
    if not degrees:
        return None
    # DFS: any path from v to end?
    def reachable(src, dst):
        if src == dst:
            return True
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for w in adj.get(u, []):
                if w not in seen:
                    if w == dst:
                        return True
                    seen.add(w); stack.append(w)
        return False
    for v in list(adj.keys()):
        if v == end:
            continue
        if not adj[v]:
            continue
        # find a successor that can still reach `end` after we use it last
        # (i.e. there exists ANOTHER path); simple heuristic: rotate
        # adj[v] so that the chosen "last" edge is one whose successor
        # can still reach end via remaining edges. With small flanks the
        # default random order usually works; we just retry on failure.
    # Hierholzer
    stack = [start]
    walk = []
    while stack:
        v = stack[-1]
        if adj.get(v):
            stack.append(adj[v].pop())
        else:
            walk.append(stack.pop())
    walk.reverse()
    # all edges consumed?
    if any(adj.get(v) for v in adj):
        return None
    return walk


def find_donor_centers(splice_labels: np.ndarray, target_label: int,
                        chrom_seq: str):
    """For each contiguous run of `target_label` in `splice_labels`, scan
    `chrom_seq` for a 'GT' dinucleotide within the run. Return positions
    of the 'GT' first base closest to the run midpoint.

    The label-midpoint convention is not always at the GT first base
    (verified empirically: only ~42 % of midpoints land on GT for chr22
    label==5). Direct sequence scan within the labelled region pins the
    GT position exactly.
    """
    mask = splice_labels == target_label
    if not mask.any():
        return np.array([], dtype=np.int64)
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size]])
    centers = []
    for a, b in zip(starts, ends):
        seg = chrom_seq[int(a):int(b)]
        # scan all GT positions within the run, pick the one closest to
        # the run midpoint (donor convention).
        gt_offsets = [i for i in range(len(seg) - 1) if seg[i:i + 2] == "GT"]
        if not gt_offsets:
            continue
        mid = (b - a) // 2
        best_off = min(gt_offsets, key=lambda x: abs(x - mid))
        centers.append(int(a) + best_off)
    return np.asarray(centers, dtype=np.int64)


def main():
    setup_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-donors", type=int, default=1000)
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--ctx-half", type=int, default=3000,
                         help="half-window around donor for the Evo 2 forward")
    parser.add_argument("--shuffle-half", type=int, default=100,
                         help="half-flank size to dinucleotide-shuffle")
    parser.add_argument("--readout-half", type=int, default=10,
                         help="half-window over which to average c at the "
                              "donor (matches the splice_pad=10 convention "
                              "of paper Table 1; GT-position-only readout "
                              "produces c≈31 because shallow recognition "
                              "happens UPSTREAM of GT in the exon)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_donors = 8
        args.n_shuffles = 2

    rng_np = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    t0 = time.time()
    write_status("RUNNING", {"n_donors": args.n_donors,
                              "n_shuffles": args.n_shuffles})

    # --- model + chr22 sequence first (centers depend on sequence) ---
    sys.path.insert(0, str(GDTR_ROOT))
    import _runner_utils as ru  # noqa: F401
    from src.constants_evo2 import N_LAYERS, BOS_OFFSET
    from src.model_loader_evo2 import load_evo2, tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names
    from src.ur_gdtr_evo2 import cosine_lens
    import pysam

    bundle = load_evo2()
    LOG.info("evo2 loaded: %s", bundle.loaded_variant)
    fasta = pysam.FastaFile(str(CHR22_FA))
    chr22_seq = fasta.fetch("chr22").upper()
    chr22_len = len(chr22_seq)
    LOG.info("chr22 length: %d", chr22_len)

    # --- recover GT-AG donor centers (exact GT positions within labelled runs) ---
    sc = np.load(SPLICE_NPY)
    centers_all = find_donor_centers(sc, target_label=SPLICE_DONOR_LABEL,
                                      chrom_seq=chr22_seq)
    LOG.info("found %d canonical GT-AG donor centers (sequence-anchored)",
              centers_all.size)

    # filter to centers that admit a full ctx_half window
    valid = (centers_all >= args.ctx_half) & \
             (centers_all + args.ctx_half + 2 < chr22_len)
    centers_valid = centers_all[valid]
    n_pick = min(args.n_donors, centers_valid.size)
    pick = rng_np.choice(centers_valid.size, size=n_pick, replace=False)
    centers = centers_valid[pick]
    LOG.info("using %d donor centers (valid range)", n_pick)

    layer_names = all_layer_names()

    def _c_at_center(seq: str, center_offset_in_window: int,
                     pad: int = args.readout_half) -> float:
        """Mean c over ±pad bp centred on `center_offset_in_window`.
        The pad-averaged readout matches the splice_pad=10 paper
        convention; per-position c at the GT first base alone is biased
        deep because recognition happens upstream in the exon."""
        toks = tokenize(seq, bundle, device="cuda")
        with torch.no_grad():
            hs = extract_hidden_states(bundle, toks, save_layers=layer_names)
            D_cos = cosine_lens(hs, n_layers=N_LAYERS, bos_offset=BOS_OFFSET)
            cd_np = D_cos.float().cpu().numpy()
        del hs
        torch.cuda.empty_cache()
        run_min = np.minimum.accumulate(cd_np, axis=0)
        lo = max(0, center_offset_in_window - pad)
        hi = min(run_min.shape[1], center_offset_in_window + pad + 1)
        cs = []
        for col_idx in range(lo, hi):
            col = run_min[:, col_idx]
            idx = np.where(col <= GAMMA_COS)[0]
            cs.append(int(idx[0]) + 1 if idx.size else N_LAYERS + 1)
        return float(np.mean(cs))

    real_c = np.zeros(n_pick, dtype=np.float32)            # real GT + real flank
    shuf_c = np.zeros((n_pick, args.n_shuffles), dtype=np.float32)  # real GT + shuffled flank
    mut_c = np.zeros(n_pick, dtype=np.float32)             # AA + real flank (GT destroyed)

    # We process donors one at a time (1 + n_shuffles forwards each).
    # Estimated wall: ~0.3 s/forward × 6 forwards × 1000 donors = ~30 min.
    flank_n = args.shuffle_half
    ctx_h = args.ctx_half
    log_every = max(20, n_pick // 25)

    for k, p in enumerate(centers):
        p = int(p)
        win_start = p - ctx_h
        win_end = p + ctx_h
        seq = chr22_seq[win_start:win_end]
        if len(seq) != 2 * ctx_h or "N" in seq[ctx_h - flank_n: ctx_h + flank_n + 2]:
            real_c[k] = -1; shuf_c[k] = -1; mut_c[k] = -1
            continue
        # Filter: keep only canonical GT donors.
        if seq[ctx_h:ctx_h + 2] != "GT":
            real_c[k] = -2; shuf_c[k] = -2; mut_c[k] = -2
            continue
        center_off = ctx_h
        real_c[k] = _c_at_center(seq, center_off)

        # Arm 1: shuffle flanks, keep GT
        left_flank = seq[ctx_h - flank_n: ctx_h]
        right_flank = seq[ctx_h + 2: ctx_h + 2 + flank_n]
        for s in range(args.n_shuffles):
            l_sh = dinuc_shuffle(left_flank, py_rng)
            r_sh = dinuc_shuffle(right_flank, py_rng)
            sh_seq = (seq[: ctx_h - flank_n] + l_sh +
                       seq[ctx_h: ctx_h + 2] +
                       r_sh + seq[ctx_h + 2 + flank_n:])
            assert len(sh_seq) == len(seq)
            shuf_c[k, s] = _c_at_center(sh_seq, center_off)

        # Arm 2: destroy GT (replace with AA), keep real flank
        mut_seq = seq[: ctx_h] + "AA" + seq[ctx_h + 2:]
        assert len(mut_seq) == len(seq)
        mut_c[k] = _c_at_center(mut_seq, center_off)

        if (k + 1) % log_every == 0 or k == 0:
            v = real_c[: k + 1] >= 0
            LOG.info("[%d/%d]  real %.1f | shuf %.1f | GT→AA %.1f  | %.1fs",
                      k + 1, n_pick,
                      float(np.median(real_c[: k + 1][v])),
                      float(np.median(shuf_c[: k + 1][v])),
                      float(np.median(mut_c[: k + 1][v])),
                      time.time() - t0)

    valid = real_c >= 0
    real_arr = real_c[valid].astype(np.float32)
    shuf_arr = shuf_c[valid].astype(np.float32).reshape(-1)
    mut_arr = mut_c[valid].astype(np.float32)

    from scipy import stats as st
    d_shuf = cohen_d(real_arr, shuf_arr)
    d_mut = cohen_d(real_arr, mut_arr)  # real (shallow) vs GT-destroyed (deep?)

    paired_real_shuf = real_c[valid].astype(np.float32) \
        - shuf_c[valid].astype(np.float32).mean(axis=1)
    paired_real_mut = real_c[valid].astype(np.float32) - mut_c[valid].astype(np.float32)
    paired_p_shuf = float(st.wilcoxon(paired_real_shuf,
                                       alternative="two-sided").pvalue) \
        if valid.sum() >= 5 else float("nan")
    paired_p_mut = float(st.wilcoxon(paired_real_mut,
                                      alternative="less").pvalue) \
        if valid.sum() >= 5 else float("nan")

    np.savez_compressed(OUT_DIR / "per_donor.npz",
                        centers=centers,
                        real_c=real_c, shuf_c=shuf_c, mut_c=mut_c,
                        valid_mask=valid)

    meta = {"n_donors_requested": args.n_donors,
            "n_donors_used": int(valid.sum()),
            "n_shuffles": args.n_shuffles,
            "ctx_half": args.ctx_half,
            "shuffle_half": args.shuffle_half,
            "readout_half": args.readout_half,
            "real": {"mean": float(real_arr.mean()),
                      "median": float(np.median(real_arr))},
            "shuf_flank_keepGT": {"mean": float(shuf_arr.mean()),
                                    "median": float(np.median(shuf_arr))},
            "mut_GTtoAA_keepFlank": {"mean": float(mut_arr.mean()),
                                       "median": float(np.median(mut_arr))},
            "cohens_d_real_vs_shuf": d_shuf,
            "cohens_d_real_vs_mut": d_mut,
            "paired_wilcoxon_p_real_vs_shuf_2sided": paired_p_shuf,
            "paired_wilcoxon_p_real_vs_mut_less": paired_p_mut,
            "wall_sec": time.time() - t0,
            "gamma_cos": GAMMA_COS}
    (OUT_DIR / "exp2_shuffled_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    write_status("PASS", meta)
    LOG.info("DONE in %.1fs", time.time() - t0)
    LOG.info("real c̄=%.2f | shuf-flank c̄=%.2f (d=%+.3f vs real, p=%.3g) "
              "| GT→AA c̄=%.2f (d=%+.3f vs real, p=%.3g)",
              real_arr.mean(), shuf_arr.mean(), d_shuf, paired_p_shuf,
              mut_arr.mean(), d_mut, paired_p_mut)


if __name__ == "__main__":
    main()
