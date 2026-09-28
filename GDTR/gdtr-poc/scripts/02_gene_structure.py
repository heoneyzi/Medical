"""02_gene_structure.py - Gate B (genomic signal in TP53 region).

Sliding window over chr17:7,668,402-7,687,550 (TP53 region, ~19 kb):
  window=6 kb, stride=500 bp -> ~26 windows.
For each window forward HyenaDNA, capture hidden states, compute BOTH:
  - JSD-gDTR settling depth per position (auxiliary)
  - UR-gDTR settling depth per position (PRIMARY post Gate A FAIL)
Stitch the central 1 kb of each window into a region-wide profile (overlap mean).

Annotation: GENCODE v44 canonical TP53 ENST00000269305 -> coding exons,
introns, 5'UTR, 3'UTR, splice donor/acceptor (+/-10 bp).

Confound controls:
  - per-100bp GC content
  - k=3 Shannon entropy
  - 5x dinuc-shuffled baselines

Outputs:
  results/tables/gdtr_by_context.csv
  results/tables/gdtr_vs_confounders.csv
  results/figures/F4_tp53_profile.{pdf,png}
  results/figures/F6_context_boxplot.{pdf,png}
  results/runs/02_gene_structure.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import GAMMA_DEFAULT, L_DEFAULT, RHO_DEFAULT, SEED_DEFAULT
from src.controls import dinuc_shuffle
from src.gdtr import settling_depth_discrete, settling_depth_interp
from src.logit_lens import jsd_lens
from src.model_loader import load_hyenadna, tokenize_sequence
from src.stats import bootstrap_ci, cohens_d, mwu_with_effect
from src.ur_gdtr import cosine_lens
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("02_gene_structure")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"
ANNO_DB = PROJECT_ROOT / "data" / "annotation" / "gencode.v44.chr17_chr13.gtf.db"

# TP53 locus (GRCh38)
TP53_CHROM = "chr17"
TP53_START = 7_668_402     # 1-based inclusive (UCSC)
TP53_END = 7_687_550       # 1-based inclusive
TP53_TRANSCRIPT = "ENST00000269305"

WINDOW = 6000
STRIDE = 500
EDGE_WARMUP = 5
CENTRAL_WIDTH = 1000          # use central 1 kb of each window
N_SHUFFLED = 5
GAMMA_JSD = GAMMA_DEFAULT     # 0.5
# UR-gDTR uses quantile-calibrated gamma_cos (post Gate A' decision)
# We compute it from the cosine distribution at the penultimate layer
# across all real windows in this region.

# Bonferroni for Gate B (5 contexts vs intron, primary on UR-gDTR)
GATE_B_ALPHA = 0.001            # Bonferroni-corrected: design 3.2
N_CONTEXTS_PRIMARY = 5          # exon, 5'UTR, 3'UTR, splice, intergenic
GATE_B_EFFECTIVE_ALPHA = GATE_B_ALPHA / N_CONTEXTS_PRIMARY


def load_region_seq() -> str:
    """Load TP53 region from chr17 fasta. Pyfaidx 0-based half-open slice."""
    from pyfaidx import Fasta
    fa = Fasta(str(CHR17_FA), as_raw=True, sequence_always_upper=True)
    # Pad both sides by half-window so sliding windows starting at TP53_START
    # cover the whole region
    pad = WINDOW // 2
    region_start_0 = max(0, TP53_START - 1 - pad)
    region_end_0 = TP53_END + pad
    seq = str(fa[TP53_CHROM][region_start_0:region_end_0])
    log.info("loaded region %s:%d-%d (length %d, padded by %d each side)",
             TP53_CHROM, region_start_0, region_end_0, len(seq), pad)
    return seq, region_start_0


def build_annotation(region_start_0: int, region_end_0_inclusive: int) -> Dict[str, np.ndarray]:
    """Build per-position context labels for TP53 region.

    Categories at each genome position (in `region_start_0..region_end_0_inclusive-1` range):
      - 'coding_exon'  (CDS overlap on TP53 transcript)
      - 'intron'       (within transcript span but not in any exon)
      - '5utr'
      - '3utr'
      - 'splice_donor_acceptor'  (within +/-10 bp of an exon-intron boundary)
      - 'intergenic'   (outside the canonical transcript span)
    Returns dict[context] -> bool array of same length as region.
    """
    import gffutils
    db = gffutils.FeatureDB(str(ANNO_DB))

    region_len = region_end_0_inclusive - region_start_0
    pos_array = np.arange(region_start_0, region_end_0_inclusive)  # 0-based

    # Find canonical TP53 transcript
    tx = None
    for t in db.features_of_type('transcript', limit=(TP53_CHROM, TP53_START, TP53_END)):
        tid = t.attributes.get('transcript_id', [''])[0]
        if TP53_TRANSCRIPT in tid:
            tx = t
            break
    if tx is None:
        raise RuntimeError(f"canonical transcript {TP53_TRANSCRIPT} not found in db")
    log.info("canonical TP53 transcript: %s strand=%s span=%d-%d (1-based GFF)",
             tx.attributes.get('transcript_id', [''])[0], tx.strand, tx.start, tx.end)

    # Helper: GFF coordinates are 1-based inclusive. Convert to 0-based [start, end).
    def gff_to_0based(s: int, e: int) -> Tuple[int, int]:
        return s - 1, e

    contexts = {
        'coding_exon': np.zeros(region_len, dtype=bool),
        'intron': np.zeros(region_len, dtype=bool),
        '5utr': np.zeros(region_len, dtype=bool),
        '3utr': np.zeros(region_len, dtype=bool),
        'splice_donor_acceptor': np.zeros(region_len, dtype=bool),
        'intergenic': np.zeros(region_len, dtype=bool),
    }

    tx_s, tx_e = gff_to_0based(tx.start, tx.end)
    transcript_mask = np.zeros(region_len, dtype=bool)
    transcript_mask[max(0, tx_s - region_start_0):max(0, tx_e - region_start_0)] = True

    # CDS
    for cds in db.children(tx, featuretype='CDS'):
        s, e = gff_to_0based(cds.start, cds.end)
        contexts['coding_exon'][max(0, s - region_start_0):max(0, e - region_start_0)] = True

    # Exons (including UTR portions)
    exon_intervals: List[Tuple[int, int]] = []
    for ex in db.children(tx, featuretype='exon'):
        s, e = gff_to_0based(ex.start, ex.end)
        exon_intervals.append((s, e))
    exon_intervals.sort()
    exon_mask = np.zeros(region_len, dtype=bool)
    for s, e in exon_intervals:
        exon_mask[max(0, s - region_start_0):max(0, e - region_start_0)] = True

    # UTRs: GFF feature 'UTR' (no 5'/3' split). Compute split using CDS start/stop.
    cds_intervals = sorted(
        gff_to_0based(c.start, c.end) for c in db.children(tx, featuretype='CDS')
    )
    if not cds_intervals:
        raise RuntimeError("no CDS for TP53 transcript")
    cds_min = min(s for s, _ in cds_intervals)
    cds_max = max(e for _, e in cds_intervals)
    for utr in db.children(tx, featuretype='UTR'):
        s, e = gff_to_0based(utr.start, utr.end)
        if tx.strand == '+':
            if e <= cds_min:
                key = '5utr'
            elif s >= cds_max:
                key = '3utr'
            else:
                # straddles - skip (rare); treat as 5utr by default
                key = '5utr'
        else:  # minus strand
            if s >= cds_max:
                key = '5utr'
            elif e <= cds_min:
                key = '3utr'
            else:
                key = '5utr'
        contexts[key][max(0, s - region_start_0):max(0, e - region_start_0)] = True

    # Intron: in transcript but not in any exon.
    intron_mask = transcript_mask & ~exon_mask
    contexts['intron'][:] = intron_mask

    # Splice donor/acceptor: +/-10 bp of each exon-intron boundary,
    # but ONLY at exon-intron boundaries (not transcript ends).
    splice_mask = np.zeros(region_len, dtype=bool)
    sorted_exons = sorted(exon_intervals)
    for i, (s, e) in enumerate(sorted_exons):
        # 5' boundary: not the first exon's 5' end
        if i > 0:
            lo = max(0, s - region_start_0 - 10)
            hi = min(region_len, s - region_start_0 + 10)
            splice_mask[lo:hi] = True
        # 3' boundary: not the last exon's 3' end
        if i < len(sorted_exons) - 1:
            lo = max(0, e - region_start_0 - 10)
            hi = min(region_len, e - region_start_0 + 10)
            splice_mask[lo:hi] = True
    contexts['splice_donor_acceptor'][:] = splice_mask

    # Intergenic = outside transcript
    contexts['intergenic'][:] = ~transcript_mask

    # Report counts
    for k, v in contexts.items():
        log.info("  context %-22s : n=%d positions", k, int(v.sum()))

    return contexts, pos_array


@torch.no_grad()
def forward_window(bundle, seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Forward one window, return (D_jsd[L,T_real], D_cos[L,T_real]) numpy."""
    input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
    out = bundle.model(input_ids, output_hidden_states=True)
    D_jsd = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head).numpy()
    D_cos = cosine_lens(out.hidden_states).numpy()
    del out
    torch.cuda.empty_cache()
    return D_jsd.astype(np.float32), D_cos.astype(np.float32)


def gc_per_pos(seq: str, win: int = 100) -> np.ndarray:
    """Per-position GC content via centered window of `win` bp."""
    arr = np.array([1.0 if c in 'GC' else 0.0 for c in seq])
    half = win // 2
    out = np.zeros(len(arr))
    cum = np.concatenate([[0], np.cumsum(arr)])
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = (cum[hi] - cum[lo]) / (hi - lo)
    return out


def shannon_entropy_kmer(seq: str, k: int = 3, win: int = 100) -> np.ndarray:
    """Per-position Shannon entropy of k-mer distribution in centered window."""
    half = win // 2
    out = np.zeros(len(seq))
    for i in range(len(seq)):
        lo = max(0, i - half)
        hi = min(len(seq), i + half + 1)
        sub = seq[lo:hi]
        counts: Dict[str, int] = {}
        for j in range(len(sub) - k + 1):
            kmer = sub[j:j + k]
            counts[kmer] = counts.get(kmer, 0) + 1
        n = sum(counts.values())
        if n == 0:
            out[i] = 0.0
            continue
        h = 0.0
        for c in counts.values():
            p = c / n
            h -= p * math.log(p, 2)
        out[i] = h
    return out


def stitch_profile(window_results: List[Tuple[int, np.ndarray]],
                   region_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch per-window central 1 kb profiles into a region-wide track.

    Args:
        window_results: list of (genome_start_0based, central_profile [CENTRAL_WIDTH]) tuples.
        region_len: target length.

    Returns:
        (mean_profile, count_profile) of length region_len. NaN where no coverage.
    """
    sum_arr = np.zeros(region_len, dtype=np.float64)
    cnt_arr = np.zeros(region_len, dtype=np.int32)
    for start_0, prof in window_results:
        end_0 = start_0 + len(prof)
        # Bound to region
        clip_lo = max(0, start_0)
        clip_hi = min(region_len, end_0)
        if clip_hi <= clip_lo:
            continue
        offset = clip_lo - start_0
        sum_arr[clip_lo:clip_hi] += prof[offset:offset + (clip_hi - clip_lo)]
        cnt_arr[clip_lo:clip_hi] += 1
    out = np.full(region_len, np.nan)
    mask = cnt_arr > 0
    out[mask] = sum_arr[mask] / cnt_arr[mask]
    return out, cnt_arr


def main():
    parser = argparse.ArgumentParser(description="Gate B: TP53 gene-structure profiling")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--rho", type=float, default=RHO_DEFAULT)
    parser.add_argument("--gamma-jsd", type=float, default=GAMMA_DEFAULT)
    parser.add_argument("--gamma-cos", type=float, default=None,
                        help="if None, calibrated as q70 of D_cos at penultimate layer")
    parser.add_argument("--n-shuffled", type=int, default=N_SHUFFLED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    setup_publication_style()
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)
    L = len(bundle.model.hyena.backbone.layers)
    log.info("L=%d", L)

    # Load TP53 region (with padding)
    region_seq, region_start_0 = load_region_seq()
    region_len = len(region_seq)
    region_end_0 = region_start_0 + region_len

    # Sliding windows (window=6 kb, stride=500 bp), pinned within region
    starts = list(range(0, region_len - WINDOW + 1, STRIDE))
    log.info("n_windows = %d", len(starts))

    # Forward each real window + collect cosine values for gamma_cos calibration
    cache_path = RUNS / "02_gene_structure_cache.npz"
    if cache_path.exists() and not args.force:
        log.info("loading cache %s", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        D_jsd_all = z["D_jsd"]
        D_cos_all = z["D_cos"]
        cache_starts = z["starts"]
        if list(cache_starts) != starts:
            log.info("cache mismatch - recomputing")
            cache_path.unlink(missing_ok=True)
    if not cache_path.exists():
        D_jsd_all = np.zeros((len(starts), L, WINDOW), dtype=np.float32)
        D_cos_all = np.zeros((len(starts), L, WINDOW), dtype=np.float32)
        for i, s in enumerate(tqdm(starts, desc="forward[real]")):
            sub = region_seq[s:s + WINDOW]
            D_jsd, D_cos = forward_window(bundle, sub)
            D_jsd_all[i] = D_jsd
            D_cos_all[i] = D_cos
        np.savez_compressed(cache_path,
                            D_jsd=D_jsd_all, D_cos=D_cos_all,
                            starts=np.array(starts))
        log.info("saved %s", cache_path)

    # Shuffled baselines
    shuf_cache = RUNS / "02_gene_structure_shuf_cache.npz"
    if shuf_cache.exists() and not args.force:
        z = np.load(shuf_cache, allow_pickle=True)
        D_jsd_shuf = z["D_jsd_shuf"]
        D_cos_shuf = z["D_cos_shuf"]
        if D_jsd_shuf.shape[0] != args.n_shuffled or D_jsd_shuf.shape[1] != len(starts):
            log.info("shuf cache mismatch - recomputing")
            shuf_cache.unlink(missing_ok=True)
    if not shuf_cache.exists():
        D_jsd_shuf = np.zeros((args.n_shuffled, len(starts), L, WINDOW), dtype=np.float32)
        D_cos_shuf = np.zeros((args.n_shuffled, len(starts), L, WINDOW), dtype=np.float32)
        for r in range(args.n_shuffled):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            log.info("shuffle replicate %d/%d (seed=%d)", r + 1, args.n_shuffled, sub_seed)
            for i, s in enumerate(tqdm(starts, desc=f"forward[shuf{r}]")):
                sub = region_seq[s:s + WINDOW]
                shuf_seq = dinuc_shuffle(sub, n_shuffles=1, seed=sub_seed + i)[0]
                D_jsd, D_cos = forward_window(bundle, shuf_seq)
                D_jsd_shuf[r, i] = D_jsd
                D_cos_shuf[r, i] = D_cos
        np.savez_compressed(shuf_cache,
                            D_jsd_shuf=D_jsd_shuf, D_cos_shuf=D_cos_shuf)
        log.info("saved %s", shuf_cache)

    # gamma_cos calibration: q70 of D_cos at penultimate layer (layer L-1, 0-indexed),
    # using post-edge positions only.
    pen_vals = D_cos_all[:, L - 2, EDGE_WARMUP:].ravel()
    if args.gamma_cos is None:
        gamma_cos = float(np.quantile(pen_vals, 0.70))
    else:
        gamma_cos = args.gamma_cos
    log.info("gamma_cos (calibrated q70 penultimate) = %.4f (default 0.10 saturates)",
             gamma_cos)
    log.info("gamma_jsd = %.3f (design default)", args.gamma_jsd)

    # Compute settling depth per (window, position) for both lenses
    def settling_for_window(D: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute c_disc[T] and c_interp[T] given D[L, T]. NumPy-only."""
        D_t = torch.from_numpy(D)
        c_disc = settling_depth_discrete(D_t, gamma=gamma).numpy()
        c_int, _ = settling_depth_interp(D_t, gamma=gamma)
        return c_disc.astype(np.float32), c_int.numpy().astype(np.float32)

    # For each window, take central CENTRAL_WIDTH positions and compute c_interp
    half = (WINDOW - CENTRAL_WIDTH) // 2

    def per_window_central(D_full: np.ndarray, gamma: float) -> List[Tuple[int, np.ndarray]]:
        results = []
        for i, s in enumerate(starts):
            D_win = D_full[i]                   # [L, T_real]
            _, c_int = settling_for_window(D_win, gamma=gamma)
            central = c_int[half:half + CENTRAL_WIDTH]
            # genome-position start (0-based, in region coords)
            results.append((s + half, central))
        return results

    # Stitch profiles
    log.info("stitching JSD profile ...")
    jsd_central = per_window_central(D_jsd_all, args.gamma_jsd)
    jsd_profile, jsd_cov = stitch_profile(jsd_central, region_len)

    log.info("stitching UR profile ...")
    ur_central = per_window_central(D_cos_all, gamma_cos)
    ur_profile, ur_cov = stitch_profile(ur_central, region_len)

    # Stitch shuffled profiles (mean over replicates per position)
    log.info("stitching shuffled profiles (n=%d replicates) ...", args.n_shuffled)
    jsd_shuf_profiles = np.zeros((args.n_shuffled, region_len), dtype=np.float64)
    ur_shuf_profiles = np.zeros((args.n_shuffled, region_len), dtype=np.float64)
    for r in range(args.n_shuffled):
        jsd_central_r = per_window_central(D_jsd_shuf[r], args.gamma_jsd)
        ur_central_r = per_window_central(D_cos_shuf[r], gamma_cos)
        jp, _ = stitch_profile(jsd_central_r, region_len)
        up, _ = stitch_profile(ur_central_r, region_len)
        jsd_shuf_profiles[r] = jp
        ur_shuf_profiles[r] = up
    jsd_shuf_mean = np.nanmean(jsd_shuf_profiles, axis=0)
    jsd_shuf_std = np.nanstd(jsd_shuf_profiles, axis=0)
    ur_shuf_mean = np.nanmean(ur_shuf_profiles, axis=0)
    ur_shuf_std = np.nanstd(ur_shuf_profiles, axis=0)

    # Build annotation context
    contexts, pos_array = build_annotation(region_start_0, region_end_0)

    # GC and entropy per position
    log.info("computing GC and k=3 Shannon entropy per position ...")
    gc = gc_per_pos(region_seq, win=100)
    ent = shannon_entropy_kmer(region_seq, k=3, win=100)

    # ----- Per-context stats (Gate B primary) -----
    def per_ctx_table(profile: np.ndarray, lens_name: str) -> pd.DataFrame:
        rows = []
        intron_vals = profile[contexts['intron'] & ~np.isnan(profile)]
        for ctx, mask in contexts.items():
            vals = profile[mask & ~np.isnan(profile)]
            if vals.size == 0:
                continue
            mean_v = float(np.mean(vals))
            med_v = float(np.median(vals))
            std_v = float(np.std(vals))
            mean_b, lo_b, hi_b = bootstrap_ci(vals, n_boot=1000, seed=args.seed)
            row = {
                "lens": lens_name,
                "context": ctx,
                "n": int(vals.size),
                "mean": mean_v,
                "ci95_low": lo_b,
                "ci95_high": hi_b,
                "median": med_v,
                "std": std_v,
            }
            if ctx != 'intron':
                U, p, r = mwu_with_effect(vals, intron_vals, alternative='two-sided')
                d = cohens_d(vals, intron_vals)
                row.update({
                    "MWU_p_vs_intron": p,
                    "MWU_U": U,
                    "rank_biserial_r_vs_intron": r,
                    "cohens_d_vs_intron": d,
                    "passes_gate_B_alpha": p < GATE_B_EFFECTIVE_ALPHA,
                })
            else:
                row.update({
                    "MWU_p_vs_intron": np.nan,
                    "MWU_U": np.nan,
                    "rank_biserial_r_vs_intron": np.nan,
                    "cohens_d_vs_intron": np.nan,
                    "passes_gate_B_alpha": np.nan,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    df_jsd = per_ctx_table(jsd_profile, 'JSD-gDTR')
    df_ur = per_ctx_table(ur_profile, 'UR-gDTR')
    df_ctx = pd.concat([df_ur, df_jsd], ignore_index=True)
    out_csv = TABLES / "gdtr_by_context.csv"
    df_ctx.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # ----- Partial Spearman vs confounders -----
    from scipy.stats import spearmanr

    def partial_spearman(x: np.ndarray, y_mask: np.ndarray, gc: np.ndarray, ent: np.ndarray):
        """Partial Spearman rho(x, y_mask | gc, ent) via residual approach.

        Treat y_mask (0/1) as continuous. Use rank-residual partial correlation:
        r = spearman(rank-resid(x|gc, ent), rank-resid(y|gc, ent)).
        """
        valid = ~np.isnan(x) & ~np.isnan(gc) & ~np.isnan(ent)
        x = x[valid]
        y = y_mask[valid].astype(float)
        gc_v = gc[valid]
        en_v = ent[valid]
        # rank everything
        from scipy.stats import rankdata
        rx = rankdata(x)
        ry = rankdata(y)
        rgc = rankdata(gc_v)
        ren = rankdata(en_v)
        # residualize via linear regression
        Z = np.stack([rgc, ren, np.ones_like(rgc)], axis=1)
        coef_x, _, _, _ = np.linalg.lstsq(Z, rx, rcond=None)
        coef_y, _, _, _ = np.linalg.lstsq(Z, ry, rcond=None)
        rx_resid = rx - Z @ coef_x
        ry_resid = ry - Z @ coef_y
        rho, p = spearmanr(rx_resid, ry_resid)
        return float(rho), float(p)

    rows_conf = []
    for lens_name, profile in [('UR-gDTR', ur_profile), ('JSD-gDTR', jsd_profile)]:
        for ctx, mask in contexts.items():
            rho_p, p_p = partial_spearman(profile, mask, gc, ent)
            rows_conf.append({
                "lens": lens_name,
                "context": ctx,
                "partial_spearman_rho": rho_p,
                "p_value": p_p,
            })
    df_conf = pd.DataFrame(rows_conf)
    out_conf = TABLES / "gdtr_vs_confounders.csv"
    df_conf.to_csv(out_conf, index=False)
    log.info("wrote %s", out_conf)

    # ----- Gate B verdict -----
    # Primary: UR-gDTR exon vs intron MWU p
    primary_row = df_ctx[(df_ctx['lens'] == 'UR-gDTR') &
                         (df_ctx['context'] == 'coding_exon')]
    if primary_row.empty:
        gate_b_pass = False
        primary_p = np.nan
        primary_d = np.nan
    else:
        primary_p = float(primary_row['MWU_p_vs_intron'].iloc[0])
        primary_d = float(primary_row['cohens_d_vs_intron'].iloc[0])
        gate_b_pass = bool(primary_p < GATE_B_EFFECTIVE_ALPHA)

    log.info("=" * 60)
    log.info("Gate B verdict (UR-gDTR primary):")
    log.info("  exon vs intron MWU p = %.3e (effective alpha = %.4f) -> %s",
             primary_p, GATE_B_EFFECTIVE_ALPHA, "PASS" if gate_b_pass else "FAIL")
    log.info("  Cohen's d = %.3f", primary_d)
    log.info("=" * 60)

    # ----- F4 figure: TP53 profile -----
    log.info("rendering F4 ...")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 4.5),
                             gridspec_kw={'height_ratios': [0.5, 1.5, 1.0, 1.0]},
                             sharex=True)

    x_genome = pos_array  # 0-based genome coords (already absolute)
    # Convert to kb display
    x_kb = (x_genome - TP53_START + 1) / 1000.0  # 0 = TP53 start

    # Panel 1: GENCODE annotation
    ax = axes[0]
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_ylabel("anno", rotation=0, ha='right', va='center')
    # exons
    cds_mask = contexts['coding_exon']
    utr5_mask = contexts['5utr']
    utr3_mask = contexts['3utr']
    intron_mask = contexts['intron']
    # find runs of contiguous True in each mask, plot as rectangles
    def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        out = []
        d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            out.append((s, e))
        return out
    for s, e in runs(intron_mask):
        ax.plot([x_kb[s], x_kb[e - 1]], [0, 0], color='gray', lw=0.5)
    for s, e in runs(cds_mask):
        ax.add_patch(Rectangle((x_kb[s], -0.5), x_kb[e - 1] - x_kb[s], 1.0,
                               facecolor=WONG_PALETTE['blue'], alpha=0.8))
    for s, e in runs(utr5_mask):
        ax.add_patch(Rectangle((x_kb[s], -0.3), x_kb[e - 1] - x_kb[s], 0.6,
                               facecolor=WONG_PALETTE['orange'], alpha=0.7))
    for s, e in runs(utr3_mask):
        ax.add_patch(Rectangle((x_kb[s], -0.3), x_kb[e - 1] - x_kb[s], 0.6,
                               facecolor=WONG_PALETTE['vermillion'], alpha=0.7))
    splice_runs = runs(contexts['splice_donor_acceptor'])
    for s, e in splice_runs:
        mid = (x_kb[s] + x_kb[e - 1]) / 2
        ax.plot(mid, 0.7, marker='v', color='black', markersize=3)
    ax.set_title("(a) TP53 GENCODE annotation (CDS=blue, 5'UTR=orange, 3'UTR=red, splice=triangles)",
                 fontsize=7.5)

    # Panel 2: UR-gDTR profile (primary)
    ax = axes[1]
    ax.plot(x_kb, ur_profile, color=WONG_PALETTE['blue'], lw=0.5,
            alpha=0.5, label='UR-gDTR (raw)')
    # rolling 50-bp smoothed
    smooth = pd.Series(ur_profile).rolling(50, center=True, min_periods=1).mean().values
    ax.plot(x_kb, smooth, color=WONG_PALETTE['blue'], lw=1.0,
            label='UR-gDTR (50-bp rolling)')
    # shuffled baseline
    ax.fill_between(x_kb, ur_shuf_mean - ur_shuf_std, ur_shuf_mean + ur_shuf_std,
                    color='gray', alpha=0.3, label='shuffled +/- 1SD')
    ax.plot(x_kb, ur_shuf_mean, color='gray', lw=0.6, ls='--')
    ax.set_ylabel("UR-gDTR settling depth c_interp")
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title(f"(b) UR-gDTR profile (primary; gamma_cos={gamma_cos:.3f})", fontsize=7.5)

    # Panel 3: JSD-gDTR profile (auxiliary)
    ax = axes[2]
    ax.plot(x_kb, jsd_profile, color=WONG_PALETTE['bluish_green'], lw=0.5,
            alpha=0.5, label='JSD-gDTR (raw)')
    smooth_j = pd.Series(jsd_profile).rolling(50, center=True, min_periods=1).mean().values
    ax.plot(x_kb, smooth_j, color=WONG_PALETTE['bluish_green'], lw=1.0,
            label='JSD-gDTR (50-bp rolling)')
    ax.fill_between(x_kb, jsd_shuf_mean - jsd_shuf_std, jsd_shuf_mean + jsd_shuf_std,
                    color='gray', alpha=0.3)
    ax.plot(x_kb, jsd_shuf_mean, color='gray', lw=0.6, ls='--')
    ax.set_ylabel("JSD-gDTR c_interp")
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title("(c) JSD-gDTR profile (auxiliary; gamma=0.5)", fontsize=7.5)

    # Panel 4: GC content
    ax = axes[3]
    ax.plot(x_kb, gc, color='black', lw=0.6)
    ax.set_ylabel("GC fraction")
    ax.set_xlabel(f"Genomic position relative to TP53 start (kb), chr17:{TP53_START:,}+")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, FIGURES / "F4_tp53_profile")
    plt.close(fig)

    (FIGURES / "F4_tp53_profile.caption.json").write_text(json.dumps({
        "figure_id": "F4",
        "title": "TP53 region gDTR profile (UR primary post Gate A FAIL)",
        "caption": (
            f"Layer-wise prediction-convergence profile for HyenaDNA-medium-160k "
            f"sliding-window forward over TP53 region "
            f"(chr17:{TP53_START:,}-{TP53_END:,}, ~19 kb). "
            f"Window=6 kb stride={STRIDE} bp central-{CENTRAL_WIDTH}-bp stitch "
            f"(n_windows={len(starts)}, {args.n_shuffled} dinuc-shuffled "
            f"replicates as gray baseline). Tracks: (a) GENCODE v44 canonical "
            f"transcript {TP53_TRANSCRIPT}; (b) UR-gDTR settling depth (primary, "
            f"gamma_cos calibrated to q70 of penultimate-layer cosine "
            f"distribution = {gamma_cos:.3f}); (c) JSD-gDTR (auxiliary, "
            f"gamma=0.5); (d) GC content per 100 bp."
        ),
    }, indent=2))

    # ----- F6 boxplot -----
    log.info("rendering F6 ...")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ctx_order = ['coding_exon', 'intron', '5utr', '3utr', 'splice_donor_acceptor', 'intergenic']

    for ax, profile, name in [(axes[0], ur_profile, 'UR-gDTR (primary)'),
                              (axes[1], jsd_profile, 'JSD-gDTR (auxiliary)')]:
        data = []
        names = []
        for ctx in ctx_order:
            vals = profile[contexts[ctx] & ~np.isnan(profile)]
            if vals.size > 0:
                data.append(vals)
                names.append(f"{ctx}\n(n={vals.size})")
        bp = ax.boxplot(data, labels=names, showfliers=False, widths=0.6,
                        patch_artist=True)
        for patch, c in zip(bp['boxes'], [WONG_PALETTE['blue'], 'gray',
                                          WONG_PALETTE['orange'],
                                          WONG_PALETTE['vermillion'],
                                          WONG_PALETTE['bluish_green'],
                                          'lightgray']):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        # MWU p-value annotation vs intron
        intron_vals = profile[contexts['intron'] & ~np.isnan(profile)]
        for i, ctx in enumerate(ctx_order):
            if ctx == 'intron':
                continue
            vals = profile[contexts[ctx] & ~np.isnan(profile)]
            if vals.size == 0:
                continue
            _, p, _ = mwu_with_effect(vals, intron_vals)
            stars = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'
            ax.text(i + 1, max(np.nanmax(profile), 1) * 0.98, stars,
                    ha='center', va='top', fontsize=6.5)
        ax.set_ylabel("settling depth c_interp")
        ax.tick_params(axis='x', labelsize=6.5, rotation=30)
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    save_figure(fig, FIGURES / "F6_context_boxplot")
    plt.close(fig)

    (FIGURES / "F6_context_boxplot.caption.json").write_text(json.dumps({
        "figure_id": "F6",
        "title": "TP53 settling depth by genomic context",
        "caption": (
            "Settling depth c_interp distributions across canonical TP53 genomic "
            "contexts for both UR-gDTR (left, primary post Gate A FAIL) and "
            "JSD-gDTR (right, auxiliary). Significance asterisks: Mann-Whitney "
            "U test vs intron (***p<1e-3, **p<1e-2, *p<5e-2). Sample sizes "
            "(n positions) printed under each context label."
        ),
    }, indent=2))

    # ----- runs/json -----
    payload = {
        "script": "02_gene_structure.py",
        "purpose": "Gate B - TP53 region gDTR profile (UR-gDTR primary)",
        "seed": args.seed,
        "n_windows": len(starts),
        "window": WINDOW,
        "stride": STRIDE,
        "central_width": CENTRAL_WIDTH,
        "edge_warmup": EDGE_WARMUP,
        "n_shuffled": args.n_shuffled,
        "L": L,
        "rho": args.rho,
        "gamma_jsd": args.gamma_jsd,
        "gamma_cos": gamma_cos,
        "tp53_region": {
            "chrom": TP53_CHROM, "start": TP53_START, "end": TP53_END,
            "transcript": TP53_TRANSCRIPT,
        },
        "context_counts": {k: int(v.sum()) for k, v in contexts.items()},
        "verdict_gate_B": {
            "primary_lens": "UR-gDTR",
            "primary_p_value": primary_p,
            "primary_cohens_d": primary_d,
            "effective_alpha_bonferroni": GATE_B_EFFECTIVE_ALPHA,
            "pass": gate_b_pass,
        },
        "outputs": {
            "table_context": "results/tables/gdtr_by_context.csv",
            "table_confounders": "results/tables/gdtr_vs_confounders.csv",
            "figure_F4": "results/figures/F4_tp53_profile.{pdf,png}",
            "figure_F6": "results/figures/F6_context_boxplot.{pdf,png}",
            "cache_real": "results/runs/02_gene_structure_cache.npz",
            "cache_shuf": "results/runs/02_gene_structure_shuf_cache.npz",
        },
        "runtime_s": time.time() - t0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "02_gene_structure.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
