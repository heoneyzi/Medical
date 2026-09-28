"""05_brca1.py - BRCA1 extension (Stage 5, conditional on Gate B PASS).

Per design Stage 5: BRCA1 chr13:43,044,295-43,170,245 ~125 kb.
We use HyenaDNA-medium-160k (already loaded) with sliding 6 kb windows
(stride 500 bp) to be consistent with Stage 2 methodology. UR-gDTR is primary.

Outputs:
  results/figures/F5_brca1_profile.{pdf,png}
  results/tables/brca1_gdtr_by_context.csv
  results/runs/05_brca1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import GAMMA_DEFAULT, SEED_DEFAULT
from src.gdtr import settling_depth_interp
from src.logit_lens import jsd_lens
from src.model_loader import load_hyenadna, tokenize_sequence
from src.stats import bootstrap_ci, cohens_d, mwu_with_effect
from src.ur_gdtr import cosine_lens
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("05_brca1")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"
ANNO_DB = PROJECT_ROOT / "data" / "annotation" / "gencode.v44.chr17_chr13.gtf.db"

BRCA1_CHROM = "chr17"
BRCA1_START = 43_044_295
BRCA1_END = 43_170_245
BRCA1_TRANSCRIPT = "ENST00000357654"

WINDOW = 6000
STRIDE = 500
EDGE_WARMUP = 5
CENTRAL_WIDTH = 1000
GAMMA_JSD = GAMMA_DEFAULT


def load_region_seq(chrom: str, start_1based: int, end_1based: int,
                    pad: int = WINDOW // 2) -> Tuple[str, int]:
    from pyfaidx import Fasta
    fa = Fasta(str(CHR17_FA), as_raw=True, sequence_always_upper=True)
    region_start_0 = max(0, start_1based - 1 - pad)
    region_end_0 = end_1based + pad
    seq = str(fa[chrom][region_start_0:region_end_0])
    log.info("loaded %s:%d-%d (length %d, pad=%d)",
             chrom, region_start_0, region_end_0, len(seq), pad)
    return seq.upper(), region_start_0


def build_brca1_annotation(region_start_0: int, region_end_0: int) -> Dict[str, np.ndarray]:
    """BRCA1 canonical transcript context masks (CDS, intron, 5'UTR, 3'UTR, splice, intergenic)."""
    import gffutils
    db = gffutils.FeatureDB(str(ANNO_DB))
    region_len = region_end_0 - region_start_0

    tx = None
    for t in db.features_of_type('transcript', limit=(BRCA1_CHROM, BRCA1_START, BRCA1_END)):
        tid = t.attributes.get('transcript_id', [''])[0]
        if BRCA1_TRANSCRIPT in tid:
            tx = t
            break
    if tx is None:
        # fallback: any BRCA1 canonical / longest
        candidates = []
        for t in db.features_of_type('transcript', limit=(BRCA1_CHROM, BRCA1_START, BRCA1_END)):
            gn = t.attributes.get('gene_name', [''])[0]
            if gn == 'BRCA1':
                candidates.append(t)
        if not candidates:
            raise RuntimeError("No BRCA1 transcript found")
        tx = max(candidates, key=lambda t: t.end - t.start)
        log.warning("Using longest BRCA1 transcript: %s",
                    tx.attributes.get('transcript_id', [''])[0])
    log.info("BRCA1 transcript: %s strand=%s span=%d-%d",
             tx.attributes.get('transcript_id', [''])[0],
             tx.strand, tx.start, tx.end)

    def gff_to_0(s, e):
        return s - 1, e

    contexts = {
        'coding_exon': np.zeros(region_len, dtype=bool),
        'intron': np.zeros(region_len, dtype=bool),
        '5utr': np.zeros(region_len, dtype=bool),
        '3utr': np.zeros(region_len, dtype=bool),
        'splice_donor_acceptor': np.zeros(region_len, dtype=bool),
        'intergenic': np.zeros(region_len, dtype=bool),
    }
    tx_s, tx_e = gff_to_0(tx.start, tx.end)
    transcript_mask = np.zeros(region_len, dtype=bool)
    transcript_mask[max(0, tx_s - region_start_0):max(0, tx_e - region_start_0)] = True

    for cds in db.children(tx, featuretype='CDS'):
        s, e = gff_to_0(cds.start, cds.end)
        contexts['coding_exon'][max(0, s - region_start_0):max(0, e - region_start_0)] = True

    exon_intervals = []
    for ex in db.children(tx, featuretype='exon'):
        s, e = gff_to_0(ex.start, ex.end)
        exon_intervals.append((s, e))
    exon_intervals.sort()
    exon_mask = np.zeros(region_len, dtype=bool)
    for s, e in exon_intervals:
        exon_mask[max(0, s - region_start_0):max(0, e - region_start_0)] = True

    cds_intervals = sorted(gff_to_0(c.start, c.end)
                           for c in db.children(tx, featuretype='CDS'))
    if cds_intervals:
        cds_min = min(s for s, _ in cds_intervals)
        cds_max = max(e for _, e in cds_intervals)
        for utr in db.children(tx, featuretype='UTR'):
            s, e = gff_to_0(utr.start, utr.end)
            if tx.strand == '+':
                key = '5utr' if e <= cds_min else ('3utr' if s >= cds_max else '5utr')
            else:
                key = '5utr' if s >= cds_max else ('3utr' if e <= cds_min else '5utr')
            contexts[key][max(0, s - region_start_0):max(0, e - region_start_0)] = True

    contexts['intron'][:] = transcript_mask & ~exon_mask

    splice_mask = np.zeros(region_len, dtype=bool)
    for i, (s, e) in enumerate(exon_intervals):
        if i > 0:
            lo = max(0, s - region_start_0 - 10)
            hi = min(region_len, s - region_start_0 + 10)
            splice_mask[lo:hi] = True
        if i < len(exon_intervals) - 1:
            lo = max(0, e - region_start_0 - 10)
            hi = min(region_len, e - region_start_0 + 10)
            splice_mask[lo:hi] = True
    contexts['splice_donor_acceptor'][:] = splice_mask
    contexts['intergenic'][:] = ~transcript_mask

    for k, v in contexts.items():
        log.info("  %s n=%d", k, int(v.sum()))
    return contexts


@torch.no_grad()
def forward_window(bundle, seq: str) -> Tuple[np.ndarray, np.ndarray]:
    input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
    out = bundle.model(input_ids, output_hidden_states=True)
    D_jsd = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head).numpy()
    D_cos = cosine_lens(out.hidden_states).numpy()
    del out
    torch.cuda.empty_cache()
    return D_jsd.astype(np.float32), D_cos.astype(np.float32)


def stitch(window_results, region_len):
    sum_arr = np.zeros(region_len, dtype=np.float64)
    cnt_arr = np.zeros(region_len, dtype=np.int32)
    for s, prof in window_results:
        e = s + len(prof)
        clip_lo = max(0, s); clip_hi = min(region_len, e)
        if clip_hi <= clip_lo: continue
        offset = clip_lo - s
        sum_arr[clip_lo:clip_hi] += prof[offset:offset + (clip_hi - clip_lo)]
        cnt_arr[clip_lo:clip_hi] += 1
    out = np.full(region_len, np.nan)
    mask = cnt_arr > 0
    out[mask] = sum_arr[mask] / cnt_arr[mask]
    return out


def main():
    parser = argparse.ArgumentParser(description="BRCA1 extension")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    setup_publication_style()

    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)
    L = len(bundle.model.hyena.backbone.layers)

    seq, region_start_0 = load_region_seq(BRCA1_CHROM, BRCA1_START, BRCA1_END)
    region_len = len(seq)
    region_end_0 = region_start_0 + region_len

    starts = list(range(0, region_len - WINDOW + 1, STRIDE))
    log.info("n_windows = %d (region_len=%d)", len(starts), region_len)

    cache = RUNS / "05_brca1_cache.npz"
    if cache.exists() and not args.force:
        z = np.load(cache, allow_pickle=True)
        D_jsd_all = z["D_jsd"]
        D_cos_all = z["D_cos"]
        if D_jsd_all.shape[0] != len(starts):
            cache.unlink(missing_ok=True)
    if not cache.exists():
        D_jsd_all = np.zeros((len(starts), L, WINDOW), dtype=np.float32)
        D_cos_all = np.zeros((len(starts), L, WINDOW), dtype=np.float32)
        for i, s in enumerate(tqdm(starts, desc="forward[BRCA1]")):
            D_jsd_all[i], D_cos_all[i] = forward_window(bundle, seq[s:s + WINDOW])
        np.savez_compressed(cache, D_jsd=D_jsd_all, D_cos=D_cos_all,
                            starts=np.array(starts))

    # gamma_cos calibration: q70 of penultimate
    pen = D_cos_all[:, L - 2, EDGE_WARMUP:].ravel()
    gamma_cos = float(np.quantile(pen, 0.70))
    log.info("gamma_cos calibrated = %.4f", gamma_cos)

    half = (WINDOW - CENTRAL_WIDTH) // 2

    def per_window(D_full, gamma):
        out = []
        for i, s in enumerate(starts):
            c_int, _ = settling_depth_interp(torch.from_numpy(D_full[i]), gamma=gamma)
            central = c_int.numpy().astype(np.float32)[half:half + CENTRAL_WIDTH]
            out.append((s + half, central))
        return out

    ur_profile = stitch(per_window(D_cos_all, gamma_cos), region_len)
    jsd_profile = stitch(per_window(D_jsd_all, GAMMA_JSD), region_len)

    contexts = build_brca1_annotation(region_start_0, region_end_0)

    # Per-context stats
    rows = []
    for lens, profile in [('UR-gDTR', ur_profile), ('JSD-gDTR', jsd_profile)]:
        intron_vals = profile[contexts['intron'] & ~np.isnan(profile)]
        for ctx, mask in contexts.items():
            vals = profile[mask & ~np.isnan(profile)]
            if vals.size == 0:
                continue
            row = {
                "lens": lens, "context": ctx, "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
            }
            if ctx != 'intron' and intron_vals.size > 0:
                _, p, _ = mwu_with_effect(vals, intron_vals)
                d = cohens_d(vals, intron_vals)
                row.update({"MWU_p_vs_intron": p, "cohens_d_vs_intron": d})
            else:
                row.update({"MWU_p_vs_intron": np.nan, "cohens_d_vs_intron": np.nan})
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = TABLES / "brca1_gdtr_by_context.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # F5 figure
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.0),
                             gridspec_kw={'height_ratios': [0.5, 1.5, 1.0]},
                             sharex=True)
    pos_array = np.arange(region_start_0, region_end_0)
    x_kb = (pos_array - BRCA1_START + 1) / 1000.0

    ax = axes[0]
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_ylabel("anno", rotation=0, ha='right', va='center')

    def runs(mask):
        out = []
        d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        for s, e in zip(st, en):
            out.append((s, e))
        return out

    for s, e in runs(contexts['intron']):
        ax.plot([x_kb[s], x_kb[e - 1]], [0, 0], color='gray', lw=0.5)
    for s, e in runs(contexts['coding_exon']):
        ax.add_patch(Rectangle((x_kb[s], -0.5), x_kb[e - 1] - x_kb[s], 1.0,
                               facecolor=WONG_PALETTE['blue'], alpha=0.8))
    for s, e in runs(contexts['5utr']):
        ax.add_patch(Rectangle((x_kb[s], -0.3), x_kb[e - 1] - x_kb[s], 0.6,
                               facecolor=WONG_PALETTE['orange'], alpha=0.7))
    for s, e in runs(contexts['3utr']):
        ax.add_patch(Rectangle((x_kb[s], -0.3), x_kb[e - 1] - x_kb[s], 0.6,
                               facecolor=WONG_PALETTE['vermillion'], alpha=0.7))
    ax.set_title(f"(a) BRCA1 GENCODE annotation (canonical transcript)", fontsize=8)

    ax = axes[1]
    smooth = pd.Series(ur_profile).rolling(200, center=True, min_periods=1).mean().values
    ax.plot(x_kb, ur_profile, color=WONG_PALETTE['blue'], lw=0.3, alpha=0.4,
            label='UR-gDTR (raw)')
    ax.plot(x_kb, smooth, color=WONG_PALETTE['blue'], lw=1.0,
            label='UR-gDTR (200-bp rolling)')
    ax.set_ylabel("UR-gDTR c_interp")
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title(f"(b) UR-gDTR profile (gamma_cos={gamma_cos:.3f})", fontsize=8)

    ax = axes[2]
    smooth_j = pd.Series(jsd_profile).rolling(200, center=True, min_periods=1).mean().values
    ax.plot(x_kb, jsd_profile, color=WONG_PALETTE['bluish_green'], lw=0.3, alpha=0.4)
    ax.plot(x_kb, smooth_j, color=WONG_PALETTE['bluish_green'], lw=1.0)
    ax.set_ylabel("JSD-gDTR c_interp")
    ax.set_xlabel(f"Genomic position relative to BRCA1 start (kb), chr13:{BRCA1_START:,}+")
    ax.set_title("(c) JSD-gDTR profile (auxiliary; gamma=0.5)", fontsize=8)

    plt.tight_layout()
    save_figure(fig, FIGURES / "F5_brca1_profile")
    plt.close(fig)

    (FIGURES / "F5_brca1_profile.caption.json").write_text(json.dumps({
        "figure_id": "F5",
        "title": "BRCA1 region UR-gDTR profile",
        "caption": (
            f"BRCA1 region (chr13:{BRCA1_START:,}-{BRCA1_END:,}, ~125 kb). "
            f"HyenaDNA-medium-160k sliding window=6 kb stride={STRIDE} bp "
            f"(n_windows={len(starts)}). Tracks: (a) GENCODE v44 canonical "
            f"BRCA1 transcript ({BRCA1_TRANSCRIPT}); (b) UR-gDTR settling depth "
            f"(primary, gamma_cos={gamma_cos:.3f}); (c) JSD-gDTR (auxiliary)."
        ),
    }, indent=2))

    # Verdict
    primary_p = float(df[(df['lens'] == 'UR-gDTR') &
                         (df['context'] == 'coding_exon')]['MWU_p_vs_intron'].iloc[0])
    primary_d = float(df[(df['lens'] == 'UR-gDTR') &
                         (df['context'] == 'coding_exon')]['cohens_d_vs_intron'].iloc[0])

    log.info("=" * 60)
    log.info("BRCA1 verdict (UR-gDTR primary):")
    log.info("  exon vs intron MWU p = %.3e Cohen's d = %.3f", primary_p, primary_d)
    log.info("=" * 60)

    payload = {
        "script": "05_brca1.py",
        "purpose": "Stage 5 - BRCA1 extension",
        "seed": args.seed,
        "model": "HyenaDNA-medium-160k (single model used; segmented sliding window)",
        "n_windows": len(starts),
        "window": WINDOW, "stride": STRIDE, "central_width": CENTRAL_WIDTH,
        "L": L,
        "gamma_jsd": GAMMA_JSD,
        "gamma_cos": gamma_cos,
        "brca1_region": {"chrom": BRCA1_CHROM, "start": BRCA1_START, "end": BRCA1_END,
                         "transcript": BRCA1_TRANSCRIPT},
        "context_counts": {k: int(v.sum()) for k, v in contexts.items()},
        "verdict": {"primary_p_value_ur": primary_p, "primary_cohens_d_ur": primary_d},
        "outputs": {
            "table": "results/tables/brca1_gdtr_by_context.csv",
            "figure_F5": "results/figures/F5_brca1_profile.{pdf,png}",
            "cache": "results/runs/05_brca1_cache.npz",
        },
        "runtime_s": time.time() - t0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "05_brca1.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
