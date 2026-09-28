"""03_variant_pilot.py - Gate C (TP53 hotspot variant pilot).

For each of 5 TP53 hotspots (R175H, R248Q, R273H, R249S, G245S):
  1. ref forward through HyenaDNA (+/-3 kb context window centered on variant)
  2. alt forward (single-nt change at center)
  3. Compute Delta-metrics for both UR-gDTR and JSD-gDTR:
     - Delta_c_discrete, Delta_c_interp
     - Delta_D(L) vector, max|Delta_D|, signed_argmax_Delta_D
  4. Null distribution: shuffle position +/-100 bp + random allele change, 100x.

Outputs:
  results/tables/tp53_hotspot_metrics.csv
  results/tables/metric_agreement.csv
  results/figures/F7_delta_jsd_heatmap.{pdf,png}
  results/figures/F8_three_metric_agreement.{pdf,png}
  results/runs/03_variant_pilot.json
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

from src.constants import GAMMA_DEFAULT, L_DEFAULT, SEED_DEFAULT
from src.gdtr import settling_depth_discrete, settling_depth_interp
from src.logit_lens import jsd_lens
from src.model_loader import load_hyenadna, tokenize_sequence
from src.ur_gdtr import cosine_lens
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("03_variant_pilot")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"
HOTSPOTS_TSV = PROJECT_ROOT / "data" / "variants" / "tp53_hotspots.tsv"

CONTEXT_BP = 3000   # +/- 3 kb context per variant -> 6 kb window
N_NULL = 100
EDGE_WARMUP = 5
GAMMA_JSD = GAMMA_DEFAULT  # 0.5
GAMMA_COS_FALLBACK = 0.50  # placeholder if no calibration cache


def load_hotspots() -> pd.DataFrame:
    df = pd.read_csv(HOTSPOTS_TSV, sep='\t')
    log.info("loaded %d hotspots", len(df))
    return df


def fetch_window(fa, chrom: str, center_pos_1based: int) -> Tuple[str, int]:
    """Return (seq, center_index_in_seq) for +/-3 kb around center.

    Args:
        fa: pyfaidx.Fasta
        chrom: chromosome name
        center_pos_1based: variant position (1-based)
    Returns:
        seq (string), variant_index (0-based, in seq)
    """
    start_0 = center_pos_1based - 1 - CONTEXT_BP
    end_0 = center_pos_1based - 1 + CONTEXT_BP + 1
    seq = str(fa[chrom][start_0:end_0])
    var_idx = CONTEXT_BP  # 0-based index of center
    return seq.upper(), var_idx


def apply_variant(seq: str, idx: int, alt_base: str) -> str:
    """Replace seq[idx] with alt_base. Asserts current base matches expected ref."""
    return seq[:idx] + alt_base + seq[idx + 1:]


@torch.no_grad()
def forward_get_D(bundle, seq: str) -> Tuple[np.ndarray, np.ndarray]:
    input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
    out = bundle.model(input_ids, output_hidden_states=True)
    D_jsd = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head).numpy()
    D_cos = cosine_lens(out.hidden_states).numpy()
    del out
    torch.cuda.empty_cache()
    return D_jsd.astype(np.float32), D_cos.astype(np.float32)


def delta_metrics_at(D_ref: np.ndarray, D_alt: np.ndarray, pos: int,
                     gamma: float) -> Dict[str, float]:
    """Compute Delta metrics at a position from D_ref/D_alt [L, T] arrays."""
    L, T = D_ref.shape
    D_ref_t = torch.from_numpy(D_ref)
    D_alt_t = torch.from_numpy(D_alt)
    c_ref_d = settling_depth_discrete(D_ref_t, gamma=gamma)[pos].item()
    c_alt_d = settling_depth_discrete(D_alt_t, gamma=gamma)[pos].item()
    c_ref_i, _ = settling_depth_interp(D_ref_t, gamma=gamma)
    c_alt_i, _ = settling_depth_interp(D_alt_t, gamma=gamma)
    delta_D = D_alt[:, pos] - D_ref[:, pos]
    abs_dD = np.abs(delta_D)
    arg = int(np.argmax(abs_dD))
    return {
        "delta_c_discrete": int(c_alt_d - c_ref_d),
        "delta_c_interp": float(c_alt_i[pos].item() - c_ref_i[pos].item()),
        "delta_D_vec": delta_D.tolist(),
        "max_abs_delta_D": float(abs_dD.max()),
        "signed_argmax_delta_D": float(delta_D[arg]),
        "argmax_layer_1based": arg + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate C: TP53 variant pilot")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--gamma-jsd", type=float, default=GAMMA_JSD)
    parser.add_argument("--gamma-cos", type=float, default=None,
                        help="if None, use calibrated value from 02_gene_structure.json")
    parser.add_argument("--n-null", type=int, default=N_NULL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # Calibration: pull gamma_cos from 02_gene_structure.json if available
    if args.gamma_cos is None:
        json02 = RUNS / "02_gene_structure.json"
        if json02.exists():
            payload = json.loads(json02.read_text())
            gamma_cos = float(payload.get("gamma_cos", GAMMA_COS_FALLBACK))
            log.info("loaded gamma_cos = %.4f from 02_gene_structure.json", gamma_cos)
        else:
            gamma_cos = GAMMA_COS_FALLBACK
            log.warning("no 02 json; using gamma_cos=%.4f", gamma_cos)
    else:
        gamma_cos = args.gamma_cos

    setup_publication_style()
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)
    L = len(bundle.model.hyena.backbone.layers)

    from pyfaidx import Fasta
    fa = Fasta(str(CHR17_FA), as_raw=True, sequence_always_upper=True)
    hotspots = load_hotspots()

    # ----- Real variant forwards -----
    cache = RUNS / "03_variant_cache.npz"
    if cache.exists() and not args.force:
        log.info("loading variant cache %s", cache)
        z = np.load(cache, allow_pickle=True)
        D_jsd_ref = z["D_jsd_ref"]
        D_cos_ref = z["D_cos_ref"]
        D_jsd_alt = z["D_jsd_alt"]
        D_cos_alt = z["D_cos_alt"]
        if D_jsd_ref.shape[0] != len(hotspots):
            log.info("cache mismatch - recomputing")
            cache.unlink(missing_ok=True)
    if not cache.exists():
        T_real = 2 * CONTEXT_BP + 1
        D_jsd_ref = np.zeros((len(hotspots), L, T_real), dtype=np.float32)
        D_cos_ref = np.zeros((len(hotspots), L, T_real), dtype=np.float32)
        D_jsd_alt = np.zeros((len(hotspots), L, T_real), dtype=np.float32)
        D_cos_alt = np.zeros((len(hotspots), L, T_real), dtype=np.float32)
        for i, row in tqdm(hotspots.iterrows(), total=len(hotspots), desc="variants"):
            chrom = row['chrom']
            pos = int(row['pos_grch38'])
            ref_b = row['ref'].upper()
            alt_b = row['alt'].upper()
            seq, var_idx = fetch_window(fa, chrom, pos)
            actual = seq[var_idx]
            log.info("%s @ %s:%d expected ref=%s observed=%s",
                     row['variant_name'], chrom, pos, ref_b, actual)
            assert actual == ref_b, f"REF mismatch at {row['variant_name']}: {actual} != {ref_b}"
            seq_alt = apply_variant(seq, var_idx, alt_b)
            D_jsd_r, D_cos_r = forward_get_D(bundle, seq)
            D_jsd_a, D_cos_a = forward_get_D(bundle, seq_alt)
            D_jsd_ref[i] = D_jsd_r
            D_cos_ref[i] = D_cos_r
            D_jsd_alt[i] = D_jsd_a
            D_cos_alt[i] = D_cos_a
        np.savez_compressed(cache,
                            D_jsd_ref=D_jsd_ref, D_cos_ref=D_cos_ref,
                            D_jsd_alt=D_jsd_alt, D_cos_alt=D_cos_alt)
        log.info("saved variant cache")

    # Compute Delta metrics for each hotspot (both lenses)
    rows = []
    delta_D_jsd_mat = np.zeros((len(hotspots), L), dtype=np.float32)
    delta_D_cos_mat = np.zeros((len(hotspots), L), dtype=np.float32)
    var_idx_pos = CONTEXT_BP   # variant position is at the central index

    for i, row in hotspots.iterrows():
        m_ur = delta_metrics_at(D_cos_ref[i], D_cos_alt[i], var_idx_pos,
                                gamma=gamma_cos)
        m_jsd = delta_metrics_at(D_jsd_ref[i], D_jsd_alt[i], var_idx_pos,
                                 gamma=args.gamma_jsd)
        delta_D_cos_mat[i] = np.array(m_ur["delta_D_vec"])
        delta_D_jsd_mat[i] = np.array(m_jsd["delta_D_vec"])
        for lens, m in [("UR-gDTR", m_ur), ("JSD-gDTR", m_jsd)]:
            rec = {
                "variant_name": row['variant_name'],
                "chrom": row['chrom'],
                "pos_grch38": int(row['pos_grch38']),
                "ref": row['ref'],
                "alt": row['alt'],
                "lens": lens,
                "delta_c_discrete": m["delta_c_discrete"],
                "delta_c_interp": m["delta_c_interp"],
                "max_abs_delta_D": m["max_abs_delta_D"],
                "signed_argmax_delta_D": m["signed_argmax_delta_D"],
                "argmax_layer_1based": m["argmax_layer_1based"],
            }
            for ell in range(L):
                rec[f"delta_D_layer{ell+1}"] = m["delta_D_vec"][ell]
            rows.append(rec)

    df_metrics = pd.DataFrame(rows)

    # ----- Null distribution (shuffled position +/-100 bp, random allele) -----
    null_cache = RUNS / "03_null_cache.npz"
    if null_cache.exists() and not args.force:
        z = np.load(null_cache, allow_pickle=True)
        null_max_abs_dD_ur = z["null_max_abs_dD_ur"]
        null_max_abs_dD_jsd = z["null_max_abs_dD_jsd"]
        if null_max_abs_dD_ur.shape != (len(hotspots), args.n_null):
            log.info("null cache mismatch - recomputing")
            null_cache.unlink(missing_ok=True)
    if not null_cache.exists():
        null_max_abs_dD_ur = np.zeros((len(hotspots), args.n_null), dtype=np.float32)
        null_max_abs_dD_jsd = np.zeros((len(hotspots), args.n_null), dtype=np.float32)
        bases = np.array(["A", "C", "G", "T"])
        for i, row in tqdm(hotspots.iterrows(), total=len(hotspots),
                           desc="null variants (per hotspot)"):
            chrom = row['chrom']
            pos = int(row['pos_grch38'])
            seq, var_idx = fetch_window(fa, chrom, pos)
            for k in tqdm(range(args.n_null), desc=f"null_{row['variant_name']}",
                          leave=False):
                # random offset within +/-100 bp, random allele change at that offset
                offset = int(rng.integers(-100, 101))
                while offset == 0:
                    offset = int(rng.integers(-100, 101))
                tgt_idx = var_idx + offset
                cur_b = seq[tgt_idx]
                alt_choices = [b for b in "ACGT" if b != cur_b]
                alt_b = rng.choice(alt_choices)
                seq_alt = seq[:tgt_idx] + str(alt_b) + seq[tgt_idx + 1:]
                D_jsd_r, D_cos_r = forward_get_D(bundle, seq)
                D_jsd_a, D_cos_a = forward_get_D(bundle, seq_alt)
                # Compute |Delta_D| at the modified position (tgt_idx)
                dDr_ur = D_cos_a[:, tgt_idx] - D_cos_r[:, tgt_idx]
                dDr_jsd = D_jsd_a[:, tgt_idx] - D_jsd_r[:, tgt_idx]
                null_max_abs_dD_ur[i, k] = float(np.max(np.abs(dDr_ur)))
                null_max_abs_dD_jsd[i, k] = float(np.max(np.abs(dDr_jsd)))
        np.savez_compressed(null_cache,
                            null_max_abs_dD_ur=null_max_abs_dD_ur,
                            null_max_abs_dD_jsd=null_max_abs_dD_jsd)
        log.info("saved null cache")

    # Per-variant percentiles vs null
    for i, row in hotspots.iterrows():
        for lens, null_vec, obs_max in [
            ("UR-gDTR", null_max_abs_dD_ur[i],
             df_metrics[(df_metrics['variant_name'] == row['variant_name']) &
                        (df_metrics['lens'] == 'UR-gDTR')]['max_abs_delta_D'].iloc[0]),
            ("JSD-gDTR", null_max_abs_dD_jsd[i],
             df_metrics[(df_metrics['variant_name'] == row['variant_name']) &
                        (df_metrics['lens'] == 'JSD-gDTR')]['max_abs_delta_D'].iloc[0]),
        ]:
            pct = float(np.mean(null_vec <= obs_max) * 100.0)
            df_metrics.loc[
                (df_metrics['variant_name'] == row['variant_name']) &
                (df_metrics['lens'] == lens),
                'percentile_vs_null'
            ] = pct
            df_metrics.loc[
                (df_metrics['variant_name'] == row['variant_name']) &
                (df_metrics['lens'] == lens),
                'null_p95'
            ] = float(np.quantile(null_vec, 0.95))
            df_metrics.loc[
                (df_metrics['variant_name'] == row['variant_name']) &
                (df_metrics['lens'] == lens),
                'null_p99'
            ] = float(np.quantile(null_vec, 0.99))
            df_metrics.loc[
                (df_metrics['variant_name'] == row['variant_name']) &
                (df_metrics['lens'] == lens),
                'exceeds_null_p95'
            ] = bool(obs_max > np.quantile(null_vec, 0.95))

    out_csv = TABLES / "tp53_hotspot_metrics.csv"
    df_metrics.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # ----- Three-metric agreement (Spearman rho) for UR-gDTR -----
    from scipy.stats import spearmanr
    sub_ur = df_metrics[df_metrics['lens'] == 'UR-gDTR']
    metric_names = ["delta_c_discrete", "delta_c_interp", "signed_argmax_delta_D"]
    rho_mat = np.zeros((3, 3), dtype=np.float64)
    p_mat = np.zeros((3, 3), dtype=np.float64)
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i == j:
                rho_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
            else:
                v1 = sub_ur[m1].values
                v2 = sub_ur[m2].values
                rho, p = spearmanr(v1, v2)
                rho_mat[i, j] = rho
                p_mat[i, j] = p
    df_agree = pd.DataFrame(rho_mat, index=metric_names, columns=metric_names)
    df_agree_csv = TABLES / "metric_agreement.csv"
    df_agree.to_csv(df_agree_csv)
    log.info("wrote %s", df_agree_csv)

    # ----- F7: variant x layer Delta_D heatmap -----
    log.info("rendering F7 ...")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0),
                             gridspec_kw={'width_ratios': [3, 1.2]})

    # UR row
    ax = axes[0, 0]
    vmax = float(np.max(np.abs(delta_D_cos_mat)))
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(delta_D_cos_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(i + 1) for i in range(L)])
    ax.set_yticks(range(len(hotspots)))
    ax.set_yticklabels(hotspots['variant_name'].tolist())
    ax.set_xlabel("Layer")
    ax.set_title(f"(a) UR-gDTR Delta_D(L) heatmap (primary)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # UR bar with null
    ax = axes[0, 1]
    sub_ur = df_metrics[df_metrics['lens'] == 'UR-gDTR']
    ys = np.arange(len(hotspots))
    obs_ur = sub_ur['max_abs_delta_D'].values
    p95_ur = sub_ur['null_p95'].values
    ax.barh(ys, obs_ur, color=WONG_PALETTE['blue'], alpha=0.75)
    ax.scatter(p95_ur, ys, marker='|', s=80, color='red',
               label='null p95')
    for k, (o, p) in enumerate(zip(obs_ur, p95_ur)):
        marker = '**' if o > sub_ur['null_p99'].iloc[k] else (
            '*' if o > p else '')
        ax.text(o * 1.02, k, marker, va='center', fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels(hotspots['variant_name'].tolist(), fontsize=7)
    ax.set_xlabel("max |Delta_D|")
    ax.set_title("(b) UR max|Delta_D| vs null", fontsize=8)
    ax.legend(fontsize=6)

    # JSD row
    ax = axes[1, 0]
    vmax_j = float(np.max(np.abs(delta_D_jsd_mat)))
    if vmax_j == 0:
        vmax_j = 1.0
    im = ax.imshow(delta_D_jsd_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax_j, vmax=vmax_j)
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(i + 1) for i in range(L)])
    ax.set_yticks(range(len(hotspots)))
    ax.set_yticklabels(hotspots['variant_name'].tolist())
    ax.set_xlabel("Layer")
    ax.set_title(f"(c) JSD-gDTR Delta_D(L) heatmap (auxiliary)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 1]
    sub_jsd = df_metrics[df_metrics['lens'] == 'JSD-gDTR']
    obs_jsd = sub_jsd['max_abs_delta_D'].values
    p95_jsd = sub_jsd['null_p95'].values
    ax.barh(ys, obs_jsd, color=WONG_PALETTE['bluish_green'], alpha=0.75)
    ax.scatter(p95_jsd, ys, marker='|', s=80, color='red')
    for k, (o, p) in enumerate(zip(obs_jsd, p95_jsd)):
        marker = '**' if o > sub_jsd['null_p99'].iloc[k] else (
            '*' if o > p else '')
        ax.text(o * 1.02, k, marker, va='center', fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels(hotspots['variant_name'].tolist(), fontsize=7)
    ax.set_xlabel("max |Delta_D|")
    ax.set_title("(d) JSD max|Delta_D| vs null", fontsize=8)

    plt.tight_layout()
    save_figure(fig, FIGURES / "F7_delta_jsd_heatmap")
    plt.close(fig)

    (FIGURES / "F7_delta_jsd_heatmap.caption.json").write_text(json.dumps({
        "figure_id": "F7",
        "title": "TP53 hotspot variant Delta_D heatmaps",
        "caption": (
            "Layer-wise Delta_D(L) = D_alt - D_ref at the variant position for "
            "5 TP53 hotspots (R175H, R248Q, R273H, R249S, G245S). Top row: "
            "UR-gDTR (cosine lens, primary post Gate A FAIL); bottom row: "
            "JSD-gDTR (auxiliary). Red bars on right show observed max|Delta_D| "
            "with red ticks at the 95th percentile of the per-variant null "
            "distribution (n=100 random alleles +/-100 bp from variant pos). "
            "* = exceeds null p95, ** = exceeds null p99."
        ),
    }, indent=2))

    # ----- F8: three-metric agreement scatter -----
    log.info("rendering F8 ...")
    fig, axes = plt.subplots(3, 3, figsize=(5.4, 5.4))
    sub_ur = df_metrics[df_metrics['lens'] == 'UR-gDTR']
    metric_disp = {
        "delta_c_discrete": "Delta_c_disc",
        "delta_c_interp": "Delta_c_interp",
        "signed_argmax_delta_D": "signed_argmax_Delta_D",
    }
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            ax = axes[i, j]
            if i == j:
                ax.hist(sub_ur[m1].values, bins=5, color=WONG_PALETTE['blue'],
                        alpha=0.7)
                ax.set_xlabel(metric_disp[m1], fontsize=7)
                ax.set_ylabel("count", fontsize=7)
            else:
                ax.scatter(sub_ur[m2].values, sub_ur[m1].values,
                           color=WONG_PALETTE['blue'], s=20)
                ax.set_xlabel(metric_disp[m2], fontsize=7)
                ax.set_ylabel(metric_disp[m1], fontsize=7)
                rho = rho_mat[i, j]
                ax.text(0.05, 0.95, f"rho={rho:.2f}",
                        transform=ax.transAxes, fontsize=6.5,
                        verticalalignment='top')
            ax.tick_params(labelsize=6.5)
    plt.tight_layout()
    save_figure(fig, FIGURES / "F8_three_metric_agreement")
    plt.close(fig)

    (FIGURES / "F8_three_metric_agreement.caption.json").write_text(json.dumps({
        "figure_id": "F8",
        "title": "Three-metric Delta agreement (UR-gDTR)",
        "caption": (
            "Pairwise scatter and Spearman rho among Delta_c_discrete, "
            "Delta_c_interp, and signed_argmax_Delta_D for the 5 TP53 hotspots "
            "under UR-gDTR (primary). Diagonal: marginal histogram. Off-"
            "diagonal: pairwise scatter with Spearman rho text annotation."
        ),
    }, indent=2))

    # ----- Verdict + JSON -----
    sub_ur = df_metrics[df_metrics['lens'] == 'UR-gDTR']
    n_pass_p95 = int(sub_ur['exceeds_null_p95'].sum())
    rho_di = rho_mat[0, 1]   # delta_c_discrete vs delta_c_interp
    rho_di_signed = rho_mat[1, 2]  # delta_c_interp vs signed_argmax

    log.info("=" * 60)
    log.info("Gate C variant pilot results (UR-gDTR primary):")
    for _, row in sub_ur.iterrows():
        log.info("  %s: max|dD|=%.3f (null p95=%.3f) exceeds=%s",
                 row['variant_name'], row['max_abs_delta_D'],
                 row['null_p95'], row['exceeds_null_p95'])
    log.info("  n_exceeding_null_p95 = %d/5", n_pass_p95)
    log.info("  Spearman rho(Delta_c_interp, signed_argmax) = %.3f", rho_di_signed)
    log.info("=" * 60)

    payload = {
        "script": "03_variant_pilot.py",
        "purpose": "Gate C - TP53 hotspot variant pilot (UR-gDTR primary)",
        "seed": args.seed,
        "n_hotspots": int(len(hotspots)),
        "n_null": args.n_null,
        "context_bp": CONTEXT_BP,
        "L": L,
        "gamma_jsd": args.gamma_jsd,
        "gamma_cos": gamma_cos,
        "verdict_gate_C": {
            "n_exceeding_null_p95_ur": n_pass_p95,
            "spearman_rho_dci_signed": float(rho_di_signed),
            "spearman_rho_disc_interp": float(rho_di),
            "informational_threshold_3_of_5": bool(n_pass_p95 >= 3),
        },
        "metric_agreement_rho_matrix": rho_mat.tolist(),
        "metric_agreement_p_matrix": p_mat.tolist(),
        "metric_names": metric_names,
        "outputs": {
            "table_metrics": "results/tables/tp53_hotspot_metrics.csv",
            "table_agreement": "results/tables/metric_agreement.csv",
            "figure_F7": "results/figures/F7_delta_jsd_heatmap.{pdf,png}",
            "figure_F8": "results/figures/F8_three_metric_agreement.{pdf,png}",
        },
        "runtime_s": time.time() - t0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "03_variant_pilot.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
