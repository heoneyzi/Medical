"""04_hp_sweep.py - HP sweep over (gamma_cos, rho) for UR-gDTR.

Reuses cached cosine distances from 02_gene_structure_cache.npz and
03_variant_cache.npz. No new forwards.

Per (gamma_cos, rho):
  - Cohen's d (UR-gDTR settling depth: exon vs intron) from Stage 2 region
  - mean |Delta_c_interp| over 5 TP53 hotspots from Stage 3

Outputs:
  results/tables/hp_sweep.csv
  results/figures/F9_hp_heatmap.{pdf,png}
  results/runs/04_hp_sweep.json
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gdtr import settling_depth_interp
from src.stats import cohens_d
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("04_hp_sweep")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"

WINDOW = 6000
STRIDE = 500
CENTRAL_WIDTH = 1000
EDGE_WARMUP = 5
TP53_START = 7_668_402
TP53_END = 7_687_550

# Sweep grid
GAMMA_COS_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
RHO_GRID = [0.5, 0.7, 0.85]


def stitch_profile(window_results: List[Tuple[int, np.ndarray]],
                   region_len: int) -> np.ndarray:
    sum_arr = np.zeros(region_len, dtype=np.float64)
    cnt_arr = np.zeros(region_len, dtype=np.int32)
    for start_0, prof in window_results:
        end_0 = start_0 + len(prof)
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
    return out


def main():
    parser = argparse.ArgumentParser(description="HP sweep")
    args = parser.parse_args()
    t0 = time.time()
    setup_publication_style()

    # ----- Stage 2 cache: TP53 region cosine forwards -----
    cache02 = RUNS / "02_gene_structure_cache.npz"
    z02 = np.load(cache02, allow_pickle=True)
    D_cos_02 = z02["D_cos"]    # [n_windows, L, T]
    starts02 = list(z02["starts"])
    n_win, L, T = D_cos_02.shape
    log.info("Stage 2 cache: n_win=%d L=%d T=%d", n_win, L, T)

    # Reload context masks via 02 results
    payload02 = json.loads((RUNS / "02_gene_structure.json").read_text())
    region_len = (T + WINDOW) - WINDOW + WINDOW  # placeholder; use proper compute below

    # Recompute region_len properly: we need to know how many positions
    # We use the same region as Stage 2; reconstruct from the cache file directly.
    # Simpler: use the stored profiles to derive coverage (load from cache with all info).
    # The shuffled cache also has the same shape.
    # We re-derive region length from starts: max(starts)+WINDOW = region_len
    region_len = (TP53_END + WINDOW // 2) - max(0, TP53_START - 1 - WINDOW // 2)
    log.info("derived region_len=%d", region_len)

    # Reconstruct context masks
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    # Re-import build_annotation from 02_gene_structure
    spec = None
    try:
        # Import at module level from local scripts dir
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_gs02", str(PROJECT_ROOT / "scripts" / "02_gene_structure.py"))
        gs02 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gs02)
    except Exception as e:
        log.error("failed to import 02_gene_structure for build_annotation: %s", e)
        raise
    pad = WINDOW // 2
    region_start_0 = max(0, TP53_START - 1 - pad)
    region_end_0 = TP53_END + pad
    contexts, pos_array = gs02.build_annotation(region_start_0, region_end_0)

    # ----- Stage 3 cache: variant forwards -----
    cache03 = RUNS / "03_variant_cache.npz"
    z03 = np.load(cache03, allow_pickle=True)
    D_cos_ref_03 = z03["D_cos_ref"]
    D_cos_alt_03 = z03["D_cos_alt"]
    n_hotspots = D_cos_ref_03.shape[0]
    log.info("Stage 3 cache: n_hotspots=%d", n_hotspots)

    # ----- Sweep -----
    half = (WINDOW - CENTRAL_WIDTH) // 2

    rows = []
    for gamma_cos in GAMMA_COS_GRID:
        # Stitch UR profile under this gamma
        win_results = []
        for i, s in enumerate(starts02):
            D_t = torch.from_numpy(D_cos_02[i])
            c_int, _ = settling_depth_interp(D_t, gamma=gamma_cos)
            central = c_int.numpy().astype(np.float32)[half:half + CENTRAL_WIDTH]
            win_results.append((s + half, central))
        profile = stitch_profile(win_results, region_len)

        # Per-context UR settling depth
        exon_vals = profile[contexts['coding_exon'] & ~np.isnan(profile)]
        intron_vals = profile[contexts['intron'] & ~np.isnan(profile)]
        if exon_vals.size == 0 or intron_vals.size == 0:
            d = float("nan")
        else:
            d = cohens_d(exon_vals, intron_vals)

        # Hotspot |delta_c_interp|
        var_idx_pos = D_cos_ref_03.shape[2] // 2
        delta_c_abs = []
        for i in range(n_hotspots):
            c_ref, _ = settling_depth_interp(torch.from_numpy(D_cos_ref_03[i]),
                                             gamma=gamma_cos)
            c_alt, _ = settling_depth_interp(torch.from_numpy(D_cos_alt_03[i]),
                                             gamma=gamma_cos)
            delta_c_abs.append(abs(c_alt[var_idx_pos].item() - c_ref[var_idx_pos].item()))
        mean_abs_dc = float(np.mean(delta_c_abs))

        for rho in RHO_GRID:
            # gDTR (exon) vs gDTR (intron) at this rho
            threshold = rho * L
            ex_gdtr = float((exon_vals > threshold).mean()) if exon_vals.size else float("nan")
            in_gdtr = float((intron_vals > threshold).mean()) if intron_vals.size else float("nan")
            rows.append({
                "gamma_cos": gamma_cos,
                "rho": rho,
                "L_threshold_count": float(threshold),
                "n_exon_positions": int(exon_vals.size),
                "n_intron_positions": int(intron_vals.size),
                "cohens_d_exon_vs_intron_c_interp": d,
                "exon_gdtr_fraction": ex_gdtr,
                "intron_gdtr_fraction": in_gdtr,
                "mean_abs_delta_c_interp_hotspots": mean_abs_dc,
            })

    df = pd.DataFrame(rows)
    out_csv = TABLES / "hp_sweep.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # ----- Recommend best (gamma_cos, rho) -----
    # Optimization: maximize |Cohen's d| (Gate B effect size) AND mean_abs_dc (variant signal).
    # We rank-normalize each then sum.
    df['rank_d'] = df['cohens_d_exon_vs_intron_c_interp'].abs().rank(method='min')
    df['rank_dc'] = df['mean_abs_delta_c_interp_hotspots'].rank(method='min')
    df['combined_rank'] = df['rank_d'] + df['rank_dc']
    best_row = df.sort_values('combined_rank', ascending=False).iloc[0]

    log.info("=" * 60)
    log.info("HP sweep recommended (gamma_cos, rho):")
    log.info("  gamma_cos=%.3f rho=%.2f", best_row['gamma_cos'], best_row['rho'])
    log.info("  Cohen's d (exon vs intron) = %.3f",
             best_row['cohens_d_exon_vs_intron_c_interp'])
    log.info("  mean |delta_c_interp| hotspots = %.3f",
             best_row['mean_abs_delta_c_interp_hotspots'])
    log.info("=" * 60)

    # ----- F9 heatmap (2 panel) -----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    pivot_d = df.pivot(index='gamma_cos', columns='rho',
                       values='cohens_d_exon_vs_intron_c_interp')
    ax = axes[0]
    im = ax.imshow(pivot_d.values, aspect='auto', cmap='RdBu_r',
                   vmin=-abs(pivot_d.values).max(), vmax=abs(pivot_d.values).max())
    ax.set_xticks(range(len(RHO_GRID)))
    ax.set_xticklabels([f"{r:.2f}" for r in RHO_GRID])
    ax.set_yticks(range(len(GAMMA_COS_GRID)))
    ax.set_yticklabels([f"{g:.3f}" for g in GAMMA_COS_GRID])
    ax.set_xlabel("rho (deep-regime threshold)")
    ax.set_ylabel("gamma_cos")
    ax.set_title("(a) Cohen's d (UR exon vs intron c_interp)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    # annotate
    for i, g in enumerate(GAMMA_COS_GRID):
        for j, r in enumerate(RHO_GRID):
            ax.text(j, i, f"{pivot_d.iloc[i, j]:.2f}",
                    ha='center', va='center', fontsize=6,
                    color='white' if abs(pivot_d.iloc[i, j]) > 0.5 else 'black')

    pivot_dc = df.pivot(index='gamma_cos', columns='rho',
                        values='mean_abs_delta_c_interp_hotspots')
    ax = axes[1]
    im = ax.imshow(pivot_dc.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(RHO_GRID)))
    ax.set_xticklabels([f"{r:.2f}" for r in RHO_GRID])
    ax.set_yticks(range(len(GAMMA_COS_GRID)))
    ax.set_yticklabels([f"{g:.3f}" for g in GAMMA_COS_GRID])
    ax.set_xlabel("rho (deep-regime threshold)")
    ax.set_ylabel("gamma_cos")
    ax.set_title("(b) mean |Delta_c_interp| (5 TP53 hotspots)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i, g in enumerate(GAMMA_COS_GRID):
        for j, r in enumerate(RHO_GRID):
            ax.text(j, i, f"{pivot_dc.iloc[i, j]:.3f}",
                    ha='center', va='center', fontsize=6,
                    color='white' if pivot_dc.iloc[i, j] < pivot_dc.values.max() * 0.5 else 'black')

    plt.tight_layout()
    save_figure(fig, FIGURES / "F9_hp_heatmap")
    plt.close(fig)

    (FIGURES / "F9_hp_heatmap.caption.json").write_text(json.dumps({
        "figure_id": "F9",
        "title": "UR-gDTR hyperparameter sweep",
        "caption": (
            "Hyperparameter sensitivity for UR-gDTR over gamma_cos in "
            f"{GAMMA_COS_GRID} and rho in {RHO_GRID}. (a) Cohen's d for "
            "exon vs intron settling depth (Gate B effect size). (b) Mean "
            "|Delta_c_interp| across the 5 TP53 hotspots. Both reuse the "
            "Stage 2 and Stage 3 cosine forwards (no additional GPU compute)."
        ),
    }, indent=2))

    # ----- runs/json -----
    payload = {
        "script": "04_hp_sweep.py",
        "purpose": "Phase 1 (gamma_cos, rho) recommendation",
        "n_grid_points": int(len(df)),
        "gamma_cos_grid": GAMMA_COS_GRID,
        "rho_grid": RHO_GRID,
        "best": {
            "gamma_cos": float(best_row['gamma_cos']),
            "rho": float(best_row['rho']),
            "cohens_d_exon_vs_intron_c_interp": float(best_row['cohens_d_exon_vs_intron_c_interp']),
            "mean_abs_delta_c_interp_hotspots": float(best_row['mean_abs_delta_c_interp_hotspots']),
        },
        "notes": (
            "rho=0.85 with L=8 means deep regime is c>6.8, i.e., c in {7, 8} only. "
            "rho=0.7 means c>5.6, i.e., c in {6, 7, 8}. rho=0.5 means c>4."
        ),
        "outputs": {
            "table": "results/tables/hp_sweep.csv",
            "figure_F9": "results/figures/F9_hp_heatmap.{pdf,png}",
        },
        "runtime_s": time.time() - t0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "04_hp_sweep.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
