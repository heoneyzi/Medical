"""01b_jsd_distribution.py — JSD effective-range characterization (Section 6.2).

Re-uses the cached forward outputs from 01_sanity_check.py (or recomputes from
scratch if missing). Writes per-layer JSD distribution stats and Figure F3.

Outputs:
  results/tables/jsd_stats.csv
  results/figures/F3_jsd_distribution.{pdf,png}
  results/runs/01b_jsd_dist.json
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
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import L_DEFAULT, SEED_DEFAULT, VOCAB_REAL
from src.viz import save_figure, setup_publication_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("01b_jsd")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

CACHE_PATH = RUNS / "01_sanity_cache.npz"

# Decision thresholds (Section 6.2)
RANGE_FULL_OK = 0.30           # log|V| sufficient
RANGE_QUANTILE_AUX = 0.15      # quantile-gamma supplementary
EDGE_WARMUP_BP = 5


def percentiles(values: np.ndarray, q: List[float]) -> Dict[str, float]:
    return {f"p{int(p*100):02d}": float(np.quantile(values, p)) for p in q}


def make_figure_F3(
    df_stats: pd.DataFrame,
    D_all: np.ndarray,
    labels: List[str],
    palette: Dict[str, str],
    out_base: Path,
    edge: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    N, L, T = D_all.shape

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)

    # Panel a: violin plot per layer
    ax = axes[0]
    data_per_layer = [D_all[:, ell, edge:].flatten() for ell in range(L)]
    parts = ax.violinplot(data_per_layer, showmeans=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor(palette["sky_blue"])
        pc.set_alpha(0.65)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    if "cmeans" in parts:
        parts["cmeans"].set_color(palette["vermillion"])
    if "cmaxes" in parts:
        parts["cmaxes"].set_color("black")
    if "cmins" in parts:
        parts["cmins"].set_color("black")
    if "cbars" in parts:
        parts["cbars"].set_color("black")
    ax.set_xticks(np.arange(1, L + 1))
    ax.set_xlabel("Layer (1-based)")
    ax.set_ylabel("Normalized JSD  D / log|V|")
    ax.set_title("(a) Per-layer distribution")
    ax.set_ylim(0, 1.05)

    # Panel b: ECDF per layer (viridis colormap)
    ax = axes[1]
    cmap = cm.get_cmap("viridis", L)
    for ell in range(L):
        vals = np.sort(data_per_layer[ell])
        if vals.size == 0:
            continue
        # subsample for plotting
        idx = np.linspace(0, vals.size - 1, num=min(2000, vals.size)).astype(int)
        ax.plot(vals[idx],
                np.linspace(0, 1, vals.size)[idx],
                color=cmap(ell), lw=0.9, label=f"L{ell + 1}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Normalized JSD")
    ax.set_ylabel("ECDF")
    ax.set_title("(b) Per-layer ECDF (viridis)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")

    # Annotate effective range from df_stats on panel a
    eff_ranges = (df_stats["p95"] - df_stats["p05"]).values
    avg_range = float(np.mean(eff_ranges))
    axes[0].text(0.02, 0.98,
                 f"mean(p95-p5) = {avg_range:.3f}",
                 transform=axes[0].transAxes, fontsize=7,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="lightgray"))
    save_figure(fig, str(out_base))
    plt.close(fig)


def _ensure_cache(args: argparse.Namespace) -> Path:
    if CACHE_PATH.exists():
        return CACHE_PATH
    log.info("cache missing — running 01_sanity_check first")
    from importlib import import_module
    sanity = import_module("scripts.01_sanity_check")  # type: ignore
    raise SystemExit("cache not present; run scripts/01_sanity_check.py first")


def main(args: argparse.Namespace) -> Dict:
    started = time.time()
    palette = setup_publication_style()
    if not CACHE_PATH.exists():
        log.error("cache %s not found; run 01_sanity_check first", CACHE_PATH)
        raise SystemExit(2)
    log.info("loading cached forward outputs from %s", CACHE_PATH)
    z = np.load(CACHE_PATH, allow_pickle=True)
    D_all = z["D"].astype(np.float32)         # [N, L, T]
    labels = list(z["labels"])
    N, L, T = D_all.shape
    log.info("loaded D shape %s (labels=%s)", D_all.shape,
             dict((l, labels.count(l)) for l in set(labels)))

    rows = []
    edge = EDGE_WARMUP_BP
    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for ell in range(L):
        vals = D_all[:, ell, edge:].flatten()
        row = {
            "layer": ell + 1,
            "n_values": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
        }
        row.update(percentiles(vals, qs))
        row["effective_range_p95_minus_p05"] = row["p95"] - row["p05"]
        rows.append(row)
    df_stats = pd.DataFrame(rows)
    out_csv = TABLES / "jsd_stats.csv"
    df_stats.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # Decision: use the median (or mean) effective range across non-final layers.
    # The final layer's distribution is degenerate (D[L]=0) so we exclude it.
    eff_per_layer = df_stats["effective_range_p95_minus_p05"].values[: L - 1]
    mean_eff = float(np.mean(eff_per_layer))
    median_eff = float(np.median(eff_per_layer))

    if median_eff >= RANGE_FULL_OK:
        recommendation = "log|V| sufficient (gamma=0.5 ok)"
    elif median_eff >= RANGE_QUANTILE_AUX:
        recommendation = "quantile-gamma supplementary"
    else:
        recommendation = "quantile-gamma primary (log|V| normalisation deprecated)"

    # Suggest a quantile gamma: 70th percentile of running-min D at layer L-1
    # (penultimate layer, since layer L is degenerate)
    if L >= 2:
        rm = np.minimum.accumulate(D_all, axis=1)         # [N, L, T]
        gamma_q70 = float(np.quantile(rm[:, L - 2, edge:].flatten(), 0.70))
    else:
        gamma_q70 = float("nan")

    make_figure_F3(df_stats, D_all, labels, palette,
                   FIGURES / "F3_jsd_distribution", edge=edge)

    runtime_s = time.time() - started
    rec = {
        "script": "01b_jsd_distribution.py",
        "seed": args.seed,
        "n_sequences": int(N),
        "n_layers": int(L),
        "edge_warmup_bp": edge,
        "vocab_real": VOCAB_REAL,
        "log_vocab_real": math.log(VOCAB_REAL),
        "per_layer_stats": df_stats.to_dict(orient="records"),
        "effective_range_per_layer": eff_per_layer.tolist(),
        "effective_range_mean": mean_eff,
        "effective_range_median": median_eff,
        "calibration_recommendation": recommendation,
        "suggested_gamma_q70_at_penultimate_layer": gamma_q70,
        "decision_thresholds": {
            "range_full_ok": RANGE_FULL_OK,
            "range_quantile_aux": RANGE_QUANTILE_AUX,
        },
        "runtime_s": runtime_s,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "01b_jsd_dist.json"
    out_json.write_text(json.dumps(rec, indent=2))
    log.info("wrote %s", out_json)

    # Required print line
    eff_str = ", ".join([f"L{ell+1}:{eff_per_layer[ell]:.3f}"
                         for ell in range(len(eff_per_layer))])
    print(f"\nJSD effective range (p95-p5) per layer: [{eff_str}]")
    print(f"median = {median_eff:.3f}; calibration recommendation: {recommendation}")
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
