"""01c_ur_sanity.py - Gate A' (UR-gDTR cosine-lens sanity).

Reuses the same 100 seqs (50 GC-matched + 50 dinuc-shuffled) as 01_sanity_check.py
(seed=42). Forwards through HyenaDNA, computes per-layer cosine distance
D_cos[l,t] = 1 - cos_sim(h_l, h_L), then:

  - M2_ur:  cosine running-min monotonicity rate per position (analog of M2)
  - Convergence-depth std/mean for both default gamma_cos=0.1 and a
    quantile-calibrated gamma_q70 (70th percentile of D_cos at penultimate
    layer).

Outputs:
  results/tables/ur_sanity.csv
  results/figures/F2b_ur_sanity.{pdf,png}
  results/figures/F2b_ur_sanity.caption.json
  results/runs/01c_ur_sanity.json
  results/runs/01c_ur_cache.npz   (D_cos[N,L,T])
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
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import L_DEFAULT, SEED_DEFAULT
from src.controls import dinuc_shuffle, extract_intergenic_chr17, gc_content, gc_match_random
from src.model_loader import load_hyenadna, tokenize_sequence
from src.stats import bootstrap_proportion_ci
from src.ur_gdtr import cosine_lens
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("01c_ur_sanity")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"

EDGE_WARMUP_BP = 5
M2_THRESHOLD_GLOBAL = 0.85
M2_THRESHOLD_PERLAYER = 0.70


def build_sequences(n_per_kind: int, length: int, seed: int) -> Tuple[List[str], List[str]]:
    """Replicate 01_sanity_check.py sequence generation deterministically."""
    intergenic, _ = extract_intergenic_chr17(
        fasta_path=str(CHR17_FA), length=length, n=n_per_kind * 2, seed=seed,
    )
    if len(intergenic) < n_per_kind * 2:
        raise RuntimeError(f"only got {len(intergenic)} intergenic windows")
    rng = np.random.default_rng(seed)
    gc_seqs: List[str] = []
    for i in range(n_per_kind):
        target_gc = gc_content(intergenic[i])
        sub_seed = int(rng.integers(0, 2**31 - 1))
        gc_seqs.extend(gc_match_random(target_gc, length, n_seqs=1, seed=sub_seed))
    shuf_seqs: List[str] = []
    for i in range(n_per_kind):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        shuf_seqs.extend(dinuc_shuffle(intergenic[n_per_kind + i],
                                       n_shuffles=1, seed=sub_seed))
    return gc_seqs, shuf_seqs


@torch.no_grad()
def forward_collect_cosine(bundle, seqs: List[str], label: str) -> np.ndarray:
    """Forward each sequence and return D_cos[N, L, T_real]."""
    Ds: List[np.ndarray] = []
    for seq in tqdm(seqs, desc=f"forward-cos[{label}]"):
        input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
        out = bundle.model(input_ids, output_hidden_states=True)
        D = cosine_lens(out.hidden_states)
        Ds.append(D.numpy().astype(np.float32))
        del out
        torch.cuda.empty_cache()
    return np.stack(Ds, axis=0)


def m2_ur_per_layer(D_cos: np.ndarray) -> np.ndarray:
    """rmin == raw at layer ell? bool [N, L, T]."""
    rmin = np.minimum.accumulate(D_cos, axis=1)
    return rmin == D_cos


def m2_ur_global(D_cos: np.ndarray) -> np.ndarray:
    rmin = np.minimum.accumulate(D_cos, axis=1)
    return np.all(rmin == D_cos, axis=1)


def convergence_depth(D_cos: np.ndarray, gamma_cos: float) -> np.ndarray:
    L = D_cos.shape[1]
    below = D_cos <= gamma_cos
    any_below = below.any(axis=1)
    first_idx = below.argmax(axis=1)
    c = np.where(any_below, first_idx + 1, L)
    return c.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Gate A' UR-gDTR sanity")
    parser.add_argument("--n-seq", type=int, default=50)
    parser.add_argument("--length", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    setup_publication_style()
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)

    log.info("building sequences seed=%d ...", args.seed)
    gc_seqs, shuf_seqs = build_sequences(args.n_seq, args.length, args.seed)
    seqs = gc_seqs + shuf_seqs
    labels = ["gc_matched"] * len(gc_seqs) + ["dinuc_shuffle"] * len(shuf_seqs)

    cache = RUNS / "01c_ur_cache.npz"
    if cache.exists() and not args.force:
        log.info("loading cached cosine forwards %s", cache)
        z = np.load(cache, allow_pickle=True)
        D_cos = z["D_cos"]
        if D_cos.shape[0] != len(seqs):
            log.info("cache mismatch - recomputing")
            cache.unlink(missing_ok=True)
    if not cache.exists():
        D_gc = forward_collect_cosine(bundle, gc_seqs, "gc_matched")
        D_sh = forward_collect_cosine(bundle, shuf_seqs, "dinuc_shuffle")
        D_cos = np.concatenate([D_gc, D_sh], axis=0)
        np.savez_compressed(cache, D_cos=D_cos.astype(np.float32),
                            labels=np.array(labels))
        log.info("cached cosine forwards to %s", cache)

    N, L, T = D_cos.shape
    log.info("D_cos shape = %s", D_cos.shape)

    D_use = D_cos[:, :, EDGE_WARMUP_BP:]

    M2_pl = m2_ur_per_layer(D_use)
    rows: List[dict] = []
    for ell in range(L):
        mask = M2_pl[:, ell, :]
        n = int(mask.size)
        s = int(mask.sum())
        rate, lo, hi = bootstrap_proportion_ci(s, n, n_boot=1000,
                                               ci=0.95, seed=args.seed)
        rows.append({
            "metric": "M2_ur",
            "layer": ell + 1,
            "n_positions": n,
            "successes": s,
            "rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "passes_perlayer_thresh": rate >= M2_THRESHOLD_PERLAYER,
        })
    M2_glob = m2_ur_global(D_use)
    n_glob = int(M2_glob.size)
    s_glob = int(M2_glob.sum())
    rate_glob, lo_glob, hi_glob = bootstrap_proportion_ci(s_glob, n_glob,
                                                          n_boot=1000, ci=0.95,
                                                          seed=args.seed)

    pen = D_use[:, L - 2, :].ravel()
    gamma_cos_q70 = float(np.quantile(pen, 0.70))
    gamma_cos_default = 0.10
    log.info("gamma_cos: default=%.3f q70_penultimate=%.4f mean_pen=%.4f",
             gamma_cos_default, gamma_cos_q70, pen.mean())

    c_q70 = convergence_depth(D_use, gamma_cos_q70)
    c_default = convergence_depth(D_use, gamma_cos_default)
    cdepth_std_q70 = float(np.std(c_q70))
    cdepth_std_default = float(np.std(c_default))
    cdepth_mean_q70 = float(np.mean(c_q70))
    cdepth_mean_default = float(np.mean(c_default))

    df_layer = pd.DataFrame(rows)
    out_csv = TABLES / "ur_sanity.csv"
    df_layer.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    m2_pl_min = float(df_layer["rate"].min())
    m2_pl_min_layer = int(df_layer.loc[df_layer["rate"].idxmin(), "layer"])
    pass_global = rate_glob >= M2_THRESHOLD_GLOBAL
    pass_perlayer_all = bool((df_layer["rate"] >= M2_THRESHOLD_PERLAYER).all())
    overall_pass = bool(pass_global and pass_perlayer_all)

    log.info("=" * 60)
    log.info("Gate A' UR-gDTR sanity verdict:")
    log.info("  global M2_ur rate = %.4f (CI %.4f-%.4f); thresh=%.2f -> %s",
             rate_glob, lo_glob, hi_glob, M2_THRESHOLD_GLOBAL,
             "PASS" if pass_global else "FAIL")
    log.info("  per-layer min M2_ur = %.4f (layer %d); thresh=%.2f all -> %s",
             m2_pl_min, m2_pl_min_layer, M2_THRESHOLD_PERLAYER,
             "PASS" if pass_perlayer_all else "FAIL")
    log.info("  OVERALL: %s", "PASS" if overall_pass else "FAIL")
    log.info("=" * 60)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    ax = axes[0]
    layers = df_layer["layer"].values
    rates = df_layer["rate"].values
    los = df_layer["ci95_low"].values
    his = df_layer["ci95_high"].values
    ax.errorbar(layers, rates, yerr=[rates - los, his - rates],
                fmt="o-", color=WONG_PALETTE["blue"], lw=1.0, capsize=2)
    ax.axhline(M2_THRESHOLD_PERLAYER, color=WONG_PALETTE["vermillion"],
               linestyle="--", lw=0.8, label=f"per-layer >= {M2_THRESHOLD_PERLAYER}")
    ax.axhline(M2_THRESHOLD_GLOBAL, color="gray", linestyle=":",
               lw=0.8, label=f"global >= {M2_THRESHOLD_GLOBAL}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("M2_ur (cos rmin monotone rate)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) UR-gDTR M2_ur per layer")
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[1]
    parts = ax.violinplot(
        [D_use[:, ell, :].ravel() for ell in range(L)],
        positions=range(1, L + 1), widths=0.7, showmeans=True,
        showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(WONG_PALETTE["sky_blue"])
        body.set_alpha(0.6)
    ax.axhline(gamma_cos_default, color=WONG_PALETTE["vermillion"],
               linestyle=":", lw=0.8, label=f"gamma_cos default={gamma_cos_default}")
    ax.axhline(gamma_cos_q70, color=WONG_PALETTE["bluish_green"],
               linestyle="--", lw=0.8, label=f"q70 pen={gamma_cos_q70:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("D_cos = 1 - cos_sim(h_l, h_L)")
    ax.set_title("(b) Cosine-distance distribution")
    ax.legend(fontsize=6)

    ax = axes[2]
    bins = np.arange(0.5, L + 1.5, 1.0)
    ax.hist(c_q70.ravel(), bins=bins, alpha=0.65, color=WONG_PALETTE["bluish_green"],
            label=f"gamma=q70={gamma_cos_q70:.3f}")
    ax.hist(c_default.ravel(), bins=bins, alpha=0.45,
            color=WONG_PALETTE["vermillion"], label=f"gamma={gamma_cos_default}")
    ax.set_xlabel("Convergence depth c (1-based)")
    ax.set_ylabel("Position count")
    ax.set_title("(c) Convergence-depth distribution")
    ax.legend(fontsize=6)

    plt.tight_layout()
    save_figure(fig, FIGURES / "F2b_ur_sanity")
    plt.close(fig)

    caption_path = FIGURES / "F2b_ur_sanity.caption.json"
    caption_path.write_text(json.dumps({
        "figure_id": "F2b",
        "title": "Gate A' UR-gDTR sanity",
        "caption": (
            "Cosine-distance lens (UR-gDTR) sanity for HyenaDNA-medium-160k "
            "on 100 random 6 kb sequences (50 GC-matched chr17 intergenic + "
            "50 dinucleotide-shuffled, seed=42). (a) Per-layer M2_ur - "
            "fraction of post-edge positions where the running-min of cosine "
            "distance to the final layer remains tied to the raw value. "
            f"(b) Cosine-distance distribution per layer with gamma_cos = 0.1 "
            f"default (red dotted) and quantile-calibrated gamma_q70 at the "
            f"penultimate layer = {gamma_cos_q70:.3f} (green dashed). "
            f"(c) Convergence-depth distribution under both gamma choices."
        ),
    }, indent=2))

    payload = {
        "script": "01c_ur_sanity.py",
        "purpose": "Gate A' UR-gDTR cosine-lens sanity (post Gate A FAIL)",
        "seed": args.seed,
        "n_seq_per_kind": args.n_seq,
        "length": args.length,
        "edge_warmup_bp": EDGE_WARMUP_BP,
        "L": L,
        "M2_ur_global_rate": rate_glob,
        "M2_ur_global_ci95": [lo_glob, hi_glob],
        "M2_ur_per_layer_min": m2_pl_min,
        "M2_ur_per_layer_min_layer": m2_pl_min_layer,
        "M2_ur_per_layer": rows,
        "thresholds": {
            "global": M2_THRESHOLD_GLOBAL,
            "per_layer": M2_THRESHOLD_PERLAYER,
        },
        "verdict": {
            "pass_global": bool(pass_global),
            "pass_perlayer_all": pass_perlayer_all,
            "overall_pass": overall_pass,
        },
        "gamma_cos_calibration": {
            "default": gamma_cos_default,
            "q70_penultimate": gamma_cos_q70,
            "convergence_depth_mean_default": cdepth_mean_default,
            "convergence_depth_mean_q70": cdepth_mean_q70,
            "convergence_depth_std_default": cdepth_std_default,
            "convergence_depth_std_q70": cdepth_std_q70,
        },
        "outputs": {
            "table": str(out_csv.relative_to(PROJECT_ROOT)),
            "figure_pdf": "results/figures/F2b_ur_sanity.pdf",
            "figure_png": "results/figures/F2b_ur_sanity.png",
            "cache": "results/runs/01c_ur_cache.npz",
        },
        "runtime_s": time.time() - t0,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    out_json = RUNS / "01c_ur_sanity.json"
    out_json.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
