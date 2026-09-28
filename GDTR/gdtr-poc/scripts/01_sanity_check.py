"""01_sanity_check.py — Gate A (logit-lens validity on Hyena conv blocks).

Generates 100 random 6 kb sequences (50 GC-matched-to-chr17-intergenic +
50 dinucleotide-shuffled), runs HyenaDNA forwards, computes per-position
M1 (top-1 monotonicity rate) and M2 (JSD running-min monotonicity rate)
metrics per layer, and writes the sanity tables and Figure F2.

Outputs:
  results/tables/sanity_M1_M2.csv
  results/tables/sanity_per_position_breakdown.csv
  results/figures/F2_sanity.{pdf,png}
  results/runs/01_sanity.json

Cache:
  results/runs/01_sanity_cache.npz   — D, top1, source labels per sequence
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import platform
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sure src is importable when running as `python scripts/01_sanity_check.py`
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (
    GAMMA_DEFAULT,
    L_DEFAULT,
    SEED_DEFAULT,
    VOCAB_REAL,
)
from src.controls import (
    dinuc_shuffle,
    extract_intergenic_chr17,
    gc_content,
    gc_match_random,
)
from src.gdtr import running_min
from src.logit_lens import jsd_lens, top1_predictions
from src.model_loader import load_hyenadna, tokenize_sequence
from src.stats import binomial_test_one_sided, bootstrap_ci, bootstrap_proportion_ci
from src.viz import save_figure, setup_publication_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("01_sanity")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# Gate A thresholds (Section 3.1)
M1_THRESHOLD_GLOBAL = 0.80
M2_THRESHOLD_GLOBAL = 0.85
M2_THRESHOLD_PERLAYER = 0.70
EDGE_WARMUP_BP = 5  # Section 4.2: skip first 5 nt per design

CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"


def build_sequences(n_per_kind: int, length: int, seed: int) -> Tuple[List[str], List[str]]:
    """Generate (gc_matched_seqs, shuffled_seqs).

    gc_matched: 50 sequences sampled by first pulling chr17 intergenic windows,
                computing per-window GC, then synthesising i.i.d. random
                sequences with that exact target GC. (Direct dinucleotide
                shuffle of the same windows generates the second set.)
    shuffled:   50 dinucleotide-shuffled chr17 intergenic windows.
    """
    log.info("sampling %d intergenic windows from chr17 ...", n_per_kind * 2)
    intergenic, coords = extract_intergenic_chr17(
        fasta_path=str(CHR17_FA), length=length, n=n_per_kind * 2, seed=seed,
    )
    if len(intergenic) < n_per_kind * 2:
        raise RuntimeError(f"only got {len(intergenic)} intergenic windows")

    gc_matched: List[str] = []
    rng = np.random.default_rng(seed)
    for i in range(n_per_kind):
        target_gc = gc_content(intergenic[i])
        # fresh seed per window for variance
        sub_seed = int(rng.integers(0, 2**31 - 1))
        gc_matched.extend(gc_match_random(target_gc, length, n_seqs=1, seed=sub_seed))

    shuffled: List[str] = []
    for i in range(n_per_kind):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        shuffled.extend(dinuc_shuffle(intergenic[n_per_kind + i],
                                      n_shuffles=1, seed=sub_seed))
    return gc_matched, shuffled


@torch.no_grad()
def forward_collect(
    bundle, seqs: List[str], label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward each sequence, return (D[N,L,T], top1[N,L,T]) numpy arrays.

    D values are normalized JSD to final layer; top1 is layer top-1 token id.
    """
    n_seq = len(seqs)
    Ds: List[np.ndarray] = []
    top1s: List[np.ndarray] = []
    for i, seq in enumerate(tqdm(seqs, desc=f"forward[{label}]")):
        input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
        out = bundle.model(input_ids, output_hidden_states=True)
        D = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head)
        top1 = top1_predictions(out.hidden_states, bundle.ln_f, bundle.lm_head)
        Ds.append(D.numpy())
        top1s.append(top1.numpy())
        # free GPU memory aggressively
        del out
        torch.cuda.empty_cache()
    return np.stack(Ds, axis=0), np.stack(top1s, axis=0)


def m1_per_layer(top1: np.ndarray) -> np.ndarray:
    """Per-layer M1 diagnostic: top1[ell, t] == top1[L, t]?

    Returns bool tensor [N, L, T]: at each layer ell, did the layer's top-1
    match the final-layer top-1 already? (Layer L is True by definition.)
    The aggregated *global* M1 (after-first-match stability) is computed
    separately by ``m1_global``.
    """
    N, L, T = top1.shape
    final = top1[:, L - 1, :]                                        # [N, T]
    return (top1 == final[:, None, :])                                # [N, L, T]


def m1_global(top1: np.ndarray) -> np.ndarray:
    """Global M1 (after-first-match stability) per (sequence, position).

    A position is M1-stable iff once its top-1 first matches the final-layer
    top-1, every subsequent layer also matches. Positions whose trajectory
    never matches are non-stable (vacuously False).

    Returns bool [N, T].
    """
    N, L, T = top1.shape
    final = top1[:, L - 1, :]
    matches = (top1 == final[:, None, :])                             # [N, L, T]
    any_match = matches.any(axis=1)                                   # [N, T]
    first_match = matches.argmax(axis=1)                              # [N, T] 0-based
    # tail-after-first-match all True per (n, t)
    out = np.zeros((N, T), dtype=bool)
    if any_match.any():
        # Vectorized: for each k from 0..L-1, mask of positions whose first_match==k
        for k in range(L):
            sel = (first_match == k) & any_match
            if sel.any():
                # Check matches[k:L, ...] all True
                tail_ok = matches[:, k:, :].all(axis=1)               # [N, T]
                out[sel] = tail_ok[sel]
    return out


def m2_per_layer(D: np.ndarray) -> np.ndarray:
    """Per-layer M2 diagnostic: D[ell, t] <= D[ell-1, t]?

    For ell=0 (1-based layer 1) the value is True by convention (no predecessor).
    Returns bool [N, L, T].
    """
    N, L, T = D.shape
    out = np.ones((N, L, T), dtype=bool)
    if L > 1:
        out[:, 1:, :] = D[:, 1:, :] <= D[:, :-1, :]
    return out


def m2_global(D: np.ndarray) -> np.ndarray:
    """Global M2 per (sequence, position): is D[:, t] non-increasing across all layers?

    Returns bool [N, T].
    """
    N, L, T = D.shape
    if L < 2:
        return np.ones((N, T), dtype=bool)
    return (D[:, 1:, :] <= D[:, :-1, :]).all(axis=1)


def trajectory_categories(top1: np.ndarray) -> Dict[str, np.ndarray]:
    """Categorize each (sequence, position) trajectory.

    Categories:
      - always-correct  : top1[ell, t] == final[t] for all ell (entire suffix)
      - late-converge   : top1 eventually matches final and stays matched (M1)
                          but does not match at layer 1
      - oscillate       : matches final at some layer but later deviates
      - never-converge  : top1[ell, t] != final[t] for all ell
    """
    N, L, T = top1.shape
    final = top1[:, L - 1, :]                                        # [N, T]
    matches = (top1 == final[:, None, :])                            # [N, L, T]
    any_match = matches.any(axis=1)                                  # [N, T]
    all_match = matches.all(axis=1)                                  # [N, T]
    # M1-stable: from first match to L-1 stays matched
    first_match_idx = matches.argmax(axis=1)                         # 0-based [N, T]
    # tail_after_first[n, t] = AND of matches[first_idx..L-1, t]
    # Build per-(n,t)
    tail_stable = np.zeros((N, T), dtype=bool)
    for n in range(N):
        for t in range(T):
            if any_match[n, t]:
                f = first_match_idx[n, t]
                tail_stable[n, t] = bool(matches[n, f:, t].all())

    always_correct = all_match
    late_converge = any_match & ~all_match & tail_stable
    oscillate = any_match & ~tail_stable
    never_converge = ~any_match
    return {
        "always_correct": always_correct,
        "late_converge": late_converge,
        "oscillate": oscillate,
        "never_converge": never_converge,
    }


def aggregate_layer_stats(
    bool_per_layer: np.ndarray,
    edge: int,
    name: str,
    seed: int,
) -> pd.DataFrame:
    """Compute mean & bootstrap CI per layer with edge warm-up trimmed."""
    N, L, T = bool_per_layer.shape
    # Trim BOS-side warm-up
    bool_trim = bool_per_layer[:, :, edge:]
    rows = []
    for ell in range(L):
        flat = bool_trim[:, ell, :].flatten()
        successes = int(flat.sum())
        n = int(flat.size)
        p = successes / n
        # Bootstrap proportion CI via resampling
        mean, low, high = bootstrap_proportion_ci(successes, n,
                                                   n_boot=1000, ci=0.95, seed=seed)
        # Per-layer one-sided binomial: H0 p = M2_THRESHOLD_PERLAYER
        binom_p = binomial_test_one_sided(successes, n, M2_THRESHOLD_PERLAYER)
        rows.append({
            "metric": name,
            "layer": ell + 1,
            "n_positions": n,
            "successes": successes,
            "rate": p,
            "ci95_low": low,
            "ci95_high": high,
            "binom_p_vs_0.70": binom_p,
        })
    return pd.DataFrame(rows)


def make_figure_F2(
    df_m1: pd.DataFrame,
    df_m2: pd.DataFrame,
    cat_counts: pd.DataFrame,
    palette: Dict[str, str],
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), constrained_layout=True)

    # Panel a: M1 per layer
    ax = axes[0]
    layers = df_m1["layer"].values
    rates = df_m1["rate"].values
    err_lo = rates - df_m1["ci95_low"].values
    err_hi = df_m1["ci95_high"].values - rates
    ax.errorbar(layers, rates,
                yerr=np.array([err_lo, err_hi]),
                fmt="o-", color=palette["blue"], capsize=2.5, lw=1.0,
                markersize=4)
    ax.axhline(M1_THRESHOLD_GLOBAL, ls="--", lw=0.7, color=palette["vermillion"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer (1-based)")
    ax.set_ylabel("M1: Top-1 monotonicity rate")
    ax.set_title("(a) Top-1 stability")

    # Panel b: M2 per layer
    ax = axes[1]
    rates = df_m2["rate"].values
    err_lo = rates - df_m2["ci95_low"].values
    err_hi = df_m2["ci95_high"].values - rates
    ax.errorbar(layers, rates,
                yerr=np.array([err_lo, err_hi]),
                fmt="s-", color=palette["bluish_green"], capsize=2.5, lw=1.0,
                markersize=4)
    ax.axhline(M2_THRESHOLD_GLOBAL, ls="--", lw=0.7,
               color=palette["vermillion"], label="global threshold")
    ax.axhline(M2_THRESHOLD_PERLAYER, ls=":", lw=0.7,
               color=palette["orange"], label="per-layer threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer (1-based)")
    ax.set_ylabel("M2: JSD running-min monotonicity")
    ax.set_title("(b) JSD monotonicity")
    ax.legend(frameon=False, loc="lower right")

    # Panel c: stacked-bar trajectory categories
    ax = axes[2]
    cats = ["always_correct", "late_converge", "oscillate", "never_converge"]
    colors = [palette["bluish_green"], palette["sky_blue"],
              palette["orange"], palette["vermillion"]]
    bottom = np.zeros(1)
    x = np.array([0])
    for c, col in zip(cats, colors):
        v = np.array([cat_counts[c].sum()])
        ax.bar(x, v, bottom=bottom, color=col, label=c.replace("_", " "),
               width=0.55, edgecolor="white", linewidth=0.5)
        bottom = bottom + v
    ax.set_xticks(x)
    ax.set_xticklabels(["Pooled"])
    ax.set_ylabel("Position count")
    ax.set_title("(c) Trajectory categories")
    ax.legend(frameon=False, fontsize=6, loc="upper right")

    save_figure(fig, str(out_base))
    plt.close(fig)


def main(args: argparse.Namespace) -> Dict:
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    palette = setup_publication_style()
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)

    log.info("building %d sequences (%d GC-matched + %d shuffled)",
             args.n_seq * 2, args.n_seq, args.n_seq)
    gc_seqs, shuf_seqs = build_sequences(n_per_kind=args.n_seq,
                                         length=args.length, seed=args.seed)
    seqs = gc_seqs + shuf_seqs
    labels = ["gc_matched"] * len(gc_seqs) + ["dinuc_shuffle"] * len(shuf_seqs)
    assert len(seqs) == args.n_seq * 2

    cache = RUNS / "01_sanity_cache.npz"
    if cache.exists() and not args.force:
        log.info("loading cached forward outputs from %s", cache)
        z = np.load(cache, allow_pickle=True)
        D_all = z["D"]
        top1_all = z["top1"]
        labels_loaded = list(z["labels"])
        if labels_loaded != labels or D_all.shape[0] != len(seqs):
            log.info("cache mismatch — recomputing")
            cache.unlink(missing_ok=True)
    if not cache.exists():
        D_gc, top1_gc = forward_collect(bundle, gc_seqs, "gc_matched")
        D_sh, top1_sh = forward_collect(bundle, shuf_seqs, "dinuc_shuffle")
        D_all = np.concatenate([D_gc, D_sh], axis=0)
        top1_all = np.concatenate([top1_gc, top1_sh], axis=0)
        np.savez_compressed(
            cache,
            D=D_all.astype(np.float32),
            top1=top1_all.astype(np.int16),
            labels=np.array(labels),
        )
        log.info("cached forwards to %s", cache)

    # Compute M1 / M2 per (seq, layer, position) and globally
    log.info("computing M1 / M2 ...")
    M1_perlayer = m1_per_layer(top1_all)             # bool [N, L, T]
    M2_perlayer = m2_per_layer(D_all)                # bool [N, L, T]
    M1_glob = m1_global(top1_all)                    # bool [N, T]
    M2_glob = m2_global(D_all)                       # bool [N, T]

    df_m1 = aggregate_layer_stats(M1_perlayer, edge=EDGE_WARMUP_BP, name="M1", seed=args.seed)
    df_m2 = aggregate_layer_stats(M2_perlayer, edge=EDGE_WARMUP_BP, name="M2", seed=args.seed)
    df_layer = pd.concat([df_m1, df_m2], ignore_index=True)
    layer_csv = TABLES / "sanity_M1_M2.csv"
    df_layer.to_csv(layer_csv, index=False)
    log.info("wrote %s", layer_csv)

    # Trajectory categories per sequence (overall, not per layer)
    cats = trajectory_categories(top1_all)
    edge = EDGE_WARMUP_BP
    cat_rows = []
    cat_counts = {k: [] for k in cats}
    for i, seq_label in enumerate(labels):
        per = {k: int(cats[k][i, edge:].sum()) for k in cats}
        per["seq_idx"] = i
        per["label"] = seq_label
        per["n_positions"] = int(top1_all.shape[2] - edge)
        cat_rows.append(per)
        for k in cat_counts:
            cat_counts[k].append(per[k])
    cat_df = pd.DataFrame(cat_rows)
    cat_csv = TABLES / "sanity_per_position_breakdown.csv"
    cat_df.to_csv(cat_csv, index=False)
    log.info("wrote %s", cat_csv)

    # Global M1 / M2 (per-position, edge-trimmed)
    M1_global_rate = float(M1_glob[:, edge:].mean())
    M2_global_rate = float(M2_glob[:, edge:].mean())
    # Per-layer M2 minimum excludes the trivially-True layer 1
    df_m2_nontriv = df_m2[df_m2["layer"] >= 2]
    M2_per_layer_min = float(df_m2_nontriv["rate"].min()) if not df_m2_nontriv.empty else 1.0
    M2_per_layer_min_layer = int(df_m2_nontriv.loc[df_m2_nontriv["rate"].idxmin(), "layer"]) \
        if not df_m2_nontriv.empty else 1

    # Gate A verdict (design 3.1)
    pass_M1 = M1_global_rate >= M1_THRESHOLD_GLOBAL
    pass_M2 = M2_global_rate >= M2_THRESHOLD_GLOBAL
    pass_perlayer_M2 = bool((df_m2_nontriv["rate"] >= M2_THRESHOLD_PERLAYER).all()) \
        if not df_m2_nontriv.empty else True
    gate_a_pass = pass_M1 and pass_M2 and pass_perlayer_M2

    # Branch decision per design 3.1
    branch = "PASS"
    if not gate_a_pass:
        if not pass_M2 or not pass_perlayer_M2:
            branch = "FAIL_M2 -> UR-gDTR primary recommended"
        elif not pass_M1:
            branch = "FAIL_M1 -> JSD-DTR valid; drop top-1 auxiliary"
        else:
            branch = "PARTIAL -> exclude failing layers"

    log.info("Gate A: %s — M1=%.3f, M2=%.3f, per-layer min M2=%.3f (layer %d)",
             "PASS" if gate_a_pass else "FAIL",
             M1_global_rate, M2_global_rate, M2_per_layer_min, M2_per_layer_min_layer)

    # Figure F2
    fig_base = FIGURES / "F2_sanity"
    make_figure_F2(df_m1, df_m2, cat_df, palette, fig_base)

    # JSON run record
    runtime_s = time.time() - started
    rec = {
        "script": "01_sanity_check.py",
        "seed": args.seed,
        "n_seq_per_kind": args.n_seq,
        "length": args.length,
        "edge_warmup_bp": EDGE_WARMUP_BP,
        "gamma": GAMMA_DEFAULT,
        "L": L_DEFAULT,
        "vocab_real": VOCAB_REAL,
        "M1_global_rate": M1_global_rate,
        "M2_global_rate": M2_global_rate,
        "M2_per_layer_min": M2_per_layer_min,
        "M2_per_layer_min_layer": M2_per_layer_min_layer,
        "M1_per_layer": df_m1.to_dict(orient="records"),
        "M2_per_layer": df_m2.to_dict(orient="records"),
        "M1_threshold_global": M1_THRESHOLD_GLOBAL,
        "M2_threshold_global": M2_THRESHOLD_GLOBAL,
        "M2_threshold_perlayer": M2_THRESHOLD_PERLAYER,
        "gate_a_pass": gate_a_pass,
        "gate_a_branch": branch,
        "runtime_s": runtime_s,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "cpu",
    }
    out_json = RUNS / "01_sanity.json"
    out_json.write_text(json.dumps(rec, indent=2))
    log.info("wrote %s", out_json)

    print(
        f"\nGate A: {'PASS' if gate_a_pass else 'FAIL'} — "
        f"M1={M1_global_rate:.3f}, M2={M2_global_rate:.3f}, "
        f"per-layer min M2={M2_per_layer_min:.3f} (layer {M2_per_layer_min_layer})"
    )
    print(f"branch: {branch}")
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--n-seq", type=int, default=50,
                   help="sequences per kind (50 GC + 50 shuffled = 100)")
    p.add_argument("--length", type=int, default=6000)
    p.add_argument("--force", action="store_true",
                   help="recompute forwards even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
