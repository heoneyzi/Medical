"""L7_d1d2.py - Diagnostics D1 (ln_f isolation) and D2 (residual-norm decomposition)
for the HyenaDNA Layer-7 anomaly identified in Phase 0 (Gate A / A').

D1: For each position, compare two cosine-distance trajectories:
    D_post(l) = 1 - cos(h_l_pre, h_post)   for l in {1..8}
    D_pre(l)  = 1 - cos(h_l_pre, h_8_pre)  for l in {1..7}
    Compute M2 per-layer monotonicity rate for both. ln_f attribution at L7.

D2: For each block l in {2..8}, compute relative residual norm per position:
    r(l, t) = ||h_l(t) - h_{l-1}(t)|| / ||h_{l-1}(t)||
    Average over positions and sequences. Report per-block.

Inputs: 50 sequences (25 GC-matched chr17 intergenic + 25 dinuc-shuffled,
seed=42, 6 kb).

Outputs (additive only):
    /path/to/gDTR-PoC/scripts/L7_d1d2.py
    /path/to/gDTR-PoC/results/runs/L7_d1d2.json
    /path/to/gDTR-PoC/results/tables/L7_d1d2_summary.csv
    /path/to/gDTR-PoC/results/figures/L7_d1d2.{pdf,png}
"""
from __future__ import annotations

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
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import BOS_OFFSET, L_DEFAULT, SEED_DEFAULT
from src.controls import (
    dinuc_shuffle,
    extract_intergenic_chr17,
    gc_content,
    gc_match_random,
)
from src.model_loader import load_hyenadna, tokenize_sequence
from src.viz import save_figure, setup_publication_style, WONG_PALETTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("L7_d1d2")

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
    """25 GC-matched + 25 dinuc-shuffled chr17 intergenic 6kb sequences.

    Mirrors 01c_ur_sanity.build_sequences but with n_per_kind=25 (50 total).
    Deterministic given (seed, length, n_per_kind).
    """
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
        shuf_seqs.extend(
            dinuc_shuffle(intergenic[n_per_kind + i], n_shuffles=1, seed=sub_seed)
        )
    return gc_seqs, shuf_seqs


@torch.no_grad()
def forward_collect_full(
    bundle, seqs: List[str], label: str, L: int = L_DEFAULT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sequence forward producing:
       D_post[N, L,   T_real] cos-dist h_l_pre vs h_post (l=1..L)
       D_pre [N, L-1, T_real] cos-dist h_l_pre vs h_L_pre (l=1..L-1)
       R_rel [N, L-1, T_real] relative residual norm for block l in 2..L

    hidden_states layout (Appendix C.3 verified):
      [0]      = embeddings
      [1..L]   = pre-ln_f block outputs (float32 due to internal upcast, C.10)
      [L+1]    = post-ln_f
    BOS sliced via [:, BOS_OFFSET:, :] (Appendix C.4).
    """
    D_post_list: List[np.ndarray] = []
    D_pre_list: List[np.ndarray] = []
    R_rel_list: List[np.ndarray] = []
    for seq in tqdm(seqs, desc=f"forward[{label}]"):
        input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
        out = bundle.model(input_ids, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != L + 2:
            raise RuntimeError(
                f"unexpected hidden_states length {len(hs)} (expected {L+2})"
            )
        h_pre = [hs[ell].float()[:, BOS_OFFSET:, :] for ell in range(1, L + 1)]
        h_post = hs[L + 1].float()[:, BOS_OFFSET:, :]

        h_post_n = F.normalize(h_post, p=2, dim=-1)
        h_L_pre_n = F.normalize(h_pre[-1], p=2, dim=-1)

        T_real = h_post.shape[1]
        D_post = torch.zeros((L, T_real), dtype=torch.float32)
        D_pre = torch.zeros((L - 1, T_real), dtype=torch.float32)
        for ell in range(L):
            h_n = F.normalize(h_pre[ell], p=2, dim=-1)
            cos_post = (h_n * h_post_n).sum(dim=-1).clamp(min=-1.0, max=1.0)
            d_post = (1.0 - cos_post).clamp(min=0.0).mean(dim=0)
            D_post[ell] = d_post.cpu()
            if ell < L - 1:
                cos_pre = (h_n * h_L_pre_n).sum(dim=-1).clamp(min=-1.0, max=1.0)
                d_pre = (1.0 - cos_pre).clamp(min=0.0).mean(dim=0)
                D_pre[ell] = d_pre.cpu()

        R_rel = torch.zeros((L - 1, T_real), dtype=torch.float32)
        for ell in range(2, L + 1):
            h_prev = h_pre[ell - 2]
            h_cur = h_pre[ell - 1]
            num = torch.linalg.vector_norm(h_cur - h_prev, dim=-1)
            den = torch.linalg.vector_norm(h_prev, dim=-1).clamp(min=1e-8)
            r = (num / den).mean(dim=0)
            R_rel[ell - 2] = r.cpu()

        D_post_list.append(D_post.numpy().astype(np.float32))
        D_pre_list.append(D_pre.numpy().astype(np.float32))
        R_rel_list.append(R_rel.numpy().astype(np.float32))
        del out, hs, h_pre, h_post
        torch.cuda.empty_cache()

    D_post_arr = np.stack(D_post_list, axis=0)
    D_pre_arr = np.stack(D_pre_list, axis=0)
    R_rel_arr = np.stack(R_rel_list, axis=0)
    return D_post_arr, D_pre_arr, R_rel_arr


def m2_post_per_layer(D_post: np.ndarray) -> np.ndarray:
    rmin = np.minimum.accumulate(D_post, axis=1)
    return rmin == D_post


def m2_pre_per_layer(D_pre: np.ndarray) -> np.ndarray:
    rmin = np.minimum.accumulate(D_pre, axis=1)
    return rmin == D_pre


def main() -> int:
    t0 = time.time()
    seed = SEED_DEFAULT
    n_per_kind = 25
    length = 6000
    L = L_DEFAULT
    np.random.seed(seed)
    torch.manual_seed(seed)

    setup_publication_style()
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)

    log.info("building 25+25=50 sequences seed=%d length=%d", seed, length)
    gc_seqs, shuf_seqs = build_sequences(n_per_kind, length, seed)
    seqs = gc_seqs + shuf_seqs
    labels = ["gc_matched"] * len(gc_seqs) + ["dinuc_shuffle"] * len(shuf_seqs)
    log.info("seqs=%d (gc=%d shuf=%d)", len(seqs), len(gc_seqs), len(shuf_seqs))

    log.info("forward pass collecting D_post, D_pre, R_rel ...")
    D_post, D_pre, R_rel = forward_collect_full(bundle, seqs, "L7_d1d2", L=L)
    log.info(
        "shapes D_post=%s D_pre=%s R_rel=%s",
        D_post.shape, D_pre.shape, R_rel.shape,
    )

    D_post_use = D_post[:, :, EDGE_WARMUP_BP:]
    D_pre_use = D_pre[:, :, EDGE_WARMUP_BP:]
    R_rel_use = R_rel[:, :, EDGE_WARMUP_BP:]

    M2_post_mask = m2_post_per_layer(D_post_use)
    M2_pre_mask = m2_pre_per_layer(D_pre_use)

    M2_post_per_layer_list = []
    for ell in range(L):
        mask = M2_post_mask[:, ell, :]
        rate = float(mask.mean())
        M2_post_per_layer_list.append({
            "layer": ell + 1,
            "metric": "M2_post",
            "rate": rate,
            "n_positions": int(mask.size),
            "successes": int(mask.sum()),
            "passes_perlayer_thresh": rate >= M2_THRESHOLD_PERLAYER,
        })
    M2_pre_per_layer_list = []
    for ell in range(L - 1):
        mask = M2_pre_mask[:, ell, :]
        rate = float(mask.mean())
        M2_pre_per_layer_list.append({
            "layer": ell + 1,
            "metric": "M2_pre",
            "rate": rate,
            "n_positions": int(mask.size),
            "successes": int(mask.sum()),
            "passes_perlayer_thresh": rate >= M2_THRESHOLD_PERLAYER,
        })

    M2_post_L7 = M2_post_per_layer_list[6]["rate"]
    M2_pre_L7 = M2_pre_per_layer_list[6]["rate"]
    if (1.0 - M2_post_L7) > 1e-9:
        ln_f_pct = (M2_pre_L7 - M2_post_L7) / (1.0 - M2_post_L7) * 100.0
    else:
        ln_f_pct = float("nan")

    block_ids = list(range(2, L + 1))
    block_means = R_rel_use.mean(axis=(0, 2))
    block_stds = R_rel_use.std(axis=(0, 2))
    r_block8 = float(block_means[-1])
    r_others = block_means[:-1]
    r_mean_2_7 = float(r_others.mean())
    r_max_2_7 = float(r_others.max())
    block8_ratio_vs_mean = r_block8 / r_mean_2_7 if r_mean_2_7 > 0 else float("nan")
    block8_ratio_vs_max = r_block8 / r_max_2_7 if r_max_2_7 > 0 else float("nan")

    rows = []
    for r in M2_post_per_layer_list:
        rows.append({
            "panel": "D1",
            "metric": "M2_post",
            "layer": r["layer"],
            "rate_or_value": r["rate"],
            "passes_perlayer_thresh": r["passes_perlayer_thresh"],
        })
    for r in M2_pre_per_layer_list:
        rows.append({
            "panel": "D1",
            "metric": "M2_pre",
            "layer": r["layer"],
            "rate_or_value": r["rate"],
            "passes_perlayer_thresh": r["passes_perlayer_thresh"],
        })
    for ell, mu, sd in zip(block_ids, block_means.tolist(), block_stds.tolist()):
        rows.append({
            "panel": "D2",
            "metric": "rel_residual_norm",
            "layer": ell,
            "rate_or_value": mu,
            "passes_perlayer_thresh": None,
        })
    rows.append({
        "panel": "D1_summary", "metric": "M2_post_L7", "layer": 7,
        "rate_or_value": M2_post_L7, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D1_summary", "metric": "M2_pre_L7", "layer": 7,
        "rate_or_value": M2_pre_L7, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D1_summary", "metric": "ln_f_attribution_pct", "layer": 7,
        "rate_or_value": ln_f_pct, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D2_summary", "metric": "r_block8", "layer": 8,
        "rate_or_value": r_block8, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D2_summary", "metric": "r_mean_blocks_2_7", "layer": -1,
        "rate_or_value": r_mean_2_7, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D2_summary", "metric": "block8_ratio_vs_mean", "layer": 8,
        "rate_or_value": block8_ratio_vs_mean, "passes_perlayer_thresh": None,
    })
    rows.append({
        "panel": "D2_summary", "metric": "block8_ratio_vs_max", "layer": 8,
        "rate_or_value": block8_ratio_vs_max, "passes_perlayer_thresh": None,
    })

    df = pd.DataFrame(rows)
    csv_path = TABLES / "L7_d1d2_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info("wrote %s", csv_path)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    axA = axes[0]
    layers_post = [r["layer"] for r in M2_post_per_layer_list]
    rates_post = [r["rate"] for r in M2_post_per_layer_list]
    layers_pre = [r["layer"] for r in M2_pre_per_layer_list]
    rates_pre = [r["rate"] for r in M2_pre_per_layer_list]
    axA.plot(layers_post, rates_post, marker="o",
             color=WONG_PALETTE["blue"], label="M2_post (with ln_f)")
    axA.plot(layers_pre, rates_pre, marker="s",
             color=WONG_PALETTE["vermillion"], label="M2_pre (no ln_f)")
    axA.axhline(M2_THRESHOLD_GLOBAL, ls=":", color="gray", lw=0.8,
                label=f"global threshold {M2_THRESHOLD_GLOBAL}")
    axA.axvline(7, ls="--", color="black", lw=0.6, alpha=0.6)
    axA.set_xlabel("layer")
    axA.set_ylabel("M2 monotonicity rate")
    axA.set_title("D1: ln_f isolation (post vs pre)")
    axA.set_ylim(0.0, 1.05)
    axA.set_xticks(list(range(1, L + 1)))
    axA.legend(loc="lower left", fontsize=6.5, frameon=False)

    axA.annotate(
        f"L7 post={M2_post_L7:.2f}\nL7 pre={M2_pre_L7:.2f}\nln_f attr={ln_f_pct:.1f}%",
        xy=(7, min(M2_post_L7, M2_pre_L7)),
        xytext=(0.55, 0.30), textcoords="axes fraction",
        fontsize=6.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.4),
    )

    axB = axes[1]
    colors = [WONG_PALETTE["sky_blue"]] * (L - 1)
    colors[-1] = WONG_PALETTE["vermillion"]
    axB.bar(block_ids, block_means.tolist(), color=colors,
            edgecolor="black", linewidth=0.5)
    axB.axhline(r_mean_2_7, ls=":", color="gray", lw=0.8,
                label=f"mean blocks 2-7 = {r_mean_2_7:.3f}")
    axB.set_xlabel("block index l")
    axB.set_ylabel("relative residual norm r(l)")
    axB.set_title("D2: residual-norm decomposition")
    axB.set_xticks(block_ids)
    axB.legend(loc="upper left", fontsize=6.5, frameon=False)
    axB.annotate(
        f"block 8 / mean(2-7) = {block8_ratio_vs_mean:.2f}x",
        xy=(8, r_block8), xytext=(0.05, 0.85), textcoords="axes fraction",
        fontsize=6.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.4),
    )

    fig.suptitle(
        "Layer-7 Anomaly Diagnostics (D1+D2) - HyenaDNA medium-160k",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, FIGURES / "L7_d1d2")
    plt.close(fig)

    runtime_s = time.time() - t0
    result = {
        "script": "L7_d1d2.py",
        "purpose": "L7 anomaly diagnostics D1 (ln_f isolation) + D2 (residual-norm decomposition)",
        "seed": seed,
        "n_per_kind": n_per_kind,
        "n_seq": len(seqs),
        "length": length,
        "edge_warmup_bp": EDGE_WARMUP_BP,
        "L": L,
        "labels": labels,
        "D1": {
            "M2_post_per_layer": M2_post_per_layer_list,
            "M2_pre_per_layer": M2_pre_per_layer_list,
            "M2_post_L7": M2_post_L7,
            "M2_pre_L7": M2_pre_L7,
            "ln_f_attribution_pct": ln_f_pct,
            "verdict": (
                "ln_f explains majority of L7 anomaly"
                if (not np.isnan(ln_f_pct) and ln_f_pct >= 50.0)
                else "ln_f explains a minority of L7 anomaly"
            ),
        },
        "D2": {
            "blocks": block_ids,
            "rel_residual_norm_mean": block_means.tolist(),
            "rel_residual_norm_std": block_stds.tolist(),
            "r_block8": r_block8,
            "r_mean_blocks_2_7": r_mean_2_7,
            "r_max_blocks_2_7": r_max_2_7,
            "block8_ratio_vs_mean": block8_ratio_vs_mean,
            "block8_ratio_vs_max": block8_ratio_vs_max,
            "verdict": (
                "block 8 dominates magnitude (consistent with hypotheses (a)+(c))"
                if block8_ratio_vs_mean >= 1.5
                else "block 8 magnitude similar to others (anomaly geometric/rotational, not magnitude)"
            ),
        },
        "thresholds": {
            "M2_perlayer": M2_THRESHOLD_PERLAYER,
            "M2_global": M2_THRESHOLD_GLOBAL,
        },
        "outputs": {
            "json": str(RUNS / "L7_d1d2.json"),
            "csv": str(csv_path),
            "figure_pdf": str(FIGURES / "L7_d1d2.pdf"),
            "figure_png": str(FIGURES / "L7_d1d2.png"),
        },
        "runtime_s": runtime_s,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    json_path = RUNS / "L7_d1d2.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("wrote %s", json_path)
    log.info("runtime=%.1fs", runtime_s)
    log.info("D1: M2_post(L7)=%.4f  M2_pre(L7)=%.4f  ln_f attr=%.1f%%",
             M2_post_L7, M2_pre_L7, ln_f_pct)
    log.info("D2: r(block8)=%.4f vs mean(2..7)=%.4f  ratio=%.2f",
             r_block8, r_mean_2_7, block8_ratio_vs_mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
