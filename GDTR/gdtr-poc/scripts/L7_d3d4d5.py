"""L7_d3d4d5.py - Mechanistic diagnostics for the Layer 7 anomaly (HyenaDNA Phase 0).

Three diagnostics, all on the same forward pass over 75 sequences (25 real + 25
dinuc-shuffled + 25 GC=0.5 random, 6 kb, seed=42):

D3 - Long-range integration test
    Per-position relative residual norm of block 8:
        r_8(t) = ||h_8(t) - h_7(t)|| / ||h_7(t)||
    Compare class means via Mann-Whitney U.

D4 - lm_head SVD alignment
    SVD of lm_head.weight[:12] (vocab-masked, shape (12, 256)).
    Per layer ell in 1..8, mean alignment energy with top-k right singular vectors:
        E_ell = mean over positions of (||h_ell V_topk||^2 / ||h_ell||^2)
    Use pre-ln_f hidden states for ell=1..7, post-ln_f for ell=8.
    Report align_jump_topk = E_8 - E_7 for k in {1,4,8,12}.

D5 - Position stratification (real_seqs only)
    Spearman correlation of per-position r_8(t) versus
        - GC content of local 100 bp window centered at t
        - Shannon entropy (k=3 mer) of local 100 bp window
        - position fraction t/T
    Subsample to 50K positions.

Outputs:
    results/runs/L7_d3d4d5.json       - all numerical results
    results/tables/L7_d3d4d5_summary.csv
    results/figures/L7_d3d4d5.{pdf,png}
"""
from __future__ import annotations

import json
import logging
import math
import platform
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import BOS_OFFSET, SEED_DEFAULT, VOCAB_REAL
from src.controls import (
    dinuc_shuffle,
    extract_intergenic_chr17,
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
log = logging.getLogger("L7_d3d4d5")

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"

N_PER_CLASS = 25
SEQ_LEN = 6000
SEED = 42
KS_FOR_D4 = (1, 4, 8, 12)
LOCAL_WIN = 100
KMER_K = 3
SUBSAMPLE_D5 = 50_000


# ----------------------------- sequence builders -----------------------------
def build_three_classes(n: int, length: int, seed: int) -> Dict[str, List[str]]:
    """Return {'real': [...], 'shuf': [...], 'rand': [...]} each of size n."""
    log.info("sampling %d intergenic chr17 windows ...", n)
    real, _coords = extract_intergenic_chr17(
        fasta_path=str(CHR17_FA), length=length, n=n, seed=seed,
    )
    if len(real) < n:
        raise RuntimeError(f"only got {len(real)} real windows; needed {n}")

    rng = np.random.default_rng(seed)
    shuf: List[str] = []
    for i, s in enumerate(real):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        shuf.extend(dinuc_shuffle(s, n_shuffles=1, seed=sub_seed))
    rand = gc_match_random(target_gc=0.5, length=length, n_seqs=n, seed=seed + 1)
    return {"real": real, "shuf": shuf, "rand": rand}


# ----------------------------- core forward pass -----------------------------
@torch.no_grad()
def forward_collect(bundle, seqs: List[str], label: str):
    """Forward each seq, compute r_8 + per-layer E on the fly to keep RAM low."""
    V_topk = forward_collect.V_topk  # tuple of np arrays shape (256, k)

    n_seq = len(seqs)
    r8s: List[np.ndarray] = []
    E_arr = np.zeros((n_seq, 8, len(KS_FOR_D4)), dtype=np.float32)
    sqnorm_arr = np.zeros((n_seq, 8), dtype=np.float32)

    for i, seq in enumerate(tqdm(seqs, desc=f"forward[{label}]")):
        input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
        out = bundle.model(input_ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple len = L+2 = 10
        # hidden_states[1..8] are pre-ln_f block outputs;
        # hidden_states[9] is post-ln_f.

        # --- D3: r_8(t) = ||h8 - h7|| / ||h7||, slice BOS ---
        h7 = hs[7][:, BOS_OFFSET:, :].float()  # [1, T_real, H]
        h8 = hs[8][:, BOS_OFFSET:, :].float()
        diff = h8 - h7
        norm_diff = diff.norm(dim=-1)          # [1, T_real]
        norm_h7 = h7.norm(dim=-1).clamp(min=1e-8)
        r8 = (norm_diff / norm_h7).squeeze(0).cpu().numpy().astype(np.float32)
        r8s.append(r8)

        # --- D4: alignment energies per layer ---
        # ell = 1..7 use pre-ln_f hidden_states[ell]
        # ell = 8 uses post-ln_f hidden_states[9]
        for ell in range(1, 9):
            idx = ell if ell <= 7 else 9
            h = hs[idx][:, BOS_OFFSET:, :].float().squeeze(0).cpu().numpy()  # [T_real, H]
            sq_norm = (h * h).sum(axis=-1)              # [T_real]
            sqnorm_arr[i, ell - 1] = float(sq_norm.mean())
            denom = np.maximum(sq_norm, 1e-12)
            for ki, V in enumerate(V_topk):
                proj = h @ V                             # [T_real, k]
                e = (proj * proj).sum(axis=-1) / denom   # [T_real]
                E_arr[i, ell - 1, ki] = float(e.mean())

        del out
        torch.cuda.empty_cache()

    r8_arr = np.stack(r8s, axis=0).astype(np.float32)  # [n, T_real]
    return {
        "r8": r8_arr,
        "E": E_arr,
        "sqnorm": sqnorm_arr,
    }


# ----------------------------- D5 helpers (real only) ------------------------
def gc_window(seq: str, t: int, win: int) -> float:
    half = win // 2
    lo = max(0, t - half)
    hi = min(len(seq), t + half)
    s = seq[lo:hi]
    if not s:
        return float("nan")
    gc = s.count("G") + s.count("C")
    nacgt = sum(s.count(c) for c in "ACGT")
    return gc / nacgt if nacgt > 0 else float("nan")


def shannon_kmer_entropy(seq: str, t: int, win: int, k: int) -> float:
    """Shannon entropy (in nats) of k-mer distribution in local window."""
    half = win // 2
    lo = max(0, t - half)
    hi = min(len(seq), t + half)
    s = seq[lo:hi]
    if len(s) < k:
        return float("nan")
    counts = Counter(s[j:j + k] for j in range(len(s) - k + 1))
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    H = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            H -= p * math.log(p)
    return H


# ----------------------------- main -----------------------------
def main() -> int:
    t0 = time.time()
    setup_publication_style()
    palette = dict(WONG_PALETTE)

    log.info("Building 75 sequences (25 real / 25 shuf / 25 rand) seq_len=%d seed=%d",
             SEQ_LEN, SEED)
    classes = build_three_classes(N_PER_CLASS, SEQ_LEN, SEED)

    log.info("Loading HyenaDNA-medium-160k bf16 ...")
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)

    # ----- Pre-compute lm_head SVD (vocab-masked) -----
    W = bundle.lm_head.weight.detach().float().cpu().numpy()
    W12 = W[:VOCAB_REAL, :]  # (12, 256)
    log.info("SVD of lm_head.weight[:%d] shape=%s", VOCAB_REAL, W12.shape)
    U_lm, S_lm, Vt = np.linalg.svd(W12, full_matrices=False)
    # Vt: (12, 256). Right singular vectors are rows of Vt; for projection
    # of row-vectors h (1, 256) we want V_topk shape (256, k).
    V_full = Vt.T  # (256, 12)
    V_topk_list: List[np.ndarray] = []
    for k in KS_FOR_D4:
        V_topk_list.append(V_full[:, :k].astype(np.float32))
    forward_collect.V_topk = V_topk_list

    # ----- Forward all three classes -----
    results: Dict[str, dict] = {}
    for cls_name in ("real", "shuf", "rand"):
        results[cls_name] = forward_collect(bundle, classes[cls_name], cls_name)

    # ----- D3: per-sequence mean r_8 per class + Mann-Whitney U -----
    r8_means = {cls: results[cls]["r8"].mean(axis=1) for cls in ("real", "shuf", "rand")}
    d3_stats = {}
    for cls in ("real", "shuf", "rand"):
        arr = r8_means[cls]
        d3_stats[cls] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def mwu(a, b):
        u, p_two = mannwhitneyu(a, b, alternative="two-sided")
        u_g, p_g = mannwhitneyu(a, b, alternative="greater")
        return {"U": float(u), "p_two_sided": float(p_two),
                "p_a_greater_b": float(p_g)}

    d3_mwu = {
        "real_vs_shuf": mwu(r8_means["real"], r8_means["shuf"]),
        "real_vs_rand": mwu(r8_means["real"], r8_means["rand"]),
        "shuf_vs_rand": mwu(r8_means["shuf"], r8_means["rand"]),
    }
    log.info("D3 means: real=%.4f shuf=%.4f rand=%.4f",
             d3_stats["real"]["mean"], d3_stats["shuf"]["mean"], d3_stats["rand"]["mean"])
    log.info("D3 MWU real_vs_shuf two-sided p=%.3e (real>shuf p=%.3e)",
             d3_mwu["real_vs_shuf"]["p_two_sided"],
             d3_mwu["real_vs_shuf"]["p_a_greater_b"])

    # ----- D4: alignment energies per layer averaged over all real_seqs -----
    d4 = {}
    for cls in ("real", "shuf", "rand"):
        E = results[cls]["E"]  # [n, 8, K]
        E_layer = E.mean(axis=0)  # [8, K]
        d4[cls] = {
            "E_per_layer_per_k": E_layer.tolist(),
            "align_jump_topk": {
                f"k={k}": float(E_layer[7, ki] - E_layer[6, ki])
                for ki, k in enumerate(KS_FOR_D4)
            },
            "E_L8_per_k": {f"k={k}": float(E_layer[7, ki]) for ki, k in enumerate(KS_FOR_D4)},
            "E_L7_per_k": {f"k={k}": float(E_layer[6, ki]) for ki, k in enumerate(KS_FOR_D4)},
        }
        log.info("D4 [%s] align_jump (L8-L7) k=1: %.4f, k=4: %.4f, k=8: %.4f, k=12: %.4f",
                 cls,
                 d4[cls]["align_jump_topk"]["k=1"],
                 d4[cls]["align_jump_topk"]["k=4"],
                 d4[cls]["align_jump_topk"]["k=8"],
                 d4[cls]["align_jump_topk"]["k=12"])
    d4["singular_values"] = S_lm.tolist()

    # ----- D5: per-position correlations on real seqs -----
    real_seqs = classes["real"]
    r8_real = results["real"]["r8"]   # [25, T_real]
    n_seq, T_real = r8_real.shape

    rng = np.random.default_rng(SEED + 7)
    total = n_seq * T_real
    if total > SUBSAMPLE_D5:
        flat_idx = rng.choice(total, size=SUBSAMPLE_D5, replace=False)
    else:
        flat_idx = np.arange(total)
    seq_idx = flat_idx // T_real
    pos_in_seq_aligned = flat_idx % T_real  # 0..T_real-1 (already BOS-stripped)

    r8_flat = r8_real.reshape(-1)[flat_idx].astype(np.float64)
    gc_arr = np.empty_like(r8_flat)
    ent_arr = np.empty_like(r8_flat)
    posfrac_arr = ((pos_in_seq_aligned + 1) / T_real).astype(np.float64)

    for i in tqdm(range(len(flat_idx)), desc="D5 features"):
        si = int(seq_idx[i])
        t = int(pos_in_seq_aligned[i])
        seq = real_seqs[si]
        gc_arr[i] = gc_window(seq, t, LOCAL_WIN)
        ent_arr[i] = shannon_kmer_entropy(seq, t, LOCAL_WIN, KMER_K)

    mask = (np.isfinite(r8_flat) & np.isfinite(gc_arr) &
            np.isfinite(ent_arr) & np.isfinite(posfrac_arr))
    r8_flat = r8_flat[mask]
    gc_arr = gc_arr[mask]
    ent_arr = ent_arr[mask]
    posfrac_arr = posfrac_arr[mask]
    log.info("D5 valid sample size = %d", len(r8_flat))

    rho_gc, p_gc = spearmanr(r8_flat, gc_arr)
    rho_ent, p_ent = spearmanr(r8_flat, ent_arr)
    rho_pos, p_pos = spearmanr(r8_flat, posfrac_arr)
    d5 = {
        "n_positions": int(len(r8_flat)),
        "spearman": {
            "gc":      {"rho": float(rho_gc),  "p": float(p_gc)},
            "entropy": {"rho": float(rho_ent), "p": float(p_ent)},
            "position":{"rho": float(rho_pos), "p": float(p_pos)},
        },
    }
    log.info("D5 Spearman: gc rho=%.3f (p=%.2e), ent rho=%.3f (p=%.2e), pos rho=%.3f (p=%.2e)",
             rho_gc, p_gc, rho_ent, p_ent, rho_pos, p_pos)

    # ----- Save JSON results -----
    out = {
        "meta": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "seed": SEED,
            "seq_len": SEQ_LEN,
            "n_per_class": N_PER_CLASS,
            "ks_for_d4": list(KS_FOR_D4),
            "local_window_bp": LOCAL_WIN,
            "kmer_k": KMER_K,
            "subsample_d5": SUBSAMPLE_D5,
            "wall_seconds": None,
        },
        "D3": {
            "per_class": d3_stats,
            "mwu": d3_mwu,
        },
        "D4": d4,
        "D5": d5,
    }

    out["D3"]["r8_means_per_seq"] = {cls: r8_means[cls].astype(float).tolist()
                                      for cls in ("real", "shuf", "rand")}
    elapsed = time.time() - t0
    out["meta"]["wall_seconds"] = round(elapsed, 1)

    json_path = RUNS / "L7_d3d4d5.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", json_path)

    # ----- Summary CSV -----
    csv_rows = []
    for cls in ("real", "shuf", "rand"):
        csv_rows.append({
            "section": "D3",
            "metric": f"r8_mean_{cls}",
            "value": d3_stats[cls]["mean"],
            "extra": f"std={d3_stats[cls]['std']:.4f},n={d3_stats[cls]['n']}",
        })
    for pair, st in d3_mwu.items():
        csv_rows.append({
            "section": "D3",
            "metric": f"MWU_{pair}_p_two",
            "value": st["p_two_sided"],
            "extra": f"U={st['U']:.1f}",
        })
    for k in KS_FOR_D4:
        csv_rows.append({
            "section": "D4",
            "metric": f"align_jump_real_k={k}",
            "value": d4["real"]["align_jump_topk"][f"k={k}"],
            "extra": (f"E_L7={d4['real']['E_L7_per_k'][f'k={k}']:.4f},"
                      f"E_L8={d4['real']['E_L8_per_k'][f'k={k}']:.4f}"),
        })
    for name, st in d5["spearman"].items():
        csv_rows.append({
            "section": "D5",
            "metric": f"spearman_r8_vs_{name}",
            "value": st["rho"],
            "extra": f"p={st['p']:.3e},n={d5['n_positions']}",
        })
    df_summary = pd.DataFrame(csv_rows)
    csv_path = TABLES / "L7_d3d4d5_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    # ----- Figures (3-panel) -----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # Panel A: boxplot per class with MWU overlay
    axA = axes[0]
    box_data = [r8_means["real"], r8_means["shuf"], r8_means["rand"]]
    bp = axA.boxplot(box_data, showfliers=True, widths=0.55, patch_artist=True,
                     medianprops=dict(color="black", lw=1.0),
                     labels=["real", "shuf", "rand"])
    cols = [palette["blue"], palette["sky_blue"], palette["orange"]]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("black")
    rng2 = np.random.default_rng(0)
    for xi, arr in enumerate(box_data, start=1):
        jitter = rng2.normal(0, 0.04, size=len(arr))
        axA.scatter(xi + jitter, arr, s=10, color="black", alpha=0.45, zorder=3)
    p_rs = d3_mwu["real_vs_shuf"]["p_two_sided"]
    p_rr = d3_mwu["real_vs_rand"]["p_two_sided"]
    p_sr = d3_mwu["shuf_vs_rand"]["p_two_sided"]
    def p_str(p):
        if p < 1e-3: return f"p={p:.1e}"
        if p < 0.05: return f"p={p:.3f}"
        return f"p={p:.2f}"
    axA.text(0.98, 0.97,
             f"real vs shuf: {p_str(p_rs)}\n"
             f"real vs rand: {p_str(p_rr)}\n"
             f"shuf vs rand: {p_str(p_sr)}",
             transform=axA.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"))
    axA.set_ylabel(r"per-seq mean $r_8(t)=\Vert h_8-h_7\Vert/\Vert h_7\Vert$")
    axA.set_xlabel("class")
    axA.set_title("D3 - Block-8 relative residual norm")

    # Panel B: alignment energy E_ell per layer, multiple curves k in {1,4,8,12}
    axB = axes[1]
    layers = np.arange(1, 9)
    k_colors = [palette["vermillion"], palette["orange"],
                palette["bluish_green"], palette["blue"]]
    E_real = np.array(d4["real"]["E_per_layer_per_k"])  # [8, K]
    for ki, k in enumerate(KS_FOR_D4):
        axB.plot(layers, E_real[:, ki], "-o", color=k_colors[ki],
                 label=f"k={k}", lw=1.4, ms=4.5)
    axB.axvspan(6.5, 7.5, color="0.85", zorder=0)
    axB.set_xlabel(r"layer $\ell$")
    axB.set_ylabel(r"alignment energy $E_\ell$ (real seqs)")
    axB.set_title("D4 - lm_head SVD alignment by layer")
    axB.set_xticks(layers)
    axB.legend(loc="best", frameon=False)

    # Panel C: scatter of r_8 vs position (rho annotated for all 3 features)
    axC = axes[2]
    plot_n = min(5000, len(r8_flat))
    pidx = rng2.choice(len(r8_flat), size=plot_n, replace=False)
    axC.scatter(posfrac_arr[pidx], r8_flat[pidx], s=4, alpha=0.25,
                color=palette["blue"])
    axC.set_xlabel("position fraction $t/T$")
    axC.set_ylabel(r"$r_8(t)$")
    axC.set_title("D5 - Position-stratified $r_8$")
    axC.text(0.02, 0.97,
             f"rho_GC      = {rho_gc:+.3f} (p={p_gc:.1e})\n"
             f"rho_entropy = {rho_ent:+.3f} (p={p_ent:.1e})\n"
             f"rho_position= {rho_pos:+.3f} (p={p_pos:.1e})\n"
             f"n = {len(r8_flat)}",
             transform=axC.transAxes, ha="left", va="top",
             family="DejaVu Sans Mono", fontsize=6.5,
             bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    fig_path = FIGURES / "L7_d3d4d5"
    save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Wall time: %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
