"""E1_tied_head_ablation.py - causal test for L7 anomaly hypothesis (c)."""
from __future__ import annotations
import json, logging, platform, socket, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd, torch
from tqdm import tqdm
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.constants import SEED_DEFAULT, VOCAB_REAL
from src.controls import dinuc_shuffle, extract_intergenic_chr17, gc_content, gc_match_random
from src.logit_lens import jsd_lens
from src.ur_gdtr import cosine_lens
from src.model_loader import load_hyenadna, tokenize_sequence
from src.viz import save_figure, setup_publication_style
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("E1_tied_head")
PROJECT_ROOT = Path("/path/to/gDTR-PoC")
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
RUNS = PROJECT_ROOT / "results" / "runs"
CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"
EDGE_WARMUP_BP = 5


def build_sequences(n_per_kind, length, seed):
    """Same logic as 01_sanity_check.build_sequences."""
    log.info("sampling %d intergenic chr17 windows", n_per_kind * 2)
    intergenic, _ = extract_intergenic_chr17(
        fasta_path=str(CHR17_FA), length=length, n=n_per_kind * 2, seed=seed,
    )
    if len(intergenic) < n_per_kind * 2:
        raise RuntimeError(f"only got {len(intergenic)} intergenic windows")
    gc_matched = []
    rng = np.random.default_rng(seed)
    for i in range(n_per_kind):
        target_gc = gc_content(intergenic[i])
        sub_seed = int(rng.integers(0, 2**31 - 1))
        gc_matched.extend(gc_match_random(target_gc, length, n_seqs=1, seed=sub_seed))
    shuffled = []
    for i in range(n_per_kind):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        shuffled.extend(dinuc_shuffle(intergenic[n_per_kind + i], n_shuffles=1, seed=sub_seed))
    return gc_matched, shuffled


@torch.no_grad()
def forward_collect(bundle, seqs, label):
    """Forward each seq, return D_jsd[N,L,T] and D_cos[N,L,T] numpy arrays."""
    Ds_jsd = []
    Ds_cos = []
    for seq in tqdm(seqs, desc=f"forward[{label}]"):
        input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
        out = bundle.model(input_ids, output_hidden_states=True)
        D_j = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head)
        D_c = cosine_lens(out.hidden_states)
        Ds_jsd.append(D_j.numpy())
        Ds_cos.append(D_c.numpy())
        del out
        torch.cuda.empty_cache()
    return np.stack(Ds_jsd, 0), np.stack(Ds_cos, 0)


def m2_per_layer_rate(D, edge=EDGE_WARMUP_BP):
    """Per-layer M2 rate: P(D[ell] <= D[ell-1]) over (seq, position) with edge trim.

    D shape [N, L, T]. Returns float array [L]; layer 0 is boundary (1.0).
    """
    N, L, T = D.shape
    assert edge < T
    out = np.ones(L, dtype=np.float64)
    for ell in range(1, L):
        comp = (D[:, ell, edge:] <= D[:, ell - 1, edge:])
        out[ell] = float(comp.mean())
    return out


def m2_runmin_rate(D, edge=EDGE_WARMUP_BP):
    """Per-layer running-min monotonicity rate.

    Defined as P(running_min(D)[ell] <= running_min(D)[ell-1]) which is True
    by definition; we instead compute the rate that the *raw* D itself
    is non-increasing. This matches the L7_d3d4d5 D1 definition used in
    Phase 0 (M2 = "raw D non-increasing across that block").
    """
    return m2_per_layer_rate(D, edge=edge)


def apply_head_condition(bundle, condition, original_lm_head_w, embedding_w, rng):
    """Mutate bundle.lm_head.weight in place based on condition.

    Returns nothing; caller is responsible for restoring original weights.
    """
    lm = bundle.lm_head
    if condition == "untied":
        lm.weight.data.copy_(original_lm_head_w)
    elif condition == "tied":
        lm.weight.data.copy_(embedding_w)
    elif condition == "random":
        H = lm.weight.shape[1]
        std = float(1.0 / np.sqrt(H))
        gen = torch.Generator(device="cpu").manual_seed(SEED_DEFAULT)
        rand = torch.randn(lm.weight.shape, generator=gen, dtype=torch.float32) * std
        lm.weight.data.copy_(rand.to(lm.weight.dtype).to(lm.weight.device))
    elif condition == "shuffled_rows":
        gen = torch.Generator(device="cpu").manual_seed(SEED_DEFAULT)
        perm = torch.randperm(original_lm_head_w.shape[0], generator=gen)
        lm.weight.data.copy_(original_lm_head_w[perm].clone())
    elif condition == "scaled_random":
        gen = torch.Generator(device="cpu").manual_seed(SEED_DEFAULT + 1)
        scale = float(original_lm_head_w.float().norm() / (original_lm_head_w.numel() ** 0.5))
        rand = torch.randn(lm.weight.shape, generator=gen, dtype=torch.float32) * scale
        lm.weight.data.copy_(rand.to(lm.weight.dtype).to(lm.weight.device))
    else:
        raise ValueError(condition)


def main():
    setup_publication_style()
    t0 = time.time()
    log.info("loading HyenaDNA-medium-160k")
    bundle = load_hyenadna(device="cuda")
    emb_w = bundle.model.hyena.backbone.embeddings.word_embeddings.weight.detach().clone()
    orig_lm_w = bundle.lm_head.weight.detach().clone()
    log.info("emb shape=%s lm_head shape=%s", tuple(emb_w.shape), tuple(orig_lm_w.shape))
    emb_lmhead_equal = bool(torch.equal(emb_w, orig_lm_w))
    emb_lmhead_frob_diff = float((emb_w.float() - orig_lm_w.float()).norm())
    log.info("emb == lm_head numerically? %s (frob diff=%.6e)", emb_lmhead_equal, emb_lmhead_frob_diff)
    n_per_kind = 50
    seq_len = 6000
    seed = SEED_DEFAULT
    log.info("building 100 sequences (50 GC-matched + 50 dinuc-shuf)")
    gc_seqs, shuf_seqs = build_sequences(n_per_kind, seq_len, seed)
    seqs = list(gc_seqs) + list(shuf_seqs)
    log.info("total %d sequences len=%d", len(seqs), len(seqs[0]))
    conditions = ["untied", "tied", "shuffled_rows", "scaled_random", "random"]
    rng = np.random.default_rng(seed)
    results = {}
    for cond in conditions:
        log.info("==== condition=%s ====", cond)
        apply_head_condition(bundle, cond, orig_lm_w, emb_w, rng)
        D_jsd, D_cos = forward_collect(bundle, seqs, label=cond)
        m2_jsd = m2_per_layer_rate(D_jsd)
        m2_cos = m2_per_layer_rate(D_cos)
        results[cond] = {
            "D_jsd_mean_per_layer": D_jsd.mean(axis=(0, 2)).tolist(),
            "D_cos_mean_per_layer": D_cos.mean(axis=(0, 2)).tolist(),
            "m2_jsd_per_layer": m2_jsd.tolist(),
            "m2_cos_per_layer": m2_cos.tolist(),
        }
        log.info("m2_jsd L7=%.4f  m2_cos L7=%.4f", m2_jsd[6], m2_cos[6])
    bundle.lm_head.weight.data.copy_(orig_lm_w)
    layers = list(range(1, 9))
    df_rows = []
    for cond in conditions:
        for ell, m_jsd, m_cos in zip(layers, results[cond]["m2_jsd_per_layer"], results[cond]["m2_cos_per_layer"]):
            df_rows.append({"condition": cond, "layer": ell, "m2_jsd": m_jsd, "m2_cos": m_cos,
                            "D_jsd_mean": results[cond]["D_jsd_mean_per_layer"][ell - 1],
                            "D_cos_mean": results[cond]["D_cos_mean_per_layer"][ell - 1]})
    df = pd.DataFrame(df_rows)
    csv_path = TABLES / "E1_tied_head.csv"
    df.to_csv(csv_path, index=False)
    log.info("wrote %s", csv_path)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    mks = ["o", "s", "D", "v", "^"]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(conditions)}
    markers = {c: mks[i % len(mks)] for i, c in enumerate(conditions)}
    ax = axes[0]
    for cond in conditions:
        ax.plot(layers, results[cond]["m2_jsd_per_layer"], marker=markers[cond],
                color=colors[cond], label=f"JSD {cond}", linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("M2 (per-layer monotonicity rate)")
    ax.set_title("Panel A: per-layer JSD M2 by lm_head condition")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7)
    ax = axes[1]
    cond_idx = np.arange(len(conditions))
    bar_jsd = [results[c]["m2_jsd_per_layer"][6] for c in conditions]
    bar_cos = [results[c]["m2_cos_per_layer"][6] for c in conditions]
    width = 0.35
    ax.bar(cond_idx - width / 2, bar_jsd, width=width, label="JSD lens",
           color=[colors[c] for c in conditions], edgecolor="black", linewidth=0.5)
    ax.bar(cond_idx + width / 2, bar_cos, width=width, label="UR cosine lens (sanity)",
           color=[colors[c] for c in conditions], edgecolor="black", linewidth=0.5, alpha=0.45, hatch="//")
    ax.set_xticks(cond_idx)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("M2 at Layer 7")
    ax.set_title("Panel B: L7 M2 per condition")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig_base = FIGURES / "E1_tied_head"
    save_figure(fig, str(fig_base))
    log.info("wrote %s.{pdf,png}", fig_base)
    plt.close(fig)
    elapsed = time.time() - t0
    meta = {
        "host": socket.gethostname(),
        "emb_lmhead_numerically_equal": emb_lmhead_equal,
        "emb_lmhead_frob_diff": emb_lmhead_frob_diff,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seed": seed,
        "n_per_kind": n_per_kind,
        "seq_len": seq_len,
        "edge_warmup_bp": EDGE_WARMUP_BP,
        "conditions": conditions,
        "wall_seconds": round(elapsed, 1),
    }
    out = {"meta": meta, "results": results,
           "L7_M2_jsd": {c: results[c]["m2_jsd_per_layer"][6] for c in conditions},
           "L7_M2_cos": {c: results[c]["m2_cos_per_layer"][6] for c in conditions}}
    json_path = RUNS / "E1_tied_head.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", json_path)
    log.info("elapsed=%.1fs", elapsed)


if __name__ == "__main__":
    main()
