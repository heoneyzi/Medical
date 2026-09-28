"""E5 — Tuned Lens (Belrose 2023) on HyenaDNA-medium-160k.

Phase 0 extension: train per-layer affine maps A_7, A_8 such that
  lm_head(A_l(h_l)) reproduces the final logits.
If hypothesis (c') (trained readout subspace alignment) is correct, A_7
should learn to "pre-rotate" h_7 into the trained 12-dim subspace and
recover M2_L7 from the Phase 0 baseline (~0.12) toward >= 0.85.

Outputs:
  results/runs/E5_tuned_lens.json
  results/runs/E5_lens_weights.npz
  results/tables/E5_tuned_lens.csv
  results/figures/E5_tuned_lens.{pdf,png}
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("/path/to/gDTR-PoC")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_hyenadna, tokenize_sequence
from src.controls import (
    extract_intergenic_chr17, dinuc_shuffle,
    gc_match_random, gc_content,
)
from src.constants import BOS_OFFSET, VOCAB_REAL, HF_REVISION
from src.logit_lens import jsd_lens
from src.ur_gdtr import cosine_lens

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("E5")

CHR17_FA = PROJECT_ROOT / "data" / "reference" / "chr17.fa"
RUNS = PROJECT_ROOT / "results" / "runs"
TABLES = PROJECT_ROOT / "results" / "tables"
FIGS = PROJECT_ROOT / "results" / "figures"
for p in (RUNS, TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)

EDGE = 5
LENGTH = 6000
HIDDEN = 256
TARGET_LAYERS = [7, 8]


def build_corpus(n_per_kind: int, length: int, seed: int) -> List[str]:
    """n_per_kind GC-matched + n_per_kind dinuc-shuffled."""
    log.info("sample windows seed=%d", seed)
    intergenic, _ = extract_intergenic_chr17(
        fasta_path=str(CHR17_FA), length=length,
        n=n_per_kind * 2, seed=seed,
    )
    if len(intergenic) < n_per_kind * 2:
        raise RuntimeError(f"got {len(intergenic)}")
    rng = np.random.default_rng(seed)
    out: List[str] = []
    for i in range(n_per_kind):
        gc = gc_content(intergenic[i])
        sub = int(rng.integers(0, 2**31 - 1))
        out.extend(gc_match_random(gc, length, 1, sub))
    for i in range(n_per_kind):
        sub = int(rng.integers(0, 2**31 - 1))
        out.extend(dinuc_shuffle(
            intergenic[n_per_kind + i], 1, sub))
    return out


def make_affine(hidden, device, dtype=torch.float32):
    A = nn.Linear(hidden, hidden, bias=True)
    nn.init.eye_(A.weight)
    nn.init.zeros_(A.bias)
    A = A.to(device=device, dtype=dtype)
    for p in A.parameters():
        p.requires_grad_(True)
    return A


@torch.no_grad()
def collect_hidden(bundle, seqs, layers_keep):
    cached = {ell: [] for ell in layers_keep}
    targets = []
    for seq in seqs:
        ids, _ = tokenize_sequence(seq, bundle.tokenizer, "cuda")
        out = bundle.model(ids, output_hidden_states=True)
        for ell in layers_keep:
            h = out.hidden_states[ell][0, BOS_OFFSET:].float().cpu()
            cached[ell].append(h)
        tgt = out.logits[0, BOS_OFFSET:, :VOCAB_REAL].float().cpu()
        targets.append(tgt)
        del out
        torch.cuda.empty_cache()
    return cached, targets


def _psl(A_dict, cached, targets, i, hw, hb, hd, edge):
    tgt = targets[i].to("cuda")
    T = tgt.shape[0]
    lo, hi = edge, T - edge
    tgt_s = tgt[lo:hi]
    loss = 0.0
    for ell, A in A_dict.items():
        h = cached[ell][i].to("cuda")[lo:hi]
        z = A(h)
        z16 = z.to(hd) @ hw.t()
        if hb is not None:
            z16 = z16 + hb
        z12 = z16[..., :VOCAB_REAL].float()
        loss = loss + torch.nn.functional.mse_loss(z12, tgt_s)
    return loss


def train_lens(Ad, cached, targets, lm_head, edge,
               n_epochs=5, bs=4, lr=1e-3):
    params=[]
    for A in Ad.values(): params.extend(A.parameters())
    opt=torch.optim.Adam(params,lr=lr)
    N=len(targets)
    hw=lm_head.weight.detach().to("cuda")
    hb=lm_head.bias
    if hb is not None: hb=hb.detach().to("cuda")
    hd=lm_head.weight.dtype
    losses=[]
    log.info("train N=%d ep=%d bs=%d",N,n_epochs,bs)
    for ep in range(n_epochs):
        perm=np.random.permutation(N)
        el,nb=0.0,0
        for s in range(0,N,bs):
            idx=perm[s:s+bs]
            opt.zero_grad()
            loss=0.0
            for i in idx:
                loss=loss+_psl(Ad,cached,targets,int(i),hw,hb,hd,edge)
            loss=loss/max(len(idx),1)
            loss.backward(); opt.step()
            el+=float(loss.item()); nb+=1
        avg=el/max(nb,1); losses.append(avg)
        log.info("ep %d loss=%.6f",ep+1,avg)
    return losses


@torch.no_grad()
def jsd_lens_tuned(hidden_states, ln_f, lm_head, A_dict,
                   vocab_real=VOCAB_REAL, bos=BOS_OFFSET):
    """Like jsd_lens but for layers in A_dict use lm_head(A(h)) without ln_f.
    For other layers ell<L use lm_head(ln_f(h)). Final layer uses lm_head(post-ln_f).
    Returns D[L, T_real]."""
    L = len(hidden_states) - 2
    final_idx = len(hidden_states) - 1
    hf = hidden_states[final_idx]
    log_v = float(np.log(vocab_real))
    hd = lm_head.weight.dtype
    # final
    lf = lm_head(hf.to(hd))[..., :vocab_real].float()
    log_pf = F.log_softmax(lf, dim=-1)
    pf = log_pf.exp()
    B, T, _ = pf.shape
    Tr = T - bos
    pf_r = pf[:, bos:, :]
    log_pf_r = log_pf[:, bos:, :]
    D = torch.zeros((L, Tr), dtype=torch.float32)
    eps = 1e-30
    for ell in range(1, L + 1):
        h = hidden_states[ell]
        if ell in A_dict:
            z = A_dict[ell](h.float())
            l_l = lm_head(z.to(hd))[..., :vocab_real].float()
        else:
            l_l = lm_head(ln_f(h.to(hd)))[..., :vocab_real].float()
        log_pl = F.log_softmax(l_l, dim=-1)[:, bos:, :]
        pl = log_pl.exp()
        m = 0.5 * (pl + pf_r)
        log_m = (m + eps).log()
        kl1 = (pl * (log_pl - log_m)).sum(-1)
        kl2 = (pf_r * (log_pf_r - log_m)).sum(-1)
        jsd = 0.5 * (kl1 + kl2)
        jsd = jsd.clamp(min=0.0)
        D[ell - 1] = jsd.mean(dim=0).cpu() / log_v
    D[L - 1].zero_()
    return D


def m2_per_layer(D):
    """D shape [N, L, T]. M2[n,l,t] = D[n,l,t] <= D[n,l-1,t]; layer 0 trivially True."""
    N, L, T = D.shape
    out = np.ones((N, L, T), dtype=bool)
    if L > 1:
        out[:, 1:, :] = D[:, 1:, :] <= D[:, :-1, :]
    return out


def m2_global(D):
    N, L, T = D.shape
    if L < 2:
        return np.ones((N, T), dtype=bool)
    return (D[:, 1:, :] <= D[:, :-1, :]).all(axis=1)


@torch.no_grad()
def eval_M2(bundle, seqs, A_dict, edge):
    """Return (D_baseline [N,L,T], D_tuned [N,L,T], D_ur [N,L,T])."""
    L = 8
    Db, Dt, Du = [], [], []
    for seq in seqs:
        ids, _ = tokenize_sequence(seq, bundle.tokenizer, "cuda")
        out = bundle.model(ids, output_hidden_states=True)
        D_b = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head)
        D_t = jsd_lens_tuned(out.hidden_states, bundle.ln_f,
                             bundle.lm_head, A_dict)
        D_u = cosine_lens(out.hidden_states)
        Db.append(D_b.numpy())
        Dt.append(D_t.numpy())
        Du.append(D_u.numpy())
        del out; torch.cuda.empty_cache()
    return (np.stack(Db), np.stack(Dt), np.stack(Du))


def aggregate_M2(D, edge):
    """Return per-layer rates and global rate, edge-trimmed."""
    Mp = m2_per_layer(D)
    Mg = m2_global(D)
    L = D.shape[1]
    per = []
    for ell in range(L):
        rate = float(Mp[:, ell, edge:].mean())
        per.append(rate)
    glob = float(Mg[:, edge:].mean())
    return per, glob



def make_figure(losses, per_b, per_t, per_u, svd_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: training loss
    ax = axes[0]
    ax.plot(range(1, len(losses) + 1), losses, marker="o", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean MSE loss (logit space)")
    ax.set_title("(A) Training loss")
    ax.grid(True, ls=":", alpha=0.5)

    # Panel B: per-layer M2 baseline vs tuned vs UR
    ax = axes[1]
    layers = np.arange(1, 9)
    ax.plot(layers, per_b, marker="o", label="JSD baseline", color="C0")
    ax.plot(layers, per_t, marker="s", label="JSD tuned (E5)", color="C2")
    ax.plot(layers, per_u, marker="^", label="UR (sanity)", color="C1", ls="--")
    ax.axhline(0.85, ls=":", color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("M2 (running-min monotonicity)")
    ax.set_title("(B) Per-layer M2: baseline vs tuned")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, ls=":", alpha=0.5)

    # Panel C: SVD spectra of A_7 and A_8
    ax = axes[2]
    s7 = np.array(svd_info["L7"]["singular_values"])
    s8 = np.array(svd_info["L8"]["singular_values"])
    ax.plot(np.arange(1, len(s7) + 1), s7, label="A_7", color="C2")
    ax.plot(np.arange(1, len(s8) + 1), s8, label="A_8", color="C3")
    ax.axhline(1.0, ls=":", color="k", lw=0.8, alpha=0.5,
               label="identity SV=1")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("(C) SVD spectrum")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGS / f"E5_tuned_lens.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_train", type=int, default=4242)
    ap.add_argument("--seed_eval", type=int, default=42)
    ap.add_argument("--n_train_per_kind", type=int, default=100)
    ap.add_argument("--n_eval_per_kind", type=int, default=50)
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    t0 = time.time()

    log.info("loading model rev=%s", HF_REVISION[:8])
    bundle = load_hyenadna(device="cuda", dtype=torch.bfloat16)
    for p in bundle.model.parameters():
        p.requires_grad_(False)

    log.info("==== build training corpus seed=%d ====", args.seed_train)
    train_seqs = build_corpus(args.n_train_per_kind, LENGTH,
                              args.seed_train)
    log.info("train corpus n=%d", len(train_seqs))

    log.info("==== collect hidden states for training ====")
    cached, targets = collect_hidden(bundle, train_seqs, TARGET_LAYERS)

    log.info("==== init A_7, A_8 (identity) ====")
    A_dict = {ell: make_affine(HIDDEN, "cuda") for ell in TARGET_LAYERS}

    # Initial loss (epoch 0)
    init_losses = []
    with torch.no_grad():
        hw = bundle.lm_head.weight.detach().to("cuda")
        hb = bundle.lm_head.bias
        if hb is not None:
            hb = hb.detach().to("cuda")
        hd = bundle.lm_head.weight.dtype
        for i in range(len(targets)):
            l = _psl(A_dict, cached, targets, i, hw, hb, hd, EDGE)
            init_losses.append(float(l.item()))
    log.info("init mean loss=%.6f", float(np.mean(init_losses)))

    log.info("==== training tuned lens ====")
    losses = train_lens(A_dict, cached, targets, bundle.lm_head, EDGE,
                        n_epochs=args.n_epochs, bs=args.bs, lr=args.lr)

    # Final loss (after training)
    final_losses = []
    with torch.no_grad():
        for i in range(len(targets)):
            l = _psl(A_dict, cached, targets, i, hw, hb, hd, EDGE)
            final_losses.append(float(l.item()))
    log.info("final mean loss=%.6f", float(np.mean(final_losses)))

    # Free training cache
    del cached, targets
    torch.cuda.empty_cache()

    log.info("==== build eval corpus seed=%d ====", args.seed_eval)
    eval_seqs = build_corpus(args.n_eval_per_kind, LENGTH,
                             args.seed_eval)
    log.info("eval corpus n=%d", len(eval_seqs))

    log.info("==== eval baseline + tuned + UR ====")
    Db, Dt, Du = eval_M2(bundle, eval_seqs, A_dict, EDGE)

    per_b, glob_b = aggregate_M2(Db, EDGE)
    per_t, glob_t = aggregate_M2(Dt, EDGE)
    per_u, glob_u = aggregate_M2(Du, EDGE)

    log.info("BASELINE per-layer M2: %s", [f"{x:.4f}" for x in per_b])
    log.info("TUNED    per-layer M2: %s", [f"{x:.4f}" for x in per_t])
    log.info("UR       per-layer M2: %s", [f"{x:.4f}" for x in per_u])

    # SVD analysis
    svd_info = {}
    for ell, A in A_dict.items():
        W = A.weight.detach().cpu().numpy()
        b = A.bias.detach().cpu().numpy()
        s = np.linalg.svd(W, compute_uv=False)
        svd_info[f"L{ell}"] = {
            "singular_values": s.tolist(),
            "frobenius_norm_W": float(np.linalg.norm(W)),
            "frobenius_norm_W_minus_I": float(np.linalg.norm(W - np.eye(HIDDEN))),
            "max_sv": float(s.max()),
            "min_sv": float(s.min()),
            "median_sv": float(np.median(s)),
            "bias_norm": float(np.linalg.norm(b)),
        }
    log.info("SVD: A7 SV range [%.3f, %.3f]; A8 SV range [%.3f, %.3f]",
             svd_info["L7"]["min_sv"], svd_info["L7"]["max_sv"],
             svd_info["L8"]["min_sv"], svd_info["L8"]["max_sv"])

    # Save weights
    npz_path = RUNS / "E5_lens_weights.npz"
    np.savez(npz_path,
             A7_W=A_dict[7].weight.detach().cpu().numpy(),
             A7_b=A_dict[7].bias.detach().cpu().numpy(),
             A8_W=A_dict[8].weight.detach().cpu().numpy(),
             A8_b=A_dict[8].bias.detach().cpu().numpy())
    log.info("saved weights to %s", npz_path)

    # Save CSV
    csv_path = TABLES / "E5_tuned_lens.csv"
    with open(csv_path, "w") as f:
        f.write("layer,M2_baseline_jsd,M2_tuned_jsd,M2_ur\n")
        for ell in range(8):
            f.write(f"{ell+1},{per_b[ell]:.6f},{per_t[ell]:.6f},{per_u[ell]:.6f}\n")
    log.info("saved csv to %s", csv_path)

    # Compose JSON results
    runtime_s = time.time() - t0
    result = {
        "script": "E5_tuned_lens.py",
        "seed_train": args.seed_train,
        "seed_eval": args.seed_eval,
        "n_train": len(train_seqs),
        "n_eval": len(eval_seqs),
        "n_epochs": args.n_epochs,
        "batch_size": args.bs,
        "lr": args.lr,
        "edge_warmup_bp": EDGE,
        "length": LENGTH,
        "L": 8,
        "vocab_real": VOCAB_REAL,
        "hf_revision": HF_REVISION,
        "target_layers": TARGET_LAYERS,
        "init_loss_mean": float(np.mean(init_losses)),
        "final_loss_mean": float(np.mean(final_losses)),
        "training_loss_per_epoch": losses,
        "M2_baseline_per_layer_jsd": per_b,
        "M2_tuned_per_layer_jsd": per_t,
        "M2_ur_per_layer": per_u,
        "M2_global_baseline_jsd": glob_b,
        "M2_global_tuned_jsd": glob_t,
        "M2_global_ur": glob_u,
        "svd": svd_info,
        "runtime_s": runtime_s,
    }

    # Verdict
    m2L7_t = per_t[6]
    if m2L7_t >= 0.85:
        verdict = "CONFIRMED"
    elif m2L7_t >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"
    result["verdict_hypothesis_c_prime"] = verdict
    result["M2_L7_baseline"] = per_b[6]
    result["M2_L7_tuned"] = m2L7_t
    log.info("VERDICT: %s (M2_L7 baseline=%.4f tuned=%.4f)",
             verdict, per_b[6], m2L7_t)

    json_path = RUNS / "E5_tuned_lens.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("saved json to %s", json_path)

    # Figure (3 panels)
    make_figure(losses, per_b, per_t, per_u, svd_info)
    log.info("DONE total runtime=%.1f s", runtime_s)


if __name__ == "__main__":
    main()
