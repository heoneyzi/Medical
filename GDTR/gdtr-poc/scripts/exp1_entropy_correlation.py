"""Exp 1 (v10) — entropy correlation Spearman ρ(c, H_t) on chr22.

Question: does the per-position settling depth c(t) merely track the model's
per-position next-token prediction entropy H_t, or does it carry independent
biological information beyond entropy?

Method:
  - Sample N_WIN random chr22 windows (6 kb each) from the phase1.6 catalogue.
  - Forward each window through Evo 2 7B and capture all residual-stream
    taps + the post-final-norm tap.
  - Per token position:
      • c(t) = first ℓ at which run-min D_cos(ℓ, t) ≤ γ_cos = 0.39663.
      • H_t  = Shannon entropy of softmax(unembed(h_norm_t)) over the
        VOCAB_SIZE = 512 head (effectively a 4-symbol biological mass).
  - Stratify positions by chr22 context (chr22_position_labels.npy
    encoding: 0 intergenic / 1 intron / 2 exon / 3 5utr / 4 3utr /
    5 splice_donor / 6 splice_acceptor — verified 2026-05-04).
  - Report:
      • Spearman ρ(c, H_t) overall + per-context.
      • Cohen's d (splice-donor vs intron) on raw c.
      • Cohen's d on residual c after regressing out H_t (linear OLS).

Output:
  results/exp1_entropy/exp1_entropy_meta.json
  results/exp1_entropy/per_position.npz
  results/exp1_entropy/_status.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
GAMMA_COS = 0.39663
SEED = 42
PHASE = "exp1_entropy"
OUT_DIR = GDTR_ROOT / "results" / PHASE
WIN_TSV = GDTR_ROOT / "data" / "baselines" / "chr22_windows.tsv"
LABELS_NPY = GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"

LOG = logging.getLogger(PHASE)


def setup_log():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def write_status(state, extra=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"step": PHASE, "status": state}
    if extra:
        payload.update(extra)
    (OUT_DIR / "_status.json").write_text(
        json.dumps(payload, indent=2, default=str))


def cohen_d(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((len(x) - 1) * sx + (len(y) - 1) * sy)
                 / (len(x) + len(y) - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def main():
    setup_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_windows = 5

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    write_status("RUNNING", {"n_windows": args.n_windows})

    # --- read window catalogue ---
    rows = []
    with WIN_TSV.open() as f:
        header = f.readline().rstrip().split("\t")
        for ln in f:
            rows.append(dict(zip(header, ln.rstrip().split("\t"))))
    win_idx = rng.choice(len(rows), size=min(args.n_windows, len(rows)),
                          replace=False)
    LOG.info("subsampled %d/%d chr22 windows", len(win_idx), len(rows))

    # --- model + data ---
    sys.path.insert(0, str(GDTR_ROOT))
    import _runner_utils as ru  # noqa: F401
    from src.constants_evo2 import N_LAYERS, BOS_OFFSET, VOCAB_SIZE
    from src.model_loader_evo2 import load_evo2, tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names
    from src.ur_gdtr_evo2 import cosine_lens
    import pysam

    bundle = load_evo2()
    LOG.info("evo2 loaded: %s", bundle.loaded_variant)
    fasta = pysam.FastaFile(str(GDTR_ROOT / "data" / "reference" / "chr22.fa"))
    labels = np.load(LABELS_NPY) if LABELS_NPY.exists() else None
    if labels is None:
        LOG.warning("chr22 labels missing — per-context split skipped")

    layer_names = all_layer_names()  # blocks.0..31 + norm
    all_c, all_H, all_lab, all_pos = [], [], [], []

    for j, ix in enumerate(win_idx):
        w = rows[int(ix)]
        chrom = w["chrom"]
        start = int(w["start"]); end = int(w["end"])
        seq = fasta.fetch(chrom, start, end).upper()
        T_seq = len(seq)
        if T_seq < 100:
            continue
        toks = tokenize(seq, bundle, device="cuda")

        with torch.no_grad():
            hs = extract_hidden_states(bundle, toks, save_layers=layer_names)
            D_cos = cosine_lens(hs, n_layers=N_LAYERS, bos_offset=BOS_OFFSET)
            cd_np = D_cos.float().cpu().numpy()  # [L, T_real]
            run_min = np.minimum.accumulate(cd_np, axis=0)
            settled = run_min <= GAMMA_COS
            argfirst = np.argmax(settled, axis=0)
            never = ~settled.any(axis=0)
            c_per_pos = (argfirst + 1).astype(np.int32)
            c_per_pos[never] = N_LAYERS + 1

            # entropy from h_norm directly (don't re-norm)
            h_norm = hs["norm"]  # [B, T, H], already post-RMSNorm
            h_cast = h_norm.to(bundle.embedding_weight.dtype)
            logits = bundle.unembed(h_cast)[..., :VOCAB_SIZE].squeeze(0)
            probs = torch.softmax(logits.float(), dim=-1)
            H = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            H_np = H.cpu().numpy()
        # Free GPU memory
        del hs
        torch.cuda.empty_cache()

        T = min(c_per_pos.size, H_np.size, T_seq)
        pos_arr = np.arange(start, start + T, dtype=np.int64)
        all_c.append(c_per_pos[:T])
        all_H.append(H_np[:T])
        all_pos.append(pos_arr)
        if labels is not None:
            all_lab.append(labels[start:start + T])

        if (j + 1) % 20 == 0 or j == 0:
            LOG.info("[%d/%d]  %s:%d-%d  T=%d  ρ_partial=%.3f  elapsed %.1fs",
                     j + 1, len(win_idx), chrom, start, end, T,
                     float(np.corrcoef(c_per_pos[:T], H_np[:T])[0, 1]),
                     time.time() - t0)

    c_arr = np.concatenate(all_c)
    H_arr = np.concatenate(all_H)
    pos_arr = np.concatenate(all_pos)
    lab_arr = np.concatenate(all_lab) if all_lab else None
    LOG.info("collected %d positions across %d windows",
             c_arr.size, len(win_idx))

    from scipy import stats as st
    rho_overall, p_overall = st.spearmanr(c_arr, H_arr)
    LOG.info("overall Spearman ρ(c,H) = %.4f  (p=%.3g, n=%d)",
             rho_overall, p_overall, c_arr.size)

    per_ctx = {}
    if lab_arr is not None:
        code_to_name = {0: "intergenic", 1: "intron", 2: "coding_exon",
                        3: "5utr", 4: "3utr",
                        5: "splice_donor", 6: "splice_acceptor"}
        for code, name in code_to_name.items():
            m = lab_arr == code
            if m.sum() < 30:
                continue
            rho, p = st.spearmanr(c_arr[m], H_arr[m])
            per_ctx[name] = {"rho": float(rho), "p": float(p),
                              "n": int(m.sum()),
                              "mean_c": float(c_arr[m].mean()),
                              "mean_H": float(H_arr[m].mean())}

    # partial: regress c on H, recompute splice-vs-intron Cohen's d on residual
    splice_d_raw = float("nan"); splice_d_residual = float("nan")
    if lab_arr is not None:
        sm = lab_arr == 5
        im = lab_arr == 1
        if sm.sum() > 50 and im.sum() > 50:
            splice_d_raw = cohen_d(c_arr[sm], c_arr[im])
            X = np.vstack([H_arr, np.ones_like(H_arr)]).T
            beta, *_ = np.linalg.lstsq(X, c_arr.astype(np.float64),
                                        rcond=None)
            c_resid = c_arr.astype(np.float64) - X @ beta
            splice_d_residual = cohen_d(c_resid[sm], c_resid[im])

    np.savez_compressed(OUT_DIR / "per_position.npz",
                        c=c_arr.astype(np.int16),
                        H=H_arr.astype(np.float32),
                        pos=pos_arr.astype(np.int64),
                        labels=(lab_arr.astype(np.int8)
                                if lab_arr is not None
                                else np.zeros(0, dtype=np.int8)))
    meta = {"n_windows": len(win_idx),
            "n_positions": int(c_arr.size),
            "rho_overall": float(rho_overall),
            "p_overall": float(p_overall),
            "per_context": per_ctx,
            "splice_donor_vs_intron_d_raw": splice_d_raw,
            "splice_donor_vs_intron_d_residual_after_H": splice_d_residual,
            "wall_sec": time.time() - t0,
            "gamma_cos": GAMMA_COS}
    (OUT_DIR / "exp1_entropy_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    write_status("PASS", meta)
    LOG.info("DONE in %.1fs — see %s", time.time() - t0,
             OUT_DIR / "exp1_entropy_meta.json")


if __name__ == "__main__":
    main()
