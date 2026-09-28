"""Exp H7 — HyenaDNA-medium chr22 forward + TDiG settling analysis.

Tests whether bidirectional settling + L29-style phase transitions are
Evo 2-specific or general across genomic foundation models.

HyenaDNA-medium-160k has 8 layers (vs Evo 2's 32). We compute the same
M3_geo curvature settling on its hidden states and check:
  1. Does splice_donor settle earlier than intron in M3_geo? (sign agreement)
  2. Is there a "last-layer rotation" pattern?

GPU: HyenaDNA-medium is ~28M params, ~1 GB. Per-sequence forward ~0.1s.
50 chr22 windows × 6kb = 50 forwards → ~30s total.

Outputs: results/hyenadna_crossarch/
  hyenadna_tier1.parquet         per-token settling cells (M3_geo only)
  hyenadna_splice_vs_intron.csv  same as chr22 splice analysis
  comparison_vs_evo2.json        side-by-side d
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/path/to/gDTR")
import torch


def load_hyenadna_medium():
    from transformers import AutoModel, AutoTokenizer
    model_name = "LongSafari/hyenadna-medium-160k-seqlen-hf"
    print(f"[load] {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                       torch_dtype=torch.float32).cuda().eval()
    return model, tokenizer


def forward_get_hidden(model, tokenizer, seq):
    """Returns h_ell tensor (L, T, H). Uses output_hidden_states."""
    ids = tokenizer(seq, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    # out.hidden_states is tuple of (n_layers+1) tensors each (1, T, H)
    h = torch.stack([h_l[0] for h_l in out.hidden_states]).float()  # (L+1, T, H)
    return h


def compute_m3_curvature_settling(h_all, gamma_q70_ref=None):
    """h_all (L, T, H). Returns per-token settling layer (T,) int.
       M3_geo curvature only: kappa[ell] = 1 - cos(step_{ell+1}, step_ell).
       Settle at first ell where kappa <= gamma_q70 for W=3 consecutive layers.
       If gamma not provided, use q70 of kappa over all tokens at L-2."""
    L, T, H = h_all.shape
    step = h_all[1:] - h_all[:-1]  # (L-1, T, H)
    step_n = torch.linalg.vector_norm(step, dim=-1)
    cos_pp = torch.nn.functional.cosine_similarity(step[1:], step[:-1], dim=-1)
    kappa = (1.0 - cos_pp).cpu().numpy()  # (L-2, T)
    if gamma_q70_ref is None:
        gamma_q70_ref = float(np.quantile(kappa.flatten(), 0.70))
    # Persistence W=3 settling
    L_kappa, T_k = kappa.shape
    max_layer = L_kappa - 1
    below = kappa <= gamma_q70_ref
    rolling = np.zeros((L_kappa, T_k), dtype=bool)
    W = 3
    for ell in range(L_kappa):
        end_k = min(ell + W - 1, max_layer)
        rolling[ell] = below[ell:end_k + 1].all(axis=0)
    c = np.full(T_k, -1, dtype=np.int32)
    any_set = rolling.any(axis=0)
    c[any_set] = np.argmax(rolling, axis=0)[any_set]
    return c, gamma_q70_ref


def build_context_map(meta_df, wid, pos_labels, n_tokens):
    row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
    start = int(row["start"])
    return pos_labels[np.clip(start + np.arange(n_tokens), 0, len(pos_labels) - 1)]


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--fasta-dir", type=Path,
                   default=Path("/path/to/gDTR/data/reference"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/hyenadna_crossarch"))
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] meta + pos_labels + fasta ...")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    import pysam
    fa = pysam.FastaFile(str(args.fasta_dir / "chr22.fa"))

    # Sample N windows
    rng = np.random.default_rng(args.seed)
    sample_meta = meta.sample(n=min(args.n_windows, len(meta)), random_state=args.seed).reset_index(drop=True)
    print(f"  sampling {len(sample_meta)} windows")

    model, tok = load_hyenadna_medium()
    print(f"[fwd] forwarding {len(sample_meta)} sequences ...")

    # Calibration pass: collect kappa from first 20 windows to set gamma
    cal_kappa = []
    cells_records = []
    t0 = time.time()
    gamma_q70 = None
    for i, row in sample_meta.iterrows():
        try:
            seq = fa.fetch("chr22", int(row.start), int(row.end)).upper()
            seq = seq.replace("N", "A")  # simple N replacement
            h_all = forward_get_hidden(model, tok, seq)
            T = h_all.shape[1]
            if i < 20:
                # collect kappa for calibration
                step = h_all[1:] - h_all[:-1]
                cos_pp = torch.nn.functional.cosine_similarity(step[1:], step[:-1], dim=-1)
                cal_kappa.append((1.0 - cos_pp).cpu().numpy().flatten())
                if i == 19:
                    gamma_q70 = float(np.quantile(np.concatenate(cal_kappa), 0.70))
                    print(f"  [calibrate] gamma_q70 = {gamma_q70:.4f} after 20 windows")
                continue
            c, _ = compute_m3_curvature_settling(h_all, gamma_q70_ref=gamma_q70)
            ctx = build_context_map(sample_meta, int(row.window_idx), pos_labels, T - 2)
            for k in range(len(c)):
                cells_records.append({"window_idx": int(row.window_idx), "tok_idx": k,
                                        "ctx_id": int(ctx[k]), "settling": int(c[k])})
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(sample_meta)}] elapsed={time.time()-t0:.1f}s")
            del h_all
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  window {i} error: {e}")
            continue

    cells_df = pd.DataFrame(cells_records)
    cells_df.to_parquet(args.out_dir / "hyenadna_tier1.parquet")
    print(f"[save] hyenadna_tier1.parquet ({len(cells_df):,} rows)")

    # Splice vs intron
    donor = cells_df[cells_df.ctx_id == 5].settling.values
    intron = cells_df[cells_df.ctx_id == 1].settling.values
    donor = donor[(donor >= 0) & (donor < 33)]
    intron = intron[(intron >= 0) & (intron < 33)]
    d = cohens_d(donor, intron)
    print(f"\nHyenaDNA splice_donor vs intron M3_geo curvature:")
    print(f"  donor n={len(donor)}, mean={donor.mean():.2f}")
    print(f"  intron n={len(intron)}, mean={intron.mean():.2f}")
    print(f"  Cohen d = {d:+.3f}  (Evo 2 reference: -0.81 for M3_geo_a0.5_b1.0)")

    summary = {
        "model": "HyenaDNA-medium-160k",
        "n_layers": int(h_all.shape[0]) if 'h_all' in locals() else None,
        "gamma_q70_curvature": gamma_q70,
        "donor_n": int(len(donor)), "intron_n": int(len(intron)),
        "donor_mean": float(donor.mean()), "intron_mean": float(intron.mean()),
        "cohen_d_donor_minus_intron": float(d),
        "evo2_reference_d_M3_geo_a0.5_b1.0": -0.81,
        "sign_agreement_with_evo2": bool(d < 0),
        "ratio_magnitude_vs_evo2": float(abs(d) / 0.81),
    }
    (args.out_dir / "comparison_vs_evo2.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(args.out_dir / "hyenadna_splice_vs_intron.csv", index=False)
    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
