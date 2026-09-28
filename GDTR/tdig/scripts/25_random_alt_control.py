"""Phase 1.4 — GPU biological validation: random-allele control for ClinVar variants.

For each of N=300 sampled P_LP + 300 B_LB variants, generate K=3 random ALT alleles
(different from ref, different from ClinVar's reported alt). Re-forward Evo 2 7B
on (ref, random_alt) and compute ΔH norm per layer.

Question answered:
  Does ΔH norm at layer L=8 distinguish *specific pathogenic mutations* from
  *any other random mutation at the same position*?

  If yes → ΔH norm captures variant-specific biology, not just position-effect
  If no → ΔH norm is mostly position-sensitivity (paper concession needed)

GPU: requires Evo 2 7B. ~1800 forwards × ~0.4s = ~12 min. Reuses gDTR
load_evo2 + forward infrastructure.

Output: results/random_alt_control/
  random_alt_delta_h.parquet     per (variant_idx, k, layer) ΔH norm
  comparison_summary.csv         real_dH vs random_dH per layer per category
  comparison.png                 distribution plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/gDTR")

import torch
import torch.nn.functional as F


def load_fasta_dir(path):
    """Per-chrom pysam FastaFile cache (matches 18_variant_forward.py setup)."""
    import pysam
    fasta_cache = {}
    fasta_dir = Path(path)

    def get_fasta(chrom):
        chrom_str = str(chrom)
        if not chrom_str.startswith("chr"):
            chrom_str = "chr" + chrom_str
        if chrom_str not in fasta_cache:
            fa_path = fasta_dir / f"{chrom_str}.fa"
            if not fa_path.exists():
                fa_path = fasta_dir / f"{chrom}.fa"
            fasta_cache[chrom_str] = pysam.FastaFile(str(fa_path))
        return fasta_cache[chrom_str], chrom_str

    return get_fasta


def build_seq(get_fasta, chrom, pos, ref_allele, alt_allele, ctx_kb=3):
    fa, chrom_str = get_fasta(chrom)
    start = max(0, pos - 1 - ctx_kb * 1000)
    end = pos - 1 + len(ref_allele) + ctx_kb * 1000
    seq = fa.fetch(chrom_str, start, end).upper()
    var_idx = pos - 1 - start
    if seq[var_idx:var_idx + len(ref_allele)] != ref_allele:
        return None, None  # ref mismatch — skip
    alt_seq = seq[:var_idx] + alt_allele + seq[var_idx + len(ref_allele):]
    return seq, alt_seq


def forward_and_delta(bundle, seq_ref, seq_alt, var_idx_ref):
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    layer_names = all_layer_names()
    ref_ids = tokenize(seq_ref, bundle, device="cuda")
    alt_ids = tokenize(seq_alt, bundle, device="cuda")
    hs_r = extract_hidden_states(bundle, ref_ids, save_layers=layer_names)
    hs_a = extract_hidden_states(bundle, alt_ids, save_layers=layer_names)
    L = 32
    h_r = torch.stack([hs_r[f"blocks.{ell}"][0, var_idx_ref] for ell in range(L)]).float()
    h_a = torch.stack([hs_a[f"blocks.{ell}"][0, var_idx_ref] for ell in range(L)]).float()
    delta = h_a - h_r
    dh_norm = torch.linalg.vector_norm(delta, dim=-1).cpu().numpy().astype(np.float32)
    del hs_r, hs_a, ref_ids, alt_ids
    torch.cuda.empty_cache()
    return dh_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--fasta-dir", type=Path,
                   default=Path("/path/to/gDTR/data/reference"))
    p.add_argument("--n-per-class", type=int, default=300,
                   help="P_LP + B_LB samples each")
    p.add_argument("--k-controls", type=int, default=3,
                   help="random ALT alleles per variant")
    p.add_argument("--ctx-kb", type=int, default=3)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/random_alt_control"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] variants ...")
    df = pd.read_parquet(args.variants)
    df_bin = df[df.category.isin({"P_LP", "B_LB"})]
    rng = np.random.default_rng(args.seed)

    # Sample
    p_lp = df_bin[df_bin.category == "P_LP"].sample(n=min(args.n_per_class, (df_bin.category == "P_LP").sum()),
                                                       random_state=args.seed)
    b_lb = df_bin[df_bin.category == "B_LB"].sample(n=min(args.n_per_class, (df_bin.category == "B_LB").sum()),
                                                       random_state=args.seed)
    sample_df = pd.concat([p_lp, b_lb], ignore_index=True)
    print(f"[load] sampled {len(p_lp)} P_LP + {len(b_lb)} B_LB")

    print(f"[load] Evo 2 ...")
    from src.model_loader_evo2 import load_evo2
    bundle = load_evo2()

    print(f"[load] reference fasta dir ...")
    get_fasta = load_fasta_dir(args.fasta_dir)

    bases = ["A", "C", "G", "T"]
    records = []
    t0 = time.time()
    n_done = 0
    n_total = len(sample_df) * (1 + args.k_controls)

    for i, row in sample_df.iterrows():
        chrom = row.chrom; pos = int(row.pos)
        ref = row.ref.upper(); real_alt = row.alt.upper()
        # Build real ref/alt
        if len(ref) != 1 or len(real_alt) != 1:
            continue  # only SNVs for simplicity
        # Real variant ΔH
        try:
            seq_ref, seq_alt = build_seq(get_fasta, chrom, pos, ref, real_alt, ctx_kb=args.ctx_kb)
            if seq_ref is None:
                continue
            var_idx = pos - 1 - max(0, pos - 1 - args.ctx_kb * 1000)
            with torch.no_grad():
                dh_real = forward_and_delta(bundle, seq_ref, seq_alt, var_idx)
            for ell in range(32):
                records.append({"variant_idx": int(i), "type": "real",
                                  "k": 0, "alt": real_alt,
                                  "layer": ell, "dh_norm": float(dh_real[ell]),
                                  "category": row.category, "gene": row.gene})
            n_done += 1
            # K random ALT controls (different from ref AND from real_alt)
            others = [b for b in bases if b != ref and b != real_alt]
            if len(others) == 0:
                continue
            for k in range(args.k_controls):
                rand_alt = rng.choice(others)
                _, seq_rand_alt = build_seq(get_fasta, chrom, pos, ref, rand_alt, ctx_kb=args.ctx_kb)
                if seq_rand_alt is None:
                    continue
                with torch.no_grad():
                    dh_rand = forward_and_delta(bundle, seq_ref, seq_rand_alt, var_idx)
                for ell in range(32):
                    records.append({"variant_idx": int(i), "type": "random",
                                      "k": k + 1, "alt": rand_alt,
                                      "layer": ell, "dh_norm": float(dh_rand[ell]),
                                      "category": row.category, "gene": row.gene})
                n_done += 1
        except Exception as e:
            print(f"  variant {i} error: {e}")
            continue
        if n_done > 0 and n_done % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{n_total}] rate={rate:.2f}/s ETA={eta/60:.1f}min")

    out_df = pd.DataFrame(records)
    out_df.to_parquet(args.out_dir / "random_alt_delta_h.parquet")
    print(f"\n[save] random_alt_delta_h.parquet ({len(out_df):,} rows)")

    # Comparison summary
    summary_records = []
    for cat in ("P_LP", "B_LB"):
        for ell in range(32):
            real = out_df[(out_df.category == cat) & (out_df.type == "real")
                            & (out_df.layer == ell)].dh_norm.values
            rand = out_df[(out_df.category == cat) & (out_df.type == "random")
                            & (out_df.layer == ell)].dh_norm.values
            if len(real) < 10 or len(rand) < 10:
                continue
            from scipy.stats import mannwhitneyu
            _, p_value = mannwhitneyu(real, rand, alternative="two-sided")
            summary_records.append({
                "category": cat, "layer": ell,
                "real_mean": float(real.mean()), "rand_mean": float(rand.mean()),
                "ratio": float(real.mean() / rand.mean()) if rand.mean() > 1e-9 else np.nan,
                "p_mannwhitney": float(p_value),
                "n_real": int(len(real)), "n_rand": int(len(rand)),
            })
    sdf = pd.DataFrame(summary_records)
    sdf.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    print(f"[save] comparison_summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cat in zip(axes, ("P_LP", "B_LB")):
        sub = sdf[sdf.category == cat]
        ax.plot(sub.layer, sub.real_mean, color="#d62728", marker="o", ms=4, lw=1.6, label="real variant ΔH")
        ax.plot(sub.layer, sub.rand_mean, color="#7f7f7f", marker="s", ms=4, lw=1.4, label="random ALT ΔH")
        ax.axvline(29, color="k", ls="--", lw=0.5, alpha=0.4)
        ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean ΔH norm")
        ax.set_title(f"{cat}: real vs random ALT (n_real≈{int(sub.n_real.median())}, n_rand≈{int(sub.n_rand.median())})")
        ax.legend(loc="best")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"comparison.{ext}", dpi=150)
    plt.close(fig)
    print("[plot] saved")
    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()
