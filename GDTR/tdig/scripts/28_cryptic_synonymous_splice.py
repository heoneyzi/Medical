"""Exp 2.1 — Cryptic synonymous splice candidate detection.

Hypothesis: synonymous variants are typically "quiet" (low ΔH, deep peak at L=27
consistent with protein-semantic identity preservation). BUT some synonymous
variants may have splice-like ΔH profiles (sharp peak at L=8 like splice/intron
variants) — these are CANDIDATE CRYPTIC SPLICE variants.

Method:
  1. Compute Δh-shape signature per variant: argmax_layer + ratio L8/L27 + AUC
  2. Within synonymous, cluster by shape signature
  3. Identify the "splice-like" sub-cluster as cryptic splice candidates
  4. Report annotations + cross-check ClinVar significance

Output: results/cryptic_synonymous/
  synonymous_signature.csv      per variant signature features
  cryptic_candidates.csv        flagged variants with shape rationale
  signature_clusters.png        UMAP of synonymous Δh shapes colored by category
  shape_examples.png            ΔH curves: typical synonymous vs cryptic candidates
  summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scalars", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--consequence", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_per_consequence/variant_with_consequence.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/cryptic_synonymous"))
    p.add_argument("--n-cryptic-percentile", type=float, default=10.0,
                   help="Top N percentile by splice-like score = cryptic candidates")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print(f"[load] variants + consequence ...")
    df = pd.read_parquet(args.scalars)
    cons = pd.read_csv(args.consequence)[["chrom", "pos", "ref", "alt", "consequence"]]
    cons["chrom"] = cons.chrom.astype(str)
    df["chrom"] = df.chrom.astype(str)
    df = df.merge(cons, on=["chrom", "pos", "ref", "alt"], how="left")

    # Filter synonymous + splice + intron for comparison
    df_syn = df[df.consequence == "synonymous"].reset_index(drop=True)
    df_spl = df[df.consequence == "splice"].reset_index(drop=True)  # may be empty by labeling
    df_int = df[df.consequence == "intron"].reset_index(drop=True)
    print(f"  synonymous n={len(df_syn)}, splice n={len(df_spl)}, intron n={len(df_int)}")

    # Build shape signature per variant: argmax_layer, mean L0-L4, mean L5-L10,
    # mean L11-L20, mean L21-L29, mean L30-L31, ratio L8/L27, integral
    def shape_features(df_in):
        if len(df_in) == 0:
            return pd.DataFrame()
        X = np.asarray(df_in.delta_h_norm_2.tolist(), dtype=np.float32)  # (N, 32)
        ag = X.argmax(axis=1)
        sig = pd.DataFrame({
            "argmax_layer": ag,
            "max_dh": X.max(axis=1),
            "mean_L0_4": X[:, 0:5].mean(axis=1),
            "mean_L5_10": X[:, 5:11].mean(axis=1),
            "mean_L11_20": X[:, 11:21].mean(axis=1),
            "mean_L21_28": X[:, 21:29].mean(axis=1),
            "mean_L29_31": X[:, 29:32].mean(axis=1),
            "ratio_L8_over_L27": X[:, 8] / (X[:, 27] + 1e-9),
            "integral_L0_29": X[:, :29].sum(axis=1),
        })
        return pd.concat([df_in[["chrom", "pos", "ref", "alt", "gene", "category",
                                   "stars", "consequence"]].reset_index(drop=True),
                          sig.reset_index(drop=True)], axis=1)

    sig_syn = shape_features(df_syn)
    sig_int = shape_features(df_int)
    print(f"[sig] synonymous shape table {sig_syn.shape}")

    # Reference distribution of intron shapes (the "L8-peakers")
    intron_l8_over_l27 = sig_int["ratio_L8_over_L27"]
    intron_argmax = sig_int["argmax_layer"]
    print(f"[ref] intron ratio_L8/L27 median={intron_l8_over_l27.median():.3f}, "
          f"argmax most common = L{int(intron_argmax.mode().iloc[0])}")

    # "Splice-like" score for each synonymous variant
    # High = looks like intron/splice (L8 peak, large L0-10 mean) vs typical synonymous (L27 peak)
    syn_score = (sig_syn.mean_L5_10 / (sig_syn.mean_L21_28 + 1e-9))
    sig_syn["splice_like_score"] = syn_score

    # Cryptic candidates = top n_cryptic_percentile% of splice_like_score
    threshold = float(np.percentile(syn_score, 100 - args.n_cryptic_percentile))
    sig_syn["is_cryptic_candidate"] = (syn_score >= threshold).astype(int)
    n_cand = int(sig_syn.is_cryptic_candidate.sum())
    print(f"[detect] threshold={threshold:.3f} ⇒ {n_cand} candidate cryptic splice (top {args.n_cryptic_percentile}%)")

    # Sanity: are candidates enriched for P_LP?
    cand = sig_syn[sig_syn.is_cryptic_candidate == 1]
    non_cand = sig_syn[sig_syn.is_cryptic_candidate == 0]
    enrich_PLP = float((cand.category == "P_LP").mean() / max((non_cand.category == "P_LP").mean(), 1e-6))
    print(f"  P_LP rate in candidates: {(cand.category == 'P_LP').mean()*100:.1f}%")
    print(f"  P_LP rate in non-candidates: {(non_cand.category == 'P_LP').mean()*100:.1f}%")
    print(f"  enrichment fold = {enrich_PLP:.2f}")

    # Save
    sig_syn.to_csv(args.out_dir / "synonymous_signature.csv", index=False)
    cand.to_csv(args.out_dir / "cryptic_candidates.csv", index=False)
    print(f"[save] synonymous_signature.csv + cryptic_candidates.csv")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Histogram of splice_like_score with intron reference
    ax = axes[0]
    bins = np.linspace(0, max(syn_score.max(), intron_l8_over_l27.max()), 50)
    ax.hist(sig_int.mean_L5_10 / (sig_int.mean_L21_28 + 1e-9), bins=bins, alpha=0.5,
             color="#1f77b4", density=True, label=f"intron reference (n={len(sig_int)})")
    ax.hist(syn_score, bins=bins, alpha=0.5,
             color="#9467bd", density=True, label=f"synonymous (n={len(sig_syn)})")
    ax.axvline(threshold, color="red", lw=1.5, ls="--",
                label=f"cryptic threshold (top {args.n_cryptic_percentile}%)")
    ax.set_xlabel("Splice-like score = mean(ΔH L5-L10) / mean(ΔH L21-L28)")
    ax.set_ylabel("Density")
    ax.set_title("Splice-like score distribution\nSynonymous variants with high score ⇒ candidate cryptic splice")
    ax.legend()

    # Example ΔH curves
    ax = axes[1]
    # 10 random typical synonymous (low score)
    typical = sig_syn[sig_syn.splice_like_score < np.percentile(syn_score, 30)].sample(min(10, len(sig_syn)//10), random_state=args.seed)
    typical_dh = np.asarray(df_syn.loc[typical.index, "delta_h_norm_2"].tolist())
    for row in typical_dh:
        ax.plot(range(32), row, color="#9467bd", alpha=0.3, lw=0.8)
    if len(typical_dh) > 0:
        ax.plot(range(32), typical_dh.mean(0), color="#9467bd", lw=2, label="typical synonymous (mean)")

    # 10 cryptic candidates (high score)
    crypt = cand.sample(min(10, len(cand)), random_state=args.seed)
    crypt_dh = np.asarray(df_syn.loc[crypt.index, "delta_h_norm_2"].tolist())
    for row in crypt_dh:
        ax.plot(range(32), row, color="#d62728", alpha=0.3, lw=0.8)
    if len(crypt_dh) > 0:
        ax.plot(range(32), crypt_dh.mean(0), color="#d62728", lw=2, label=f"cryptic candidates (mean, n={len(cand)})")

    # Intron reference mean
    int_dh = np.asarray(df_int.delta_h_norm_2.tolist())
    if len(int_dh) > 0:
        ax.plot(range(32), int_dh.mean(0), color="#1f77b4", lw=2, ls="--",
                 label=f"intron reference (mean, n={len(df_int)})")
    ax.axvline(29, color="k", ls=":", lw=0.5)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("ΔH norm L2")
    ax.set_title("ΔH layer profile shapes — typical synonymous vs cryptic candidates")
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"shape_examples.{ext}", dpi=150)
    plt.close(fig)
    print(f"[plot] shape_examples saved")

    summary = {
        "n_synonymous_total": len(df_syn),
        "n_cryptic_candidates": n_cand,
        "candidate_threshold_score": float(threshold),
        "P_LP_rate_in_candidates": float((cand.category == "P_LP").mean() * 100),
        "P_LP_rate_in_non_candidates": float((non_cand.category == "P_LP").mean() * 100),
        "P_LP_enrichment_fold": float(enrich_PLP),
        "n_candidate_PLP": int((cand.category == "P_LP").sum()),
        "n_candidate_BLB": int((cand.category == "B_LB").sum()),
        "n_candidate_VUS": int((cand.category == "VUS").sum()),
        "top_candidate_genes": cand.gene.value_counts().head(10).to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
