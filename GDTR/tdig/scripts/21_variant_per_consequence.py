"""Phase 1.1b — Variant per-consequence analysis.

Joins ClinVar VCF molecular consequence (MC field) to our variant_scalars
and computes per-consequence AUROC + Δh distribution per layer.

Mirrors gDTR Paper 1 §3.3 (consequence ordering) but uses our per-layer Δh.

Output under --out-dir (default: results/variant_per_consequence/):
  variant_with_consequence.csv       per-variant table with MC joined
  per_consequence_auroc.csv          (consequence, layer, AUROC, n)
  per_consequence_auroc.png          curves per consequence class
  per_consequence_kw.csv             Kruskal-Wallis at each layer (consequence ordering)
  per_consequence_delta_distribution.png  Δh distribution by consequence per layer
  summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32

# Sequence Ontology consequence categories — collapse to broad classes
CONSEQUENCE_MAP = {
    "intron_variant": "intron",
    "missense_variant": "missense",
    "synonymous_variant": "synonymous",
    "frameshift_variant": "frameshift",
    "stop_gained": "nonsense",
    "stop_lost": "nonsense",
    "splice_donor_variant": "splice",
    "splice_acceptor_variant": "splice",
    "splice_region_variant": "splice",
    "5_prime_UTR_variant": "5utr",
    "3_prime_UTR_variant": "3utr",
    "inframe_insertion": "inframe_indel",
    "inframe_deletion": "inframe_indel",
    "initiator_codon_variant": "start_loss",
    "non-coding_transcript_variant": "noncoding",
    "genic_upstream_transcript_variant": "upstream",
    "genic_downstream_transcript_variant": "downstream",
    "no_sequence_alteration": "no_alt",
}

CONSEQUENCE_ORDER = ["intron", "synonymous", "missense", "splice",
                      "nonsense", "frameshift", "inframe_indel",
                      "5utr", "3utr", "start_loss", "noncoding"]

CONSEQUENCE_COLORS = {
    "intron": "#1f77b4", "synonymous": "#9467bd", "missense": "#ff7f0e",
    "splice": "#d62728", "nonsense": "#8c564b", "frameshift": "#e377c2",
    "inframe_indel": "#17becf", "5utr": "#bcbd22", "3utr": "#7f7f7f",
    "start_loss": "#2ca02c", "noncoding": "#aec7e8",
}


def parse_clinvar_mc(vcf_path: Path, key_set: set):
    """Stream VCF, extract MC for variants in our key_set {(chrom, pos, ref, alt)}.

    Returns {key: consequence_label}.
    """
    mc_pat = re.compile(r"MC=([^;]+)")
    so_pat = re.compile(r"SO:\d+\|([a-zA-Z0-9_]+)")
    out = {}
    found = 0
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, _, ref, alt, *_ = cols
            # VCF chrom might have "chr" prefix or not — match our parquet (no prefix)
            chrom_clean = chrom.replace("chr", "")
            key = (chrom_clean, int(pos), ref, alt)
            if key not in key_set:
                continue
            info = cols[7]
            m = mc_pat.search(info)
            if not m:
                out[key] = "no_MC"
                continue
            mc_field = m.group(1)
            # Multiple consequences separated by ',' — take first SO term
            so_terms = [s.split("|", 1)[-1] for s in mc_field.split(",")]
            # Pick most severe consequence per SO ordering (simple: first match)
            label = "other"
            for term in so_terms:
                if term in CONSEQUENCE_MAP:
                    label = CONSEQUENCE_MAP[term]
                    break
            else:
                if so_terms:
                    label = so_terms[0]
            out[key] = label
            found += 1
            if found == len(key_set):
                break
    return out


def list_to_array(df_col):
    return np.asarray(df_col.tolist(), dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--clinvar-vcf", type=Path,
                   default=Path("/path/to/gDTR/data/variants/clinvar.vcf.gz"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_per_consequence"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print(f"[load] {args.variants}")
    df = pd.read_parquet(args.variants)
    print(f"[load]   {len(df):,} variants")

    # Build key set
    df["chrom_str"] = df["chrom"].astype(str)
    keys = set(zip(df.chrom_str, df.pos.astype(int), df.ref, df.alt))
    print(f"[load] {len(keys):,} unique keys to join")

    print(f"[VCF] parsing {args.clinvar_vcf} for MC field ...")
    mc_map = parse_clinvar_mc(args.clinvar_vcf, keys)
    print(f"[VCF] matched {len(mc_map):,} / {len(keys):,} variants")

    df["consequence"] = df.apply(
        lambda r: mc_map.get((r["chrom_str"], int(r["pos"]), r["ref"], r["alt"]), "unmatched"),
        axis=1,
    )
    print(f"[join] consequence distribution:")
    print(df.consequence.value_counts().head(20).to_string())

    # Save annotated table (keep only essential cols + delta features)
    keep_cols = ["chrom", "pos", "ref", "alt", "gene", "category", "stars", "consequence"]
    df[keep_cols].to_csv(args.out_dir / "variant_with_consequence.csv", index=False)
    print(f"[save] variant_with_consequence.csv")

    # Per-consequence AUROC (P_LP vs B_LB)
    df_bin = df[df.category.isin({"P_LP", "B_LB"})].copy()
    df_bin["y"] = (df_bin.category == "P_LP").astype(int)
    print(f"\n[A] per-consequence AUROC, n_PLP={int(df_bin.y.sum())}, n_BLB={int((1-df_bin.y).sum())}")

    auroc_records = []
    delta_records = []
    for cons in CONSEQUENCE_ORDER:
        sub = df_bin[df_bin.consequence == cons]
        n_pos, n_neg = int(sub.y.sum()), int((1 - sub.y).sum())
        if n_pos < 5 or n_neg < 5:
            print(f"  skip {cons}: n_PLP={n_pos}, n_BLB={n_neg}")
            continue
        X = list_to_array(sub.delta_h_norm_2)
        for ell in range(N_LAYERS):
            a = roc_auc_score(sub.y.values, X[:, ell])
            auroc_records.append({
                "consequence": cons, "layer": ell, "AUROC": float(a),
                "n_pos": n_pos, "n_neg": n_neg,
            })
        best_l = int(np.argmax([r["AUROC"] for r in auroc_records[-N_LAYERS:]]))
        best_a = max(r["AUROC"] for r in auroc_records[-N_LAYERS:])
        print(f"  {cons:18s} n=({n_pos},{n_neg})  best L={best_l:2d}  AUROC={best_a:.3f}")

    # Per-consequence Δh distribution (full cohort, both P+B), per layer
    for cons in CONSEQUENCE_ORDER:
        sub = df[df.consequence == cons]
        if len(sub) < 20:
            continue
        X = list_to_array(sub.delta_h_norm_2)
        for ell in range(N_LAYERS):
            delta_records.append({
                "consequence": cons, "layer": ell, "n": len(sub),
                "mean_delta_h": float(X[:, ell].mean()),
                "median_delta_h": float(np.median(X[:, ell])),
                "std_delta_h": float(X[:, ell].std()),
            })

    df_au = pd.DataFrame(auroc_records)
    df_au.to_csv(args.out_dir / "per_consequence_auroc.csv", index=False)
    df_d = pd.DataFrame(delta_records)
    df_d.to_csv(args.out_dir / "per_consequence_delta_distribution.csv", index=False)

    # Kruskal-Wallis at each layer (consequence ordering, full cohort)
    kw_records = []
    consequences_present = [c for c in CONSEQUENCE_ORDER
                             if (df.consequence == c).sum() >= 20]
    for ell in range(N_LAYERS):
        groups = []
        for c in consequences_present:
            sub = df[df.consequence == c]
            X = list_to_array(sub.delta_h_norm_2)
            groups.append(X[:, ell])
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        # Find which consequence has max mean Δh at this layer
        argmax_c = consequences_present[int(np.argmax([g.mean() for g in groups]))]
        argmin_c = consequences_present[int(np.argmin([g.mean() for g in groups]))]
        kw_records.append({
            "layer": ell, "H": float(H), "p": float(p),
            "max_consequence": argmax_c, "min_consequence": argmin_c,
        })
    pd.DataFrame(kw_records).to_csv(args.out_dir / "per_consequence_kw.csv", index=False)
    print(f"\n[KW] saved per-layer Kruskal-Wallis (n_classes={len(consequences_present)})")

    # === Figure 1 — AUROC curves per consequence ===
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for cons in consequences_present:
        g = df_au[df_au.consequence == cons]
        if g.empty:
            continue
        n_p = int(g.n_pos.iloc[0]); n_n = int(g.n_neg.iloc[0])
        ax.plot(g.layer, g.AUROC, marker="o", ms=3.5, lw=1.5,
                color=CONSEQUENCE_COLORS.get(cons, "k"),
                label=f"{cons} (n={n_p}+{n_n})")
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("AUROC (P_LP vs B_LB) per consequence")
    ax.set_title("(1.1b) Variant pathogenicity AUROC per layer — stratified by molecular consequence\n"
                 "ΔH norm L2 as classifier; consequences with ≥5 P_LP + 5 B_LB shown")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_consequence_auroc.{ext}", dpi=150)
    plt.close(fig)
    print(f"[fig1] saved")

    # === Figure 2 — mean Δh per consequence per layer (full cohort) ===
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for cons in consequences_present:
        g = df_d[df_d.consequence == cons].sort_values("layer")
        n_ = int(g.n.iloc[0])
        ax.plot(g.layer, g.mean_delta_h, marker="o", ms=3.5, lw=1.6,
                color=CONSEQUENCE_COLORS.get(cons, "k"),
                label=f"{cons} (n={n_})")
    ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.set_xlabel("Layer ℓ"); ax.set_ylabel("Mean ΔH norm L2")
    ax.set_title("(1.1b) Mean ΔH per layer per consequence — full cohort (P + B + VUS)\n"
                 "Direct view of where variants of each class perturb hidden states most")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_consequence_delta_distribution.{ext}", dpi=150)
    plt.close(fig)
    print(f"[fig2] saved")

    # === Summary ===
    summary = {
        "n_variants_total": len(df),
        "n_consequence_classes_present": len(consequences_present),
        "consequences_present": consequences_present,
        "per_consequence_best": {},
        "kw_L7_8_27_29": {
            int(ell): {"H": float(r["H"]), "p": float(r["p"]),
                        "max_c": r["max_consequence"], "min_c": r["min_consequence"]}
            for ell, r in zip([7, 8, 27, 29],
                              [next((rec for rec in kw_records if rec["layer"] == ell), {}) for ell in (7, 8, 27, 29)])
            if r
        },
    }
    for cons in consequences_present:
        g = df_au[df_au.consequence == cons]
        if g.empty:
            continue
        best = g.loc[g.AUROC.idxmax()]
        summary["per_consequence_best"][cons] = {
            "best_layer": int(best.layer),
            "best_AUROC": float(best.AUROC),
            "L7_AUROC": float(g[g.layer == 7].AUROC.iloc[0]),
            "L8_AUROC": float(g[g.layer == 8].AUROC.iloc[0]),
            "L27_AUROC": float(g[g.layer == 27].AUROC.iloc[0]),
            "L29_AUROC": float(g[g.layer == 29].AUROC.iloc[0]),
            "n_pos": int(best.n_pos), "n_neg": int(best.n_neg),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print("Headline:")
    print(json.dumps(summary["per_consequence_best"], indent=2))


if __name__ == "__main__":
    main()
