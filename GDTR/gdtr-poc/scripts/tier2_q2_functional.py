#!/usr/bin/env python3
"""Tier-2 functional validation of Q2 conservation discordance regions.
Compute overlap of Q2 with GTEx eQTL, GWAS Catalog, and ENCODE cCRE-ELS,
with hypergeometric and shuffle-based p-values.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, binomtest

ROOT = Path("/path/to/gDTR")
EXT = ROOT / "data" / "external"
OUT = ROOT / "results" / "tier2_q2_functional"
OUT.mkdir(parents=True, exist_ok=True)

CHR22_LEN = 50818468
Q2_BED = OUT / "q2_sorted.bed"
GENOME_FILE = OUT / "chr22.genome"
GENOME_FILE.write_text(f"chr22\t{CHR22_LEN}\n")


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, **kw)


def bed_total_bp(bed_path):
    """Sum of (end-start) over a sorted, possibly overlapping BED. Use bedtools merge first."""
    out = run(f"sort -k1,1 -k2,2n {bed_path} | bedtools merge -i - | awk '{{s+=$3-$2}} END{{print s+0}}'")
    return int(out.stdout.strip())


def n_records(bed_path):
    out = run(f"wc -l < {bed_path}")
    return int(out.stdout.strip())


def overlap_bp(a_bed, b_bed):
    """bp of intersection (both can have records). Use sortBed | mergeBed first to avoid double counting."""
    cmd = (
        f"bedtools intersect -a {a_bed} -b {b_bed} -wo "
        f"| awk '{{s+=$NF}} END{{print s+0}}'"
    )
    # But that double-counts if a_bed has overlapping records. Q2 regions are non-overlapping.
    out = run(cmd)
    return int(out.stdout.strip())


def n_q2_overlap(a_bed, b_bed):
    out = run(f"bedtools intersect -a {a_bed} -b {b_bed} -u -wa | wc -l")
    return int(out.stdout.strip())


def pair_overlap(a_bed, b_bed, out_path):
    run(f"bedtools intersect -a {a_bed} -b {b_bed} -wa -wb > {out_path}")
    return n_records(out_path)


def shuffle_null(q2_bed, dataset_bed, n_shuffles=100, seed=42):
    """For region-level overlap: count Q2 regions that hit dataset; compare to shuffled."""
    obs = n_q2_overlap(q2_bed, dataset_bed)
    null = []
    for i in range(n_shuffles):
        cmd = (
            f"bedtools shuffle -i {q2_bed} -g {GENOME_FILE} -seed {seed + i} -chrom -noOverlapping "
            f"| sort -k1,1 -k2,2n "
            f"| bedtools intersect -a - -b {dataset_bed} -u -wa | wc -l"
        )
        try:
            r = run(cmd)
            null.append(int(r.stdout.strip()))
        except subprocess.CalledProcessError:
            # noOverlapping may fail if can't fit; fall back without it
            cmd = (
                f"bedtools shuffle -i {q2_bed} -g {GENOME_FILE} -seed {seed + i} -chrom "
                f"| sort -k1,1 -k2,2n "
                f"| bedtools intersect -a - -b {dataset_bed} -u -wa | wc -l"
            )
            r = run(cmd)
            null.append(int(r.stdout.strip()))
    null = np.array(null)
    pval = float((null >= obs).sum() + 1) / (n_shuffles + 1)
    return {
        "n_obs": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_max": int(null.max()),
        "null_min": int(null.min()),
        "pval_shuffle": pval,
        "n_shuffles": n_shuffles,
    }


def analyze(name, dataset_bed):
    print(f"\n=== {name} ===")
    n_dataset = n_records(dataset_bed)
    n_dataset_bp = bed_total_bp(dataset_bed)
    q2_total_bp = bed_total_bp(Q2_BED)
    inter_bp = overlap_bp(Q2_BED, dataset_bed)
    n_q2_hit = n_q2_overlap(Q2_BED, dataset_bed)
    n_q2_total = n_records(Q2_BED)

    # bp-level hypergeometric (population = chr22 bp)
    # M = chr22 bp, n = dataset bp on chr22 (merged), N = q2 bp, k = intersect bp
    M = CHR22_LEN
    n_succ = n_dataset_bp
    N = q2_total_bp
    k = inter_bp
    expected_k = N * n_succ / M
    fold = (k / expected_k) if expected_k > 0 else float("nan")
    # one-sided p-value: P(X >= k)
    pval_hg = float(hypergeom.sf(k - 1, M, n_succ, N))

    # region-level binomial: each Q2 region has prob = n_dataset_bp / chr22 of being hit (rough)
    # But region size matters, so use shuffle null for region-level.
    p_region = n_dataset_bp / M  # prob a single bp is in dataset
    # Approximate prob a Q2 region hits something: ~1 - (1 - p)^len(region). Compute exactly.
    q2_lengths = []
    with open(Q2_BED) as fh:
        for ln in fh:
            f = ln.strip().split("\t")
            q2_lengths.append(int(f[2]) - int(f[1]))
    q2_lengths = np.array(q2_lengths)
    p_hits = 1.0 - np.power(1.0 - p_region, q2_lengths)
    expected_n_hits = float(p_hits.sum())
    fold_region = n_q2_hit / expected_n_hits if expected_n_hits > 0 else float("nan")

    # shuffle null
    print(f"  Running 100 shuffles...")
    shuf = shuffle_null(Q2_BED, dataset_bed, n_shuffles=100, seed=42)

    # paired output
    pair_out = OUT / f"q2_x_{name}.bed"
    n_pairs = pair_overlap(Q2_BED, dataset_bed, str(pair_out))

    res = {
        "dataset": name,
        "n_dataset_chr22": n_dataset,
        "dataset_chr22_bp_merged": n_dataset_bp,
        "q2_n_regions": n_q2_total,
        "q2_total_bp": q2_total_bp,
        "n_q2_overlap": n_q2_hit,
        "intersect_bp": inter_bp,
        "expected_intersect_bp": expected_k,
        "fold_enrichment_bp": fold,
        "pval_hypergeom_bp": pval_hg,
        "expected_n_q2_hits_binom": expected_n_hits,
        "fold_enrichment_region": fold_region,
        "shuffle_null": shuf,
        "n_pair_records": n_pairs,
        "paired_bed": str(pair_out),
    }
    print(f"  n_dataset={n_dataset}, dataset_bp={n_dataset_bp}, intersect_bp={inter_bp}")
    print(f"  q2_total_bp={q2_total_bp}, fold_bp={fold:.3f}, pval_hg={pval_hg:.3e}")
    print(f"  n_q2_hit={n_q2_hit}/{n_q2_total} regions ({100*n_q2_hit/n_q2_total:.1f}%)")
    print(f"  expected_region_hits={expected_n_hits:.1f}, fold_region={fold_region:.3f}")
    print(f"  shuffle: obs={shuf['n_obs']}, null_mean={shuf['null_mean']:.1f}+/-{shuf['null_std']:.1f}, pval_shuf={shuf['pval_shuffle']:.4f}")
    return res


def main():
    datasets = {
        "eqtl": EXT / "gtex_eqtl_chr22.bed",
        "gwas": EXT / "gwas_chr22.bed",
        "cre_els": EXT / "ccre_els_only_chr22.bed",
    }
    results = {}
    for name, bed in datasets.items():
        if not bed.exists():
            print(f"[SKIP] {name}: {bed} not found")
            continue
        results[name] = analyze(name, str(bed))
        with open(OUT / f"{name}_overlap.json", "w") as fh:
            json.dump(results[name], fh, indent=2)

    # consolidated summary
    summary = {
        "tier": "tier2_q2_functional",
        "chrom": "chr22",
        "chr22_len": CHR22_LEN,
        "q2_n_regions": 5090,
        "q2_total_bp_min100": int(bed_total_bp(Q2_BED)),
        "phase5_reference": {
            "ENCODE_cCRE_all_fold": 1.2839,
            "note": "Phase 5 used position-level (bp) ENCODE cCRE annotation; Tier-2 uses region-level Q2 BED (≥100 bp regions only) ∩ chr22-filtered datasets.",
        },
        "datasets": results,
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    (OUT / "_done").touch()
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
