# TDiG — Think Deep in Genome

**YAICON 8th 1st Prize · Team 띵디지놈**

Layer-wise residual-stream settling analysis for genomic foundation models.

---

## Overview

Genomic foundation models — Evo 2, HyenaDNA, NT-v2, DNABERT-2 — are increasingly accurate at variant scoring, yet their internal layer-wise computation is hard to interpret. TDiG asks a complementary question:

> **When and how does a genomic foundation model converge?**

We approach this through a family of *training-free, geometric settling-depth* metrics applied to the residual stream of Evo 2 7B, and we map each metric onto biological context (splice sites, regulatory elements, intronic regions, ClinVar variants).

The project unfolds in two phases:

1. **gDTR (Phase 1)** — a single-axis cosine settling-depth metric `c(t)`, validated on chr22 + chr17 and four genomic foundation models.
2. **TDiG (this repo, Phase 2)** — the framework is extended into a **17-cell taxonomy** (5 metric families × 3 reference variants) computed from a single forward pass, enabling cross-axis dissociation patterns that a single metric cannot resolve.

## Background — gDTR

gDTR (Genomic Deep Think Ratio) adapts the Deep-Thinking Ratio idea from NLP transformers to nucleotide-level causal language models. For an input sequence of length T processed by an L-layer model, we extract the residual stream `h_ℓ(t)` at every layer ℓ and token position t, and define the per-token settling depth

```
c(t) = min { ℓ : run-min D_cos(ℓ, t) ≤ γ }
```

where `D_cos(ℓ, t) = 1 − cos(h_ℓ(t), h_norm(t))`, the run-min envelope enforces monotone descent on non-monotone genomic-model trajectories, and γ is a per-layer q70 calibration threshold. Lower `c(t)` means the representation stabilises earlier.

Three findings frame the message:

- **Bidirectional readout.** Motif edits deepen `c̄` while flank-shuffles lift it — `c(t)` reads out grammar integration rather than motif strength alone.
- **Splice sites and enhancer-like cCREs settle ~2 layers earlier than intronic and coding contexts** (splice donor Cohen's d = −0.43; ENCODE cCRE-ELS d = −0.19). chr22 calibration transfers to held-out chr17 with 94.6 % effect retention.
- **Variant-consequence layer encodes the disrupted biological information level.** Synonymous variants peak at the deepest layers (consistent with protein-semantic information consolidating late); intron / frameshift / nonsense peak earlier.

The Phase-1 paper has been accepted as an **Oral at the ICML 2026 GenBio Workshop** (see [Paper](#paper)).

## TDiG framework

TDiG generalises gDTR's single direction-axis into five complementary settling axes, each measuring a different question about convergence:

| ID | Name | What it measures | Reference variants |
|---|---|---|---|
| M1 | `c_dir` | Direction settling (cosine), persistence W = 3 | A, B, C |
| M2 | residual-accumulation magnitude | Pre-norm magnitude ratio `|‖h_ℓ‖/‖h_ref‖ − 1|` (diagnostic) | A (B, C degenerate) |
| M3 | `c_geo` | Trajectory dynamics: standardised velocity + curvature, 5 α/β cells | reference-free |
| M4 | `c_M,set` | Reference-whitened settling distance with Ledoit–Wolf shrinkage on Σ_ref | A, B, C |
| M5 | `c_τ` | Path tortuosity — remaining cumulative path / straight-line distance | A, B, C |

Reference variants A/B/C apply different RMSNorm treatments to `h_ℓ` and `h_ref`, isolating the effect of γ asymmetry. The full design gives **17 settling cells** per token, all computed from one forward pass.

A single forward pass over chr22 + chr17 (Evo 2 7B) produces a per-position 17-cell tensor that is then mapped onto 7 biological contexts (intergenic, intron, coding exon, 5′UTR, 3′UTR, splice donor, splice acceptor) and 10,910 ClinVar variants across 15 cancer-associated genes.

## Selected results

- **Cross-chromosome generalisation.** chr22 → chr17 Spearman ρ = 0.989 across 13 valid cells; median retention 97 %; sign preserved 13/13.
- **Context separation.** Coding-exon vs intron Cohen's d = −0.94 (M3_geo curvature-only), larger than splice-donor vs intron (−0.81). M3 wins 13 / 21 pairwise contexts as best discriminator.
- **L29 phase transition.** Independent confirmation from 5 measurements (probing AUROC crash 0.980 → 0.799 at L29; SVD condition number 10¹⁰–¹²; activation patching ΔH explosion of 12 orders of magnitude with P/B ratio preserved; per-position 7-way multitask accuracy crashes from L24's 74 % to L29's 40 %). Block 31 is confirmed as an idle passthrough (linear-fit R² = 1.000).
- **Variant peak layer encodes biology.** Synonymous L = 27, missense L = 27, 5′UTR L = 27 versus intron / 3′UTR L = 8. ΔH-norm L2 at L = 8 reaches AUROC 0.855 (full P/LP vs B/LB cohort, 8,008 variants).
- **VUS reclassification.** A GBM on the 64-dim per-layer log-ΔH + Δcos feature reaches 5-fold CV AUROC 0.949 (+9.4 pp over the single-layer ΔH baseline), reclassifying 1,303 / 2,902 VUS as Likely Pathogenic and 746 as Likely Benign.
- **Cryptic functional-element discovery in introns.** The top 0.5 % M5_τ_refB intron outliers are enriched 2.70× for proximity to annotated splice sites (21.0 % within ±200 bp vs 7.8 % random) — an unsupervised handle on splice-related intronic regulatory elements.
- **Cross-architecture.** HyenaDNA-medium reproduces the splice-donor vs intron sign (Cohen's d = −0.14 vs Evo 2 −0.81); architecture shares the geometric structure, depth controls the magnitude.

## Repository layout

```
TDiG/
├── docs/
│   ├── thesis.md                       motivation and scope decisions
│   ├── metric_definitions.md           all 5 metric formulas + 3 reference variants
│   ├── design_decisions.md             iteration log
│   ├── reference_variants.md           A / B / C variant rationale
│   ├── reproduction.md                 3-phase pipeline (population stats → forward → analysis)
│   └── PAPER_OUTLINE.md                Phase-2 paper draft outline
├── PLAN.md                             consolidated experimental plan
├── METRICS_GUIDE.md                    how to access each of the 17 cells + derived metrics
├── scripts/                            numbered analysis scripts (10b → 36)
├── src/tdig/                           module skeletons (references, metrics)
├── results/
│   ├── RESULTS_v3.md                   comprehensive analysis tracker
│   ├── figures/                        publication-ready figures (V1–V9, advanced, zoom)
│   ├── analysis_BD/                    probing + metric ↔ PCA
│   ├── analysis_T123/                  PCA biological-meaning checks
│   ├── chr17_replication/              cross-chromosome retention
│   ├── bootstrap_chr22_ci/             chr22 95 % bootstrap CIs
│   ├── variant_analysis_scalars/       per-layer ΔH AUROC
│   ├── variant_per_consequence/        per-consequence AUROC and Δ-trajectories
│   ├── vus_reclassification/           VUS predictions + classifier metrics
│   ├── intron_outlier/                 cryptic functional-element candidates
│   ├── L29_svd/                        L29 phase-transition mechanism
│   ├── multitask_per_position/         7-way per-position context prediction
│   ├── context_separation/             7×7 best-cell context heatmap
│   ├── hyenadna_crossarch/             cross-architecture replication
│   └── data_cache_minimal_archive/     small artifacts + pointers to large caches
└── tests/
```

Large hidden-state caches (chr22 / chr17 / variant H5 files, ~120 GB) are stored on the project's GPU server and partially mirrored as a HuggingFace dataset. See `METRICS_GUIDE.md` for the access matrix.

## Reproducibility

```bash
# 1. Population stats and γ calibration (must precede all forward passes)
python scripts/10_population_stats.py
python scripts/10b_calibrate_v2.py

# 2. Forward passes (Evo 2 7B)
python scripts/15_chr22_forward.py
python scripts/16_chr17_forward.py
python scripts/18_variant_forward.py

# 3. Analysis (each script writes to its own results/ subdirectory)
python scripts/19_variant_analysis_scalars.py
python scripts/21_variant_per_consequence.py
python scripts/22_chr17_replication.py
python scripts/23_bootstrap_chr22_ci.py
python scripts/27_vus_reclassification.py
python scripts/29_context_separation_matrix.py
python scripts/30_intron_outlier_discovery.py
python scripts/31_L29_svd_mechanism.py
python scripts/34_per_position_multitask.py
python scripts/36_hyenadna_crossarch.py
```

Full reproduction recipe — including the 3-phase server pipeline (~60–75 min on a single H200) and the data-cache pull script — is documented in [`docs/reproduction.md`](docs/reproduction.md).

## Team

Team **띵디지놈** (Think Deep in Genome), YAICON 8th, Yonsei University.

| Name | Role | Affiliation | GitHub |
|---|---|---|---|
| Cho Yoonjin | Team Lead | College of Medicine | [@darejinn](https://github.com/darejinn) |
| Kim Minseok | Member | College of Medicine | [@0304michael](https://github.com/0304michael) |
| Koo Minsun | Member | College of Medicine (premed) | [@minseongu1123](https://github.com/minseongu1123) |
| Kang Jiheon | Member | Electrical and Electronic Engineering | [@heoneyzi](https://github.com/heoneyzi) |
| Sim Jaeyun | Member | Statistics and Data Science (MS) | [@JaeyoonShim729](https://github.com/JaeyoonShim729) |

## Paper

Phase 1 of this work has been accepted as an **Oral at the ICML 2026 GenBio Workshop**:

> *GDTR: Layer-wise Settling Depth Reveals Biological Grammar in Genomic Foundation Models.*
> OpenReview: [openreview.net/forum?id=Z9h1jiPbus](https://openreview.net/forum?id=Z9h1jiPbus)

## License

MIT — see [`LICENSE`](LICENSE).
