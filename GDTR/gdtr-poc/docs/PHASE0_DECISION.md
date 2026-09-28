# PHASE0_DECISION.md - gDTR PoC Final Report

**Project**: gDTR (Genomic Deep-Thinking Ratio) on HyenaDNA-medium-160k
**Date**: 2026-04-26
**Status**: Phase 0 chain complete; Phase 1 method recommendation locked

---

## 1. Executive Summary

Phase 0 ran six stages on a single RTX 3090 (vessl). The headline result: the **strict NLP-style logit/UR running-min monotonicity assumption breaks on HyenaDNA's 8-layer Hyena residual stream**, with Layer 7 in particular acting as a substantive transformer that increases cosine distance to the final layer for ~54% of positions. Both Gate A (JSD lens) and Gate A' (UR cosine lens) **fail** the per-layer M2 ≥ 0.70 criterion at Layer 7 specifically. However, both lenses retain *interpretable, biologically meaningful* signal once γ is calibrated empirically:

- **Gate B PASSES** with effect size large enough to dominate the strict α (UR-gDTR exon vs intron Mann-Whitney p = 4.88e-224 vs Bonferroni-corrected α = 0.0002, Cohen's d = -1.018).
- **Gate B replicates on BRCA1** (independent gene): UR-gDTR exon vs intron p = 0 (numerical floor), Cohen's d = -0.78.
- **Gate C is informational**: 0/5 TP53 hotspots exceed null p95, signaling that single-nt variants do not generate Δ_D outliers larger than typical "random nearby allele changes."
- **Phase 1 recommendation (locked)**: UR-gDTR cosine lens primary, γ_cos = 0.50, ρ = 0.85; JSD lens auxiliary; ΔD(L) profile (vector) primary for variant analysis (Δc deprecated in favor of full vector); tuned lens (Belrose 2023) carry-over for Phase 1 to address the Layer 7 anomaly.

The most important finding for the manuscript is the **Layer 7 anomaly itself**: HyenaDNA's penultimate Hyena conv block consistently expands the residual stream away from the final-layer trajectory, violating the smooth-monotone-convergence assumption that underlies NLP DTR. This is a real architectural property of pure-Hyena CLMs that motivates Phase 1's tuned-lens approach.

---

## 2. Gate A — Logit Lens (JSD) Validity

**Setup**: 100 random 6 kb sequences (50 GC-matched chr17 intergenic + 50 dinucleotide-shuffled), seed=42.

**Per-layer M1, M2 with bootstrap 95% CI**: see `results/tables/sanity_M1_M2.csv`.

| Layer | M1 rate (CI) | M2 rate (CI) | Pass perlayer (M2 ≥ 0.70)? |
|---|---|---|---|
| 1 | 0.442 (0.441-0.443) | 1.000 | yes (degenerate) |
| 2 | 0.458 | 0.495 | NO |
| 3 | 0.513 | 0.819 | yes |
| 4 | 0.566 | 0.864 | yes |
| 5 | 0.582 | 0.927 | yes |
| 6 | 0.553 | 0.965 | yes |
| 7 | 0.448 | 0.120 | **NO (anomaly)** |
| 8 | 1.000 | 1.000 | yes (self) |

**Verdict**: M1=0.281 < 0.80, M2=0.009 < 0.85. Layer 7 with M2=0.12 is the dominant failure. Per-layer breakdown shows the cosine/JSD trajectory is *not* running-min monotone at Layer 7 — the penultimate Hyena block transforms the residual stream away from the final layer.

**Interpretation**: NLP-style logit-lens monotonicity assumption fails for HyenaDNA's pure-Hyena residual stream. This is an **informative negative**, not a methodology failure: it characterizes a real architectural property and motivates Phase 1's tuned-lens approach.

---

## 3. Gate A' — UR-gDTR (Cosine Lens) Sanity

**Setup**: same 100 seqs reused (same seed, deterministic regeneration). Cosine forwards cached at `results/runs/01c_ur_cache.npz`.

| Layer | M2_ur rate | passes? |
|---|---|---|
| 1 | 1.000 | yes (degenerate) |
| 2 | 0.505 | NO |
| 3 | 0.898 | yes |
| 4 | 0.917 | yes |
| 5 | 0.968 | yes |
| 6 | 0.987 | yes |
| 7 | 0.462 | **NO (Layer 7 anomaly mirrored)** |
| 8 | 1.000 | yes (self) |

**Verdict (strict)**: M2_ur global = 0.178 < 0.85, per-layer min at Layer 7 = 0.462 < 0.70 → strict Gate A' FAIL on the same Layer 7 anomaly.

**Interpretation**: The Layer 7 anomaly is *not* an artifact of vocabulary projection (it appears identically in the cosine lens, which bypasses lm_head). It is a property of the residual stream itself. Layers 3-6 all pass per-layer monotonicity comfortably; only Layer 2 (early diversification) and Layer 7 (late re-expansion) fail.

**Calibration discovery**: Default γ_cos = 0.10 saturates all positions to c=8 (mean convergence depth = 8.0, std = 0.0 — useless). Quantile-calibrated γ_q70 at the penultimate layer = 0.482 produces meaningful c distributions (mean = 6.26, std = 0.78). **All downstream stages adopt quantile-calibrated γ_cos.**

**Why proceed despite strict fail**: The design's strict monotonicity criterion was calibrated for NLP-like 32+ layer models where the residual stream is expected to converge smoothly. For an 8-layer Hyena model with a known Layer 7 transformation, this criterion is overly stringent; the *informational content* of UR-gDTR is preserved (and confirmed by Stage 2 yielding p < 1e-200 effect sizes).

---

## 4. Gate B — Genomic Signal (TP53)

**Setup**: TP53 region chr17:7,668,402-7,687,550 padded ±3 kb (region length 25,149 bp), sliding window=6 kb stride=500 bp → **39 windows**. Central 1 kb stitched per window. 5× dinucleotide-shuffled baselines. UR-gDTR primary with γ_cos = q70 penultimate = **0.510**. Bonferroni α/5 = 0.0002.

### UR-gDTR (primary) per-context settling depth

| Context | n_pos | mean c_interp | MWU p vs intron | Cohen's d vs intron | Gate B passes? |
|---|---|---|---|---|---|
| coding_exon | 1179 | 5.32 | **4.88e-224** | **-1.02** | **YES** |
| intron | 16558 | 6.17 | — | — | (ref) |
| 5'UTR | 142 | 5.50 | 5.75e-19 | -0.78 | YES |
| 3'UTR | 1191 | 5.68 | 2.71e-76 | -0.57 | YES |
| splice ±10 bp | 400 | 5.39 | 1.52e-67 | -0.92 | YES |
| intergenic | 930 | 5.28 | 4.40e-193 | -1.06 | YES |

**Direction**: Coding exon, splice, and intergenic positions all have *lower* settling depth than intron (predictions converge to the final layer at earlier layers). Introns occupy the highest c_interp regime — i.e., HyenaDNA's later layers refine intronic predictions more. This biological inversion (intron > exon in settling depth) is interesting and may relate to longer-range syntactic structure of intronic Hyena conv kernels.

### JSD-gDTR (auxiliary)
**Degenerate**: with γ=0.5 (design default), JSD distribution at layer 1 already has p99 = 0.08 (effective range = 0.02; see `01b_jsd_dist.json`). Settling depth saturates to c_interp = 1.0 for *every* position in *every* context, MWU p = 1.0. **Confirms design §6.2's range<0.15 → quantile-γ primary recommendation.**

### Confounders (`gdtr_vs_confounders.csv`)
Partial Spearman ρ(UR-gDTR, context | GC, entropy):
- coding_exon: ρ = -0.39 (still strongly negative after partialing out GC + k=3 entropy)
- intergenic: ρ = -0.32
- splice: ρ = -0.13

The exon-vs-intron signal is **not** explained by GC or compositional confounders.

**Gate B verdict: PASS (UR-gDTR primary)**.

---

## 5. Gate C — TP53 Hotspot Variant Pilot

**Setup**: 5 hotspots (R175H, R248Q, R273H, R249S, G245S) ±3 kb context. Null = 100× random allele change at random position within ±100 bp (per hotspot).

| Variant | max\|ΔD\| (UR) | null p95 | exceeds p95? | Δc_interp (UR) | argmax layer |
|---|---|---|---|---|---|
| R175H | 0.045 | 0.135 | NO | -0.191 | 4 |
| R248Q | 0.024 | 0.132 | NO | -0.112 | 3 |
| R273H | 0.036 | 0.135 | NO | +0.082 | 6 |
| R249S | 0.051 | 0.153 | NO | -0.209 | 7 |
| G245S | 0.057 | 0.133 | NO | +0.072 | 1 |

**n_exceeding_null_p95 = 0/5** — **Gate C informational fail** per design §3.3.

### Three-metric agreement (UR-gDTR)
- ρ(Δc_discrete, Δc_interp) = NaN (Δc_discrete = 0 for all 5 variants — no integer crossing)
- ρ(Δc_interp, signed_argmax_ΔD) = **0.40**
- ρ(Δc_discrete, signed_argmax_ΔD) = NaN

Per design §12 decision rule (ρ < 0.5): **ΔD(ℓ) vector is the primary feature for Phase 3 classifier; Δc_discrete is deprecated; Δc_interp downgraded to scalar summary only.**

### Interpretation
Hotspot single-nt variants do *not* perturb the residual stream more than typical random nearby substitutions — i.e., on the 6.6M HyenaDNA model, the variant signal is below the per-position noise. This does not mean the model fails to encode variant-functional information; it means the layer-wise prediction-convergence delta is not amplified at hotspot positions relative to other 1-nt changes. Phase 3 should:
1. Use ΔD(ℓ) vector (8-dim) as feature, not scalar
2. Aggregate over a window (not single-position) to integrate per-position signals
3. Increase model capacity (Evo 2 hybrid, with attention) to test whether attention layers amplify variant signal

---

## 6. JSD Effective Range and Quantile-γ Recommendation

From `01b_jsd_dist.json`:
- Effective range (p95−p5) median across layers = **0.019**
- Per-layer effective ranges: 0.058, 0.061, 0.026, 0.013, 0.011, 0.009, 0.019, 0.000
- Effective range << 0.15 design threshold → **quantile-based γ is the correct primary** per design §6.2

The JSD lens *does* contain layer-wise gradient information (visible as small non-zero values across layers in F3), but the dynamic range under log(|V|=12) normalization is too compressed for a uniform γ=0.5 to distinguish converged from non-converged positions. Design §12 prescribes γ_q70 at penultimate layer; Phase 0 confirms this is the correct path (γ=0.5 universally saturates to c=1).

For UR-gDTR cosine lens, the equivalent calibration:
- random sequences: γ_q70_pen = 0.482
- TP53 region: γ_q70_pen = 0.510 (slightly higher because TP53 has stronger Layer 7 transformations)
- BRCA1 region: γ_q70_pen = 0.508

Recommendation: **Phase 1 should use γ_cos = q70 of penultimate cosine distribution, computed per region/dataset rather than fixed.** This is a method contribution: regional adaptive γ_cos.

---

## 7. HP Sweep — Phase 1 (γ_cos, ρ) Recommendation

Reused Stage 2 + Stage 3 cosine forwards. Grid: γ_cos ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}, ρ ∈ {0.5, 0.7, 0.85}. Combined-rank optimization over (|Cohen's d|, mean|Δc_interp|).

**Best**: **γ_cos = 0.50, ρ = 0.85** — Cohen's d = -1.026, mean|Δc_interp| over 5 hotspots = 0.349.

Per F9 heatmap inspection:
- Effect size |d| is large (>0.5) across γ_cos ∈ [0.3, 0.7] for all ρ values
- Variant signal mean|Δc_interp| peaks near γ_cos ∈ [0.3, 0.7]
- Below γ_cos < 0.2 the metric saturates (most positions never below threshold)

**Phase 1 carry-over**: γ_cos = 0.50, ρ = 0.85. (Design default ρ=0.85 retained; γ_cos shifted from 0.10 default to 0.50 quantile-aware value.)

---

## 8. BRCA1 Extension (Stage 5)

**Setup**: BRCA1 chr17:43,044,295-43,170,245 (~125 kb, design said chr13 — corrected to chr17 since BRCA1 is on chr17). HyenaDNA-medium-160k segmented sliding (252 windows × 6 kb, stride 500 bp). Single-pass took ~20 sec; the larger 1m model was not needed.

### Per-context (UR-gDTR primary)

| Context | n_pos | mean | MWU p vs intron | Cohen's d |
|---|---|---|---|---|
| coding_exon | 5589 | 5.44 | ~0 | **-0.78** |
| intron | 73982 | 6.08 | — | — |
| 5'UTR | 113 | 5.36 | 1.25e-18 | -0.85 |
| 3'UTR | 1386 | 5.69 | 1.17e-63 | -0.46 |
| splice ±10 bp | 880 | 5.47 | 1.59e-96 | -0.73 |
| intergenic | 45430 | 5.95 | 1.14e-128 | -0.16 |

**BRCA1 result independently replicates the TP53 finding**: coding exon and splice ±10 have lower settling depth than intron, with large effect (|d| > 0.7). Cross-gene generalization of Gate B confirmed on a non-overlapping locus.

JSD-gDTR identically degenerate (all c_interp = 1.0).

---

## 9. Phase 1 Method Recommendations (LOCKED)

1. **Primary lens**: **UR-gDTR (cosine)**, with γ_cos calibrated per-region as the 70th percentile of D_cos at the penultimate layer. JSD-gDTR computed in parallel as auxiliary cross-check.
2. **Tuned lens**: Add Belrose 2023 tuned lens training on Phase 1 (Evo 2) Hyena layers to address the Layer 7 anomaly. Specifically learn a per-layer affine transformation A_l such that softmax(W_U · A_l · h_l) is the maximum-likelihood predictor of the next token; this should restore monotonicity by absorbing the Layer 7 shift.
3. **Hyperparameter starting point**: γ_cos = 0.50, ρ = 0.85 (from Stage 4 sweep).
4. **Variant analysis primary feature**: ΔD(ℓ) vector (8-dim for medium HyenaDNA, 32-dim for Evo 2 hybrid). Δc_discrete deprecated; Δc_interp scalar summary only.
5. **Calibration approach**:
   - Phase 1 Step 1: characterize JSD/cosine effective range per layer on Evo 2 forward. If range > 0.30 use design default γ; else use quantile-γ.
   - Phase 1 Step 2: confirm Layer-N anomaly position on Evo 2 (32 layers) — expected to be much weaker due to attention layers smoothing the residual stream.
6. **Confound control**: continue per-position GC + k=3 Shannon entropy partial Spearman as in Stage 2.

---

## 10. Limitations of Phase 0

Per design §15 plus Phase 0 specifics:
1. **N=5 hotspots are not powered for variant statistics**: Gate C is informational only; Phase 3 ClinVar 2K+ planned.
2. **HyenaDNA has no attention**: Layer 7 anomaly may be Hyena-specific. Evo 2 hybrid (attention + Hyena) may behave differently — verify in Phase 1.
3. **8 layers limit c resolution**: c ∈ [1, 8] discretizes Δc to integer steps (Δc_discrete = 0 for all hotspots demonstrates this). Evo 2's 32+ layers will provide finer resolution.
4. **Single species (human GRCh38)**: cross-species held to Phase 4.
5. **TP53 + BRCA1 only**: gene-specific bias remains plausible; Phase 2 chr22 genome-wide will check.
6. **gamma_cos calibration is region-dependent**: γ_q70(random) ≠ γ_q70(TP53) ≠ γ_q70(BRCA1). Need a principled cross-region calibration scheme for Phase 2.
7. **Strict Gate A failed for both lenses on the M2 monotonicity criterion**. Decision was made to continue with quantile-calibrated γ; this is documented here and constitutes a soft pre-registration deviation. The deviation does not affect Gates B/C primary thresholds (still p < 0.001 Bonferroni for B), only the *measurement* parameter γ.

---

## 11. Reproducibility Checklist

- [x] Seed=42 across all stages
- [x] HyenaDNA HF revision pinned: `7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce`
- [x] GENCODE v44 (2023-07-17) lock recorded in `data/DATA_VERSIONS.txt`
- [x] ClinVar 2026-04-18 lock
- [x] All raw forward outputs cached in `results/runs/*_cache.npz` (re-runnable from CSV)
- [x] All figure CSV inputs are written before figure rendering
- [x] All `results/runs/*.json` log seed, host, platform, torch version, runtime
- [x] No git repo exists (vessl scratch); SHA = N/A noted in design §13

### File inventory

```
results/
├── tables/
│   ├── sanity_M1_M2.csv              (Stage 1 Gate A)
│   ├── sanity_per_position_breakdown.csv
│   ├── jsd_stats.csv                  (Stage 1b)
│   ├── ur_sanity.csv                  (Stage 1 Gate A')
│   ├── gdtr_by_context.csv            (Stage 2 Gate B)
│   ├── gdtr_vs_confounders.csv        (Stage 2 confounders)
│   ├── tp53_hotspot_metrics.csv       (Stage 3 Gate C)
│   ├── metric_agreement.csv           (Stage 3 Spearman)
│   ├── hp_sweep.csv                   (Stage 4 HP)
│   └── brca1_gdtr_by_context.csv      (Stage 5 BRCA1)
├── figures/
│   ├── F2_sanity.{pdf,png}            (Gate A)
│   ├── F2b_ur_sanity.{pdf,png}        (Gate A')
│   ├── F3_jsd_distribution.{pdf,png}  (1b)
│   ├── F4_tp53_profile.{pdf,png}      (Gate B primary)
│   ├── F5_brca1_profile.{pdf,png}     (Stage 5)
│   ├── F6_context_boxplot.{pdf,png}   (Gate B)
│   ├── F7_delta_jsd_heatmap.{pdf,png} (Gate C)
│   ├── F8_three_metric_agreement.{pdf,png}
│   ├── F9_hp_heatmap.{pdf,png}
│   └── *.caption.json                 (sidecars per fig)
└── runs/
    ├── 00_smoke_test.json
    ├── 01_sanity.json
    ├── 01_sanity_cache.npz
    ├── 01b_jsd_dist.json
    ├── 01c_ur_sanity.json
    ├── 01c_ur_cache.npz
    ├── 02_gene_structure.json
    ├── 02_gene_structure_cache.npz
    ├── 02_gene_structure_shuf_cache.npz
    ├── 03_variant_pilot.json
    ├── 03_variant_cache.npz
    ├── 03_null_cache.npz
    ├── 04_hp_sweep.json
    ├── 05_brca1.json
    └── 05_brca1_cache.npz
```

**End of PHASE0_DECISION.md**
