# Tier 1 extensions — diagnostic & mechanism analyses

**Project**: gDTR on Evo 2 7B.
**Tier**: 1 (Phase-3 diagnostics: per-layer ablation, interpretability baselines, mechanism case studies, bootstrap stability).
**Status**: T1.1 / T1.3 / T1.4 DONE; T1.2 in progress on the H200 server.
**Predecessor**: [`phase3_variant_pathogenicity.md`](phase3_variant_pathogenicity.md) (same 8,008-variant P_LP+B_LB cohort).
**Independent verification**: all four CPU bundle outputs (T1.1, T1.4, T2.2, T2.3) re-derived from raw OOF scores with `|Δ| < 0.001` across six spot-checked AUROCs — see `results/_verification/cpu_bundle_verification.md`.

---

## 0. TL;DR

| Task | Status | Headline number |
|---|---|---|
| T1.1 per-layer ΔD AUROC ablation | DONE | best single-layer JSD = L29 (0.794); best single-layer cos = L30 (0.729); 32-d vector ≫ best single layer |
| T1.2 interpretability baselines (delta-H, attention rollout, IG, comparison, cost) | in progress on server (rollout 12/85 min, IG/compare/cost queued) | — |
| T1.3 mechanism case studies | DONE | 3 P + 3 B variants; ΔD_max P > B in all 3 pairs (11×, 8×, 1.5×); BRCA1 splice peaks at L7, TP53 R175H peaks at L28, BRCA1 c.5266 peaks at L24 |
| T1.4 bootstrap AUROC stability | DONE | ΔD_cos 32-d 0.8436 [0.833, 0.853]; std 0.0050; 1000 bootstraps × 4 models |

---

## 1. T1.1 — Per-layer ΔD AUROC ablation

**Source**: `results/tier1_per_layer/per_layer_auroc.csv` (128 rows = 32 layers × 2 lenses × 2 CV schemes), `results/tier1_per_layer/summary.json`.

**Pipeline**: `Pipeline([StandardScaler, LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')])`, `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, OOF score pooling, plus LOGO over 15 cancer genes.

### 1.1 Vector-level numbers (32-d, all layers)

| Feature | CV scheme | AUROC mean | 95% CI |
|---|---|---:|---|
| ΔD_jsd vector (32-d) | stratified 10-fold | 0.8225 | [0.8122, 0.8332] |
| ΔD_cos vector (32-d) | stratified 10-fold | **0.8436** | [0.8331, 0.8533] |
| ΔD_jsd vector (32-d) | LOGO | 0.8114 | [0.8007, 0.8212] |
| ΔD_cos vector (32-d) | LOGO | 0.8253 | [0.8153, 0.8349] |

n_train = 8008 (P_LP = 3514, B_LB = 4494), n_genes = 15, elapsed = 288.8 sec.

### 1.2 Best single-layer feature (1-d)

| Lens | Best layer | AUROC (stratified) |
|---|---:|---:|
| JSD | **L29 (hcm)** | 0.7940 |
| cos | **L30** | 0.7291 |

Note (post-hoc verification) — L30 vs L31 cosine columns are bit-identical
(the post-norm representation is duplicated by the lens implementation),
and L31 jsd is identically zero (jsd cannot be computed at the post-norm tap
because lm_head consumes that representation directly). Both lenses converge
on the same model location: **the deepest interpretively-meaningful tap**
(jsd: last block before post-norm; cos: post-norm itself). The manuscript
adds a footnote on this equivalence.

### 1.3 Take-away

The 32-d vector outperforms the best single layer by ≥ 0.05 AUROC for both
lenses, confirming Phase 0's design choice of vector-level features. Single-layer
AUROCs trace a U-shape (early L0 ≈ 0.6, mid-zone L11–L17 around 0.66–0.74,
peak around L29–L30, then drop at the degenerate post-norm taps).

---

## 2. T1.3 — Mechanism case studies (3 P + 3 B variants)

**Source**: `results/tier1_case_studies/case_studies.json` (17 KB), `case_studies.npz` (4 KB per-layer arrays), `_done`.

**Setup**: 6 variants (3 pathogenic + 3 benign controls) on chr17, ±3 kb context, evo2_7b_base 8K. All Phase 3 cached values match (rel_err = 0 across all 3 P variants); 9.95 sec wall, 17.1 GB GPU peak.

### 2.1 Per-variant summary

| Variant | Category | argmax layer | max\|ΔD_jsd\| | Evo 2 ΔLL | Pair ratio P/B max\|ΔD\| |
|---|---|---:|---:|---:|---:|
| BRCA1 splice (43076602 G→T, near c.5074+1G>A) | P_LP (3★) | **L7 (hcs, shallow)** | 0.0390 | +0.275 | **11.1×** vs LB control 0.0035 (L13) |
| TP53 R175H (7674220 C→T, c.524G>A neg-strand) | P_LP (3★) | **L28 (hcs, deep)** | 0.0206 | -1.505 | **8.0×** vs LB control 0.0026 (L1) |
| BRCA1 c.5266 region (43057063 G→A) | P_LP (3★) | **L24 (attn, deep)** | 0.0367 | +0.017 | **1.5×** vs LB control 0.0244 (L24) |

Benign controls (matched 2–10 bp away from the P variant):

| Variant | Category | argmax layer | max\|ΔD_jsd\| | Evo 2 ΔLL |
|---|---|---:|---:|---:|
| BRCA1 LB at 43076592 (Δ=10 bp from splice P) | B_LB (3★) | L13 | 0.0035 | -0.026 |
| TP53 LB at 7674222 (Δ=2 bp from R175H) | B_LB (2★) | L1 | 0.0026 | -1.603 |
| BRCA1 LB at 43057061 (Δ=2 bp from c.5266 P) | B_LB (3★) | L24 | 0.0244 | +0.381 |

### 2.2 Mechanistic interpretation

- **BRCA1 splice variant** peaks at a **shallow** layer (L7) — splice grammar
  is a local pattern that disrupts logit prediction early, consistent with the
  Phase 1+2 finding that splice positions converge early (low c).
- **TP53 R175H** (canonical missense in DNA-binding domain) peaks at a **deep**
  layer (L28) — protein-level effect needs long-range integration; matches the
  per-layer ablation peak at L29 for the population.
- **BRCA1 c.5266** (region near a frameshift hotspot, indels excluded) peaks at
  a **mid-deep** attention block (L24) — likely reflects long-range linkage to
  domain context.
- **Pair-ratio test**: every P variant has max\|ΔD\| > paired B control, with
  ratios 1.5×–11.1×. This is the local mechanism evidence backing the
  population-level AUROC = 0.844 finding.

### 2.3 Note on "argmax deep" count

`n_pathogenic_deep_argmax = 2` in the JSON refers to TP53 (L28) and BRCA1
c.5266 (L24); BRCA1 splice (L7) is shallow. So 2/3 P variants peak deep, 1/3
peaks shallow — the population result aggregates over many such variants and
is consistent with the case-study mix.

---

## 3. T1.4 — Bootstrap AUROC stability (1000 bootstraps × 4 models)

**Source**: `results/tier1_bootstrap/auroc_distributions.csv` (4000 rows = 4 models × 1000 bootstraps), `summary.json`.

| Feature | Point AUROC | Mean | 95% CI | Std |
|---|---:|---:|---|---:|
| ΔD_cos vec (32-d) | 0.8437 | 0.8436 | [0.8331, 0.8533] | 0.0050 |
| ΔD_jsd vec (32-d) | 0.8225 | 0.8226 | [0.8123, 0.8324] | 0.0051 |
| Evo 2 ΔLL (1-d) | 0.7514 | 0.7512 | [0.7390, 0.7622] | 0.0061 |
| Ensemble cos+ΔLL (33-d) | 0.8607 | 0.8605 | [0.8513, 0.8696] | 0.0047 |

Per-model std ∈ [0.0047, 0.0061] — all within the [0.005, 0.020] expected
range for 8008-variant bootstraps. The 95% CIs are tight enough to justify the
+0.017 ensemble gain over ΔD alone (DeLong p = 3.6e-15) reported in
[`phase3_variant_pathogenicity.md`](phase3_variant_pathogenicity.md) §3.2.

---

## 4. T1.2 — Interpretability baselines (in progress)

Pipeline (running on H200 in tmux session `gdtr` window 18):
1. **delta-H baseline** — per-layer entropy difference of the residual stream
   under ref vs alt context.
2. **Attention rollout** — Abnar & Zuidema (2020) rollout of attention weights
   from the input embedding to the position of interest.
3. **Integrated gradients** — IG from baseline (random sequence) to the
   variant context, attributing to per-position embeddings.
4. **Comparison pipeline** — runs the three baselines on the same 8008
   variants, AUROC by feature; bootstrapped CIs.
5. **Compute cost benchmark** (T2.4) — wall-time, peak VRAM, FLOPs per variant
   for ΔD vs each baseline.

Status as of 2026-04-28: rollout 12/85 min, IG / compare / cost queued. Files
will land in `results/tier1_baselines/` and `results/tier2_compute/`. They are
**not** committed in this repo reorganization to avoid touching paths the
running chain script writes to.

---

## 5. Files

| Tier-1 task | Result dir | Script |
|---|---|---|
| T1.1 | `results/tier1_per_layer/` | `scripts/40_t11_per_layer_ablation.py` |
| T1.2 (running) | `results/tier1_baselines/` (pending) | `scripts/45_t12_delta_h.py`, `46_t12_rollout.py`, `47_t12_ig.py`, `48_t12_compare_pipeline.py`, `44_t12_smoke.py` |
| T1.3 | `results/tier1_case_studies/` | `scripts/44_t13_case_studies.py` |
| T1.4 | `results/tier1_bootstrap/` | `scripts/41_t14_bootstrap.py` |

Verification: `results/_verification/cpu_bundle_verification.md` (re-derives
T1.1, T1.4, T2.2, T2.3 with `|Δ| < 0.001` and confirms bit-for-bit
reproducibility of T2.2 with seed=42).

---

**Document version**: 2026-04-28 (new — Tier 1 was not part of the legacy `PHASE1_FINDINGS.md`).
