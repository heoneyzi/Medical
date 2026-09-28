<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../../README.md) › [GDTR](../../../README.md) › [Line A](../../README.md) › **E7 · Robustness**</sub>

# 🔬 E7 · Tier 2 — Sensitivity, failure modes, compute cost, Q2 overlaps

> **Question —** Do the variant results depend on arbitrary modelling choices, where does the classifier fail, what does the lens cost to run, and do E5's Q2 regions overlap known functional annotations?

| | |
|---|---|
| **Status** | ✅ done (2026-04-28) |
| **Model / data** | E3's 8,008 ClinVar SNVs; E5's 5,090 Q2 regions on chr22; GTEx v8 eQTL, GWAS Catalog, ENCODE cCRE-ELS (chr22) |
| **Compute** | CPU, plus a 100-variant GPU cost benchmark on one H200 |
| **Headline** | 9 classifier settings → AUROC 0.8421–0.8438 (range 0.0017) |

## Results

| Analysis | Result | File |
|---|---|---|
| Hyper-parameter grid (L1 / L2 / elastic-net × C) | AUROC 0.8421–0.8438 | [`tier2_sensitivity/hp_grid.csv`](../../results/tier2_sensitivity/hp_grid.csv) |
| Operating point | Youden threshold 0.476: sensitivity 0.722, specificity 0.904 | [`tier2_failure/youden_threshold.json`](../../results/tier2_failure/youden_threshold.json) |
| Where it fails | false-negative rate highest for PALB2 (0.42) and BRCA1 (0.41); 0.46 when CADD disagrees vs 0.026 when it agrees | [`tier2_failure/failure_breakdown.csv`](../../results/tier2_failure/failure_breakdown.csv) |
| Cost per variant | ΔD_cos 0.54 s · ‖Δh‖₂ 0.52 s · rollout 0.52 s · integrated gradients (8 steps) 1.75 s; peak VRAM 16–20 GB | [`tier2_compute/cost_benchmark.csv`](../../results/tier2_compute/cost_benchmark.csv) |
| Q2 ∩ annotations (bp fold enrichment) | cCRE-ELS **1.90×** · eQTL 1.62× (*p* = 5.4 × 10⁻⁵⁶) · GWAS 1.50× (*p* = 1.6 × 10⁻⁷); each above all 100 shuffles | [`tier2_q2_functional/summary.json`](../../results/tier2_q2_functional/summary.json) |

<p align="center"><img src="../../results/figures_v2/S6_failure_analysis.png" width="620" alt="Failure analysis of the variant classifier by gene and stratum"></p>
<p align="center"><sub>Failure analysis — <code>results/figures_v2/S6_failure_analysis.png</code>.</sub></p>

## Takeaway

- The variant AUROC is not an artefact of classifier tuning, and the lens is as cheap as other single-pass attribution methods.
- Errors concentrate where an orthogonal predictor (CADD) also disagrees with ClinVar — hard cases, not random noise.
- The Q2 overlap enrichments are real at base-pair level, but E10's functional positive control reverses the direction expected for "functional" DNA, so they are not read as functional discovery.

## Files

| File | What it is |
|---|---|
| [`scripts/42_t22_hp_sensitivity.py`](../../scripts/42_t22_hp_sensitivity.py), [`43_t23_failure_analysis.py`](../../scripts/43_t23_failure_analysis.py), [`49_t24_cost.py`](../../scripts/49_t24_cost.py), [`tier2_q2_functional.py`](../../scripts/tier2_q2_functional.py) | the four Tier 2 analyses |
| [`results/tier2_*`](../../results/) | outputs (large BED intersections omitted) |
| [`results/figures_v2/S1_hp_sensitivity.png`](../../results/figures_v2/S1_hp_sensitivity.png) | sensitivity figure |
| [`docs/findings/tier2_extensions.md`](../../docs/findings/tier2_extensions.md) | write-up |

---
<sub>[← E6](../E6_tier1_variant_diagnostics/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [E8 →](../E8_entropy_control/README.md)</sub>
