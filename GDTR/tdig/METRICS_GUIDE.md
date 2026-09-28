# TDiG Metrics — How to Obtain Each One

This guide explains how to access or recompute each of the **17 settling-depth
cells** plus all downstream metrics, at every derivation stage. Use this when:
- You want a specific Cohen d / AUROC / retention value → read a CSV (Stage 6)
- You want per-token settling integers for a custom analysis → read tier1 parquet (Stage 4)
- You want to redefine cells (new γ, new α/β) → use tier2 scalars (Stage 2)
- You want to compute brand-new metrics → use tier3 raw hidden states (Stage 1, HF)

---

## The 17 settling cells

| Cell ID | Formula at layer ℓ | γ anchor layer | Reference vector |
|---|---|---|---|
| **M1_dir_refA** | `c = first ℓ where (1 − cos(h_ℓ, h_29)) ≤ γ` for W=3 consecutive layers | L=28 | h_29 raw |
| **M1_dir_refB** | same, with RMSnorm(h_ℓ) vs RMSnorm(h_29) | L=28 | RMSnorm(h_29) |
| **M1_dir_refC** | same, with h_ℓ vs h_norm | L=28 | h_norm (post-final-norm output) |
| **M2_mag_refA** | `c = first ℓ where ‖‖h_ℓ‖/‖h_29‖ − 1‖ ≤ γ`, W=3 | L=28 | h_29 raw |
| **M2_mag_refB_diag** | same, RMSnorm-based; γ = q99 of distribution | L=28 | RMSnorm(h_29) |
| **M2_mag_refC_diag** | same, h_norm-based; γ = q99 | L=L-1 | h_norm |
| **M3_geo_a0.0_b1.0** | `g_ℓ = β·κ_z(ℓ)`, settling-persistence over standardized curvature only | L=26 | reference-free |
| **M3_geo_a0.5_b1.0** | `g_ℓ = 0.5·v_z + 1.0·κ_z` | L=26 | reference-free |
| **M3_geo_a1.0_b1.0** | `g_ℓ = v_z + κ_z` (symmetric) | L=26 | reference-free |
| **M3_geo_a1.0_b0.5** | `g_ℓ = v_z + 0.5·κ_z` | L=26 | reference-free |
| **M3_geo_a1.0_b0.0** | `g_ℓ = v_z` (velocity only) | L=26 | reference-free |
| **M4_set_refA** | `c = first ℓ where √((h_ℓ−h_29)ᵀ Σ_ref⁻¹ (h_ℓ−h_29)) ≤ γ` (monotone direct, no W) | L=28 | h_29 + Σ_ref⁻¹_A |
| **M4_set_refB** | same, RMSnorm(h_29) + Σ_ref⁻¹_B | L=28 | RMSnorm(h_29) + Σ_B |
| **M4_set_refC** | same, h_norm + Σ_ref⁻¹_C | L=28 | h_norm + Σ_C |
| **M5_tau_refA** | `c = first ℓ where remaining_path(ℓ→L*)/‖h_ℓ−h_29‖ ≤ γ`, W=3 | L=27 | h_29 raw |
| **M5_tau_refB** (Option B) | same, with RMSnorm trajectory + RMSnorm res_norm | L=27 | RMSnorm(h_29) |
| **M5_tau_refC** | same, with h_norm res_norm | L=27 | h_norm |

Where:
- `v(ℓ, t) = ‖h_{ℓ+1} − h_ℓ‖ / ‖h_ℓ‖` (relative velocity)
- `κ(ℓ, t) = 1 − cos(h_{ℓ+1}−h_ℓ, h_ℓ−h_{ℓ−1})` (curvature)
- `v_z, κ_z` = population-z-scored (using `gamma_calibration_v2.json::_geo_pop_stats`)
- W = 3 persistence (3 consecutive layers must satisfy threshold)
- γ = q70 of the metric distribution at the anchor layer, computed from 100 chr22 sanity sequences

Forward script `15_chr22_forward.py::compute_v2_settling()` is authoritative.

---

## 6 derivation stages — where each lives

| Stage | What | Storage | Size |
|---|---|---|---|
| 1 | Raw hidden states h_ℓ ∈ ℝ⁴⁰⁹⁶ | 🤗 HF `*_tier3_raw.h5` | 108 GB |
| 2 | Per-(window, layer, token) scalars (cos, norm, step, D_Mset) | 🤗 HF `*_tier2_scalars.h5` | 1.2 GB |
| 3 | γ thresholds + Σ_ref⁻¹ | 🐙 GitHub `data_cache_minimal_archive/.../population_stats/` | ~200 MB |
| 4 | **Per-token settling integers c(t) for all 17 cells** | 🐙 GitHub `data_cache_minimal_archive/{chr22,chr17}_tier1.parquet` | 883 MB |
| 5 | Variant ΔH per layer (scalars) | 🐙 GitHub `data_cache_minimal_archive/variant_scalars.parquet` | 11 MB |
| 6 | Derived aggregates (Cohen d, AUROC, retention, CIs) | 🐙 GitHub `results/*.csv` | ~250 MB |

---

## Recipes — how to get specific values

### 🟢 Recipe A: "Give me the chr22 splice_donor vs intron Cohen d for M3_geo_a0.5_b1.0"

**Pre-computed (instant, 1 line):**
```python
import pandas as pd
df = pd.read_csv("results/splice_vs_intron.csv")
print(df[df.cell == "M3_geo_a0.5_b1.0"].cohens_d_donor_minus_intron.iloc[0])
# → -0.8886
```

### 🟢 Recipe B: "Give me 95% CI on that d value"

```python
df = pd.read_csv("results/bootstrap_chr22_ci/bootstrap_d_ci.csv")
row = df[df.cell == "M3_geo_a0.5_b1.0"].iloc[0]
print(f"{row.mean_d:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]")
# → -0.801 [-0.827, -0.775]
```

### 🟢 Recipe C: "Same d but on chr17"

```python
df17 = pd.read_csv("results/chr17_replication/splice_vs_intron_chr17.csv")
df17[df17.cell == "M3_geo_a0.5_b1.0"].cohen_d.iloc[0]   # → -0.8019
```

### 🟢 Recipe D: "Per-token settling integer of M5_tau_refB at window_idx=42, token=130 on chr22"

```python
import pandas as pd
df = pd.read_parquet("data_cache_minimal_archive/chr22_tier1.parquet")
row = df[df.window_idx == 42].iloc[0]
cell_arr = row["M5_tau_refB"]   # python list, length T
print(cell_arr[130])   # → int in [-1, 32]
```

### 🟡 Recipe E: "Recompute M1_dir_refA with a different γ (q50 instead of q70)"

```python
import h5py, json, numpy as np
import sys; sys.path.insert(0, "scripts")
from importlib import import_module

# 1. Load tier2 scalars
with h5py.File("chr22_tier2_scalars.h5", "r") as f:
    cos_A = f["cos_refA"][:].astype(np.float32)   # (100, 32, 6000)
# 2. Load γ q50 (already calibrated)
gamma_v2 = json.loads(open("data_cache_minimal_archive/population_stats/gamma_calibration_v2.json").read())
gamma = gamma_v2["D_dir_refA_at_28"]["q50"]  # q50 instead of q70
# 3. Apply settling_persistence (see scripts/15_chr22_forward.py)
from scripts._helpers import settling_persistence  # or copy function from 15_chr22_forward.py
D_dir = 1.0 - cos_A
cells_q50 = np.zeros((100, 6000), dtype=np.int32)
for i in range(100):
    cells_q50[i] = settling_persistence(D_dir[i], gamma, W=3, max_layer=29)
```

This is exactly what `scripts/20_gamma_ablation.py` does — see that script for the full
loop including ALL 17 cells × {q50, q70, q90}.

### 🟡 Recipe F: "Variant Δc = c(alt) − c(ref) for M5_tau_refC"

```python
# Pre-computed:
df = pd.read_csv("results/variant_settling_cells/variant_settling_per_cell.csv")
sub = df[df.cell == "M5_tau_refC"]
print(sub[["variant_idx", "category", "c_ref", "c_alt", "delta_c"]].head())
```

### 🟡 Recipe G: "VUS reclassification probabilities (the 0.949 AUROC classifier)"

```python
df = pd.read_csv("results/vus_reclassification/vus_predictions.csv")
# Columns: chrom, pos, ref, alt, gene, stars, consequence,
#          prob_PLP, prediction, dh_baseline_high, agreement_with_dh_baseline
df[df.prob_PLP > 0.9].head()  # high-confidence VUS → likely pathogenic
```

### 🔴 Recipe H: "Compute brand-new metric (e.g. layer-asymmetric tortuosity) from raw h_ℓ"

```bash
# 1. Download tier3 from HuggingFace (108 GB total)
huggingface-cli download darejinn/TDiG-evo2-hidden-states --repo-type dataset

# 2. Load + compute in Python
python -c "
import h5py, numpy as np
with h5py.File('chr22_tier3_raw.h5', 'r') as f:
    h = f['raw_h_ell_rmsnormed'][:]   # (100, 32, 600, 4096) fp16
    # your new metric ...
"
```

### 🔴 Recipe I: "Re-run forward pass (108 GB regenerate, only if both clouds vanish)"

```bash
# Need: 1× H200 GPU (e.g. DigitalOcean droplet), Evo 2 7B model weights
huggingface-cli download arcinstitute/evo2_7b_base
python scripts/10b_calibrate_v2.py       # ~5 min: γ_v2 + Σ_ref⁻¹
python scripts/15_chr22_forward.py       # ~3 h: chr22 tier1/2/3
python scripts/16_chr17_forward.py       # ~7 h: chr17
python scripts/18_variant_forward.py     # ~1.5 h: ClinVar variants
# Total ~12 h, ~$60 H200 droplet cost
```

---

## Cross-reference table — which CSV has which metric

| Metric value | File | Columns of interest |
|---|---|---|
| chr22 cell distribution (mean/median/std/never%) | `results/per_cell_summary.csv` | cell, mean, median, std, never% |
| chr22 splice vs intron Cohen d | `results/splice_vs_intron.csv` | cell, n_donor, n_intron, cohens_d_donor_minus_intron, mannwhitney_p |
| chr22 canonical vs non-canonical | `results/canonical_vs_noncanonical.csv` | cell, ..., cohens_d |
| chr22 per-context summary | `results/per_context_distributions.csv` | context, cell, n, mean, median, std |
| chr17 cell distribution | `results/chr17_replication/per_cell_summary_chr17.csv` | same as chr22 version |
| chr17 splice vs intron | `results/chr17_replication/splice_vs_intron_chr17.csv` | same |
| chr22→chr17 retention | `results/chr17_replication/retention_table.csv` | cell, chr22_d, chr17_d, retention_pct, sign_preserved |
| 200-iter bootstrap CI on chr22 | `results/bootstrap_chr22_ci/bootstrap_d_ci.csv` | cell, n_iter, mean_d, ci_low, ci_high, std |
| γ q50/q70/q90 ablation | `results/gamma_ablation/cell_d_under_gamma.csv` | cell, gamma, cohen_d, range_min, range_max, never_settled_pct |
| 7×7 context pair separation | `results/context_separation/pairwise_d.csv` | cell, context_i, context_j, d_i_minus_j |
| Best cell per context pair | `results/context_separation/best_cell_per_pair.csv` | context_i, context_j, best_cell, d |
| Per-variant Δc per cell | `results/variant_settling_cells/variant_settling_per_cell.csv` | variant_idx, gene, category, cell, c_ref, c_alt, delta_c |
| Variant cell AUROC P_LP vs B_LB | `results/variant_settling_cells/cell_auroc.csv` | cell, AUROC, n_PLP, n_BLB |
| Variant per-consequence × cell | `results/variant_settling_cells/per_consequence_cell.csv` | cell, consequence, AUROC |
| Per-layer ΔH baseline AUROC | `results/variant_analysis_scalars/per_layer_auroc.csv` | layer, feature, AUROC |
| Per-gene AUROC | `results/variant_analysis_scalars/per_gene_auroc.csv` | gene, feature, best_layer, best_AUROC |
| Per-consequence AUROC | `results/variant_per_consequence/per_consequence_auroc.csv` | consequence, layer, AUROC |
| VUS predictions (n=2902) | `results/vus_reclassification/vus_predictions.csv` | variant cols + prob_PLP, prediction |
| VUS classifier metrics | `results/vus_reclassification/classifier_metrics.csv` | classifier, CV_AUROC, CV_AUPRC |
| VUS feature importance | `results/vus_feature_importance/gbm_feature_importance.csv` | feature_type, layer, importance |
| Per-layer-only VUS AUROC | `results/vus_feature_importance/per_layer_only_auroc.csv` | layer, AUROC |
| 7-way multitask per layer | `results/multitask_per_position/per_layer_metrics.csv` | layer, class_id, class_name, f1, AUROC_ovr |
| Intron outlier enrichment | `results/intron_outlier/enrichment_summary.csv` | cell, outlier_near_splice_frac, random_near_splice_frac, enrichment_fold |
| Cryptic synonymous candidates | `results/cryptic_synonymous/cryptic_candidates.csv` | per-variant shape + splice_like_score |
| Random-alt control (GPU validation) | `results/random_alt_control/comparison_summary.csv` | category, layer, real_mean, rand_mean, ratio, p_mannwhitney |
| L29 SVD spectra | `results/L29_svd/singular_values.csv` + `alignment_metrics.json` | transition, k, singular_value; cond, R², rotation_score |
| HyenaDNA cross-arch | `results/hyenadna_crossarch/comparison_vs_evo2.json` | model, cohen_d, sign_agreement_with_evo2, ratio_magnitude_vs_evo2 |
| Activation patching | `results/activation_patching/patching_results.csv` | variant_idx, category, layer, delta_h, delta_cos |
| Metric ↔ PCA correlation | `results/analysis_BD/metric_pca_corr.csv` | layer, metric, PC, spearman_r |
| PC-alone AUROC | `results/analysis_T123/T1_pc_alone_auroc.csv` | layer, PC, pair, AUROC |
| Per-context PC distribution | `results/analysis_T123/T2_pc_context_distribution.csv` | layer, PC, context, mean, std, ci_low, ci_high |
| PC ↔ position confounder | `results/analysis_T123/T3_pc_position_correlation.csv` | layer, PC, spearman_r |

## Script ↔ output map

| Script | Produces |
|---|---|
| `scripts/10b_calibrate_v2.py` | γ_v2 + Σ_ref⁻¹ in population_stats/ |
| `scripts/15_chr22_forward.py` | chr22 tier1/2/3 (GPU forward) |
| `scripts/16_chr17_forward.py` | chr17 tier1/2/3 |
| `scripts/18_variant_forward.py` | variant_h_ell_ref/alt.h5 + variant_scalars.parquet |
| `scripts/13_analyze_chr22_v2.py` | results/{per_cell_summary, splice_vs_intron, canonical_vs_noncanonical, per_context_distributions}.csv |
| `scripts/14e_probing_and_metric_pca.py` | results/analysis_BD/ — per-layer probing + D correlation |
| `scripts/14f_pc_biological_meaning.py` | results/analysis_T123/ — T1+T2+T3 |
| `scripts/19_variant_analysis_scalars.py` | results/variant_analysis_scalars/ |
| `scripts/20_gamma_ablation.py` | results/gamma_ablation/ |
| `scripts/21_variant_per_consequence.py` | results/variant_per_consequence/ |
| `scripts/22_chr17_replication.py` | results/chr17_replication/ |
| `scripts/23_bootstrap_chr22_ci.py` | results/bootstrap_chr22_ci/ |
| `scripts/24_variant_settling_cells.py` | results/variant_settling_cells/ |
| `scripts/25_random_alt_control.py` | results/random_alt_control/ (GPU) |
| `scripts/26_variant_clustering.py` | results/variant_clustering/ |
| `scripts/27_vus_reclassification.py` | results/vus_reclassification/ |
| `scripts/28_cryptic_synonymous_splice.py` | results/cryptic_synonymous/ |
| `scripts/29_context_separation_matrix.py` | results/context_separation/ |
| `scripts/30_intron_outlier_discovery.py` | results/intron_outlier/ |
| `scripts/31_L29_svd_mechanism.py` | results/L29_svd/ |
| `scripts/32_activation_patching.py` | results/activation_patching/ (GPU) |
| `scripts/34_per_position_multitask.py` | results/multitask_per_position/ |
| `scripts/35_vus_feature_importance.py` | results/vus_feature_importance/ |
| `scripts/36_hyenadna_crossarch.py` | results/hyenadna_crossarch/ (GPU) |
