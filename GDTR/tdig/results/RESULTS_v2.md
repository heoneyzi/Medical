# chr22 v2 Results Summary (2026-05-24)

End-to-end analysis of TDiG design v2 metric family on Evo 2 7B chr22 forward.

**Forward**: 12,978 windows × 6,000 tokens = 77,868,000 per-position records.
**Wall time**: 179 min on H200 (1.22 win/s).
**Output**: 17 settling cells per token via the v2 metric family with persistence W=3, M4_set inline, M5 Option B locked, M3 c_geo with 5 α/β cells.

---

## 1. Headline findings

### 1.1 Splice donor vs intron (187,236 vs 41,477,463 positions, all p < 1e-300)

The strongest discriminators by Cohen's d:

| Cell | Cohen's d | Sign interpretation |
|---|---|---|
| **M3_geo_a0.0_b1.0** (curvature-only) | **−1.053** | splice donor's trajectory *stops curving* ~7 layers earlier |
| **M3_geo_a0.5_b1.0** | **−0.889** | curvature-weighted Def 1 |
| **M3_geo_a1.0_b1.0** (symmetric) | **−0.644** | M3 default |
| **M5_tau_refB** (Option B locked) | **−0.437** | path tortuosity Def 2 (RMSNormed trajectory) |
| **M3_geo_a1.0_b0.5** | −0.372 | velocity-weighted Def 1 |
| **M1_dir_refA** | **+0.302** | **Def 2: donor settles LATER toward h_29** |
| M2_mag_refA | +0.174 | residual accumulation weak |
| M5_tau_refA | −0.155 | path geometry weak (norm-growth artifact) |
| M4_set_refA | −0.155 | reference-whitened weak |
| M1_dir_refB | +0.146 | Def 2 DTR-style |
| M1_dir_refC | −0.099 | Def 3 weak (gDTR baseline -0.43 NOT matched — persistence + γ differ) |
| M3_geo_a1.0_b0.0 (velocity-only) | +0.154 | inverse direction vs curvature |
| Degenerate (d ≈ 0): M2_mag_refB/C, M4_set_refB/C, M5_tau_refC | — | as predicted |

### 1.2 Empirical confirmation of "settling 3-definition split"

The **same biological question** (donor vs intron) gives **opposite-sign d** depending on settling definition:

- **Def 1 (trajectory stops)** — M3_geo: donor settles **earlier** (negative d, up to −1.05)
- **Def 2 (toward h_29)** — M1_dir_refA: donor settles **later** (positive d, +0.30)
- **Def 3 (toward h_norm)** — M1_dir_refC: donor settles slightly earlier (−0.10), gDTR-consistent in sign

This validates the iter-13 design decision to report the three definitions separately — they are genuinely measuring **different concepts**.

### 1.3 Canonical vs non-canonical donor (193,148 vs 5,808)

For the first time at full scale (Pre-V was n=2 non-canonical, now n=5,808):

| Cell | Cohen's d (canon − noncanon) | p |
|---|---|---|
| M3_geo_a0.5_b1.0 | **−0.325** | 7e-64 |
| M3_geo_a1.0_b1.0 | **−0.230** | 1e-51 |
| M3_geo_a1.0_b0.0 | **+0.218** | 7e-47 |
| M1_dir_refB | +0.187 | 7e-45 |
| M1_dir_refA | +0.184 | 2e-43 |
| M3_geo_a0.0_b1.0 | −0.183 | 1e-24 |
| M4_set_refA | −0.168 | 2e-36 |

Multiple cells distinguish canonical from non-canonical at moderate effect size.

---

## 2. Figure family (in `figures/`)

| Figure | What it shows |
|---|---|
| **fig_v7_context_heatmap** | 17 cells × 7 contexts heatmap (Cohen's d vs intron). The densest single biological-mapping view. |
| **fig_summary_splice_d** | Per-cell horizontal bar chart of donor−intron Cohen's d. Sign mixed (Def 1 negative, Def 2 positive). |
| **fig_summary_canonical_d** | Same structure for canonical−non-canonical. |
| **fig_v1_trajectory_pca** | (LEFT) per-layer-mean-centered PCA on RMSNormed h_ell, all (window, layer, token) colored by context; (RIGHT) example trajectories per context with start ○ and end × markers. |
| **fig_v2_velocity_heatmap** | log10(velocity) per layer × context. Shows L29 spike (~10⁵) and L30 collapse (~10⁻¹²) — Evo 2 idle-block + rotation signature. |
| **fig_v3_curvature_heatmap** | curvature κ_ℓ per layer × context. L1, L8, L15 peaks; L27 trough (settling region). |
| **fig_v4_cumulative_path** | (LEFT) cumulative relative velocity Σv_ℓ — flat until L29 jump; (RIGHT) raw cumulative ‖Δh‖ on log scale showing growth 1 → 10¹². |
| **fig_v5_2d_signature** | (LEFT) hexbin density of (c_dir_refA, c_geo_a0.0_b1.0) — Def 2 vs Def 1; (RIGHT) per-context KDE of Δc — bimodal distribution at Δc ≈ -25 (curvature settles ~25 layers before direction) vs Δc ≈ 0. |
| **fig_v9_tortuosity_profile** | per-context mean τ(ℓ) on log scale, ℓ ∈ [0, 29). All contexts ≥ 1; M5 tortuosity drops toward 1 as ℓ → 29. |

---

## 3. Key biological observations

1. **Splice sites (donor + acceptor) trajectory settles EARLIEST** in curvature space (M3_geo_β=1 cells, d=-1.06 / -1.29). This means the local trajectory direction stabilizes well before the magnitude-dominated rotation event.

2. **Coding exon also shows strong M3 negative d (-1.29)** — coding regions stabilize early in trajectory curvature too.

3. **Untranslated regions (5'UTR, 3'UTR) show weakest signals**, intermediate between coding/splice and intron.

4. **Intergenic regions distinct from intron** in some cells (M3 curvature-only +0.19 in canonical-vs-noncanonical-like comparison).

5. **Layer 15 is the curvature peak** — across all contexts, this is where trajectory direction changes most abruptly. Phase 1 follow-up (32-layer tuned-lens) had also flagged L11–L17 as the hardest-to-linearly-recover band.

6. **Layer 27 is the curvature trough** — just before the canonical interpretively-distinct tap L*=29. Trajectory smooths in the last layers before the rotation.

---

## 4. Known limitations + caveats

1. **Degenerate cells (predicted)**: M2_refB/C, M4_set_refB/C all collapse to settling=0 due to scale mismatch.
2. **M1_dir_refC γ too loose (1.018)** — produces median c=4 for all tokens. Post-hoc γ recalibration at deeper layer needed for Def 3 cell.
3. **M5_tau_refA median 0** — Evo 2's monotonically growing residual magnitude causes τ(0) ≈ 1 trivially. Predicted geometric artifact.
4. **gDTR baseline NOT matched** — M1_dir_refC d=−0.099 vs gDTR −0.43. Expected due to v2 persistence W=3 + different γ. Documented; v1-compatible γ would reproduce gDTR exactly but loses v2 robustness improvements.
5. **Ledoit-Wolf shrinkage hit λ=0.95 clip** — M4_set Σ_ref ≈ scaled identity. M4_set effectively reduces to scaled L2 (still useful, but loses ideal "Mahalanobis-whitened" property). v3 future work: better shrinkage estimator.
6. **All comparisons at fixed γ_v2 q70**. γ ablation (q50/q90) computed but not yet plotted/analyzed.

---

## 5. Files in `results/`

```
results/
├── RESULTS_v2.md                    this document
├── per_cell_summary.csv             17 cells × {mean, median, std, never-settled%}
├── splice_vs_intron.csv             17 cells × {n, means, Cohen's d, p}
├── canonical_vs_noncanonical.csv    17 cells × same structure
├── per_context_distributions.csv    17 cells × 7 contexts × {n, mean, median, std}
└── figures/
    ├── fig_v1_trajectory_pca.{png,pdf}       headline: trajectory shape per context
    ├── fig_v2_velocity_heatmap.{png,pdf}     diagnostic: log10 velocity per layer
    ├── fig_v3_curvature_heatmap.{png,pdf}    diagnostic: curvature per layer
    ├── fig_v4_cumulative_path.{png,pdf}      diagnostic: cumulative path
    ├── fig_v5_2d_signature.{png,pdf}          headline: Def 2 vs Def 1 signature
    ├── fig_v7_context_heatmap.{png,pdf}      headline: 17×7 d heatmap
    ├── fig_v9_tortuosity_profile.{png,pdf}   diagnostic: τ(ℓ) per context
    ├── fig_summary_splice_d.{png,pdf}        summary bar chart
    └── fig_summary_canonical_d.{png,pdf}     summary bar chart
```

Generated by `scripts/13_analyze_chr22_v2.py` + `scripts/14_visualize_chr22_v2.py`.

---

## 6. Next steps (chr17, variants in progress)

- chr17 forward: running in tmux 'tdig' window 0 (~6-7h, single-process rate slowed by GPU sharing with variants)
- variant forward: running in 'variants' window (~3h)
- Once both complete: replicate this analysis on chr17, then variant-level analysis (P/LP vs B/LB by gene)

Future work:
- γ ablation (q50 / q70 / q90) per cell — data already saved in `gamma_calibration_v2.json`
- M4_set with full-rank Σ (try sklearn LedoitWolf or alternative shrinkage)
- Cross-architecture v2 (HyenaDNA-large, NT-v2, DNABERT-2) — gDTR Phase 4 carry-over
- ClinVar variant pathogenicity classification using v2 features
