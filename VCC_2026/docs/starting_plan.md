<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › [docs](README.md) › **Starting plan**</sub>

# Starting plan: metric-first baselines → statistical transfer → learned models

> [!NOTE]
> Condensed English version of the Korean plan [`ko/02_start_plan_2026-08-22_ko.md`](ko/02_start_plan_2026-08-22_ko.md) (22 Aug 2026). Its basis: the 2026 `cell-eval2` metric specification (`configs/vcc2026.yaml`, rule version 3), Arc's 2025 evaluation and wrap-up posts, and the team's survey of multi-context CRISPRi data. The plan's reference values (replicate ranges, noise floors) are quoted as the plan states them.

## 0. The one correction to the original idea

The proposed order was right: analyse the metrics → build a complete baseline → find which metric matters → compare statistical against learned methods. One step was reframed:

- ❌ "One of the six metrics matters most, so optimise it first."
- ✅ "The six scaled metrics carry **equal weight**, so find which metric is the current **bottleneck**, which is already near its ceiling, and which model changes move which metric."

Metric analysis is a way to diagnose **which biological failure mode** the method has, not to find a loophole.

## 1. From three 2025 metrics to six 2026 metrics

| 2025 | 2026 | In plain words |
|---|---|---|
| PDS (L1 distance) | `pds_cosine` | Still "tell perturbations apart", but on the *direction* of the change, so inflating magnitudes no longer helps |
| MAE | `expr_mse_unbiased_capped_norm` | Whole-profile error with sampling noise removed (jackknife) and normalised by how far the real perturbation moved; clipped to [0, 1] |
| DES (one score) | four DE metrics | "Did you get the DE genes?" split into *which* genes, *direction*, *confidence ranking* and *magnitude* |
| mean-baseline normalisation | baseline = 0, replicate = 1 | Distance from an uninformative prediction to experimental reproducibility |

The four DE metrics, all built on per-perturbation Wilcoxon tests:

| Metric | Question | Note from the plan |
|---|---|---|
| Direction fidelity + yield | Do predicted responders move the right way, and are enough of them called? | Narrow baseline–replicate span, so small raw gains move the scaled score a lot; yield blocks "call three safe genes" gaming |
| Direction reach | Going down the confidence ranking, how long does direction accuracy stay ≥ 90 %? | Replicates already reach ~0.96–0.98, so little headroom |
| Significant-gene Jaccard | Overlap of predicted and true significant-gene sets | Replicates only reach raw ~0.38–0.42, so scaled scores can exceed 2.5: the most headroom, but gaming the DE count harms the other axes |
| LFC NMAE | Are the fold-change magnitudes right? | A sign-only prediction is not enough |

In short, 2025's DES becomes four questions: *which* genes (Jaccard), *which direction* (fidelity), *how confidently* (reach) and *how much* (LFC NMAE). The metrics police each other: PDS-only, sign-only, mean-only or DE-set-only strategies are each caught by another axis.

## 2. Analysis quantities to track

- **Gap to replicate** per scaled metric: $\mathrm{Gap}_m = \max(0, 1 - s_m)$. The largest gaps show where the method fails biologically.
- Also weigh **headroom**, **split / repeat noise** and **engineering cost**: roughly, priority ≈ gap × attainable gain ÷ cost.
- **Noise floors** quoted from the official reference-split analysis. Differences below these are not evidence: PDS 0.5–1.3 %, expression 0.8–2.2 %, fidelity 1.5–4.8 %, reach 0.2–0.9 %, Jaccard 0.8–2.2 %, LFC 0.7–2.7 %.
- Per experiment, store: experiment id, model, context, metric, raw score, baseline, replicate anchor, scaled score, overall, seed and git commit. Also record the number of training contexts, targets, cells, target overlap, context similarity, the distance to the nearest training context, and model size and time.

## 3. Phases

| Phase | Goal | Content |
|---|---|---|
| 0 | Reproduce the evaluator | Pin `cell-eval2` and `vcc2026.yaml`; run all six metrics locally; store raw, anchors, scaled and overall per context |
| 1 | Metric sanity tests | Deliberately odd predictions: A no-effect, B all-perturbation mean, C global gene effect, D magnitude sweep α ∈ {0.25, 0.5, 1, 1.5, 2, 4}, E right sign / wrong magnitude, F right DE set / flipped sign, G only the top DE genes, H too many DE genes. Output: a `metric_behavior_matrix.csv` |
| 2 | Shadow benchmark | Rebuild the 2026 structure from public data with truth: GSE281048 leave-one-line-out (BxPC3, A549, MCF7 folds) plus stronger OOD holdouts (VCC 2025 H1, HepG2, Jurkat); select by $\mathrm{Mean}(S_c) - \lambda\,\mathrm{Std}(S_c)$ and track the worst context |
| 3 | Complete statistical ladder | B0 control · B1 global perturbation mean · B2 gene-specific global effect · B3 nearest context · B4 similarity-weighted (TransPert-like) · B5 gene-specific weighting (pathway, target baseline, DepMap), all without neural networks |
| 4 | Statistical vs learned | Same data, split, generator and evaluator: L1 ridge, L2 MLP (context PCA + ESM-2 + target baseline), L3 statistical + neural residual $\Delta_{\mathrm{pred}} = \Delta_{\mathrm{stat}} + f_\theta(z_c, z_g, \Delta_{\mathrm{stat}})$; a 7-row ablation over {statistical prior, context encoder, ESM-2}, read per metric |
| 5 | Pseudobulk vs single cell | The submission needs 400 cells per target, so compare generators with the *same* mean effect: G0 control-cell bootstrap with multiplicative effect, G1 library-size-preserving multinomial, G2 negative binomial, G3 conditional flow (the code later settled on G0 multinomial, G1 Poisson, G2 negative binomial). DE metrics use single-cell Wilcoxon tests, so variance and zero fraction matter even when means match |

A diagnostic table maps a low metric to its likely cause: PDS → gene-specific effects (ESM-2, target-specific Δ); expression MSE → magnitude calibration; fidelity → sign / recall (pathway or context conditioning); reach → confidence ranking; Jaccard → DE-count calibration and distribution; LFC NMAE → effect magnitude.

## 4. Research questions and decision gates

- **Q1** How far does cell-agnostic statistical transfer go (B2–B5)? **Q2** What does a neural model add, metric by metric? **Q3** Is a context encoder needed (raw cosine vs PCA vs pretrained encoder)? **Q4** Do gene embeddings help on held-out genes (one-hot vs ESM-2 vs + STRING vs + DepMap)? **Q5** Does single-cell distribution modelling matter for the metrics?
- **Gates:** (1) a strong B4/B5 → go hybrid-residual; a weak one → improve the context representation first. (2) ESM-2 helps held-out genes → multi-prior gene encoder; otherwise prioritise perturbation data coverage. (3) Good pseudobulk but poor DE metrics → generator problem. (4) Scores collapse with OOD distance → better context encoder / similarity model. (5) Neural models do not beat statistics consistently → **do not scale up**, the biggest lesson from 2025.
- **Adopt a new model only if:** local shadow-CV overall improves; the worst context improves or holds; it does not raise one metric by sacrificing three; the effect reproduces on at least two held-out contexts; and the gain exceeds the metric noise floor.
- **Scaling questions:** does performance track the number of *cells* or of *contexts* (same total cells, 2 contexts vs 8)? Is gene coverage (X-Atlas-like) or context coverage (GSE281048-like) the bottleneck?

## 5. First two weeks and first figures

Day 1–2 evaluator and sanity predictions → Day 3 metric-behaviour report → Day 4–5 shadow datasets → Day 6–8 B0–B5 → Day 9–11 ridge, MLP + ESM-2, statistical + residual → Day 12 ablation → Day 13 generator comparison → Day 14 decide statistical-first, hybrid, or fully neural. First figures: the baseline ladder, a method × metric heatmap, context robustness, statistical vs neural per metric, OOD distance vs score, and number of contexts vs score.

**One line:** understand the metrics → establish the statistical ceiling → prove what neural models add → only then add complexity.

## What happened next

The frozen-baseline code in this project implements Phase 0 (the exact `cell-eval2` wrapper), the Phase 2 shadow benchmarks and the Phase 3 ladder with separate generators. Its results are in [E1](../experiments/E1_jiang24_bxpc3_proxy/README.md)–[E3](../experiments/E3_replogle_only_strict/README.md). As the 2025 lesson suggested, the statistical transfer rung proved hard to beat.
