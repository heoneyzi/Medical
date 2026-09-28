<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › [docs](README.md) › **Experiments & metrics**</sub>

# Two core experiments and a robustness extension: design, conditions, results, metrics

> [!NOTE]
> Condensed English version of the Korean report [`ko/03_experiments_and_metrics_ko.md`](ko/03_experiments_and_metrics_ko.md) (late Aug 2026). The numeric tables are in [E2](../experiments/E2_six_metric_robustness/README.md) and [E3](../experiments/E3_replogle_only_strict/README.md) (as CSVs). All scores come from the public `cell-eval2` implementation run on **external** data with **dataset-local anchors**; they are not challenge-server scores. The sweeps were run after seeing the truth, so they measure sensitivity; they do not pick a new "best" setting.

## Summary

- **What was run.** One pipeline uses frozen foundation models (STATE, STACK, UCE, TranscriptFormer, scPRINT-2) only as **context-similarity calculators**, never as response generators; the response itself is transferred from a statistical effect library. It ran on two zero-shot splits: Jiang24 (a new cell line) and GSE270828 (a new batch), scored with the six-axis 2026 metrics. A follow-up then checked stability across **temperature τ, seed and predicted cell count** with 144 extra GPU runs.
- **Three conclusions.**
  1. **The effect library (statistics), not the embedding (neural network), makes the score.** The jump from no-effect to transfer is large; the differences between raw / PCA and frozen embeddings are small and flip with conditions.
  2. **"New batch" and "new cell line" are different problems.** A held-out batch scores positive Overall (~0.75), while a genuinely new line (BxPC3) scores −0.76 to −2.5, below the trivial baseline. The challenge asks for the latter.
  3. **The metric you look at picks the winner.** Jiang24 Overall is effectively one axis (Jaccard); GSE270828's ranking follows PDS. Removing one axis changes the winner, so "Overall #1" is often one axis's shadow.

## 1. Glossary

- **Zero-shot context transfer.** Predict each knockdown's response in an unseen context from that context's control cells only; truth is used only for scoring.
- **Effect library.** Log-fold changes (perturbed vs control) per target from reference contexts with known responses.
- **Frozen context encoder.** A pretrained model used as-is to embed control cells, only to measure *which source context the holdout resembles*.
- **`gwps_nearest` vs `gwps_weighted`.** Copy the single most similar source's effect, or blend all sources with cosine → softmax(τ) weights. Small τ ≈ nearest; large τ ≈ flat average.
- **Generator.** Turns the predicted mean profile into integer-count cells (multinomial here).
- **Local-anchor scaling.** Per dataset, the generic-response baseline = 0 and the replicate split-half anchor = 1, so negative values and values above 1 are normal.

## 2. Why the experiment was designed this way

Two training-free routes compete: **(a)** large foundation models "understand" the unseen context, or **(b)** simply *moving* the large, already-measured public effect tables is strong. To compare them fairly, the models are **not allowed to generate responses**. Every row shares the same effect library, generator, seed, cell count and scorer; only the embedding space that measures context similarity changes. Any row difference is then attributable to the context representation, as far as the benchmark allows. If each model generated responses its own way, differences would mix representation, generation and library effects and could not be interpreted.

- **Experiment 1** (method axis): STATE / STACK against raw / PCA, across `no_effect`, `global_mean`, `gwps_direct`, `gwps_nearest`, `gwps_weighted`. Is an embedding better than a statistical baseline, and is nearest or weighted safer?
- **Experiment 2** (representation axis): the same pipeline with UCE, TranscriptFormer and scPRINT-2 added, so that five frozen encoders plus raw and PCA are compared under identical `gwps_weighted` conditions.
- **Robustness extension**: no new models. It asks whether the conclusions survive noise in τ, seed and sampling depth, and whether removing a single axis changes the winner.

## 3. Conditions

| | Jiang24 (IFNG, BxPC3 holdout) | GSE270828 (rep3 holdout) |
|---|---|---|
| Split | leave-one-cell-line-out | leave-one-batch-out |
| Sources | A549, HAP1, HT29, K562, MCF7 (five different lines) | rep1, rep2 (two replicate batches of one experiment) |
| Targets | 56 genes | 184 human-accelerated regions (regulatory elements) |
| Encoder-input / evaluation genes | 15,473 / 4,017 | 19,698 / 4,095 (141 unique mapped genes) |
| Held-out truth cells | 5,979 (500 control + 5,479 perturbed) | 17,974 (484 + 17,490) |
| Predicted cells (50 per target) | 2,800 | 9,200 |

Pooled embedding sizes: STATE 1,034 · STACK 1,600 · UCE-4L 1,280 · TranscriptFormer 2,048 · scPRINT-2 424; PCA 5–50 components. scGPT and scFoundation were excluded fail-closed (their manually downloaded checkpoints were not available; nothing was substituted). Target-specific priors (Geneformer deletion, scPRINT-2 GRN, TranscriptFormer-CGE) were excluded on purpose.

Every row follows the same five steps: (1) build a logFC library from source contexts; (2) embed the holdout's **control cells only**; (3) select or weight sources by cosine similarity; (4) apply the same multinomial generator, seed 2026, 50 cells per target, scale 1.0, negative clipping and no controls; (5) score against the same truth and dataset-local scale with `cell-eval2==0.16.0` (`vcc2026`, GPU `gpudge` backend).

**The four zero axes of GSE270828.** Its targets are regulatory elements, so a truth-only support audit finds **0 / 184** targets with LFC-NMAE support. The public evaluator then sets MSE, NMAE, reach and Jaccard to an explicit 0, and its Overall is effectively PDS + fidelity. That is why GSE scores look stable: fewer axes are scored. On Jiang24 the normalised MSE is also 0 for every row.

**Robustness design.** Starting from τ 0.1, seed 2026 and 50 cells, one variable changes at a time: τ ∈ {0.01, 0.03, 0.3, 1.0}; seed ∈ {2024, 2025, 2027}; cells ∈ {25, 100}. That gives 9 conditions × 2 datasets × 8 rows = **144 new GPU rows**, all `ok`, covering **912,000** predicted cells (212,800 Jiang24 + 699,200 GSE270828). The combined analysis table has 192 rows (the 16 reference rows are reused in each of the three sweep groups). Overall recomputation error ≤ 2.22e-16; no non-finite metrics.

## 4. How good are the baselines? (reference setting)

- **Jiang24: every Overall is negative.** Even with frozen-context weighting, transfer stays below the generic baseline on a truly new line. The best row, scPRINT-2 (−0.760), wins only because its Jaccard is *less bad* (−2.36 vs −5 to −10 for the others); it leads on no other axis. STACK has the best fidelity (0.735) but the worst Overall (−1.703), because its Jaccard of −10.11 swamps everything. Raw has the best PDS (0.390) and ranks third overall.
- **GSE270828: every Overall is positive (0.556–0.753).** A new batch is "easy", partly because only two axes are scored. Fidelity sits near 4 for every row, so it sets the level but not the order; the order follows PDS, and PCA wins.
- **Scale of difficulty.** The same pipeline gives about +0.75 on a new batch and −0.76 to −1.7 on a new line. That gap is the difference between "predict another replicate of a known experiment" and "predict a cell line never seen", and the challenge asks for the latter. The two datasets have different anchors and must not be compared on one scale.

## 5. Six questions, answered

1. **How good are frozen models?** Usable, but they do not reliably beat statistical baselines. On Jiang24, scPRINT-2 edges out PCA and raw only through Jaccard, and the lead vanishes under other seeds, τ values or cell counts. On GSE270828 all frozen encoders lose to PCA; in the strict Replogle-only auxiliary run, raw beats every frozen encoder (dual-source stratum: raw −0.145 best). In this pipeline, frozen models did not justify their cost as similarity calculators.
2. **How sensitive is each metric?** Jaccard dominates Jiang24 in both level and ranking (dominant axis in 85/96 results; 98.2 / 93.4 / 163.1 % of Overall variance across the τ / seed / cell sweeps; removing it changes the winner in 11/12 conditions). In GSE270828, fidelity sets the level (96/96) but PDS sets the order (86.3 / 86.7 / 81.3 %; removing PDS changes the winner in 7/12). Per axis: Jaccard is the most volatile and extremely sensitive to cell count; PDS flips sign often (for example, GSE PCA falls from 0.423 to −0.762 at 100 cells); fidelity is large but barely separates models; NMAE depends strongly on sampling depth; MSE is uninformative here; reach is small and secondary.
3. **Why do the two datasets differ?** Jiang24 sources are genuinely different lines, so "which source resembles BxPC3?" is a real question, and every representation answers it poorly. GSE sources are two replicates, so the similarity question is nearly trivial. Jiang24 targets are genes that match the effect-library logic; GSE targets are regulatory elements whose DE axes are undefined. The GSE +0.75 is a lower-bound check on an easy problem; the Jiang24 negatives measure the real difficulty.
4. **Frozen vs statistical?** Most of the value is statistical. Step 1, from no-effect to any effect transfer, gives large gains (Jiang24: −1.343 → −0.98 to −0.76; GSE: 0.606 → 0.753). Step 2, from raw / PCA to frozen similarity, gives small, condition-dependent differences. For STATE and STACK on Jiang24, `nearest` was more stable than `weighted`: when similarity signals are weak and noisy, blending sources with wrong weights contaminates the effect.
5. **Why do identical pipelines score differently?** The representations disagree on *which* line BxPC3 resembles. Raw, PCA, UCE and TranscriptFormer pick **A549**; scPRINT-2, STATE and STACK pick **HT29**. τ controls how firmly: raising it lowers the top weight and spreads mass over ~5 sources. So a τ sweep genuinely moves between a sharp single choice and a flat mixture.
6. **What do the metrics really measure, and what should be optimised?**
   - **PDS**: can the prediction be told apart from other targets' truths? The most honest "is the direction right?" axis on a new line; volatile, so average it over seeds.
   - **MSE**: profile squared error; uninformative on these two datasets.
   - **NMAE**: are magnitudes right? Depends on sampling depth; chasing it alone falls into the MAE trap.
   - **Fidelity**: large and stable, but weak at separating models.
   - **Reach**: how many true DE genes are recovered in the right direction; secondary.
   - **Jaccard**: overlap of predicted and true DE sets. Its Wilcoxon basis makes it hypersensitive to cell count, and losing here collapses Overall on a new line.
   - **Practical rules:** expect the dominant axis to differ by dataset type (Jaccard for new lines, PDS for batch-like shifts); never trust a single Overall number (check leave-one-axis-out); don't spend effort on axes that look big but don't separate models; remember the NMAE ↔ Jaccard trade-off and find its optimum by shadow CV.

## 6. Robustness findings

- **Temperature.** On both datasets, all five τ values produce different winners. Averaged over τ, stable leaders exist: PCA on GSE270828 (mean rank 2.0, Overall 0.719 ± 0.046) and raw on Jiang24.
- **Seed.** No representation wins all four seeds (Jiang24: scPRINT-2 wins 2 of 4; GSE270828: raw wins 2 of 4). Single-sample score gaps are not decisive; report means over seeds with uncertainty.
- **Cell count, the most important extra finding.** The number of predicted cells sets the Monte Carlo depth of the submitted distribution, and it pushes axes in opposite directions. More cells make NMAE and fidelity better (stabler means; at 25 cells NMAE sits on its floor of about −6.0 and every row ties), but make **Jaccard worse**: the Wilcoxon test gains power to expose a wrong DE set, and at 100 cells Jaccard collapses to −15 to −17. The number of submitted cells is therefore a tuning knob that decides which axis wins.
- **Stable vs single-condition winners.** GSE270828: PCA is the most stable, followed by UCE and STATE. Jiang24: camps split; scPRINT-2 has the best mean rank among frozen encoders but a large spread, and raw is the steadiest "least bad".

## 7. Strategic implications

1. Invest in the effect library: broader Replogle-style coverage (more targets, more lines) and real ESM-2 nearest-gene fallbacks for uncovered targets beat swapping embeddings.
2. Treat frozen embeddings as optional ensemble members, never as a single bet.
3. Aim at the deciding axis: Jaccard for new lines (rank-preserving sparsification, transfer only the top-|logFC| genes, per-gene confidence weights); PDS for batch-like shifts (source-agreement confidence).
4. Tune only with shadow CV and multiple seeds; scale, τ, cells and shrinkage all trade axes against each other. Fix them on a separate holdout before evaluating.
5. Treat the submitted cell count as an experimental variable.
6. Always report all six axes and a leave-one-axis-out check.

## 8. Honest limits

Not official scores (dataset-local anchors); never compare Jiang24 and GSE270828 on one scale (different anchors; five vs two defined axes); the zero axes of GSE270828 are a fallback, not a result, so its strict run is effectively a missing-target negative control; the sweeps are post-hoc, so a "best" τ or cell count would be selection bias unless re-confirmed on a fresh holdout; tied `nearest` rows reflect the same `argmax`, not equal embeddings; pretraining overlap is unknown, so this is "foundation-model augmented", not pretraining-clean zero-shot.

## 9. Reproduction and audit trail

Original artefacts (not included; too large): experiment 1 `data_shadow/two_dataset_vcc2026_metrics/comparison_wide.csv`; experiment 2 `data_experiments/external_frozen_matrix/shadow-existing/comparison_wide.csv`; strict Replogle-only `…/replogle-only/comparison_wide.csv`; robustness `new/metric_robustness_20260825/` (192-row combined table, axis contributions, leave-one-axis-out, variance decomposition, rank stability, source weights, validation report). Runners in this repository: [`scripts/run_two_dataset_full_matrix.sh`](../scripts/run_two_dataset_full_matrix.sh) and [`scripts/run_two_dataset_external_frozen_matrix.sh`](../scripts/run_two_dataset_external_frozen_matrix.sh).
