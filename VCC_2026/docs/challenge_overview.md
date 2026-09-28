<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › [docs](README.md) › **Challenge overview**</sub>

# VCC 2026 explained: task, data, evaluation, leaderboard, and a strategy

> [!NOTE]
> Condensed English version of the Korean write-up [`ko/01_challenge_explainer_2026-08-22_ko.md`](ko/01_challenge_explainer_2026-08-22_ko.md), written on **22 Aug 2026**, two days after launch. It is based on Arc Institute's launch announcement, the challenge site and Arc's public repositories. Items marked *to verify* were not yet readable on the JavaScript-rendered site. Later findings are flagged as **Update**. Official description: [Cell commentary, doi:10.1016/j.cell.2026.08.004](https://doi.org/10.1016/j.cell.2026.08.004).

## 1. The challenge in one sentence

Given only the **non-targeting control** transcriptomes of a cell line the model has never seen, predict the **single-cell transcriptomic response** to a **CRISPRi knockdown** of each listed gene, **zero-shot**.

## 2. What changed from 2025

| | 2025 | 2026 |
|---|---|---|
| Generalisation axis | Unseen perturbations in one new context (H1 hESC) | Zero-shot perturbation prediction across several new contexts |
| Contexts | H1 | 3 validation lines + 3 *different* final-test lines |
| Challenge training labels | 150 H1 perturbations released | None for any target context |
| Validation | 50 H1 genes | 3 unseen lines × 300 perturbations |
| Final test | 100 H1 genes | 3 further unseen lines (October phase) |
| Adaptation | Fine-tune on H1 labels | No label-based fine-tuning on the target context |
| Evaluation | PDS + DES + MAE | 6 complementary metrics |
| Weighting | Open to single-metric gaming | Equal weight over metrics × contexts |
| Key capability | Perturbation transfer | Cell-context transfer **and** perturbation transfer |

Launch facts: CRISPRi knockdowns; per validation line, control profiles plus 300 target-gene IDs (up to 3 × 300 = 900 context–target conditions); metrics scaled per context between a context-mean baseline and a real replicate experiment; one unweighted overall score; submissions and leaderboard open at launch on 20 Aug 2026 with more than 1,800 registrants; USD 175,000 total prizes; sponsors NVIDIA, 10x Genomics and Ultima Genomics.

## 3. The task as a function

For a new context *c* and target gene *g*, the model sees $X^{(c)}_{\mathrm{control}}$ and must output $\widehat X^{(c,g)}_{\mathrm{perturbed}} = f(X^{(c)}_{\mathrm{control}}, g)$. It has to combine *what gene g generally does*, *what state this cell is in*, and *how that state modulates g's effect*. A useful decomposition is

$$\Delta(c,g) = \Delta_{\mathrm{shared}}(g) + \Delta_{\mathrm{context}}(c,g), \qquad \widehat X_{\mathrm{pert}} = X_{\mathrm{control}} + \widehat\Delta(c,g),$$

which matches the residual ("predict the change") framing of the 2025 runner-up.

**Why 2026 is harder.** In 2025 a model could learn how H1 responds from 150 H1 examples. In 2026 the new line provides zero perturbation examples, so the control transcriptome is the only window into the new context: baseline expression, cell cycle, stress, lineage, tissue programmes, signalling, the target's own baseline expression, its pathway neighbours, heterogeneity. The controls act like a **prompt** describing the new cell ("cellular prompt → perturbation response"), which fits Arc's set-of-cells / in-context direction (STATE, STACK). The context encoder matters far more than in 2025.

## 4. Evaluation

- **Six complementary metrics**, each scaled **per context** so that 0 ≈ the context-mean baseline and 1 ≈ an independent replicate experiment. The overall score is the unweighted mean of 6 metrics × 3 contexts = 18 components.
- **Why a replicate ceiling:** sampling noise, technical noise, heterogeneity and guide efficacy mean two real experiments never match perfectly, so "as good as a replicate" is a fair target.
- **Do not hand-roll the scaling** (metric directions differ); use the official evaluator.
- **Trap:** the public `cell-eval --profile vcc` historically carries the **2025** metric profile and should not be assumed to equal the 2026 scorer. *To verify at the time: exact metric names, directions, thresholds, clipping, normalisation, missing-gene and cell-count handling.*
  **Update:** the 2026 scorer is `cell-eval2`'s `vcc2026` preset: `pds_cosine`, `expr_mse_unbiased_capped_norm`, `de_wilcoxon_lfc_nmae`, `de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`, `de_wilcoxon_sig_jaccard` ([`TWO_DATASET_SIX_METRIC.md`](TWO_DATASET_SIX_METRIC.md)).
- **Why the redesign:** in 2025 the mean baseline was so strong on MAE that almost no model beat it, so top teams optimised PDS and DES instead, which amounted to metric gaming. 2026 counters with more metrics, per-context scoring, replicate normalisation and equal weights, favouring *generalist* models. The 2025 seven-metric Generalist Prize was effectively the test bed.

## 5. How to use the leaderboard

The live leaderboard scores validation lines A/B/C, but the final ranking uses different lines D/E/F. Tuning against the leaderboard therefore overfits the validation contexts: $\arg\max$ of the validation leaderboard need not be $\arg\max$ of final-test generalisation. Treat it as a noisy signal, trust local leave-one-context-out CV, and log every submission (commit, model, external data, CV mean, **CV worst context**, leaderboard overall). Prefer robust models, e.g. select by $\mathrm{mean}(S_c) - \lambda\,\mathrm{std}(S_c)$: a model scoring 0.62 / 0.61 / 0.60 is safer than one scoring 0.8 / 0.8 / 0.2.

## 6. Data strategy

| Tier | Dataset | Role in 2026 |
|---|---|---|
| 1 | **GSE281048** (Jiang24 / Mixscale): the same pathway perturbations in 6 cell lines × 5 signalling contexts | Learn context dependence directly; build a "fake 2026" by hiding one line (e.g. BxPC3) and giving the model only its controls |
| 1 | **Replogle 2022** (K562 genome-scale, ~9,867 targets) | Gene-effect backbone: "what does knocking down *g* usually do?" |
| 1 | **GSE264667** (Jurkat, HepG2 CRISPRi) | Cheap extra contexts (immune-like, liver-like) |
| 1 | **VCC 2025 H1** | Better used as a zero-shot *shadow test* than as training data |
| 1 | **X-Atlas / Orion** (~18,903 protein-coding targets, mostly HCT116 / HEK293T) | Target coverage and gene prior; weak on its own for context generalisation |
| 2 | **Marson primary CD4⁺ T-cell CRISPRi** | Primary-cell out-of-distribution test |
| 2 | **KOLF2.1J iPSC** | Pluripotent perturbation rules, keeping H1 held out |
| 2 | **scBaseCount / CELLxGENE Census** | Pretraining a context encoder on broad cell states |
| 3 | Parse PBMC, Tahoe-100M, OP3, PerturbFate, sci-Plex, KO / CRISPRa sets | Auxiliary only; different modality, never merged with CRISPRi labels |

The "core triangle" is GSE281048 (context generalisation) + Replogle (robust gene effects) + X-Atlas (gene coverage), with scBaseCount / CELLxGENE underneath for context representation. Priors worth precomputing: ESM-2 (protein sequence), STRING (interaction neighbours), DepMap (gene × cell-line dependency, a natural context prior) and PerturbAtlas-style historical DE statistics.

## 7. Lessons from the 2025 winners

- **1st, xTrimoSCPerturb:** not a pure foundation model. It combined public perturbation data, protein embeddings, pseudobulk, DEG frequency and mean expression, i.e. classical statistics plus deep learning.
- **2nd, XLearning "Model X":** control pseudobulk + ESM-2 gene embedding + a simple fully connected network predicting Δ expression. For 2026, add a target-context control embedding and a multi-context prior.
- **3rd, TransPert:** similarity-weighted transfer of effects measured in reference lines, which fits the 2026 task almost exactly.

## 8. Baseline ladder and a recommended architecture

The classical baseline to beat: $w_r = \mathrm{softmax}(\mathrm{sim}(z_{c^*}, z_r)/\tau)$, $\widehat\Delta_{c^*,g} = \sum_r w_r \Delta_{r,g}$, $\widehat X = X^{c^*}_{\mathrm{ctrl}} + \widehat\Delta$. It is explainable, fast, and makes leakage and context similarity easy to inspect.

| Rung | Model |
|---|---|
| B0 | Control mean (predict nothing) |
| B1 | Global gene effect averaged over reference contexts |
| B2 | Nearest-context transfer |
| B3 | Similarity-weighted, TransPert-like transfer |
| B4 | Pseudobulk MLP on context embedding + ESM-2(g) + baseline expression + historical summary |
| B5 | Hybrid context × gene model (set encoder, multi-prior gene encoder, cross-attention / FiLM, shared effect + context residual) |
| B6 | Distributional model (flow matching / diffusion / STATE-style), only once B0–B5 work |

Recommended structure: a **context branch** (set encoder over control cells → $z_c$, similarity to a reference atlas, pathway and baseline features) crossed with a **gene branch** (ESM-2, STRING, DepMap, PerturbAtlas; never one-hot IDs), producing a shared gene effect plus a context residual, with a pseudobulk head and later a cell-distribution head. The most practical first neural model feeds the statistical prediction in as an input and learns only a correction: $\widehat\Delta = \widehat\Delta_{\mathrm{stat}} + f_\theta(z_c, z_g, \widehat\Delta_{\mathrm{stat}})$. Further ideas: confidence-aware blending between statistical and neural predictions; several similarity definitions (pseudobulk, PCA, pretrained encoder, pathway activity, DepMap) and gene-specific similarity $S(c,r,g) = \alpha S_{\mathrm{global}} + \beta S_{\mathrm{pathway}(g)} + \gamma S_{\mathrm{DepMap}}$; monitoring the out-of-distribution distance $\min_r d(z_c, z_r)$; continuous mixture-of-experts gating; test-time adaptation on the provided controls *only if the rules allow it*.

## 9. Local validation

Random-cell splits test nothing about zero-shot. Use (A) leave-one-cell-line-out, (B) gene zero-shot, (C) double zero-shot (new line and new gene) and (D) dataset / lab holdout. The recommended "shadow VCC 2026" holds out three public contexts at once (e.g. HepG2, BxPC3, H1), exposes only their controls and ~300 sampled targets, and reports the mean *and* the worst context.

## 10. Submission engineering

Keep the submission pipeline separate from the model: gene-order validation → cell-count formatting → AnnData → metadata → official prep. Common failures: gene order or name mismatch, duplicated genes, wrong control or target labels, NaN / inf, negative or non-count values, wrong normalisation space, wrong cell counts, metadata mismatch, corrupted compression. Always start from the official sample submission and schema.

## 11. What not to do

Search for public data matching the validation lines; brag about random-cell splits; train on X-Atlas alone; pretrain only on observational atlases; trust a foundation model without baselines; merge CRISPRa, knockout or drug labels with CRISPRi; change the architecture daily to chase the leaderboard.

## 12. First steps (as planned on 22 Aug 2026)

1. Download the validation package; confirm the 3 lines, the 300-target lists and the file schema. **Update:** the challenge format was implemented as raw integer counts, 18,533 genes, 400 cells per perturbation, contexts A/B/C, controls excluded (`src/vcc_baselines/config.py`).
2. Build the external target-coverage and context-similarity matrices. **Update:** teammate Yumin Jung found that 272 of the 300 validation targets are measured in the Replogle K562 genome-wide screen (see the team table in the [project README](../README.md#-team-results-on-the-live-validation-leaderboard)).
3. Build a GSE281048 leave-one-context-out shadow challenge ([E1](../experiments/E1_jiang24_bxpc3_proxy/README.md)).
4. Finish a TransPert-like statistical baseline ([E1](../experiments/E1_jiang24_bxpc3_proxy/README.md)–[E3](../experiments/E3_replogle_only_strict/README.md)).
5. Add an ESM-2 + context-embedding residual MLP on top, and only then consider large flow or diffusion models.

## 13. Confirmed vs to verify (as of 22 Aug 2026)

| Item | Status then |
|---|---|
| Multi-context zero-shot task; no challenge perturbation labels; 3 validation lines with controls + 300 CRISPRi targets; 3 different final-test lines in October; 6 metrics; mean ↔ replicate normalisation; equal-weight overall; leaderboard open; USD 175k prizes | ✅ confirmed from the launch material |
| Names of the validation lines; whether the three 300-target sets are identical; exact metric names; leaderboard columns; external-data cutoff; submission limits; whether test-time adaptation is allowed; final deadline | ⚠️ to verify on the live site or package |

**Strategy in one line:** strong statistical transfer + control-context representation + gene biological priors + neural residual correction + multi-context out-of-distribution validation, rather than blindly scaling a single foundation model.

<sub>Sources listed in the original: virtualcellchallenge.org (main, data, evaluation, leaderboard, rules); Arc's 2026 launch announcement; ArcInstitute/cell-eval, ArcInstitute/state, arc-virtual-cell-atlas; the 2025 wrap-up and the 2025 *Cell* benchmark paper; GSE281048, Replogle 2022, X-Atlas/Orion, GSE264667, scBaseCount / CELLxGENE, DepMap, STRING, ESM-2, PerturbAtlas, PRiMeFlow, PerturBench, X-Cell.</sub>
