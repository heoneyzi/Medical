<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [Experiments](../README.md) › **④ GeoACMG**</sub>

# 🧬 ④ GeoACMG — the agent on real variant-interpretation evidence

> **Question —** On tasks built from ClinGen expert records, do the tasks really need several tools, does the frozen MedCPT space carry execution information that distances can read, and does observe→replan still help?

| | |
|---|---|
| **Why this stage** | ClinGen records both applied (**Met**) and rejected (**Not Met**) evidence, so an environment built from them contains tool calls that cost something and return nothing useful — exactly where planning should matter |
| **Data** | ClinGen Evidence Repository: 13,278 records → 4,054 tasks → 2,921 after a 100-tasks-per-gene cap (1,753 train / 584 dev / 584 test; 155 genes, 47 in dev); exact search over 448,704 train+dev states |
| **Models** | Frozen MedCPT (768-d); goal-conditioned geometry heads (5 energies × seeds); State Flow planner |
| **Compute** | GPU container; geometry runs batched per seed to share one feature store |
| **Headline** | Goal completion **0.5993** observe→replan vs 0.1169 one-shot vs 0.0274 blind on the 584 development tasks; raw frozen distances order states at chance (0.5055 / 0.5078 / 0.4966) |

## Setup

- **Environment.** A task asks the agent to reach the panel's ACMG classification of a variant. Actions are five evidence tool families — population frequency, in-silico predictors, clinical assertions, gene mechanism, molecular consequence — and five report classes (benign … pathogenic). Evidence comes from the experts' Met and Not Met records, replayed as offline snapshots; tool costs are fixed (probe 1.0, report 0.5).
- **Pre-registration.** Claims B0 and C1–C5, each with a falsifier, were fingerprinted before analysis; findings outside the pre-registered primary set are reported as exploratory. An amendment ([R6](../../results/geoacmg_final/findings/R6_amendment.json)) records the revised claims (C5 withdrawn, C4 constrained).
- **Pipeline.** corpus → benchmark diagnostic (B0) → exact search oracle → frozen embeddings → probes → 5 energies × 3 seeds (cosine, Euclidean, Poincaré, directed quasimetric, pair MLP) → paired-seed, readout, horizon and cross-model analyses (R1–R4) → batched State Flow evaluation on the 584 development task roots (R7).

## Results

| Question | Result (development split unless noted) | Source |
|---|---|---|
| Do tasks need more than one tool family? *(whole corpus, 2,921 tasks, 155 gene clusters)* | The best single tool family (population frequency) suffices for 0.3102 of tasks; gap **0.6898** [0.6323, 0.7391] — a property of the benchmark | [`b0.jsonl`](../../results/geoacmg_final/findings/b0.jsonl) |
| Is execution information in frozen MedCPT states? *(linear probe, gene hold-out)* | Remaining-cost R² **0.3355** vs −0.0173 for a random projection; next-action accuracy 0.5523 vs permutation null 0.2896 | [`rq1.jsonl`](../../results/geoacmg_final/findings/rq1.jsonl) |
| Do raw distances expose it? *(depth-matched ordering, 141,517 pairs)* | cosine 0.5055 · L2 0.5078 · dot 0.4966 — indistinguishable from chance | [`R2_readout_decomposition.json`](../../results/geoacmg_final/findings/R2_readout_decomposition.json) |
| Can small learned heads read it? *(5 energies × seeds 17/29/43)* | Own-energy ordering from 0.5870 (Euclidean) to 0.8266 (pair MLP); dev policy accuracy only from 0.8512 (Euclidean) to 0.8699 (pair MLP) | same + [`geometry_comparison/`](../../results/geoacmg_latest_checkpoints/geometry_comparison/) |
| Is the ordering gap just parameters? *(R1: cosine vs Euclidean, zero energy parameters, 335,942 total, matched initial trunk)* | Ordering +0.1432 [0.0950, 0.1802], clustered over 5 seeds (17/29/43/59/71) | [`R1_paired_seeds.json`](../../results/geoacmg_final/findings/R1_paired_seeds.json) |
| Do new seeds agree on *planning*? *(R1b, seeds 59/71)* | Policy −0.0099 [−0.0167, −0.0028] — the earlier cosine planning advantage **reversed** (task-bootstrap CI over the 2-seed average) | [`R1b_new_seed_planning_axis.json`](../../results/geoacmg_final/findings/R1b_new_seed_planning_axis.json) |
| Is the learned space readable only by its own energy? *(R2, 15 runs)* | Own energy 0.7405 vs standardized cosine 0.7199 vs plain cosine 0.5761 — standardizing recovers most of it | [`R2_readout_decomposition.json`](../../results/geoacmg_final/findings/R2_readout_decomposition.json) |
| Does ordering quality drive policy? *(R4, 19 runs, cyclic model mismatch)* | Matched − mismatched ordering×policy +0.000424 [0.000169, 0.000766] — above 0 but tiny; C5 ("planner validity = representation validity") was **withdrawn** because matched and mismatched model pairs correlated equally (+0.3282 vs +0.3289) | [`R4_crossmodel_control.json`](../../results/geoacmg_final/findings/R4_crossmodel_control.json), [`R6_amendment.json`](../../results/geoacmg_final/findings/R6_amendment.json) |
| Does observe→replan still help? *(R7, 584 development roots × 4 samples, one flow checkpoint)* | Goal completion 0.5993 vs 0.1169 one-shot vs 0.0274 blind; verifier STOP guard 0.6233; replan − one-shot **+0.4824** [0.4439, 0.5223] | [`R7_flow_dev_root_only.json`](../../results/geoacmg_final/findings/R7_flow_dev_root_only.json) |

<p align="center"><img src="../../assets/geoacmg_dev_dissociation.png" width="760" alt="Left: depth-matched ordering accuracy of raw frozen MedCPT distances (near 0.5) versus five learned geometry heads (0.587 to 0.827). Right: development-split policy accuracy per geometry, all between about 0.85 and 0.87, with per-seed dots."></p>
<p align="center"><sub>Figure: representation ordering differs a lot between geometries, planning barely does. Drawn by <code>assets/make_figures.py</code> from <code>R2_readout_decomposition.json</code> and the per-run <code>value_metrics.json</code> files.</sub></p>

The R1–R4 statistics were recomputed from the raw readout-margin arrays: 276/276 checks pass ([`geoacmg_verified_metrics.json`](../../verification/geoacmg_verified_metrics.json)).

## What it showed

- **A genuine planning problem.** Unlike stages ①–②, most tasks cannot be decided with the single best tool family.
- **Information is there; raw distances cannot read it.** Frozen MedCPT states contain execution information (probe), none of the three raw distances orders states, and the learned heads differ a lot in how well they *read* the space.
- **Ordering quality and planning quality separate.** Large ordering gaps between geometries became small policy gaps, and two new seeds flipped the planning sign.
- **Observe → replan is again the clearest effect** — the same pattern as on the hard-v2 test, now on expert-derived tasks (one flow checkpoint, task-bootstrap interval).

## What it led to

Stage ⑤ tests whether the geometry differences survive more training seeds and other readouts, and whether the agent really uses the MedCPT view.

## Files

| File | What it is |
|---|---|
| [`results/geoacmg_final/findings/`](../../results/geoacmg_final/findings/) | All findings of this stage: `b0.jsonl`, `rq1*`, `R0`–`R7` JSON, pre-registration, capacity/ordering/mechanism analyses, the auto-generated `REPORT.md` |
| [`results/geoacmg_latest_checkpoints/`](../../results/geoacmg_latest_checkpoints/README.md) | Per-run development metrics: `geometry_comparison/` (15 runs), `geometry_r1_pairs/` (seeds 59/71), `capacity_matched/`, `state_flow/` |
| [`configs/geoacmg.yaml`](../../configs/geoacmg.yaml) | GeoACMG run record (MedCPT view, seeds, training settings) |
| [`src/geoflowagent/geoacmg/`](../../src/README.md) | ClinGen parsing, ACMG evidence/tool map, tasks, splits, probes, paired/readout/strata analyses, sealing, claims, batched flow evaluation |
| [`scripts_run/`](../../scripts_run/README.md) | Geometry batch runners and analysis scripts; `export_stage/` holds the R1–R4/R7 finishing scripts |
| [`docs/original/geoacmg/EXPERIMENT_JOURNAL_20260919.md`](../../docs/original/geoacmg/EXPERIMENT_JOURNAL_20260919.md), [`00_ORIENTATION.md`](../../docs/original/geoacmg/00_ORIENTATION.md) | Full experiment journal and design rationale (Korean, original) |
| [`verification/verify_geoacmg_metrics.py`](../../verification/verify_geoacmg_metrics.py) | Recomputes R1–R4 from the readout-margin arrays (saved output: 276/276 PASS) |

---
<sub>[← ③ hard-v2](../03_hard_v2/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [⑤ Follow-ups →](../05_geoacmg_followups/README.md)</sub>
