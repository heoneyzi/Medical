# Real-data shadow VCC evaluation

This benchmark is deliberately separate from the challenge validation set. It asks the same operational question—predict CRISPRi responses in a cell line whose perturbed cells are hidden—without claiming to reproduce the private leaderboard distribution or server-side normalization.

## Primary split used here

The supplied dataset catalogue identifies Jiang24/GSE281048 as the closest public shadow benchmark. The PerturBench release contains six cancer cell lines and five signaling treatments. The checked-in recipe fixes the signaling treatment to IFNG, hides all perturbed BxPC3 cells, and transfers effects from A549, HAP1, HT29, K562, and MCF7.

The completed split has:

- 245,240 IFNG cells and 15,473 encoder-input genes from raw `layers['counts']`;
- 56 CRISPRi targets supported by at least 30 cells in every source and the holdout;
- 4,017 evaluation genes;
- at most 500 control cells per context for frozen encoders; and
- no challenge validation data.

Primary sources: [GSE281048 at NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281048) and the [official PerturBench release](https://huggingface.co/datasets/altoslabs/perturbench/tree/main).

## Reproduce Jiang24 IFNG → BxPC3 LOCO

The current archive is about 15 GB compressed and expands to about 88 GB. Put the temporary uncompressed file on a volume with enough space:

```bash
JIANG_H5AD_OUT=/large/tmp/jiang24_processed.h5ad \
  bash scripts/download_shadow_data.sh data_public jiang24

bash scripts/run_jiang24_loco.sh \
  /large/tmp/jiang24_processed.h5ad \
  data_public/perturbench/jiang24_IFNG_counts.h5ad \
  data_shadow/jiang24_ifng_bxpc3_loco
```

The download script verifies SHA-256 `dd890a8019b8a615010963b32e6e28ce42fd0b94a749602bc93efb2fa8af56cc`. `run_jiang24_loco.sh` first streams only IFNG rows and the raw count layer into an approximately 878 MB counts-only H5AD, then prepares and evaluates the split. Once compaction succeeds, the large temporary uncompressed file can be removed; retain the verified gzip and compact H5AD.

The equivalent explicit commands are:

```bash
python -m vcc_baselines compact-shadow \
  --input /large/tmp/jiang24_processed.h5ad \
  --where treatment=IFNG --matrix counts \
  --out data_public/perturbench/jiang24_IFNG_counts.h5ad

python -m vcc_baselines prepare-shadow \
  --input data_public/perturbench/jiang24_IFNG_counts.h5ad \
  --context-col cell_type --holdout-context bxpc3 \
  --pert-col condition --control-label control --matrix X \
  --min-cells 30 --max-targets 100 --max-genes 4000 \
  --max-controls 500 --max-cells-per-pert 100 \
  --cells-per-pert 50 --seed 2026 \
  --out data_shadow/jiang24_ifng_bxpc3_loco
```

The Tian–Kampmann iPSC → day-7-neuron script remains available as a complementary differentiation/state-shift sanity check:

```bash
bash scripts/download_shadow_data.sh data_public tian
bash scripts/run_shadow_real.sh
```

It is a strong OOD test, not a literal cell-line LOCO benchmark.

## Leakage barriers

- Challenge-like or validation paths are rejected by `prepare-shadow`.
- Held-out model input contains BxPC3 non-targeting controls only.
- BxPC3 perturbation responses are isolated in `truth.h5ad` and read only by evaluation.
- Target selection uses perturbation labels and cell counts, not held-out response values.
- Gene selection uses held-out controls only.
- Source controls are stored separately from source perturbation references.
- `manifest.json` records paths, filters, matrix origin, QC, cell counts, and split assertions.
- `compact-shadow --matrix counts` fails rather than silently treating normalized values as raw counts.

## Frozen STATE and STACK context encoders

The comparison uses the exact same effect library and generator for every representation. Only the control-cell context representation changes.

Official upstreams are [STATE](https://github.com/ArcInstitute/state) and [STACK](https://github.com/ArcInstitute/stack). Download their official checkpoints into `models/SE-100M` and `models/Stack-Large`, then run:

```bash
bash scripts/run_frozen_contexts.sh data_shadow/jiang24_ifng_bxpc3_loco
```

In the validated run:

- STATE-SE used `model.safetensors` SHA-256 `7ee823583ad10c6a12befba6df12dfcd61ac0972deab57be3b4db24a0045af86`, mapped 9,996/15,473 input genes, and produced 1,034-dimensional embeddings.
- STACK used `bc_large.ckpt` SHA-256 `d388c75891fb26618e6e36054c9819b6a82d6f5bc608baac75994e33ddc8bb81`, matched 8,255/15,012 model genes, and produced 1,600-dimensional embeddings.
- Each model embedded the same deterministic sample of 500 controls per context.

The current PyPI STATE CLI searches for Lightning `*.ckpt`, while the official SE-100M release supplies `model.safetensors`. `scripts/state_safetensors_embed.py` bridges that packaging mismatch, derives the checkpoint's actual gene-head size, and requires a strict all-key load. It never substitutes a random/PCA embedding. STACK's H5AD output stores embeddings in `X`; STATE stores them in `obsm['X_state']`. `pool-embeddings` supports both.

Arc checkpoints have their own non-commercial licenses; the repository's MIT license does not replace them.

## Observed BxPC3 results

Local metrics (`pdisc_norm_rank` and MAE lower is better; Pearson delta higher is better):

| representation / method | pdisc rank | MAE | Pearson delta |
|---|---:|---:|---:|
| no effect | 0.5013 | 0.1806 | 0.0791 |
| raw nearest | **0.4386** | 0.1807 | **0.1514** |
| PCA weighted | 0.4922 | 0.1844 | 0.0326 |
| STATE nearest | 0.4773 | 0.1888 | 0.0800 |
| STATE weighted | 0.4705 | 0.1872 | -0.0295 |
| STACK nearest | 0.4773 | 0.1888 | 0.0800 |
| STACK weighted | 0.5101 | 0.1851 | -0.1228 |

Public `cell-eval --profile vcc` proxy (`PDS` higher is better):

| representation / method | PDS | MAE | OverlapN |
|---|---:|---:|---:|
| no effect | 0.5102 | 0.1806 | 0.00003 |
| raw nearest | **0.5794** | 0.1807 | 0.00136 |
| PCA weighted | 0.5137 | 0.1844 | 0.00072 |
| STATE nearest | 0.5316 | 0.1888 | **0.00633** |
| STATE weighted | 0.5389 | 0.1872 | 0.00295 |
| STACK nearest | 0.5316 | 0.1888 | **0.00633** |
| STACK weighted | 0.4959 | 0.1851 | 0.00364 |

Raw nearest-neighbour transfer is strongest on this one held-out split. STATE weighted improves on PCA weighted in discrimination, but neither frozen encoder beats raw nearest. That negative result is useful: a frozen representation is not assumed better merely because it is larger.

The exact outputs are `benchmark_local.csv`, `benchmark_frozen_local.csv`, and `benchmark_cell_eval.csv` under the split directory. These are public proxy scores, not official VCC 2026 validation or leaderboard scores.

## Reproducibility contract

Context names are sorted before saving/loading the effect library, seeds are fixed, and repeated frozen benchmark runs are byte-identical. Missing model artifacts are reported as unavailable; the code does not relabel PCA or random vectors as STATE/STACK. For a stronger conclusion, repeat the recipe across every feasible treatment × held-out-line combination rather than selecting the best split after observing its score.
