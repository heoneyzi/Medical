# Independent external frozen-model experiment

This experiment leaves the existing STATE/STACK runner unchanged. It enrolls UCE-4L,
TranscriptFormer-Sapiens cell embeddings, scGPT whole-human, scFoundation cell embeddings, and
scPRINT-2 as frozen context encoders through isolated command adapters.

## Protocols

- `replogle-only` (default): Replogle 2022 K562 GWPS and RPE1 are the only explicit response
  sources. Global magnitude is selected only by symmetric K562↔RPE1 internal CV.
- `shadow-existing`: reproduces the existing same-dataset source-context transfer with more
  encoders. It is non-strict and requires `--acknowledge-non-strict`.

Both protocols use `cell-eval2==0.16.0`, preset `vcc2026`, and report PDS, MSE, NMAE, fidelity,
reach, Jaccard, and their exact unweighted Overall. These are external-dataset scores with local
baseline/replicate anchors, not Challenge-server scores.

GSE270828 perturbs HAR regulatory elements rather than genes. All 184 labels are missing from the
strict exact-gene Replogle library, so strict frozen context weighting cannot change its prediction;
those rows are a negative-control/audit, not evidence that the encoders are equivalent.

## Setup

The current machine cannot hold every environment/checkpoint comfortably at once. Install only the
models needed for a run; each has an independent environment:

```bash
cd 01_Medical/VCC_2026    # project root

bash scripts/setup_external_frozen_models.sh gene-map
bash scripts/setup_external_frozen_models.sh replogle-data
bash scripts/setup_external_frozen_models.sh uce
bash scripts/setup_external_frozen_models.sh transcriptformer
bash scripts/setup_external_frozen_models.sh scgpt
bash scripts/setup_external_frozen_models.sh scfoundation
bash scripts/setup_external_frozen_models.sh scprint2
```

UCE setup downloads and integrity-checks the official 4-layer checkpoint, token tensor, chromosome
metadata, offsets, and protein-embedding archive. These files total about 9.1 GB before archive
extraction. A zero-byte, truncated, or checksum-mismatched file is `UNAVAILABLE`; setup resumes a
`.part` transfer but never enrolls it as a checkpoint.

scGPT and scFoundation print `AUDIT_REQUIRED` because their official checkpoint hosts require a
manual folder/file download. Put the files at the exact paths reported by preflight. No random or
partial checkpoint fallback exists.

If `/root` is space-constrained, place a new model's repository, environment, and checkpoint on a
larger mounted filesystem while retaining the registry paths as symlinks:

```bash
MODEL_STORAGE_ROOT=/path/to/large_disk/vcc_model_assets \
  bash scripts/setup_external_frozen_models.sh transcriptformer
```

Existing non-symlink model paths are never moved or overwritten by this option.
When `MODEL_STORAGE_ROOT` is set, the setup script also places the `uv` package cache there unless
`UV_CACHE_DIR` was explicitly supplied. This matters for CUDA-heavy environments such as scPRINT-2.

## Preflight and run

All requested models must be ready by default:

```bash
bash scripts/run_two_dataset_external_frozen_matrix.sh --preflight-only
bash scripts/run_two_dataset_external_frozen_matrix.sh
```

For an explicitly incomplete smoke run, enroll only ready models or allow unavailable models:

```bash
bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --models=uce,transcriptformer_cell

bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --allow-unavailable
```

The non-strict ablation is separate:

```bash
bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --protocol=shadow-existing --acknowledge-non-strict
```

To start with one model and see every nested stage in the terminal:

```bash
bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --models=uce
```

Monitor from a second terminal:

```bash
tail -f data_experiments/external_frozen_matrix/replogle-only/progress.tsv
tail -f data_experiments/external_frozen_matrix/replogle-only/logs/latest.log
```

## Outputs

```text
data_experiments/external_frozen_matrix/<protocol>/
├── jiang24_embeddings/<model>/<context>/{cells.npy,context.npy,metadata.json,run.log}
├── gse270828_embeddings/<model>/<context>/{cells.npy,context.npy,metadata.json,run.log}
├── jiang24_details/representations/<model>/...
├── gse270828_details/representations/<model>/...
├── jiang24_vcc2026.csv
├── gse270828_vcc2026.csv
├── jiang24_coverage_vcc2026.csv
├── gse270828_coverage_vcc2026.csv
├── comparison_wide.csv
├── metric_changes_long.csv
├── coverage_comparison_wide.csv
├── coverage_metric_changes_long.csv
├── progress.tsv
└── logs/latest.log
```

Unlike the older detailed output layout, raw/PCA/model prediction and scorer artifacts are separated
by representation and no longer overwrite one another. Completed standardized embeddings and
successful metric rows are reused only when their input/checkpoint/repository provenance matches.

Coverage tables report `ALL_TARGETS`, `DUAL_SOURCE`, `K562_ONLY`, and `MISSING` when present. Every
non-empty stratum is independently passed through the same exact six-metric `cell-eval2` wrapper;
scores are not replaced with local proxy metrics. Per-model `*.qc.json` files record library size,
zero fraction, detected genes, embedding norms, and pairwise context distances.

TranscriptFormer-CGE, Geneformer deletion, and scPRINT-2 GRN are deliberately excluded from this
context matrix: they are target-specific priors and require the gates documented in the implementation
specification. They are visible in preflight/registry with an explicit exclusion reason rather than
being presented as failed context encoders.

## Interpretation boundary

The runner compares frozen representations, not direct perturbation predictions emitted by those
foundation models. Every representation sees the same Replogle-derived effect library, generator,
cell counts, and scorer scale within a dataset. Consequently, a difference between rows isolates
the context-similarity representation as far as this benchmark permits. Unknown checkpoint
pretraining overlap remains explicitly `unknown`, so this is `B_FM_AUGMENTED`, not a claim of
checkpoint-pretraining-clean zero-shot.
