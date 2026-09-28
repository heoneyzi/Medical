# Strict two-dataset zero-shot evaluation

This protocol is stricter than the earlier Jiang24 leave-one-cell-line-out (LOCO) benchmark. LOCO hides BxPC3 responses but transfers measured responses from five other Jiang24 cell lines. That is valid out-of-cell-line transfer, but it is **not** complete zero-shot when the rule forbids every perturbation response from the evaluation dataset.

## Information contract

Prediction may read only:

- raw non-targeting control cells from the evaluation context;
- the intervention identifier list; and
- the output expression-gene list.

Prediction may not read:

- perturbed cells from any batch or context in the evaluation dataset;
- pseudobulk, differential-expression statistics, or effect-size summaries derived from those cells;
- GSE270828 HAR-to-nearby-gene annotations;
- a same-dataset effect library; or
- evaluation results for scale selection, method selection, or any other tuning.

`prepare-zero-shot` writes the permitted files and `predict.yaml` under `blind/`. It writes all perturbed cells under the separate `sealed/` directory. The config explicitly disables GWPS sources and contains no real truth path. `predict` must finish before `score-zero-shot` receives the sealed truth path. SHA-256 hashes and the contract are recorded on both sides.

Target identifiers are part of the experimental design, as they are in a challenge target panel. Every eligible non-control label is included; targets are not selected using response strength. Genes are ranked from controls only.

## Datasets

### Jiang24 / GSE281048

- Context: IFNG-treated BxPC3 cancer cells.
- Intervention: coding-gene CRISPRi.
- Strict protocol: no responses from A549, HAP1, HT29, K562, MCF7, or BxPC3 are available to prediction.

### GSE270828 / PRJNA1128583

- Context: H23555 iPSC-derived neural stem cells.
- Intervention: direct-capture CRISPRi of 180 human accelerated regions, plus five positive controls and six non-targeting guides.
- Converted official object: 87,896 post-QC/single-guide cells × 19,698 genes, 498,320,076 nonzero raw counts, including 1,518 NTC cells and 185 non-NT target labels across three batches.
- The official 2.2 GB RDS is double-gzipped. `export_gse270828_rds.R` and `build_gse270828_h5ad.py` preserve its sparse raw-count matrix without a dense or MatrixMarket expansion.
- The feature README is retained for provenance, but its HAR-to-gene mapping is not copied into blind inputs.
- During scoring only, that published map is supplied to `cell-eval2` so the official target-gene-exclusion rule can be applied. It is never visible to prediction.

Official sources: [GSE270828 at NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270828), [PRJNA1128583 at NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1128583), and [the authors' analysis code](https://github.com/NoonanLab/Noble_et_al_2024).

## Frozen-model eligibility

Being frozen prevents fitting on the evaluation response, but does not by itself prove zero-shot cleanliness. Two additional conditions matter:

1. the checkpoint must expose a target-specific transition interface for the intervention modality; and
2. its training manifest must establish that the evaluation dataset was absent when prior-data exclusion is required.

The local STATE `SE-100M` artifact is an embedding checkpoint, not a perturbation transition model. Its embedding can describe NTC state, but cannot turn an unseen HAR identifier into a response. STACK-Large is an in-context generator, but its generation workflow requires response examples and does not provide a direct HAR-ID intervention interface. Both published training descriptions include large evolving corpora (including scBaseCount for STACK), and the available manifests do not certify exclusion of GSE270828. They are therefore reported in `frozen_model_eligibility.csv` as **not eligible for a clean strict score**, not assigned fabricated predictions.

This does not say the checkpoints definitely trained on GSE270828. It says absence cannot currently be proven from the released metadata.

An independently audited frozen model can still be added without changing the split. Generate its H5AD using **only** `blind/`, then score the already-created artifact:

```bash
python -m vcc_baselines score-zero-shot \
  --config BUNDLE/blind/predict.yaml \
  --pred AUDITED_FROZEN_PREDICTION.h5ad \
  --truth BUNDLE/sealed/CONTEXT/truth.h5ad \
  --method-name audited_frozen_model --engine cell-eval2 \
  --out BUNDLE/results/audited_frozen_model
```

The scorer rejects wrong genes, ordering, target sets, and any config with a GWPS response source. A name does not certify a checkpoint; its training-data exclusion and intervention interface still need an audit entry.

## Run

```bash
# Download and verify the official GSE files.
bash scripts/download_shadow_data.sh data_public gse270828

# Build both blind bundles, predict without opening sealed truth, run the exact
# public cell-eval2 vcc2026 preset, audit frozen checkpoints, and merge rows.
bash scripts/run_two_dataset_zero_shot.sh
```

Outputs are written under `data_zero_shot/`:

```text
jiang24_ifng_bxpc3_strict/{blind,sealed,results}/
gse270828_nsc_strict/{blind,sealed,results}/
frozen_model_eligibility.csv
two_dataset_strict_comparison.csv
```

All reported components come from `cell-eval2==0.16.0` with the exact `vcc2026` suite: `pds`, `mse`, `nmae`, `fid`, `reach`, and `jac`; `overall` is their unweighted mean. Each external dataset has its own generic-response and five-split replicate anchors. The implementation is exact, but the values are not official leaderboard scores because they do not use the Challenge server's reference panel and anchor bundle. The two biological modalities are shown side by side and never averaged across datasets.
