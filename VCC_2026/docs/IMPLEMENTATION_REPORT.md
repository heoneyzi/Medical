# External frozen-model implementation report

## Delivered boundary

The existing STATE/STACK and two-dataset runners remain in place. The new entry point is
`scripts/run_two_dataset_external_frozen_matrix.sh`; it writes only below
`data_experiments/external_frozen_matrix/<protocol>/` unless `OUT_ROOT` is supplied.

The default `replogle-only` protocol consumes explicit perturbation responses only from the
official Replogle 2022 K562 and RPE1 raw-pseudobulk files. Evaluation truth is used by the scorer
after prediction and is not used for source selection, scale selection, representation fitting, or
checkpoint fitting. The alternative `shadow-existing` protocol is isolated and requires an
explicit non-strict acknowledgement.

## Model roles

| Model | Implemented role | Enrollment rule |
|---|---|---|
| UCE-4L | frozen cell-context encoder | exact official repo commit, checkpoint/runtime assets, output key and dimension validated |
| TranscriptFormer-Sapiens | frozen cell-context encoder | unique symbol-to-Ensembl staging and official cell embedding output validated |
| scGPT whole-human | frozen cell-context encoder | complete official checkpoint folder and `X_scGPT` output required |
| scFoundation 100M | frozen cell-context encoder | official checkpoint and exact row-count NPY output required |
| scPRINT-2 small-v2 | frozen cell-context encoder | official checkpoint and `scprint_emb` output required |
| TranscriptFormer-CGE | gated target prior | excluded until target-presence gate is implemented and passed |
| Geneformer deletion | gated confidence prior | excluded until Replogle magnitude-correlation gate passes |
| scFoundation gene retrieval | gated missing-target retriever | excluded until Replogle leave-one-target-out reconstruction passes |
| scPRINT-2 GRN | gated GRN prior | excluded until Replogle DEG-enrichment gate passes |

Unavailable, partial, mismatched, or unverified artifacts fail closed as `AUDIT_REQUIRED`; the
runner never substitutes random weights or another representation.

## Reproducibility and analysis artifacts

Each model/context stores the input and checkpoint digests, pinned repository commit, exact command,
native output, standardized cell embedding matrix, pooled context vector, elapsed time, and run log.
Prediction/scorer directories are separated by representation. A verified local scorer scale is
shared within each dataset, while cross-dataset score aggregation is deliberately prohibited.

The final tables contain PDS, MSE, NMAE, fidelity, reach, Jaccard, unweighted Overall, and deltas
from the designated baseline. These are exact public VCC 2026 metric computations with external
dataset-local anchors, not Challenge-server validation scores.

The same six metrics are also emitted separately for `ALL_TARGETS`, `DUAL_SOURCE`, `K562_ONLY`,
and `MISSING` whenever the stratum is non-empty. Each subset has its own exact public scorer scale;
cross-stratum averaging is forbidden. Model QC sidecars include median library size, zero fraction,
median detected genes, embedding norms, and pairwise source/holdout distances.

## Data audit findings and coverage

- K562: 11,258 pseudobulk rows, 9,866 perturbation labels, 514 selected core-control rows,
  8,246 canonical genes. The downloaded file matches the official 374,587,922-byte MD5.
- RPE1: 2,679 rows, 2,393 perturbation labels, 113 selected core-control rows, 8,748
  canonical genes. The downloaded file matches the official 95,350,546-byte MD5.
- Duplicate `TBCE`/`HSPA14` loci are resolved only by the HGNC-approved exact Ensembl ID and the
  discarded IDs are preserved in `replogle_data_audit.json`.
- Jiang24: 56 targets = 8 `DUAL_SOURCE`, 41 `K562_ONLY`, 7 `MISSING`.
- GSE270828: 184 targets = 184 `MISSING` under the exact-gene contract.

The union HGNC map contains 24,209 symbols: 17,856 exact, 892 alias, 12 ambiguous, and 5,449
missing. Unique exact/alias coverage is 12,361/15,473 genes for Jiang24 and 16,149/19,698 for
GSE270828. Ambiguous mappings fail closed rather than being chosen by row order.

## Internal CV and fixed hyperparameters

Symmetric K562-to-RPE1 and RPE1-to-K562 internal CV used 2,390 dual targets and 7,093 common genes.
It selected global lnFC scale `0.25` by mean/worst normalized MAE. The projected library bakes this
scale exactly once; generated configs then keep `predict.scale=1.0`. G0 uses pooled control gene
probabilities, target relfold, deterministic SHA-256-derived seeds, fixed library sizes, sparse
integer multinomial output, and no control-cell zero lock.

## Smoke and validation status

- Python/unit regression: passing; includes deterministic G0, exact-source projection, command
  adapter reuse, missing runtime assets, checkpoint tampering, and non-count input rejection.
- Replogle-only response-source policy audit: passing for both generated configs.
- UCE-4L real-input smoke: official checkpoint/runtime assets passed preflight; the official command
  retained 128/128 Replogle K562 cells and emitted finite 1,280-dimensional `X_uce` embeddings.
- TranscriptFormer-Sapiens real-input smoke: the official `tf_sapiens` checkpoint passed its pinned
  directory digest; official inference retained 128/128 cells and emitted finite 2,048-dimensional
  embeddings after unique HGNC-to-Ensembl staging (96.07% mapping).
- Authoritative smoke records live under `docs/model_smoke_tests/` and the run artifact
  `metadata.json`; the other models remain unavailable until their own preflight and smoke pass.
- Challenge submission validation: not applicable to these external-dataset score runs. They do not
  claim a `.vcc` or server validation score; the repository's existing challenge submission path is
  unchanged.

## Skipped/gated experiments

TranscriptFormer-CGE, Geneformer deletion confidence, scFoundation missing-gene retrieval, and
scPRINT-2 GRN are disabled. Each changes a target-specific prior and requires its own Replogle-only
gate; none is silently treated as a context embedding or folded into a preselected hybrid. Direct
STATE/STACK code remains unchanged; this independent runner is specifically for the additional
external encoders.

## Known hard boundary

GSE270828 targets 184 HAR regulatory elements rather than genes. None is an exact perturbation target
in the Replogle gene library. In strict mode all such predictions therefore remain at the no-effect
fallback; this dataset is a negative-control/audit for the strict exact-target contract, not a valid
ranking experiment for the frozen context encoders.

Checkpoint pretraining overlap with Replogle or VCC cannot currently be ruled out. The experiment is
therefore labeled `B_FM_AUGMENTED`, even though explicit perturbation-response use is strict
Replogle-only. Different coverage strata and different datasets have different local scorer anchors,
so neither scores nor deltas may be averaged across them.

## Exact reproduction commands

```bash
cd 01_Medical/VCC_2026    # project root
bash scripts/setup_external_frozen_models.sh gene-map
bash scripts/setup_external_frozen_models.sh replogle-data
bash scripts/setup_external_frozen_models.sh uce       # repeat for each requested model

bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --models=uce --preflight-only
bash scripts/run_two_dataset_external_frozen_matrix.sh \
  --models=uce
```

Replace the model list with a comma-separated subset of
`uce,transcriptformer_cell,scgpt,scfoundation,scprint2`. The terminal shows outer stages, model/context
embedding progress, method-matrix progress, coverage-stratum progress, and official scorer substeps.
The full run is resumable from standardized embeddings and successful CSV rows.
