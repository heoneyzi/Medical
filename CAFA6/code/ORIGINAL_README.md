# CAFA6: Protein Function Prediction

Hierarchical multi-label Gene Ontology prediction using protein foundation-model representations, label-space JEPA, and ontology-aware post-processing.

This is Jiheon Kang's fork of the team repository [SOL1archive/CAFA6](https://github.com/SOL1archive/CAFA6). The original implementation and contributor history are retained. Jiheon's CV records participation as a **team member** in the **bronze-medal** CAFA 6 project, with **82,404 training proteins** and a **224,309-protein test superset**. The test superset is broader than the hidden scored test set. These are CV-reported details; this update does not independently reproduce a leaderboard result or assign every component to one contributor.

[Jiheon Kang's research portfolio](https://heoneyzi.github.io/) · [Upstream team repository](https://github.com/SOL1archive/CAFA6)

## Environment and external resources

The authoritative package requirements are [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock): **Python 3.12+**, PyTorch **2.9.1+**, Transformers, ESM, and the listed bioinformatics dependencies.

```bash
# From the repository root, with uv installed:
uv sync --frozen
source .venv/bin/activate
```

Training and model inference require suitable accelerator hardware, external data, and model downloads. Memory depends on the backbone, sequence length, and batch size; this repository does not establish a universal minimum. No data, checkpoints, or model weights are bundled.

Acquire the competition resources from [CAFA 6 on Kaggle](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data), then arrange them as described in [the dataset guide](docs/en/dataset_description.md). Use the competition's ontology snapshot and record resource release dates when comparing experiments.

## Actual notebook map

| Notebook | Contents | External requirements |
| --- | --- | --- |
| [eda.ipynb](eda.ipynb) | Sequence, GO-label, taxonomy, information-accretion and submission-format exploration | Competition files under `data/` |
| [prot-t5-pred.ipynb](notebooks/prot-t5-pred.ipynb) | Calls the repository's ProtT5 training/prediction modules; training is controlled by `DO_TRAIN` | Competition files and the selected checkpoint |
| [cafa-6-goa-prott5-ensemble-0-370.ipynb](notebooks/cafa-6-goa-prott5-ensemble-0-370.ipynb) | Weighted ensemble and GOA+ post-processing of existing predictions | Kaggle-mounted data and two precomputed TSV files |

The `0-370` filename is inherited and is not a score reproduced by this release. JEPA implementations are Python modules under `src/`; this checkout contains no JEPA training notebooks.

## Source entrypoints

| Pipeline | Source | Documentation |
| --- | --- | --- |
| Label-space JEPA | [train_label_jepa.py](src/jepa_go/train_label_jepa.py), [predict_label_jepa.py](src/jepa_go/predict_label_jepa.py) | [Pipeline guide](docs/en/pipelines.md) |
| JEPA encoder + classifier | [train_jepa_encoder.py](src/jepa_go/train_jepa_encoder.py), [predict_jepa_encoder.py](src/jepa_go/predict_jepa_encoder.py) | [Pipeline guide](docs/en/pipelines.md) |
| ProtT5 classifier | [prott5_go_train.py](src/train/prott5_go_train.py), [prott5_go_predict.py](src/test/prott5_go_predict.py) | [Pipeline guide](docs/en/pipelines.md) |
| ESM-C embeddings + classifier | [esm-c_model](src/esm-c_model) | [Pipeline guide](docs/en/pipelines.md) |
| GOA + ProtT5 ensemble | [make_ensemble_submission.py](scripts/make_ensemble_submission.py) | [English](docs/en/reproducing_kaggle_ensemble.md) / [한국어](docs/ko/reproducing_kaggle_ensemble.md) |

After obtaining the data and a trained model, prediction uses an existing entrypoint:

```bash
mkdir -p submissions
python -m src.jepa_go.predict_label_jepa \
  --model_dir models/your-trained-label-jepa-model \
  --test_fasta data/Test/testsuperset.fasta \
  --output submissions/label-jepa.tsv \
  --min_score 0.001
```

The model path is a placeholder for your checkpoint, not a bundled artifact. Optional `--enforce_consistency --obo_path data/Train/go-basic.obo` activates the available hierarchy-consistency post-processing.

The actual SLURM prediction script is [label-jepa-predict.sh](job-scripts/label-jepa-predict.sh). Job scripts retain their original cluster paths, partitions, and resource settings; adapt them and create the SLURM log directories before submitting. `job-scripts/train.sh` trains the **ProtT5 classifier**, not ESM-C.

## Reproducibility and attribution

This update corrects documentation against the checked-in code. It does not retrain models, regenerate external predictions, rerun the competition evaluation, or change the research pipeline. [Release scope](docs/en/release_scope.md) records the checks and remaining requirements.

The local ensemble script filters output IDs against the supplied test FASTA. That restricts the output universe; it does not establish that external GOA or model predictions satisfy temporal-data rules. Document their origin and snapshot dates separately.

The upstream README identifies the project as MIT-licensed, but this checkout has no standalone `LICENSE` file. Preserve upstream notices and verify applicable licensing with the [upstream project](https://github.com/SOL1archive/CAFA6) before redistribution. External datasets and pretrained models retain their own terms.
