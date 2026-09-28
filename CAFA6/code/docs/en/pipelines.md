# Training and prediction entrypoints

This guide maps the current source. It does not claim that every pipeline has been rerun or that every experiment produces the medal-winning submission. See the [README](../../README.md) for environment setup and team attribution.

## External requirements

- Competition files under `data/Train/`: `train_sequences.fasta`, `train_terms.tsv`, `train_taxonomy.tsv`, `go-basic.obo`; plus `data/IA.tsv` and `data/Test/testsuperset.fasta` as needed.
- Configured model downloads: ESM2 through `facebook/esm2_t33_650M_UR50D`, ProtT5 through `Rostlab/prot_t5_xl_uniref50`, or ESM-C through the `esm` SDK. Respect each model's access requirements and terms.
- Compatible pretrained JEPA or classifier checkpoints for inference-only runs.
- Optional Weights & Biases access when enabling `--use_wandb`. The direct CLI does not enable this flag by default.

The CV records 82,404 training sequences and 224,309 test-superset sequences. Counts were not recomputed during this update; the hidden scored test set is a subset of the superset.

## Label-space JEPA

The model combines a protein encoder, projection, online/EMA label embeddings, masked label prediction, and classification. See [models.py](../../src/jepa_go/models.py), [config.py](../../src/jepa_go/config.py), [training](../../src/jepa_go/train_label_jepa.py), and [prediction](../../src/jepa_go/predict_label_jepa.py).

```bash
python -m src.jepa_go.train_label_jepa \
  --model_family esm \
  --backbone_ckpt facebook/esm2_t33_650M_UR50D \
  --train_fasta data/Train/train_sequences.fasta \
  --train_terms_tsv data/Train/train_terms.tsv \
  --output_dir models/label-jepa

mkdir -p submissions
python -m src.jepa_go.predict_label_jepa \
  --model_dir models/label-jepa \
  --test_fasta data/Test/testsuperset.fasta \
  --output submissions/label-jepa.tsv --min_score 0.001
```

These launch actual training/inference and require the resources above. They were not executed for this documentation update.

Selected direct-CLI defaults, from the argument parsers:

| Setting | Default |
| --- | --- |
| `--model_family` / `--backbone_ckpt` | `prott5` / `Rostlab/prot_t5_xl_uniref50` |
| Training `--batch_size` / `--epochs` | `8` / `10` |
| `--max_length` | `1024` |
| `--label_embed_dim` / `--context_ratio` | `512` / `0.6` |
| `--ema_decay` | `0.999` |
| Prediction `--min_score` | `0.001` |

The full SLURM script overrides some defaults, including ESM2, batch size 16, and maximum length 1536. Its environment variables are not the direct Python CLI defaults. Training offers `--use_ontology_propagation` and optional taxonomy features; prediction offers `--enforce_consistency` and `--obo_path`. These features are opt-in in the current parsers.

Prediction uses the configuration, labels, and weights from a compatible trained model directory and writes headerless `(protein ID, GO term, score)` TSV rows. Output size depends on the checkpoint and score cutoff; no fixed file size or predictions-per-protein count is guaranteed.

## JEPA encoder + classifier

See [train_jepa_encoder.py](../../src/jepa_go/train_jepa_encoder.py), [predict_jepa_encoder.py](../../src/jepa_go/predict_jepa_encoder.py), and [pretraining](../../src/pretrain/jepa_protein_train.py).

```bash
python -m src.jepa_go.train_jepa_encoder \
  --jepa_model_dir models/your-pretrained-jepa-model \
  --output_dir models/jepa-go

python -m src.jepa_go.predict_jepa_encoder \
  --model_dir models/jepa-go \
  --output submissions/jepa-go.tsv --min_score 0.001
```

The direct training parser defaults to ProtT5, batch size 8, and 5 epochs. The pretrained-model path must point to an appropriate checkpoint. Model weights are excluded from Git.

## ProtT5 classifier

[prot-t5-pred.ipynb](../../notebooks/prot-t5-pred.ipynb) calls [prott5_go_train.py](../../src/train/prott5_go_train.py) and [prott5_go_predict.py](../../src/test/prott5_go_predict.py). Inspect its `DO_TRAIN`, paths, model directory, and namespace before running it.

[train.sh](../../job-scripts/train.sh) trains separate MF/BP/CC ProtT5 classifiers; [test.sh](../../job-scripts/test.sh) and [train-test.sh](../../job-scripts/train-test.sh) support the related workflow. They retain the original cluster settings.

## ESM-C embeddings and classifier

The actual directory is [`src/esm-c_model`](../../src/esm-c_model), with a hyphen. Historical examples use a different package spelling. Python's `-m` accepts the checked-in module path:

```bash
python -m src.esm-c_model.esmc_embed \
  --base-path data \
  --train-fasta data/Train/train_sequences.fasta \
  --train-ids data/helpers/feats/train_ids.npy \
  --model esmc_300m --modes mean --run train --batch-size 8

python -m src.esm-c_model.train_base \
  --config configs/esmc_base_example.yaml
```

Before using the example configuration, set `paths.base_path` to your own prepared data workspace. The embedding builder needs ordered `train_ids.npy`; the classifier additionally expects preprocessed label features and folds through [ESMCPaths](../../src/esm-c_model/paths.py). These helper artifacts are not created merely by downloading the competition FASTA files. Inspect [the trainer](../../src/esm-c_model/trainer.py) and preserve alignment between IDs, embeddings, labels, and folds.

No source directory was renamed during this update. See [esmc_embed.py](../../src/esm-c_model/esmc_embed.py), [train_base.py](../../src/esm-c_model/train_base.py), and [esmc_base_example.yaml](../../configs/esmc_base_example.yaml) for actual interfaces. The original [AGENT.md](../../AGENT.md) is an archived implementation guide containing legacy examples.

## Ensemble

[make_ensemble_submission.py](../../scripts/make_ensemble_submission.py) combines external GOA and ProtT5+InterPro predictions and applies ontology post-processing. The [ensemble guide](reproducing_kaggle_ensemble.md) distinguishes its local defaults from the original Kaggle notebook.

## Metrics

The JEPA metrics module exposes `compute_fmax(predictions, targets, ia_weights=None, num_thresholds=51)` and separate `compute_auprc(predictions, targets)`:

```python
from src.jepa_go.metrics import compute_fmax, compute_auprc

fmax, best_threshold, semantic_proxy = compute_fmax(
    y_prob, y_true, ia_weights=ia_weights, num_thresholds=51
)
auprc = compute_auprc(y_prob, y_true)
```

The third F-max result is the implementation's IA-weighted false-positive/false-negative proxy at the best F1 threshold, named `smin` in source. It is not AUPRC or a threshold-minimized official S-min computation. Local validation metrics do not establish an official competition score. ESM-C also has its own [metrics](../../src/esm-c_model/metrics.py).

## SLURM scripts

| Existing script | Actual role |
| --- | --- |
| [label-jepa-full.sh](../../job-scripts/label-jepa-full.sh) | Label-space JEPA training followed by prediction |
| [label-jepa-train.sh](../../job-scripts/label-jepa-train.sh) | Label-space JEPA training |
| [label-jepa-predict.sh](../../job-scripts/label-jepa-predict.sh) | Label-space JEPA prediction; `MIN_SCORE=0.001` by default |
| [jepa-go-train-predict.sh](../../job-scripts/jepa-go-train-predict.sh) | JEPA encoder classifier training/prediction |
| [jepa-go-predict.sh](../../job-scripts/jepa-go-predict.sh) | JEPA encoder classifier prediction |
| [train.sh](../../job-scripts/train.sh) | ProtT5 MF/BP/CC training |

Adapt the working directory, `#SBATCH` output/error locations, partition, resources, and environment before submitting. Log directories must exist when SLURM opens them. Dated default model paths reflect the original environment; those models are not present in this clone.
