# ESM-C Model for CAFA6 - Implementation Guide

> Historical implementation notes, preserved from the upstream team repository. Some examples below use obsolete package names, configuration files, or prediction scripts. Use the current [pipeline guide](docs/en/pipelines.md) for verified file names, command flags, environment requirements, and external-data prerequisites.

This document describes the implementation details of the ESM-C based protein function prediction pipeline for CAFA6.

## Overview

The `esm-c_model` package provides a complete pipeline for:
1. **Embedding Generation**: Generate protein embeddings using ESM-C language models
2. **Base Predictor Training**: Train a predictor to map embeddings to GO term predictions
3. **Cross-Validation**: Taxonomy-aware k-fold cross-validation
4. **Inference**: Generate predictions for test sequences

## Architecture

```
src/esm-c_model/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses
├── paths.py             # Path management
├── models.py            # Neural network models
├── dataset.py           # Data loading utilities
├── metrics.py           # Evaluation metrics
├── trainer.py           # Training orchestration
├── esmc_embed.py        # Embedding generation script
└── train_base.py        # Main training entry point
```

## Module Descriptions

### 1. Configuration (`config.py`)

Provides type-safe configuration management using dataclasses:

- **`EmbedConfig`**: ESM-C model settings (model name, pooling mode, batch size)
- **`BaseTrainConfig`**: Training hyperparameters (learning rate, epochs, negative sampling)
- **`CVConfig`**: Cross-validation settings (seed, number of folds)
- **`TrainConfig`**: Complete training configuration combining all above

```python
from src.esm_c_model import TrainConfig

config = TrainConfig.from_yaml("configs/esmc_base.yaml")
print(config.base.lr)  # 0.001
```

### 2. Path Management (`paths.py`)

Centralized path handling for all input/output files:

```python
from src.esm_c_model import ESMCPaths

paths = ESMCPaths(base_path="/path/to/cafa6")

# Input paths
paths.train_fasta          # Train/train_sequences.fasta
paths.train_ids_npy        # helpers/feats/train_ids.npy
paths.Y_sparse_npz         # helpers/feats/Y_sparse_float32.npz

# Embedding paths
paths.embed_dir("esmc_300m", "mean")  # cache/embeds/esmc/esmc_300m/mean/

# Output paths
paths.oof_logits_npy("exp1")  # models/exp1/base/oof_logits.npy
```

### 3. Models (`models.py`)

Neural network architectures for GO term prediction:

#### MLPTrunk
Multi-layer perceptron for feature transformation:
- Architecture: Linear → GELU → Dropout (repeated)
- Configurable hidden layer sizes and dropout

#### BasePredictor
Main prediction model:
- **Linear head**: Direct projection from embeddings to labels
- **MLP head**: MLPTrunk + linear classifier
- Supports efficient partial logit computation for negative sampling

```python
from src.esm_c_model import BasePredictor

model = BasePredictor(
    in_dim=1280,      # ESM-C embedding dimension
    n_labels=45000,   # Number of GO terms
    head="mlp",
    hidden=[1024, 512],
    dropout=0.2,
)
```

### 4. Dataset (`dataset.py`)

Data loading utilities for extreme multi-label classification:

- **`TrainDataset`**: PyTorch dataset wrapping embeddings and sparse labels
- **`collate_fn`**: Custom collation for variable-length positive labels
- **`sample_negatives`**: Efficient negative sampling using rejection sampling
- **`build_batch_indices`**: Build padded tensors for BCE loss with negative sampling

Key optimization: **Global negative sampling per batch** reduces overhead while maintaining diversity.

### 5. Metrics (`metrics.py`)

Evaluation metrics for protein function prediction:

- **`fmax_micro`**: Micro-averaged F-max score (primary CAFA metric)
- **`compute_precision_recall`**: Precision, recall, F1 at a threshold
- **`MetricsTracker`**: Track training progress and best checkpoints

```python
from src.esm_c_model import fmax_micro

best_f1, best_threshold = fmax_micro(Y_true_csr, Y_prob, thresholds=[0.1, 0.2, 0.3])
```

### 6. Trainer (`trainer.py`)

Main training orchestration class:

```python
from src.esm_c_model import ESMCTrainer, TrainConfig

config = TrainConfig.from_yaml("config.yaml")
trainer = ESMCTrainer(config)

trainer.load_data()
summary = trainer.train()

print(f"Mean F1: {summary['mean_fold_score']:.4f}")
```

Features:
- Automatic Mixed Precision (AMP) training
- Gradient clipping
- Early stopping based on validation F-max
- Memory-mapped OOF logits for large datasets
- GPU memory cleanup between folds

### 7. Embedding Generation (`esmc_embed.py`)

Generate protein embeddings from FASTA sequences:

```bash
python -m src.esm_c_model.esmc_embed \
    --base-path /path/to/cafa6 \
    --model esmc_300m \
    --modes mean,meanmax,cls \
    --run train \
    --batch-size 8 \
    --sort-by-length
```

Pooling modes:
- **mean**: Mean pooling over sequence length `[D]`
- **meanmax**: Concatenation of mean and max pooling `[2D]`
- **cls**: CLS token (first position) `[D]`

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. Embedding Generation                      │
│  FASTA sequences → ESM-C model → Pooled embeddings (.npy)       │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                     2. Base Predictor Training                   │
│  Embeddings + Labels → Cross-validation → OOF & Test logits     │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                     3. Post-processing (Optional)                │
│  Logits → Threshold optimization → Final predictions            │
└─────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Step 1: Generate Embeddings

```bash
# Generate train embeddings
python -m src.esm_c_model.esmc_embed \
    --model esmc_300m \
    --modes mean \
    --run train \
    --batch-size 8

# Generate test embeddings
python -m src.esm_c_model.esmc_embed \
    --model esmc_300m \
    --modes mean \
    --test-fasta data/Test/testsuperset.fasta \
    --run test
```

### Step 2: Train Base Predictor

```bash
python -m src.esm_c_model.train_base --config configs/esmc_base.yaml
```

### Step 3: Check Outputs

```
models/{exp_name}/base/
├── fold_0.pt           # Fold 0 checkpoint
├── fold_1.pt           # Fold 1 checkpoint
├── ...
├── oof_logits.npy      # Out-of-fold predictions [N_train, L]
├── test_logits.npy     # Test predictions [N_test, L]
├── train_ids.npy       # Training IDs (aligned)
├── test_ids.npy        # Test IDs
├── labels.npy          # GO term labels
├── meta.json           # Experiment metadata
└── metrics.txt         # Fold scores summary
```

## Configuration Reference

### YAML Configuration

```yaml
name: experiment_name

paths:
  base_path: /path/to/cafa6

embed:
  model: esmc_300m        # esmc_300m | esmc_600m | esmc_1b
  mode: mean              # mean | meanmax | cls

base:
  head: mlp               # linear | mlp
  hidden: [1024, 512]     # MLP hidden sizes
  dropout: 0.2
  lr: 0.001
  epochs: 5
  batch_size: 512
  neg_k: 1024             # Negative samples per batch
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: true

cv:
  seed: 42
```

## Key Design Decisions

### 1. Negative Sampling

For extreme multi-label classification with ~45k labels:
- **Global batch negatives**: Sample negatives shared across the batch
- **Efficient BCE**: Compute loss only on selected (positive + negative) indices
- **Memory efficient**: Avoid computing full [B, L] logit matrices

### 2. Taxonomy-Aware Splits

Uses precomputed `folds.npy` from GroupKFold:
- Groups proteins by taxonomy (species/genus)
- Prevents data leakage from related organisms
- More realistic evaluation for novel proteins

### 3. Memory Management

- **Memory-mapped OOF logits**: Handle large prediction matrices
- **Float16 storage**: Reduce disk footprint by 2x
- **GPU cache cleanup**: Clear between folds

### 4. Modular Architecture

Separation of concerns enables:
- Easy experimentation with different components
- Unit testing of individual modules
- Reuse across different experiments

## Dependencies

```
torch>=2.0
numpy
scipy
tqdm
pyyaml
esm  # ESM-C SDK
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `neg_k` (negative samples)
- Use `--device cpu` for debugging

### Import Errors
- Ensure `esm` package is installed for ESM-C
- Check Python path includes project root

### Embedding Mismatch
- Ensure train/test embeddings use same model and mode
- Verify `train_ids.npy` alignment with labels

## Future Improvements

1. **Multi-GPU Training**: Distributed data parallel for larger models
2. **Hierarchical Labels**: Leverage GO term hierarchy (is-a relationships)
3. **Ensemble Methods**: Combine multiple embedding types/models
4. **GCN Refinement**: Graph convolutional network on label co-occurrence

---

# JEPA-based GO Term Prediction

This section describes the JEPA-based approaches for GO term prediction implemented in `src/jepa_go/`.

## Overview

Two JEPA-based approaches are provided:

### 1. JEPA Encoder Approach
Uses a pretrained JEPA encoder (e.g., `models/jepa-prott5-1`) as a frozen feature extractor with a trainable classifier head.

### 2. Label-Space JEPA (Option C)
A novel approach that:
- Learns GO term embeddings jointly with protein representations
- Uses JEPA-style masked prediction in label space
- Predicts hidden target labels from context labels
- Uses EMA target embeddings for stable training

## Architecture

```
src/jepa_go/
├── __init__.py                # Package exports
├── config.py                  # JepaGoConfig, LabelJepaConfig
├── dataset.py                 # Dataset utilities
├── models.py                  # JepaGoModel, LabelJepaModel
├── metrics.py                 # F-max, AUPRC evaluation
├── train_jepa_encoder.py      # JEPA encoder training
├── predict_jepa_encoder.py    # JEPA encoder prediction
├── train_label_jepa.py        # Label-Space JEPA training
└── predict_label_jepa.py      # Label-Space JEPA prediction
```

## Model Descriptions

### JepaGoModel (JEPA Encoder Approach)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Protein Sequence                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            JEPA Pretrained Encoder (frozen)                      │
│            (ProtT5 or ESM with JEPA weights)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Mean Pooling                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Classifier Head (MLP, trainable)                    │
│              [1024] → [512] → [num_labels]                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    GO Term Predictions                           │
└─────────────────────────────────────────────────────────────────┘
```

### LabelJepaModel (Label-Space JEPA)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Protein Sequence                              │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    Protein Encoder (ProtT5/ESM)                       │
│                         + Mean Pooling                                │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    Projection to Label Space                          │
│                    protein_dim → label_embed_dim                      │
└──────────────────────────────────────────────────────────────────────┘
             ↓                                           ↓
┌────────────────────────┐                  ┌────────────────────────────┐
│   Context Labels       │                  │   Target Labels (masked)    │
│   (60% of positives)   │                  │   (40% of positives)        │
└────────────────────────┘                  └────────────────────────────┘
             ↓                                           ↓
┌────────────────────────┐                  ┌────────────────────────────┐
│  Online Label Embed.   │                  │   EMA Target Label Embed.   │
│  (trainable)           │                  │   (exponential moving avg)  │
└────────────────────────┘                  └────────────────────────────┘
             ↓                                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         JEPA Predictor                                │
│              (protein_embed + context_embed) → target_pred            │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    JEPA Loss: cosine(pred, target)                    │
│                    + Classification Loss (auxiliary)                  │
└──────────────────────────────────────────────────────────────────────┘
```

## Configuration

### JepaGoConfig (JEPA Encoder)

```python
from src.jepa_go import JepaGoConfig

config = JepaGoConfig(
    # Model
    model_family="prott5",  # or "esm"
    jepa_model_dir=Path("models/jepa-prott5-1"),
    
    # Training
    batch_size=8,
    epochs=5,
    learning_rate=1e-4,
    dtype="bfloat16",  # default
    
    # Labels
    namespace=None,  # all GO terms, or "MF"/"BP"/"CC"
    max_labels=8192,
    
    # Architecture
    classifier_hidden=[1024, 512],
    freeze_encoder=True,
    
    # Logging
    use_wandb=True,
    wandb_project="cafa6",
)
```

### LabelJepaConfig (Label-Space JEPA)

```python
from src.jepa_go import LabelJepaConfig

config = LabelJepaConfig(
    # Model
    model_family="prott5",
    backbone_ckpt="Rostlab/prot_t5_xl_uniref50",
    
    # Label-Space JEPA specific
    label_embed_dim=512,
    context_ratio=0.6,  # 60% context, 40% target
    ema_decay=0.999,
    jepa_loss_weight=1.0,
    classification_loss_weight=1.0,
    
    # Training
    epochs=10,
    dtype="bfloat16",
)
```

## Job Scripts

### JEPA Encoder (Train + Predict)

```bash
# Submit job
sbatch job-scripts/jepa-go-train-predict.sh

# Or with custom settings
EPOCHS=10 BATCH_SIZE=4 OUTPUT_DIR=models/jepa-go-v2 \
    sbatch job-scripts/jepa-go-train-predict.sh
```

### Label-Space JEPA

Three separate scripts for flexibility:

```bash
# Training only
sbatch job-scripts/label-jepa-train.sh

# Prediction only (requires trained model)
MODEL_DIR=models/label-jepa sbatch job-scripts/label-jepa-predict.sh

# Full pipeline (train + predict)
sbatch job-scripts/label-jepa-full.sh
```

### ESM Backbone

To use ESM instead of ProtT5:

```bash
MODEL_FAMILY=esm \
BACKBONE_CKPT=facebook/esm2_t33_650M_UR50D \
    sbatch job-scripts/label-jepa-full.sh
```

## Training Details

### Precision
- Default: **bfloat16** (best for modern GPUs)
- Options: `bfloat16`, `float16`, `float32`

### Encoder Freezing
- Default: Encoder frozen (only classifier/projection trainable)
- Option: `--no_freeze_encoder` for full fine-tuning
- Option: `--unfreeze_last_n_layers N` for partial fine-tuning

### Label-Space JEPA Loss

The Label-Space JEPA uses two losses:

1. **JEPA Loss**: Cosine similarity between predicted and target label embeddings
   ```
   L_jepa = 1 - cos(pred_embed, target_embed)
   ```

2. **Classification Loss**: Standard BCE loss for all labels (auxiliary)
   ```
   L_cls = BCE(logits, all_labels)
   ```

Total: `L = jepa_weight * L_jepa + cls_weight * L_cls`

## Outputs

### Model Checkpoints

```
models/{name}/
├── config.json           # Model configuration
├── labels.json           # Label list and mapping
├── classifier.pt         # Classifier weights (JEPA encoder)
├── model.pt              # Full model weights (Label JEPA)
├── predictions.tsv       # GO term predictions
└── submission.tsv        # Kaggle submission format
```

### Prediction Format

```tsv
Protein_ID    GO_term    Probability
P12345        GO:0005515 0.9234
P12345        GO:0006810 0.8123
...
```

## Switching Between Backbones

Both approaches support ProtT5 and ESM backbones:

| Backbone | Checkpoint | Hidden Dim |
|----------|------------|------------|
| ProtT5-XL | `Rostlab/prot_t5_xl_uniref50` | 1024 |
| ESM2-650M | `facebook/esm2_t33_650M_UR50D` | 1280 |
| ESM2-3B | `facebook/esm2_t36_3B_UR50D` | 2560 |

Example with ESM:
```bash
python -m src.jepa_go.train_label_jepa \
    --model_family esm \
    --backbone_ckpt facebook/esm2_t33_650M_UR50D \
    --output_dir models/label-jepa-esm
```

## Why Label-Space JEPA?

The Label-Space JEPA approach has several advantages:

1. **Captures Label Relationships**: GO terms form a hierarchical ontology; learning embeddings allows the model to capture semantic relationships.

2. **Self-Supervised Signal**: JEPA-style prediction provides additional training signal beyond supervised classification.

3. **Regularization**: The dual-loss setup (JEPA + classification) acts as regularization, potentially improving generalization.

4. **Flexible Inference**: At inference time, only the classification path is used, so no overhead.

5. **EMA Stability**: Target embeddings using EMA provide stable training targets, similar to momentum contrast methods.

---

## Prediction Threshold

> **Important**: The default prediction threshold is **0.01** to ensure comprehensive predictions.

The threshold determines how many GO term predictions are output per protein:

| Threshold | Predictions/Protein | Output Size |
|-----------|---------------------|-------------|
| 0.5 | ~3 | ~20MB |
| 0.2 | ~5-10 | ~50MB |
| 0.01 | ~30-50 | ~1.45GB |

Higher thresholds result in smaller files but may miss valid predictions. The CAFA evaluation uses all predictions, so lower thresholds are preferred.

### Setting Threshold

```bash
# In job scripts
THRESHOLD=0.01 sbatch job-scripts/label-jepa-predict-v2.sh

# In Python
python -m src.jepa_go.predict_label_jepa \
    --model_dir models/label-jepa \
    --threshold 0.01
```

---

## Inference Optimization

The prediction scripts use optimized batched inference for maximum performance:

### 1. `torch.inference_mode()`

More efficient than `torch.no_grad()` — disables view tracking and version counter bumps:

```python
@torch.inference_mode()
def predict(...):
    # All operations in this function are optimized for inference
```

### 2. Vectorized Batch Processing

Instead of processing one protein at a time, operations are vectorized:

```python
# Batched threshold mask (entire batch at once)
mask = probs >= threshold  # [batch_size, num_labels]

# Vectorized sorting per protein
sorted_order = selected_probs.argsort(descending=True)
sorted_indices = indices[sorted_order]
```

### 3. Efficient Tensor Extraction

Uses `.tolist()` once instead of converting elements individually:

```python
predictions = list(zip(
    [label_list[idx] for idx in sorted_indices.tolist()],
    sorted_probs.tolist()
))
```

---

## Updated Job Scripts (2026-02-02)

### New Prediction-Only Scripts

| Script | Purpose |
|--------|---------|
| `label-jepa-predict-v2.sh` | Label-Space JEPA prediction with low threshold (0.01) |
| `jepa-go-predict.sh` | JEPA Encoder prediction only |

### Environment Variables

All job scripts support customization via environment variables:

```bash
# Training parameters
EPOCHS=20 BATCH_SIZE=8 LR=1e-4

# Model selection
MODEL_FAMILY=esm BACKBONE_CKPT=facebook/esm2_t33_650M_UR50D

# Prediction settings
THRESHOLD=0.01 MODEL_DIR=models/label-jepa-20260202

# Example
EPOCHS=5 BATCH_SIZE=4 sbatch job-scripts/label-jepa-full.sh
```
