"""
JEPA-based GO term prediction package.

This package provides two approaches for GO term prediction using JEPA concepts:

1. JEPA Encoder-based (train_jepa_encoder, predict_jepa_encoder):
   - Uses pretrained JEPA encoder (e.g., jepa-prott5-1) for protein embeddings
   - Trains classifier head on top of frozen encoder
   - Simple and effective transfer learning approach

2. Label-Space JEPA (train_label_jepa, predict_label_jepa):
   - Learns GO term embeddings jointly with protein representations
   - Uses JEPA-style masked prediction in label space
   - Context labels → Target label prediction
   - EMA target embeddings for stable training

Features:
- Ontology-based label propagation (child → parent, true path rule)
- Hierarchical consistency enforcement (P(parent) >= P(child))
- Taxonomy embedding for species-aware predictions

Modules:
- config: Configuration classes (JepaGoConfig, LabelJepaConfig)
- dataset: Dataset utilities (JepaGoDataset, LabelJepaDataset, InferenceDataset)
- models: Model definitions (JepaGoModel, LabelJepaModel)
- metrics: Evaluation metrics (compute_fmax, compute_auprc)
- ontology: GO hierarchy (GOntology, label propagation, consistency enforcement)
- taxonomy: Species embeddings (TaxonomyEmbedding, TaxonomyFusion)
- train_jepa_encoder: Training script for JEPA encoder approach
- predict_jepa_encoder: Prediction script for JEPA encoder approach
- train_label_jepa: Training script for Label-Space JEPA
- predict_label_jepa: Prediction script for Label-Space JEPA

Example usage:
    # JEPA Encoder approach
    python -m src.jepa_go.train_jepa_encoder \\
        --jepa_model_dir models/jepa-prott5-1 \\
        --output_dir models/jepa-go \\
        --use_wandb
    
    # Label-Space JEPA approach
    python -m src.jepa_go.train_label_jepa \\
        --model_family prott5 \\
        --output_dir models/label-jepa \\
        --use_wandb

Job scripts:
    - job-scripts/jepa-go-train-predict.sh: JEPA encoder train + predict
    - job-scripts/label-jepa-train.sh: Label-Space JEPA training only
    - job-scripts/label-jepa-predict.sh: Label-Space JEPA prediction only
    - job-scripts/label-jepa-full.sh: Label-Space JEPA full pipeline

Supports both ProtT5 and ESM backbone models with bfloat16 training.
"""

from .config import JepaGoConfig, LabelJepaConfig
from .dataset import (
    JepaGoDataset,
    LabelJepaDataset,
    InferenceDataset,
    load_terms,
    load_ia_weights,
    load_taxonomy_mapping,
    train_val_split,
)
from .models import (
    JepaGoModel,
    LabelJepaModel,
    build_encoder,
    load_jepa_go_model,
    load_label_jepa_model,
)
from .metrics import (
    compute_fmax,
    compute_auprc,
    MetricsAccumulator,
)
from .ontology import (
    GOntology,
    load_ontology,
    load_with_goatools,
    load_with_pronto,
)
from .taxonomy import (
    load_taxonomy,
    get_taxonomy_stats,
    TaxonomyEmbedding,
    TaxonomyFusion,
)

__all__ = [
    # Config
    "JepaGoConfig",
    "LabelJepaConfig",
    # Dataset
    "JepaGoDataset",
    "LabelJepaDataset",
    "InferenceDataset",
    "load_terms",
    "load_ia_weights",
    "load_taxonomy_mapping",
    "train_val_split",
    # Models
    "JepaGoModel",
    "LabelJepaModel",
    "build_encoder",
    "load_jepa_go_model",
    "load_label_jepa_model",
    # Metrics
    "compute_fmax",
    "compute_auprc",
    "MetricsAccumulator",
    # Ontology
    "GOntology",
    "load_ontology",
    "load_with_goatools",
    "load_with_pronto",
    # Taxonomy
    "load_taxonomy",
    "get_taxonomy_stats",
    "TaxonomyEmbedding",
    "TaxonomyFusion",
]

