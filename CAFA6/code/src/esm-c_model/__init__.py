"""
ESM-C Model Package for CAFA6

This package provides functionality for:
1. Generating protein embeddings using ESM-C models
2. Training base predictors for GO term prediction
3. Cross-validation with taxonomy-aware splits
"""

from .config import ESMCConfig, TrainConfig
from .paths import ESMCPaths
from .models import MLPTrunk, BasePredictor
from .dataset import TrainDataset, collate_fn, sample_negatives, build_batch_indices
from .metrics import fmax_micro, compute_precision_recall
from .trainer import ESMCTrainer

__all__ = [
    "ESMCConfig",
    "TrainConfig",
    "ESMCPaths",
    "MLPTrunk",
    "BasePredictor",
    "TrainDataset",
    "collate_fn",
    "sample_negatives",
    "build_batch_indices",
    "fmax_micro",
    "compute_precision_recall",
    "ESMCTrainer",
]
