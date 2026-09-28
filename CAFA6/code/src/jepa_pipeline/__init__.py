"""
JEPA Pipeline Package for CAFA6

This package provides:
1. JEPA-ProtT5 embedding generation
2. Base predictor training on JEPA embeddings
3. Final prediction generation
"""

from .jepa_embed import load_jepa_encoder, embed_sequences
from .jepa_train import JEPATrainer
from .jepa_predict import generate_predictions

__all__ = [
    "load_jepa_encoder",
    "embed_sequences",
    "JEPATrainer",
    "generate_predictions",
]
