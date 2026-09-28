"""
Path management for ESM-C model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_BASE_PATH = "/path/to/cafa6"


@dataclass(frozen=True)
class ESMCPaths:
    """Centralized path management for ESM-C model training and inference."""
    
    base_path: str = DEFAULT_BASE_PATH
    
    # =========================
    # Input Directories
    # =========================
    
    @property
    def train_dir(self) -> str:
        """Directory containing training data."""
        return os.path.join(self.base_path, "Train")
    
    @property
    def test_dir(self) -> str:
        """Directory containing test data."""
        return os.path.join(self.base_path, "Test")
    
    @property
    def feats_dir(self) -> str:
        """Directory containing precomputed features."""
        return os.path.join(self.base_path, "helpers", "feats")
    
    @property
    def cache_dir(self) -> str:
        """Cache directory for embeddings."""
        return os.path.join(self.base_path, "cache")
    
    @property
    def models_dir(self) -> str:
        """Directory for saved models."""
        return os.path.join(self.base_path, "models")
    
    # =========================
    # Input Files
    # =========================
    
    @property
    def train_fasta(self) -> str:
        """Path to training FASTA file."""
        return os.path.join(self.train_dir, "train_sequences.fasta")
    
    @property
    def test_fasta(self) -> str:
        """Path to test FASTA file."""
        return os.path.join(self.test_dir, "testsuperset.fasta")
    
    @property
    def train_ids_npy(self) -> str:
        """Path to train IDs numpy file."""
        return os.path.join(self.feats_dir, "train_ids.npy")
    
    @property
    def labels_npy(self) -> str:
        """Path to labels numpy file."""
        return os.path.join(self.feats_dir, "labels.npy")
    
    @property
    def Y_sparse_npz(self) -> str:
        """Path to sparse label matrix."""
        return os.path.join(self.feats_dir, "Y_sparse_float32.npz")
    
    @property
    def folds_npy(self) -> str:
        """Path to fold assignments numpy file."""
        return os.path.join(self.feats_dir, "folds.npy")
    
    # =========================
    # Embedding Directories
    # =========================
    
    def embed_dir(self, model: str, mode: str) -> str:
        """
        Get embedding directory for a specific model and mode.
        
        Args:
            model: ESM-C model name (e.g., 'esmc_300m')
            mode: Pooling mode (e.g., 'mean', 'meanmax', 'cls')
        
        Returns:
            Path to embedding directory.
        """
        return os.path.join(self.cache_dir, "embeds", "esmc", model, mode)
    
    def train_embeds_npy(self, model: str, mode: str) -> str:
        """Path to train embeddings numpy file."""
        return os.path.join(self.embed_dir(model, mode), "train_embeds.npy")
    
    def train_embed_ids_npy(self, model: str, mode: str) -> str:
        """Path to train embedding IDs numpy file."""
        return os.path.join(self.embed_dir(model, mode), "train_ids.npy")
    
    def test_embeds_npy(self, model: str, mode: str) -> str:
        """Path to test embeddings numpy file."""
        return os.path.join(self.embed_dir(model, mode), "test_embeds.npy")
    
    def test_embed_ids_npy(self, model: str, mode: str) -> str:
        """Path to test embedding IDs numpy file."""
        return os.path.join(self.embed_dir(model, mode), "test_ids.npy")
    
    # =========================
    # Output Directories
    # =========================
    
    def experiment_dir(self, exp_name: str) -> str:
        """Get experiment output directory."""
        return os.path.join(self.models_dir, exp_name)
    
    def base_output_dir(self, exp_name: str) -> str:
        """Get base predictor output directory."""
        return os.path.join(self.experiment_dir(exp_name), "base")
    
    # =========================
    # Output Files
    # =========================
    
    def oof_logits_npy(self, exp_name: str) -> str:
        """Path to out-of-fold logits."""
        return os.path.join(self.base_output_dir(exp_name), "oof_logits.npy")
    
    def test_logits_npy(self, exp_name: str) -> str:
        """Path to test logits."""
        return os.path.join(self.base_output_dir(exp_name), "test_logits.npy")
    
    def fold_checkpoint(self, exp_name: str, fold: int) -> str:
        """Path to fold checkpoint."""
        return os.path.join(self.base_output_dir(exp_name), f"fold_{fold}.pt")
    
    def meta_json(self, exp_name: str) -> str:
        """Path to experiment metadata."""
        return os.path.join(self.base_output_dir(exp_name), "meta.json")
    
    def metrics_txt(self, exp_name: str) -> str:
        """Path to metrics summary."""
        return os.path.join(self.base_output_dir(exp_name), "metrics.txt")
    
    # =========================
    # Utilities
    # =========================
    
    @staticmethod
    def ensure_dir(path: str) -> str:
        """Create directory if it doesn't exist and return the path."""
        os.makedirs(path, exist_ok=True)
        return path
    
    def ensure_embed_dir(self, model: str, mode: str) -> str:
        """Ensure embedding directory exists."""
        return self.ensure_dir(self.embed_dir(model, mode))
    
    def ensure_output_dir(self, exp_name: str) -> str:
        """Ensure experiment output directory exists."""
        return self.ensure_dir(self.base_output_dir(exp_name))
