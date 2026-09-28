"""
Configuration classes for JEPA-based GO term prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, List


ModelFamily = Literal["prott5", "esm"]


@dataclass
class JepaGoConfig:
    """
    Configuration for JEPA encoder-based GO term prediction.
    
    Uses a pretrained JEPA encoder to generate protein embeddings,
    then trains a classifier head for GO term prediction.
    """
    
    # Experiment
    name: str = "jepa-go"
    seed: int = 42
    debug_mode: bool = False
    
    # Model
    model_family: ModelFamily = "prott5"
    jepa_model_dir: Path = Path("models/jepa-prott5-1")
    
    # If jepa_model_dir is None, use this checkpoint directly
    backbone_ckpt: str = "Rostlab/prot_t5_xl_uniref50"
    
    # Data
    train_fasta: Path = Path("data/Train/train_sequences.fasta")
    train_terms_tsv: Path = Path("data/Train/train_terms.tsv")
    test_fasta: Path = Path("data/Test/testsuperset.fasta")
    go_obo: Path = Path("data/Train/go-basic.obo")
    ia_tsv: Path = Path("data/IA.tsv")
    
    # Label filtering
    namespace: Optional[str] = None  # None = all, or "MF", "BP", "CC"
    min_label_count: int = 5
    max_labels: int = 8192
    
    # Tokenization
    max_length: int = 1024
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Validation
    val_fraction: float = 0.03
    eval_steps: int = 500
    eval_thresholds: int = 51
    
    # Model architecture
    classifier_hidden: List[int] = field(default_factory=lambda: [1024, 512])
    dropout: float = 0.2
    freeze_encoder: bool = True
    
    # Precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    
    # Output
    output_dir: Path = Path("models/jepa-go")
    save_every_epoch: bool = True
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "cafa6"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_every: int = 50
    
    @classmethod
    def with_debug(cls, **kwargs):
        kwargs["debug_mode"] = True
        kwargs.setdefault("max_labels", 100)
        kwargs.setdefault("epochs", 1)
        kwargs.setdefault("eval_steps", 10)
        kwargs.setdefault("log_every", 5)
        return cls(**kwargs)
    
    def get_dtype(self):
        import torch
        if self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float16":
            return torch.float16
        return torch.float32


@dataclass
class LabelJepaConfig:
    """
    Configuration for Label-Space JEPA.
    
    Learns GO term embeddings jointly with protein representations,
    using JEPA-style masked prediction in label space.
    """
    
    # Experiment
    name: str = "label-jepa"
    seed: int = 42
    debug_mode: bool = False
    
    # Backbone Model
    model_family: ModelFamily = "prott5"
    backbone_ckpt: str = "Rostlab/prot_t5_xl_uniref50"
    
    # Optional: use JEPA pretrained encoder
    jepa_model_dir: Optional[Path] = None
    
    # Data
    train_fasta: Path = Path("data/Train/train_sequences.fasta")
    train_terms_tsv: Path = Path("data/Train/train_terms.tsv")
    test_fasta: Path = Path("data/Test/testsuperset.fasta")
    go_obo: Path = Path("data/Train/go-basic.obo")
    ia_tsv: Path = Path("data/IA.tsv")
    
    # Label filtering
    namespace: Optional[str] = None
    min_label_count: int = 5
    max_labels: int = 8192
    
    # Tokenization
    max_length: int = 1024
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Label-Space JEPA specific
    label_embed_dim: int = 512
    context_ratio: float = 0.6  # fraction of labels to use as context
    predictor_hidden_mult: int = 2
    ema_decay: float = 0.999
    
    # Loss weights
    jepa_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    
    # Validation
    val_fraction: float = 0.03
    eval_steps: int = 500
    eval_thresholds: int = 51
    
    # Model architecture
    projection_hidden: List[int] = field(default_factory=lambda: [1024])
    dropout: float = 0.2
    freeze_encoder: bool = True
    unfreeze_last_n_layers: int = 0
    
    # Precision
    dtype: str = "bfloat16"
    
    # Output
    output_dir: Path = Path("models/label-jepa")
    save_every_epoch: bool = True
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "cafa6"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_every: int = 50
    
    @classmethod
    def with_debug(cls, **kwargs):
        kwargs["debug_mode"] = True
        kwargs.setdefault("max_labels", 100)
        kwargs.setdefault("epochs", 2)
        kwargs.setdefault("eval_steps", 10)
        kwargs.setdefault("log_every", 5)
        return cls(**kwargs)
    
    def get_dtype(self):
        import torch
        if self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float16":
            return torch.float16
        return torch.float32
