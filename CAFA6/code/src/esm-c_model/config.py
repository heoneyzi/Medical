"""
Configuration classes for ESM-C model training and embedding generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class EmbedConfig:
    """Configuration for ESM-C embedding generation."""
    
    model: str = "esmc_300m"
    mode: str = "mean"  # mean, meanmax, cls
    device: str = "cuda:0"
    batch_size: int = 8
    cls_index: int = 0
    sort_by_length: bool = True
    
    @classmethod
    def from_dict(cls, d: dict) -> "EmbedConfig":
        return cls(
            model=d.get("model", "esmc_300m"),
            mode=d.get("mode", "mean"),
            device=d.get("device", "cuda:0"),
            batch_size=d.get("batch_size", 8),
            cls_index=d.get("cls_index", 0),
            sort_by_length=d.get("sort_by_length", True),
        )


@dataclass
class BaseTrainConfig:
    """Configuration for base predictor training."""
    
    head: str = "linear"  # linear or mlp
    hidden: List[int] = field(default_factory=lambda: [1024, 512])
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 512
    val_batch_size: int = 1024
    neg_k: int = 1024  # number of negative samples
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True  # automatic mixed precision
    label_limit: int = 0  # 0 = no limit
    fmax_thresholds: Optional[List[float]] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "BaseTrainConfig":
        return cls(
            head=d.get("head", "linear"),
            hidden=d.get("hidden", [1024, 512]),
            dropout=float(d.get("dropout", 0.2)),
            lr=float(d.get("lr", 1e-3)),
            epochs=int(d.get("epochs", 5)),
            batch_size=int(d.get("batch_size", 512)),
            val_batch_size=int(d.get("val_batch_size", 1024)),
            neg_k=int(d.get("neg_k", 1024)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            grad_clip=float(d.get("grad_clip", 1.0)),
            amp=bool(d.get("amp", True)),
            label_limit=int(d.get("label_limit", 0)),
            fmax_thresholds=d.get("fmax_thresholds"),
        )


@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    
    seed: int = 42
    n_folds: int = 5
    
    @classmethod
    def from_dict(cls, d: dict) -> "CVConfig":
        return cls(
            seed=int(d.get("seed", 42)),
            n_folds=int(d.get("n_folds", 5)),
        )


@dataclass
class ESMCConfig:
    """Configuration for ESM-C embedding generation."""
    
    base_path: str
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ESMCConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            base_path=cfg.get("paths", {}).get("base_path", "/path/to/cafa6"),
            embed=EmbedConfig.from_dict(cfg.get("embed", {})),
        )


@dataclass
class TrainConfig:
    """Complete configuration for training."""
    
    name: str
    base_path: str
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    base: BaseTrainConfig = field(default_factory=BaseTrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    device: str = "cuda:0"
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        
        name = cfg.get("name", os.path.splitext(os.path.basename(path))[0])
        base_path = cfg.get("paths", {}).get("base_path", "/path/to/cafa6")
        device = cfg.get("device") or cfg.get("embed", {}).get("device") or "cuda:0"
        
        return cls(
            name=name,
            base_path=base_path,
            embed=EmbedConfig.from_dict(cfg.get("embed", {})),
            base=BaseTrainConfig.from_dict(cfg.get("base", {})),
            cv=CVConfig.from_dict(cfg.get("cv", {})),
            device=device,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "paths": {"base_path": self.base_path},
            "embed": {
                "model": self.embed.model,
                "mode": self.embed.mode,
                "device": self.embed.device,
                "batch_size": self.embed.batch_size,
            },
            "base": {
                "head": self.base.head,
                "hidden": self.base.hidden,
                "dropout": self.base.dropout,
                "lr": self.base.lr,
                "epochs": self.base.epochs,
                "batch_size": self.base.batch_size,
                "neg_k": self.base.neg_k,
            },
            "cv": {
                "seed": self.cv.seed,
                "n_folds": self.cv.n_folds,
            },
            "device": self.device,
        }


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
