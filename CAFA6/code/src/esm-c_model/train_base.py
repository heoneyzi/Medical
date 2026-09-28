#!/usr/bin/env python3
"""
Train base predictor for CAFA6 (embedding -> logits).

This script trains a base predictor model using precomputed protein embeddings
from ESM-C models. It supports cross-validation with taxonomy-aware splits
and produces out-of-fold (OOF) and test predictions.

Outputs:
  - oof_logits.npy: Out-of-fold predictions [N_train, L] (float16)
  - test_logits.npy: Test predictions [N_test, L] (float16)
  - fold_{k}.pt: Model checkpoints per fold
  - metrics.txt: Summary of fold scores

Example usage:
    python -m src.esm_c_model.train_base --config configs/esmc_base.yaml

Configuration (YAML):
    name: experiment_name
    paths:
      base_path: /path/to/cafa6
    embed:
      model: esmc_300m
      mode: mean
    base:
      head: mlp
      hidden: [1024, 512]
      dropout: 0.2
      lr: 1e-3
      epochs: 5
      batch_size: 512
      neg_k: 1024
    cv:
      seed: 42
"""

from __future__ import annotations

import argparse
import torch

from .config import TrainConfig, load_yaml
from .paths import ESMCPaths
from .trainer import ESMCTrainer


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train ESM-C base predictor for CAFA6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device", 
        default=None, 
        help="Override device (e.g., cuda:0, cpu)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = TrainConfig.from_yaml(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Setup device
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[train_base] Experiment: {config.name}")
    print(f"[train_base] Device: {device}")
    
    # Create paths
    paths = ESMCPaths(base_path=config.base_path)
    
    # Create trainer and run
    trainer = ESMCTrainer(
        config=config,
        device=device,
        paths=paths,
    )
    
    # Load data and train
    trainer.load_data()
    summary = trainer.train()
    
    print("\n[train_base] Training complete!")
    print(f"  Mean F1: {summary['mean_fold_score']:.4f}")
    print(f"  Fold scores: {summary['fold_scores']}")


if __name__ == "__main__":
    main()
