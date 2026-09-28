"""
Training script for JEPA encoder-based GO term prediction.

Uses a pretrained JEPA encoder (from jepa-prott5-1 or similar) with
a classifier head for GO term prediction.

Usage:
    python -m src.jepa_go.train_jepa_encoder \
        --jepa_model_dir models/jepa-prott5-1 \
        --output_dir models/jepa-go-v1 \
        --epochs 5 \
        --use_wandb
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.fasta import read_fasta
from src.jepa_go.config import JepaGoConfig
from src.jepa_go.dataset import (
    JepaGoDataset,
    load_ia_weights,
    load_terms,
    train_val_split,
)
from src.jepa_go.metrics import MetricsAccumulator, compute_fmax
from src.jepa_go.models import JepaGoModel, build_encoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer_scheduler(
    model: nn.Module,
    config: JepaGoConfig,
    num_training_steps: int,
):
    """Create optimizer and learning rate scheduler."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    
    # Only optimize classifier (encoder may be frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=num_training_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="cos",
    )
    
    return optimizer, scheduler


def train_epoch(
    model: JepaGoModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: JepaGoConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    wandb_run=None,
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Use bfloat16 autocast
    dtype = config.get_dtype()
    use_amp = dtype != torch.float32
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with autocast(enabled=use_amp, dtype=dtype):
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging
            if global_step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": loss.item() * config.gradient_accumulation_steps, "lr": lr})
                
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": loss.item() * config.gradient_accumulation_steps,
                        "train/lr": lr,
                        "train/step": global_step,
                    })
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: JepaGoModel,
    dataloader: DataLoader,
    config: JepaGoConfig,
    device: torch.device,
    ia_weights: torch.Tensor = None,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    dtype = config.get_dtype()
    use_amp = dtype != torch.float32
    
    accumulator = MetricsAccumulator()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with autocast(enabled=use_amp, dtype=dtype):
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        probs = torch.sigmoid(logits)
        accumulator.update(probs, labels)
    
    metrics = accumulator.compute(ia_weights, num_thresholds=config.eval_thresholds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    
    return metrics


def save_checkpoint(
    model: JepaGoModel,
    config: JepaGoConfig,
    label_list: list,
    label_to_idx: dict,
    output_dir: Path,
    epoch: int = None,
):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classifier weights
    if epoch is not None:
        classifier_path = output_dir / f"classifier_epoch{epoch}.pt"
    else:
        classifier_path = output_dir / "classifier.pt"
    torch.save(model.classifier.state_dict(), classifier_path)
    
    # Save config
    config_dict = {
        "model_family": config.model_family,
        "backbone_ckpt": config.backbone_ckpt,
        "jepa_model_dir": str(config.jepa_model_dir) if config.jepa_model_dir else None,
        "num_labels": len(label_list),
        "classifier_hidden": config.classifier_hidden,
        "dropout": config.dropout,
        "max_length": config.max_length,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save label mapping
    with open(output_dir / "labels.json", "w") as f:
        json.dump({"label_list": label_list, "label_to_idx": label_to_idx}, f)
    
    print(f"Saved checkpoint to {output_dir}")


def train(config: JepaGoConfig):
    """Main training function."""
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {config}")
    
    # Initialize wandb
    wandb_run = None
    if config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or config.name,
            config=vars(config),
        )
    
    # Load data
    print("Loading sequences...")
    train_sequences = dict(read_fasta(config.train_fasta))
    print(f"Loaded {len(train_sequences)} training sequences")
    
    print("Loading GO terms...")
    protein_to_terms, label_list, label_to_idx = load_terms(
        config.train_terms_tsv,
        namespace=config.namespace,
        min_count=config.min_label_count,
        max_labels=config.max_labels,
    )
    print(f"Loaded {len(label_list)} labels for {len(protein_to_terms)} proteins")
    
    # Load IA weights
    ia_weights = load_ia_weights(config.ia_tsv, label_to_idx)
    ia_weights = ia_weights.to(device)
    
    # Train/val split
    train_ids, val_ids = train_val_split(
        protein_to_terms, val_fraction=config.val_fraction, seed=config.seed
    )
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Build encoder
    print("Building encoder...")
    encoder, hidden_dim, tokenizer = build_encoder(
        model_family=config.model_family,
        backbone_ckpt=config.backbone_ckpt,
        jepa_model_dir=config.jepa_model_dir,
        dtype=config.get_dtype(),
    )
    
    # Build model
    model = JepaGoModel(
        encoder=encoder,
        hidden_dim=hidden_dim,
        num_labels=len(label_list),
        classifier_hidden=config.classifier_hidden,
        dropout=config.dropout,
        freeze_encoder=config.freeze_encoder,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    train_protein_to_terms = {pid: protein_to_terms[pid] for pid in train_ids if pid in protein_to_terms}
    val_protein_to_terms = {pid: protein_to_terms[pid] for pid in val_ids if pid in protein_to_terms}
    
    train_dataset = JepaGoDataset(
        sequences=train_sequences,
        protein_to_terms=train_protein_to_terms,
        label_to_idx=label_to_idx,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    
    val_dataset = JepaGoDataset(
        sequences=train_sequences,
        protein_to_terms=val_protein_to_terms,
        label_to_idx=label_to_idx,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    
    print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    optimizer, scheduler = create_optimizer_scheduler(model, config, num_training_steps)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    global_step = 0
    best_fmax = 0.0
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, epoch, global_step, wandb_run
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, config, device, ia_weights)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val F-max: {val_metrics['fmax']:.4f} @ {val_metrics['threshold']:.2f}")
        print(f"Val AUPRC: {val_metrics['auprc']:.4f}")
        
        if wandb_run is not None:
            wandb_run.log({
                "val/loss": val_metrics["loss"],
                "val/fmax": val_metrics["fmax"],
                "val/threshold": val_metrics["threshold"],
                "val/smin": val_metrics["smin"],
                "val/auprc": val_metrics["auprc"],
                "epoch": epoch,
            })
        
        # Save checkpoint
        if config.save_every_epoch:
            save_checkpoint(
                model, config, label_list, label_to_idx,
                config.output_dir, epoch=epoch
            )
        
        # Save best model
        if val_metrics["fmax"] > best_fmax:
            best_fmax = val_metrics["fmax"]
            save_checkpoint(model, config, label_list, label_to_idx, config.output_dir)
            print(f"New best F-max: {best_fmax:.4f}")
    
    print(f"\nTraining complete. Best F-max: {best_fmax:.4f}")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    return model, label_list, label_to_idx


def main():
    parser = argparse.ArgumentParser(description="Train JEPA encoder-based GO predictor")
    
    # Model
    parser.add_argument("--model_family", type=str, default="prott5", choices=["prott5", "esm"])
    parser.add_argument("--backbone_ckpt", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--jepa_model_dir", type=str, default="models/jepa-prott5-1")
    
    # Data
    parser.add_argument("--train_fasta", type=str, default="data/Train/train_sequences.fasta")
    parser.add_argument("--train_terms_tsv", type=str, default="data/Train/train_terms.tsv")
    parser.add_argument("--go_obo", type=str, default="data/Train/go-basic.obo")
    parser.add_argument("--ia_tsv", type=str, default="data/IA.tsv")
    parser.add_argument("--namespace", type=str, default=None, choices=["MF", "BP", "CC"])
    parser.add_argument("--min_label_count", type=int, default=5)
    parser.add_argument("--max_labels", type=int, default=8192)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    
    # Validation
    parser.add_argument("--val_fraction", type=float, default=0.03)
    parser.add_argument("--eval_thresholds", type=int, default=51)
    
    # Model architecture
    parser.add_argument("--classifier_hidden", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder")
    
    # Precision
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="models/jepa-go")
    parser.add_argument("--name", type=str, default="jepa-go")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cafa6")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Create config
    config = JepaGoConfig(
        name=args.name,
        seed=args.seed,
        debug_mode=args.debug,
        model_family=args.model_family,
        jepa_model_dir=Path(args.jepa_model_dir) if args.jepa_model_dir else None,
        backbone_ckpt=args.backbone_ckpt,
        train_fasta=Path(args.train_fasta),
        train_terms_tsv=Path(args.train_terms_tsv),
        go_obo=Path(args.go_obo),
        ia_tsv=Path(args.ia_tsv),
        namespace=args.namespace,
        min_label_count=args.min_label_count,
        max_labels=args.max_labels,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        val_fraction=args.val_fraction,
        eval_thresholds=args.eval_thresholds,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        dtype=args.dtype,
        output_dir=Path(args.output_dir),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        log_every=args.log_every,
    )
    
    if args.debug:
        config = JepaGoConfig.with_debug(
            model_family=args.model_family,
            jepa_model_dir=Path(args.jepa_model_dir) if args.jepa_model_dir else None,
        )
    
    train(config)


if __name__ == "__main__":
    main()
