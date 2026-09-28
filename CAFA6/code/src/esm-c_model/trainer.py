"""
Trainer class for ESM-C model training.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader

from .config import TrainConfig, BaseTrainConfig
from .dataset import (
    TrainDataset,
    collate_fn,
    build_batch_indices,
    create_train_dataloader,
    load_embeddings,
    align_embeddings_to_labels,
)
from .metrics import fmax_micro, MetricsTracker
from .models import BasePredictor, create_model
from .paths import ESMCPaths


def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class ESMCTrainer:
    """
    Trainer for ESM-C based protein function prediction.
    
    Supports:
        - Linear or MLP head architectures
        - Automatic Mixed Precision (AMP)
        - Negative sampling for extreme multi-label classification
        - GroupKFold cross-validation
        - Early stopping based on F-max
    
    Args:
        config: Training configuration.
        device: PyTorch device.
        paths: Path configuration.
    """
    
    def __init__(
        self,
        config: TrainConfig,
        device: torch.device = None,
        paths: ESMCPaths = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.paths = paths or ESMCPaths(base_path=config.base_path)
        
        # Initialize random state
        self.seed = config.cv.seed
        seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        
        # Data placeholders
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.train_ids: Optional[np.ndarray] = None
        self.test_ids: Optional[np.ndarray] = None
        self.Y_csr: Optional[sparse.csr_matrix] = None
        self.labels: Optional[np.ndarray] = None
        self.folds: Optional[np.ndarray] = None
        
        # Output directory
        self.out_dir = self.paths.ensure_output_dir(config.name)
    
    def load_data(self) -> None:
        """Load all required data for training."""
        print(f"[ESMCTrainer] Loading data...")
        
        # Load label information
        train_ids_feat = np.load(self.paths.train_ids_npy, allow_pickle=True).astype(object)
        self.labels = np.load(self.paths.labels_npy, allow_pickle=True).astype(object)
        self.Y_csr = sparse.load_npz(self.paths.Y_sparse_npz).tocsr()
        self.folds = np.load(self.paths.folds_npy).astype(np.int64)
        
        # Validate shapes
        if self.Y_csr.shape[0] != len(train_ids_feat):
            raise ValueError(f"Y rows != train_ids: {self.Y_csr.shape[0]} vs {len(train_ids_feat)}")
        if self.Y_csr.shape[0] != len(self.folds):
            raise ValueError(f"folds != train size: {len(self.folds)} vs {self.Y_csr.shape[0]}")
        
        # Apply label limit if specified
        label_limit = self.config.base.label_limit
        if label_limit and 0 < label_limit < self.Y_csr.shape[1]:
            print(f"[ESMCTrainer] Applying label_limit={label_limit}")
            self.labels = self.labels[:label_limit]
            self.Y_csr = self.Y_csr[:, :label_limit]
        
        # Load embeddings
        embed_dir = self.paths.embed_dir(self.config.embed.model, self.config.embed.mode)
        print(f"[ESMCTrainer] Loading embeddings from: {embed_dir}")
        
        X_tr, tr_ids_embed, self.X_test, self.test_ids = load_embeddings(embed_dir)
        
        # Align embeddings to label order
        self.X_train, self.train_ids = align_embeddings_to_labels(
            train_ids_feat, tr_ids_embed, X_tr
        )
        
        print(f"[ESMCTrainer] Data loaded:")
        print(f"  - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"  - Labels: {self.Y_csr.shape[1]}")
        print(f"  - Folds: {sorted(set(self.folds.tolist()))}")
    
    def _create_model(self) -> BasePredictor:
        """Create a new model instance."""
        return create_model(
            in_dim=int(self.X_train.shape[1]),
            n_labels=int(self.Y_csr.shape[1]),
            head=self.config.base.head,
            hidden=self.config.base.hidden,
            dropout=self.config.base.dropout,
            device=self.device,
        )
    
    @torch.no_grad()
    def _predict_full(
        self,
        model: BasePredictor,
        X: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict logits for all samples."""
        model.eval()
        outputs = []
        
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(
                np.asarray(X[i:i + batch_size], dtype=np.float32)
            ).to(self.device)
            
            logits = model(xb).detach().cpu().float().numpy()
            outputs.append(logits)
        
        return np.concatenate(outputs, axis=0)
    
    def train_one_fold(
        self,
        fold: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, str, BasePredictor]:
        """
        Train model for one fold.
        
        Args:
            fold: Fold index.
        
        Returns:
            Tuple of (val_indices, val_logits, best_f1, checkpoint_path, model).
        """
        cfg = self.config.base
        
        # Split indices
        val_idx = np.where(self.folds == fold)[0].astype(np.int64)
        tr_idx = np.where(self.folds != fold)[0].astype(np.int64)
        
        print(f"[Fold {fold}] Train: {len(tr_idx)}, Val: {len(val_idx)}")
        
        # Create dataloader
        dl_train = create_train_dataloader(
            self.X_train, self.Y_csr, tr_idx,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        
        # Create model and optimizer
        model = self._create_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.device.type == "cuda")
        )
        
        # Training loop
        metrics = MetricsTracker()
        best_state = None
        n_labels = self.Y_csr.shape[1]
        
        thresholds = cfg.fmax_thresholds or [0.05, 0.1, 0.2, 0.3, 0.4]
        
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            
            for _, xb, pos_list in dl_train:
                xb = xb.to(self.device, non_blocking=True)
                
                # Build batch indices with negative sampling
                idx_pad, tgt_pad, msk = build_batch_indices(
                    pos_list, n_labels, cfg.neg_k, self.rng
                )
                idx_pad = idx_pad.to(self.device, non_blocking=True)
                tgt_pad = tgt_pad.to(self.device, non_blocking=True)
                msk = msk.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=(cfg.amp and self.device.type == "cuda")):
                    h = model.forward_hidden(xb)
                    logits_sel = model.logits_selected(h, idx_pad)
                    
                    # Apply mask and compute loss
                    logits_flat = logits_sel[msk]
                    tgt_flat = tgt_pad[msk]
                    loss = F.binary_cross_entropy_with_logits(logits_flat, tgt_flat)
                
                scaler.scale(loss).backward()
                
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += float(loss.detach().cpu())
                n_batches += 1
            
            # Validation
            val_logits = self._predict_full(model, self.X_train[val_idx], cfg.val_batch_size)
            val_prob = 1 / (1 + np.exp(-val_logits))
            val_f1, val_t = fmax_micro(self.Y_csr[val_idx], val_prob, thresholds)
            
            avg_loss = running_loss / max(n_batches, 1)
            is_best = metrics.update(epoch, avg_loss, val_f1, val_t)
            
            if is_best:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{cfg.epochs}: loss={avg_loss:.4f}, val_f1={val_f1:.4f} (t={val_t}), time={elapsed:.0f}s")
        
        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Save checkpoint
        ckpt_path = self.paths.fold_checkpoint(self.config.name, fold)
        torch.save(model.state_dict(), ckpt_path)
        
        # Final validation logits
        val_logits = self._predict_full(model, self.X_train[val_idx], cfg.val_batch_size)
        
        return val_idx, val_logits.astype(np.float32), metrics.best_f1, ckpt_path, model
    
    def train(self) -> Dict:
        """
        Run full cross-validation training.
        
        Returns:
            Summary dictionary with fold scores and output paths.
        """
        if self.X_train is None:
            self.load_data()
        
        # Save metadata
        meta = {
            "exp_name": self.config.name,
            "embed_model": self.config.embed.model,
            "embed_mode": self.config.embed.mode,
            "n_train": int(self.X_train.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "embed_dim": int(self.X_train.shape[1]),
            "n_labels": int(self.Y_csr.shape[1]),
            "head": self.config.base.head,
            "device": str(self.device),
            "seed": self.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(self.paths.meta_json(self.config.name), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Initialize output arrays
        n_train, n_labels = self.Y_csr.shape
        n_test = self.X_test.shape[0]
        
        oof_path = self.paths.oof_logits_npy(self.config.name)
        test_path = self.paths.test_logits_npy(self.config.name)
        
        oof_logits = np.memmap(oof_path, mode="w+", dtype=np.float16, shape=(n_train, n_labels))
        test_logits_acc = np.zeros((n_test, n_labels), dtype=np.float32)
        
        # Get unique folds
        unique_folds = sorted(set(self.folds.tolist()))
        print(f"[ESMCTrainer] Folds: {unique_folds}")
        print(f"[ESMCTrainer] Fold counts: {dict((k, int((self.folds == k).sum())) for k in unique_folds)}")
        
        fold_scores = {}
        checkpoints = []
        
        # Train each fold
        for fold in unique_folds:
            print(f"\n{'=' * 80}")
            print(f"[ESMCTrainer] Training fold {fold}")
            
            val_idx, val_logits, best_f1, ckpt_path, model = self.train_one_fold(fold)
            
            # Store OOF predictions
            oof_logits[val_idx] = val_logits.astype(np.float16)
            
            # Accumulate test predictions
            test_logits = self._predict_full(model, self.X_test, self.config.base.val_batch_size)
            test_logits_acc += test_logits
            
            fold_scores[int(fold)] = float(best_f1)
            checkpoints.append(ckpt_path)
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
        
        # Average test predictions
        test_logits_final = (test_logits_acc / max(1, len(unique_folds))).astype(np.float16)
        np.save(test_path, test_logits_final)
        
        # Flush OOF memmap
        oof_logits.flush()
        
        # Save IDs and labels
        np.save(os.path.join(self.out_dir, "train_ids.npy"), self.train_ids, allow_pickle=True)
        np.save(os.path.join(self.out_dir, "test_ids.npy"), self.test_ids, allow_pickle=True)
        np.save(os.path.join(self.out_dir, "labels.npy"), self.labels, allow_pickle=True)
        
        # Save summary
        summary = {
            "fold_scores": fold_scores,
            "mean_fold_score": float(np.mean(list(fold_scores.values()))) if fold_scores else None,
            "oof_logits": oof_path,
            "test_logits": test_path,
            "checkpoints": checkpoints,
        }
        
        with open(self.paths.metrics_txt(self.config.name), "w") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        
        print(f"\n[ESMCTrainer] Training complete!")
        print(f"  - OOF logits: {oof_path}")
        print(f"  - Test logits: {test_path}")
        print(f"  - Mean F1: {summary['mean_fold_score']:.4f}")
        
        return summary
