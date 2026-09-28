"""
Metrics for GO term prediction evaluation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def compute_fmax(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ia_weights: torch.Tensor = None,
    num_thresholds: int = 51,
) -> Tuple[float, float, float]:
    """
    Compute protein-centric F-max metric.
    
    Args:
        predictions: (N, L) sigmoid probabilities
        targets: (N, L) binary labels
        ia_weights: (L,) IA weights per label (optional)
        num_thresholds: Number of thresholds to search
    
    Returns:
        fmax: Maximum F1 score
        best_threshold: Threshold achieving fmax
        smin: S-min (semantic distance, only if ia_weights provided)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.float().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.float().cpu().numpy()
    if ia_weights is not None and isinstance(ia_weights, torch.Tensor):
        ia_weights = ia_weights.float().cpu().numpy()
    
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in thresholds:
        preds_binary = (predictions >= threshold).astype(np.float32)
        
        # Per-protein precision and recall
        tp = (preds_binary * targets).sum(axis=1)
        pred_pos = preds_binary.sum(axis=1) + 1e-8
        actual_pos = targets.sum(axis=1) + 1e-8
        
        precision = tp / pred_pos
        recall = tp / actual_pos
        
        # Macro average over proteins
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        if mean_precision + mean_recall > 0:
            f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        else:
            f1 = 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = mean_precision
            best_recall = mean_recall
    
    # Compute S-min if IA weights provided
    smin = 0.0
    if ia_weights is not None:
        # S-min at best threshold
        preds_binary = (predictions >= best_threshold).astype(np.float32)
        
        # Remaining uncertainty (false negatives weighted by IA)
        fn_weighted = ((1 - preds_binary) * targets * ia_weights).sum(axis=1)
        
        # Misinformation (false positives weighted by IA)
        fp_weighted = (preds_binary * (1 - targets) * ia_weights).sum(axis=1)
        
        # S-min = average semantic distance
        smin = (fn_weighted + fp_weighted).mean()
    
    return float(best_f1), float(best_threshold), float(smin)


def compute_auprc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute micro-averaged Area Under Precision-Recall Curve.
    
    Args:
        predictions: (N, L) sigmoid probabilities
        targets: (N, L) binary labels
    
    Returns:
        auprc: Micro-averaged AUPRC
    """
    from sklearn.metrics import average_precision_score
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.float().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.float().cpu().numpy()
    
    # Flatten for micro-average
    preds_flat = predictions.ravel()
    targets_flat = targets.ravel()
    
    # Filter samples with no positive labels
    mask = targets_flat > 0
    if mask.sum() == 0:
        return 0.0
    
    return float(average_precision_score(targets_flat, preds_flat))


class MetricsAccumulator:
    """
    Accumulate predictions and targets for batch-wise evaluation.
    """
    
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.protein_ids = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        protein_ids: list = None,
    ):
        self.predictions.append(predictions.detach().float().cpu())
        self.targets.append(targets.detach().float().cpu())
        if protein_ids is not None:
            self.protein_ids.extend(protein_ids)
    
    def compute(
        self,
        ia_weights: torch.Tensor = None,
        num_thresholds: int = 51,
    ) -> dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with fmax, threshold, smin, auprc
        """
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Apply sigmoid if logits
        if all_preds.min() < 0 or all_preds.max() > 1:
            all_preds = torch.sigmoid(all_preds)
        
        fmax, threshold, smin = compute_fmax(
            all_preds, all_targets, ia_weights, num_thresholds
        )
        auprc = compute_auprc(all_preds, all_targets)
        
        return {
            "fmax": fmax,
            "threshold": threshold,
            "smin": smin,
            "auprc": auprc,
        }
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.protein_ids = []
