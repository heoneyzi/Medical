"""
Evaluation metrics for protein function prediction.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy import sparse


@torch.no_grad()
def fmax_micro(
    y_true_csr: sparse.csr_matrix,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Compute micro-averaged F-max score over multiple thresholds.
    
    F-max is a key metric for CAFA protein function prediction:
        F1 = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        y_true_csr: Ground truth labels [N, L] in CSR sparse format.
        y_prob: Predicted probabilities [N, L] in range [0, 1].
        thresholds: List of thresholds to try. Defaults to common values.
    
    Returns:
        Tuple of (best_f1_score, best_threshold).
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    best_f1 = 0.0
    best_threshold = thresholds[0]
    
    for t in thresholds:
        y_hat = (y_prob >= t)
        
        # Compute micro-averaged metrics
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(y_true_csr.shape[0]):
            pos = y_true_csr[i].indices
            pred = np.nonzero(y_hat[i])[0]
            
            if len(pred) == 0 and len(pos) == 0:
                continue
            
            set_pos = set(pos.tolist())
            set_pred = set(pred.tolist())
            
            tp += len(set_pos & set_pred)
            fp += len(set_pred - set_pos)
            fn += len(set_pos - set_pred)
        
        # Compute F1
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_f1, best_threshold


def compute_precision_recall(
    y_true_csr: sparse.csr_matrix,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Compute micro-averaged precision, recall, and F1 at a given threshold.
    
    Args:
        y_true_csr: Ground truth labels [N, L] in CSR sparse format.
        y_prob: Predicted probabilities [N, L] in range [0, 1].
        threshold: Classification threshold.
    
    Returns:
        Tuple of (precision, recall, f1).
    """
    y_hat = (y_prob >= threshold)
    
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(y_true_csr.shape[0]):
        pos = y_true_csr[i].indices
        pred = np.nonzero(y_hat[i])[0]
        
        if len(pred) == 0 and len(pos) == 0:
            continue
        
        set_pos = set(pos.tolist())
        set_pred = set(pred.tolist())
        
        tp += len(set_pos & set_pred)
        fp += len(set_pred - set_pos)
        fn += len(set_pos - set_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_coverage(
    y_true_csr: sparse.csr_matrix,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute prediction coverage (fraction of samples with at least one prediction).
    
    Args:
        y_true_csr: Ground truth labels [N, L] in CSR sparse format.
        y_prob: Predicted probabilities [N, L] in range [0, 1].
        threshold: Classification threshold.
    
    Returns:
        Coverage fraction.
    """
    y_hat = (y_prob >= threshold)
    has_prediction = np.any(y_hat, axis=1)
    return np.mean(has_prediction)


def compute_average_predictions(
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute average number of predictions per sample.
    
    Args:
        y_prob: Predicted probabilities [N, L] in range [0, 1].
        threshold: Classification threshold.
    
    Returns:
        Average number of predictions.
    """
    y_hat = (y_prob >= threshold)
    return np.mean(np.sum(y_hat, axis=1))


class MetricsTracker:
    """
    Tracks training metrics over epochs.
    """
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_f1": [],
            "val_threshold": [],
            "val_precision": [],
            "val_recall": [],
        }
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.best_threshold = 0.0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_f1: float,
        val_threshold: float,
        val_precision: float = None,
        val_recall: float = None,
    ):
        """Update metrics history."""
        self.history["train_loss"].append(train_loss)
        self.history["val_f1"].append(val_f1)
        self.history["val_threshold"].append(val_threshold)
        
        if val_precision is not None:
            self.history["val_precision"].append(val_precision)
        if val_recall is not None:
            self.history["val_recall"].append(val_recall)
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = epoch
            self.best_threshold = val_threshold
            return True  # New best
        
        return False
    
    def summary(self) -> dict:
        """Get metrics summary."""
        return {
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_f1": self.history["val_f1"][-1] if self.history["val_f1"] else None,
        }
