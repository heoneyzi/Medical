"""
Dataset and data loading utilities for ESM-C model training.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    """
    Dataset for training base predictor.
    
    Args:
        X: Embedding matrix [N, D].
        Y_csr: Sparse label matrix [N, L] in CSR format.
        indices: Row indices to include in this dataset.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y_csr: sparse.csr_matrix,
        indices: np.ndarray,
    ):
        self.X = X
        self.Y = Y_csr
        self.indices = indices.astype(np.int64)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, i: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (row_index, embedding, positive_label_indices).
        """
        ridx = int(self.indices[i])
        x = self.X[ridx].astype(np.float32)
        row = self.Y[ridx]
        pos = row.indices.astype(np.int64)  # positive label indices
        return ridx, x, pos


def collate_fn(batch: List[Tuple]) -> Tuple[np.ndarray, torch.Tensor, List[np.ndarray]]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (row_index, embedding, positive_indices) tuples.
    
    Returns:
        Tuple of (row_indices, embeddings_tensor, list_of_positive_indices).
    """
    rows, xs, poss = zip(*batch)
    X = torch.from_numpy(np.stack(xs, axis=0))  # [B, D]
    return np.array(rows, dtype=np.int64), X, list(poss)


def sample_negatives(
    n_labels: int,
    pos_set: set,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample k negative indices uniformly from [0, n_labels-1] excluding positives.
    
    Uses rejection sampling which is efficient when positives are sparse.
    
    Args:
        n_labels: Total number of labels.
        pos_set: Set of positive label indices to exclude.
        k: Number of negatives to sample.
        rng: NumPy random generator.
    
    Returns:
        Array of k negative indices.
    """
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    
    out = np.empty((k,), dtype=np.int64)
    filled = 0
    
    # Rejection sampling
    while filled < k:
        # Sample more than needed to reduce iterations
        draw = rng.integers(0, n_labels, size=(2 * (k - filled)), dtype=np.int64)
        draw = [d for d in draw if d not in pos_set]
        
        if draw:
            take = min(len(draw), k - filled)
            out[filled:filled + take] = np.array(draw[:take], dtype=np.int64)
            filled += take
    
    return out


def build_batch_indices(
    pos_list: List[np.ndarray],
    n_labels: int,
    neg_k: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded index tensor for BCE loss with negative sampling.
    
    Strategy:
        - Sample a global negative set shared across the batch
        - Each sample uses its own positives + shared negatives
        - This reduces sampling overhead while maintaining diversity
    
    Args:
        pos_list: List of positive label indices per sample.
        n_labels: Total number of labels.
        neg_k: Number of negative samples to include.
        rng: NumPy random generator.
    
    Returns:
        idx_pad: [B, Kmax] int64, padded label indices (-1 for padding).
        tgt_pad: [B, Kmax] float32, target labels (1 for positive, 0 for negative).
        msk: [B, Kmax] bool, mask for valid positions.
    """
    B = len(pos_list)
    
    # Compute union of positives to avoid sampling them as negatives
    union_pos = set()
    for p in pos_list:
        union_pos.update(p.tolist())
    
    # Sample global negatives
    global_negs = sample_negatives(n_labels, union_pos, neg_k, rng) if neg_k > 0 else np.empty((0,), np.int64)
    
    # Compute padded tensor dimensions
    lens = [len(p) + len(global_negs) for p in pos_list]
    Kmax = max(lens) if lens else 0
    
    # Initialize padded tensors
    idx_pad = np.full((B, Kmax), -1, dtype=np.int64)
    tgt_pad = np.zeros((B, Kmax), dtype=np.float32)
    msk = np.zeros((B, Kmax), dtype=bool)
    
    # Fill tensors
    for i, p in enumerate(pos_list):
        p = np.unique(p)  # Remove duplicates
        k = len(p) + len(global_negs)
        
        # Concatenate positives and negatives
        idx = np.concatenate([p, global_negs]) if len(global_negs) else p
        
        idx_pad[i, :k] = idx
        tgt_pad[i, :len(p)] = 1.0  # Positives first
        msk[i, :k] = True
    
    return (
        torch.from_numpy(idx_pad),
        torch.from_numpy(tgt_pad),
        torch.from_numpy(msk),
    )


def create_train_dataloader(
    X: np.ndarray,
    Y_csr: sparse.csr_matrix,
    indices: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        X: Embedding matrix [N, D].
        Y_csr: Sparse label matrix [N, L].
        indices: Row indices for this split.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
    
    Returns:
        Configured DataLoader.
    """
    dataset = TrainDataset(X, Y_csr, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def load_embeddings(embed_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from directory.
    
    Args:
        embed_dir: Directory containing embedding files.
    
    Returns:
        Tuple of (train_embeds, train_ids, test_embeds, test_ids).
    """
    import os
    
    train_X = np.load(os.path.join(embed_dir, "train_embeds.npy"), mmap_mode="r")
    train_ids = np.load(os.path.join(embed_dir, "train_ids.npy"), allow_pickle=True).astype(object)
    test_X = np.load(os.path.join(embed_dir, "test_embeds.npy"), mmap_mode="r")
    test_ids = np.load(os.path.join(embed_dir, "test_ids.npy"), allow_pickle=True).astype(object)
    
    return train_X, train_ids, test_X, test_ids


def align_embeddings_to_labels(
    train_ids_feat: np.ndarray,
    train_ids_embed: np.ndarray,
    X_embed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align embedding rows to match feature train_ids order.
    
    Args:
        train_ids_feat: Train IDs from feature files (target order).
        train_ids_embed: Train IDs from embedding files (current order).
        X_embed: Embedding matrix to reorder.
    
    Returns:
        Tuple of (aligned_embeddings, aligned_ids).
    
    Raises:
        ValueError: If sizes don't match.
        KeyError: If IDs are missing.
    """
    if len(train_ids_feat) != len(train_ids_embed):
        raise ValueError(
            f"Train IDs size mismatch: feats={len(train_ids_feat)}, embed={len(train_ids_embed)}"
        )
    
    # Check if already aligned
    if np.all(train_ids_feat == train_ids_embed):
        return X_embed, train_ids_feat
    
    # Build mapping and reorder
    idx_map = {pid: i for i, pid in enumerate(train_ids_embed)}
    
    missing = [pid for pid in train_ids_feat if pid not in idx_map]
    if missing:
        raise KeyError(f"{len(missing)} train IDs missing in embeddings. Example: {missing[:5]}")
    
    order = np.array([idx_map[pid] for pid in train_ids_feat], dtype=np.int64)
    X_aligned = X_embed[order]
    
    return X_aligned, train_ids_feat
