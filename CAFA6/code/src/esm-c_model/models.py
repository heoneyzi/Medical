"""
Neural network models for ESM-C based protein function prediction.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPTrunk(nn.Module):
    """
    Multi-layer perceptron trunk for feature transformation.
    
    Architecture:
        Linear -> GELU -> Dropout -> ... (repeated for each hidden layer)
    
    Args:
        in_dim: Input dimension.
        hidden: List of hidden layer dimensions.
        dropout: Dropout probability.
    """
    
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.2):
        super().__init__()
        
        dims = [in_dim] + list(hidden)
        layers = []
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP trunk."""
        return self.net(x)


class BasePredictor(nn.Module):
    """
    Base predictor model for GO term prediction.
    
    Architecture:
        Embedding -> Trunk (Identity or MLP) -> Linear Classifier
    
    Supports:
        - Linear head: Direct linear projection from embeddings
        - MLP head: MLP trunk + linear classifier
        - Efficient partial logit computation for negative sampling
    
    Args:
        in_dim: Input embedding dimension.
        n_labels: Number of GO term labels.
        head: Head type ('linear' or 'mlp').
        hidden: Hidden layer dimensions for MLP trunk.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        in_dim: int,
        n_labels: int,
        head: str = "linear",
        hidden: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        hidden = hidden or [1024, 512]
        head = head.lower()
        
        if head == "linear":
            self.trunk = nn.Identity()
            trunk_dim = in_dim
        elif head == "mlp":
            self.trunk = MLPTrunk(in_dim, hidden=hidden, dropout=dropout)
            trunk_dim = self.trunk.out_dim
        else:
            raise ValueError(f"Unknown head type: {head}. Expected 'linear' or 'mlp'.")
        
        self.classifier = nn.Linear(trunk_dim, n_labels)
        self.head_type = head
        self.n_labels = n_labels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing logits for all labels.
        
        Args:
            x: Input embeddings [B, D].
        
        Returns:
            Logits [B, L] for all labels.
        """
        h = self.trunk(x)
        return self.classifier(h)
    
    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk only (for negative sampling).
        
        Args:
            x: Input embeddings [B, D].
        
        Returns:
            Hidden representations [B, H].
        """
        return self.trunk(x)
    
    def logits_selected(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for selected label indices only.
        
        This is more efficient than computing all logits when using negative sampling.
        
        Args:
            h: Hidden representations [B, H].
            idx: Label indices [B, K] (may contain -1 for padding).
        
        Returns:
            Logits [B, K] for selected labels.
        """
        W = self.classifier.weight  # [L, H]
        b = self.classifier.bias    # [L]
        
        # Clamp -1 padding to 0 for gather (will be masked later)
        idx_clamped = idx.clamp_min(0)
        
        # Gather weights and biases for selected indices
        W_sel = W[idx_clamped]  # [B, K, H]
        b_sel = b[idx_clamped]  # [B, K]
        
        # Compute logits: h @ W_sel.T + b_sel
        return torch.einsum("bh,bkh->bk", h, W_sel) + b_sel
    
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for all labels.
        
        Args:
            x: Input embeddings [B, D].
        
        Returns:
            Probabilities [B, L].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the trunk output."""
        if isinstance(self.trunk, nn.Identity):
            return self.classifier.in_features
        return self.trunk.out_dim


def create_model(
    in_dim: int,
    n_labels: int,
    head: str = "linear",
    hidden: List[int] = None,
    dropout: float = 0.2,
    device: torch.device = None,
) -> BasePredictor:
    """
    Factory function to create a BasePredictor model.
    
    Args:
        in_dim: Input embedding dimension.
        n_labels: Number of GO term labels.
        head: Head type ('linear' or 'mlp').
        hidden: Hidden layer dimensions for MLP trunk.
        dropout: Dropout probability.
        device: Device to place model on.
    
    Returns:
        Initialized BasePredictor model.
    """
    model = BasePredictor(
        in_dim=in_dim,
        n_labels=n_labels,
        head=head,
        hidden=hidden,
        dropout=dropout,
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def load_model(
    checkpoint_path: str,
    in_dim: int,
    n_labels: int,
    head: str = "linear",
    hidden: List[int] = None,
    dropout: float = 0.2,
    device: torch.device = None,
) -> BasePredictor:
    """
    Load a trained BasePredictor model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        in_dim: Input embedding dimension.
        n_labels: Number of GO term labels.
        head: Head type ('linear' or 'mlp').
        hidden: Hidden layer dimensions for MLP trunk.
        dropout: Dropout probability.
        device: Device to place model on.
    
    Returns:
        Loaded BasePredictor model.
    """
    model = create_model(in_dim, n_labels, head, hidden, dropout, device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model
