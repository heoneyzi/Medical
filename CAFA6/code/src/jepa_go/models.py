"""
Model definitions for JEPA-based GO term prediction.

Supports:
- Protein encoding with ProtT5/ESM backbones
- Learnable GO term embeddings with JEPA-style training
- Optional taxonomy embedding for species-aware predictions
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooler(nn.Module):
    """Mean pooling over sequence dimension with attention mask."""
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D)
        # attention_mask: (B, L)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)  # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        return summed / counts


class GoClassifierHead(nn.Module):
    """
    Multi-label classifier head for GO term prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [1024, 512]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class JepaGoModel(nn.Module):
    """
    JEPA encoder-based GO term prediction model.
    
    Uses a pretrained JEPA encoder (frozen or fine-tuned) with
    a classifier head for multi-label GO term prediction.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_labels: int,
        classifier_hidden: List[int] = None,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.pooler = MeanPooler()
        self.classifier = GoClassifierHead(
            input_dim=hidden_dim,
            num_labels=num_labels,
            hidden_dims=classifier_hidden,
            dropout=dropout,
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Get encoder output
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Handle different encoder output formats
        if hasattr(encoder_output, "last_hidden_state"):
            hidden = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            hidden = encoder_output[0]
        else:
            hidden = encoder_output
        
        # Pool and classify
        pooled = self.pooler(hidden, attention_mask)
        logits = self.classifier(pooled)
        
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get pooled embeddings without classification."""
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(encoder_output, "last_hidden_state"):
                hidden = encoder_output.last_hidden_state
            elif isinstance(encoder_output, tuple):
                hidden = encoder_output[0]
            else:
                hidden = encoder_output
            
            return self.pooler(hidden, attention_mask)


class LabelEmbedding(nn.Module):
    """
    Learnable embeddings for GO term labels.
    """
    
    def __init__(self, num_labels: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        
        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, label_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Get label embeddings.
        
        Args:
            label_indices: Optional tensor of label indices. If None, return all.
        
        Returns:
            Embeddings of shape (N, D) or (L, D) if indices is None
        """
        if label_indices is None:
            return self.embedding.weight
        return self.embedding(label_indices)


class LabelJepaPredictor(nn.Module):
    """
    Predictor network for Label-Space JEPA.
    
    Predicts target label embeddings from context label embeddings
    and protein representation.
    """
    
    def __init__(
        self,
        protein_dim: int,
        label_dim: int,
        hidden_mult: int = 2,
    ):
        super().__init__()
        
        hidden_dim = label_dim * hidden_mult
        
        # Protein projection
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Context aggregation
        self.context_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, label_dim),
        )
    
    def forward(
        self,
        protein_embed: torch.Tensor,
        context_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target label embeddings.
        
        Args:
            protein_embed: (B, protein_dim) protein representations
            context_embed: (B, label_dim) aggregated context label embeddings
        
        Returns:
            (B, label_dim) predicted target label embeddings
        """
        protein_h = self.protein_proj(protein_embed)  # (B, H)
        context_h = self.context_proj(context_embed)  # (B, H)
        
        combined = torch.cat([protein_h, context_h], dim=-1)  # (B, 2H)
        pred = self.predictor(combined)  # (B, label_dim)
        
        return pred


class LabelJepaModel(nn.Module):
    """
    Label-Space JEPA model for GO term prediction.
    
    Architecture:
    - Protein encoder (ProtT5 or ESM)
    - Learnable GO term embeddings (online and target/EMA)
    - Predictor network
    - Classification head
    
    Training:
    1. Encode protein sequence
    2. Sample context/target split of positive labels
    3. Aggregate context label embeddings
    4. Predict target label embeddings
    5. Compute JEPA loss (prediction vs EMA target embeddings)
    6. Also train classification head (optional auxiliary loss)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        protein_dim: int,
        num_labels: int,
        label_embed_dim: int = 512,
        predictor_hidden_mult: int = 2,
        projection_hidden: List[int] = None,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        # Taxonomy support
        num_taxa: int = 0,
        taxon_embed_dim: int = 64,
        taxon_fusion_type: str = "gate",
    ):
        super().__init__()
        
        self.encoder = encoder
        self.pooler = MeanPooler()
        self.num_labels = num_labels
        self.label_embed_dim = label_embed_dim
        self.num_taxa = num_taxa
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Optionally unfreeze last N layers
        if unfreeze_last_n_layers > 0 and hasattr(self.encoder, "encoder"):
            layers = list(self.encoder.encoder.block)[-unfreeze_last_n_layers:]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Taxonomy embeddings (optional)
        self.taxon_embedding = None
        self.taxon_fusion = None
        if num_taxa > 0:
            from src.jepa_go.taxonomy import TaxonomyEmbedding, TaxonomyFusion
            self.taxon_embedding = TaxonomyEmbedding(
                num_taxa=num_taxa,
                embed_dim=taxon_embed_dim,
                dropout=dropout,
            )
            self.taxon_fusion = TaxonomyFusion(
                protein_dim=protein_dim,
                taxon_dim=taxon_embed_dim,
                output_dim=protein_dim,
                fusion_type=taxon_fusion_type,
                dropout=dropout,
            )
        
        # Protein projection to label space
        proj_hidden = projection_hidden or [1024]
        proj_layers = []
        in_dim = protein_dim
        for h_dim in proj_hidden:
            proj_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        proj_layers.append(nn.Linear(in_dim, label_embed_dim))
        self.protein_projection = nn.Sequential(*proj_layers)
        
        # Label embeddings - online (trained with gradients)
        self.label_embedding = LabelEmbedding(num_labels, label_embed_dim)
        
        # Label embeddings - target (EMA, no gradients)
        self.target_label_embedding = LabelEmbedding(num_labels, label_embed_dim)
        for param in self.target_label_embedding.parameters():
            param.requires_grad = False
        
        # Predictor for JEPA loss
        self.predictor = LabelJepaPredictor(
            protein_dim=label_embed_dim,
            label_dim=label_embed_dim,
            hidden_mult=predictor_hidden_mult,
        )
        
        # Classification head (auxiliary)
        self.classifier = nn.Linear(label_embed_dim, num_labels)
    
    @torch.no_grad()
    def update_target_embedding(self, ema_decay: float = 0.999):
        """Update target label embeddings with EMA."""
        for online_param, target_param in zip(
            self.label_embedding.parameters(),
            self.target_label_embedding.parameters(),
        ):
            target_param.data.mul_(ema_decay).add_(
                online_param.data, alpha=1 - ema_decay
            )
    
    def encode_protein(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        taxon_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode protein and project to label space.
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            taxon_ids: Optional (B,) taxonomy indices
            
        Returns:
            (B, label_embed_dim) protein embeddings in label space
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        if hasattr(encoder_output, "last_hidden_state"):
            hidden = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            hidden = encoder_output[0]
        else:
            hidden = encoder_output
        
        pooled = self.pooler(hidden, attention_mask)  # (B, protein_dim)
        
        # Fuse taxonomy information if available
        if taxon_ids is not None and self.taxon_embedding is not None:
            taxon_embed = self.taxon_embedding(taxon_ids)  # (B, taxon_dim)
            pooled = self.taxon_fusion(pooled, taxon_embed)  # (B, protein_dim)
        
        projected = self.protein_projection(pooled)  # (B, label_embed_dim)
        
        return projected
    
    def forward_jepa(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        taxon_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for JEPA training.
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            context_mask: (B, num_labels) binary mask of context labels
            target_mask: (B, num_labels) binary mask of target labels
            taxon_ids: Optional (B,) taxonomy indices
        
        Returns:
            pred_embed: (B, label_dim) predicted embeddings
            target_embed: (B, label_dim) target embeddings (detached)
            logits: (B, num_labels) classification logits
        """
        # Encode protein
        protein_embed = self.encode_protein(input_ids, attention_mask, taxon_ids)  # (B, D)
        
        # Get all label embeddings
        all_label_embed = self.label_embedding()  # (L, D)
        all_target_embed = self.target_label_embedding()  # (L, D)
        
        # Aggregate context embeddings (mean of positive context labels)
        # context_mask: (B, L), all_label_embed: (L, D)
        context_expanded = context_mask.unsqueeze(-1)  # (B, L, 1)
        label_expanded = all_label_embed.unsqueeze(0)  # (1, L, D)
        context_sum = (context_expanded * label_expanded).sum(dim=1)  # (B, D)
        context_count = context_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        context_embed = context_sum / context_count  # (B, D)
        
        # Predict target embeddings
        pred_embed = self.predictor(protein_embed, context_embed)  # (B, D)
        
        # Get target embeddings (mean of positive target labels, from EMA)
        target_expanded = target_mask.unsqueeze(-1)  # (B, L, 1)
        target_label_expanded = all_target_embed.unsqueeze(0)  # (1, L, D)
        target_sum = (target_expanded * target_label_expanded).sum(dim=1)  # (B, D)
        target_count = target_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        target_embed = target_sum / target_count  # (B, D)
        
        # Classification logits
        logits = self.classifier(protein_embed)  # (B, num_labels)
        
        return pred_embed, target_embed.detach(), logits
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        taxon_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for inference (classification only).
        
        Args:
            input_ids: (B, L) input token IDs
            attention_mask: (B, L) attention mask
            taxon_ids: Optional (B,) taxonomy indices
        
        Returns:
            logits: (B, num_labels) classification logits
        """
        protein_embed = self.encode_protein(input_ids, attention_mask, taxon_ids)
        logits = self.classifier(protein_embed)
        return logits


def build_encoder(
    model_family: str,
    backbone_ckpt: str,
    jepa_model_dir: Optional[Path] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, int, any]:
    """
    Build protein encoder.
    
    Args:
        model_family: "prott5" or "esm"
        backbone_ckpt: HuggingFace checkpoint name
        jepa_model_dir: Optional path to pretrained JEPA model
        dtype: Model dtype
    
    Returns:
        encoder: Encoder model
        hidden_dim: Hidden dimension
        tokenizer: Tokenizer
    """
    if model_family == "prott5":
        from transformers import T5EncoderModel, T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained(backbone_ckpt, legacy=True)
        encoder = T5EncoderModel.from_pretrained(backbone_ckpt, torch_dtype=dtype)
        hidden_dim = encoder.config.d_model
        
    elif model_family == "esm":
        from transformers import AutoModel, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(backbone_ckpt)
        encoder = AutoModel.from_pretrained(backbone_ckpt, torch_dtype=dtype)
        hidden_dim = encoder.config.hidden_size
        
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    
    # Load JEPA pretrained weights if available
    if jepa_model_dir is not None:
        jepa_model_dir = Path(jepa_model_dir)
        encoder_path = jepa_model_dir / "encoder.pt"
        if encoder_path.exists():
            print(f"Loading JEPA encoder from {encoder_path}")
            state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict, strict=False)
    
    return encoder, hidden_dim, tokenizer


def load_jepa_go_model(
    model_dir: Path,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[JepaGoModel, any, Dict]:
    """
    Load a trained JepaGoModel from checkpoint.
    
    Args:
        model_dir: Directory containing model files
        device: Device to load model to
        dtype: Model dtype
    
    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        config: Model config dict
    """
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Build encoder
    encoder, hidden_dim, tokenizer = build_encoder(
        model_family=config["model_family"],
        backbone_ckpt=config["backbone_ckpt"],
        jepa_model_dir=config.get("jepa_model_dir"),
        dtype=dtype,
    )
    
    # Build model
    model = JepaGoModel(
        encoder=encoder,
        hidden_dim=hidden_dim,
        num_labels=config["num_labels"],
        classifier_hidden=config.get("classifier_hidden", [1024, 512]),
        dropout=config.get("dropout", 0.2),
        freeze_encoder=True,
    )
    
    # Load classifier weights
    classifier_path = model_dir / "classifier.pt"
    if classifier_path.exists():
        state_dict = torch.load(classifier_path, map_location="cpu", weights_only=True)
        model.classifier.load_state_dict(state_dict)
    
    if device is not None:
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, config


def load_label_jepa_model(
    model_dir: Path,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[LabelJepaModel, any, Dict]:
    """
    Load a trained LabelJepaModel from checkpoint.
    """
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Build encoder
    encoder, hidden_dim, tokenizer = build_encoder(
        model_family=config["model_family"],
        backbone_ckpt=config["backbone_ckpt"],
        jepa_model_dir=config.get("jepa_model_dir"),
        dtype=dtype,
    )
    
    # Build model
    model = LabelJepaModel(
        encoder=encoder,
        protein_dim=hidden_dim,
        num_labels=config["num_labels"],
        label_embed_dim=config.get("label_embed_dim", 512),
        predictor_hidden_mult=config.get("predictor_hidden_mult", 2),
        projection_hidden=config.get("projection_hidden", [1024]),
        dropout=config.get("dropout", 0.2),
        freeze_encoder=True,
        # Taxonomy support
        num_taxa=config.get("num_taxa", 0),
        taxon_embed_dim=config.get("taxon_embed_dim", 64),
        taxon_fusion_type=config.get("taxon_fusion_type", "gate"),
    )
    
    # Load weights
    weights_path = model_dir / "model.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    if device is not None:
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, config

