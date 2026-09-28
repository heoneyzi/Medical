"""
Taxonomy handling module for species-aware protein function prediction.

Provides utilities for:
- Loading protein → taxonomy ID mappings
- Taxonomy embedding module for PyTorch
- Export to pandas for EDA

Usage:
    from src.jepa_go.taxonomy import load_taxonomy, TaxonomyEmbedding
    
    # Load taxonomy mapping
    taxon_map, taxon_to_idx = load_taxonomy("data/Train/train_taxonomy.tsv")
    
    # Create embedding layer
    embed = TaxonomyEmbedding(num_taxa=len(taxon_to_idx), embed_dim=64)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import torch.nn as nn


def load_taxonomy(
    tsv_path: Path | str,
    min_count: int = 1,
) -> Tuple[Dict[str, int], Dict[int, int], List[int]]:
    """
    Load protein → taxonomy ID mapping from TSV file.
    
    Args:
        tsv_path: Path to train_taxonomy.tsv (protein_id, taxon_id)
        min_count: Minimum count for a taxon to be included (default 1)
        
    Returns:
        protein_to_taxon: Dict mapping protein ID → original taxon ID
        taxon_to_idx: Dict mapping original taxon ID → integer index [1, N]
                      (0 is reserved for unknown/padding)
        taxon_list: List of original taxon IDs in index order
    """
    tsv_path = Path(tsv_path)
    
    # Read protein → taxon mapping
    protein_to_taxon: Dict[str, int] = {}
    taxon_counts: Counter = Counter()
    
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                protein_id = parts[0]
                taxon_id = int(parts[1])
                protein_to_taxon[protein_id] = taxon_id
                taxon_counts[taxon_id] += 1
    
    # Filter by min_count
    valid_taxa = {t for t, c in taxon_counts.items() if c >= min_count}
    
    # Create taxon → index mapping (0 is reserved for unknown)
    taxon_list = sorted(valid_taxa)
    taxon_to_idx = {taxon: idx + 1 for idx, taxon in enumerate(taxon_list)}
    
    return protein_to_taxon, taxon_to_idx, taxon_list


def get_taxon_idx(
    protein_id: str,
    protein_to_taxon: Dict[str, int],
    taxon_to_idx: Dict[int, int],
) -> int:
    """
    Get the index of a protein's taxonomy ID.
    
    Args:
        protein_id: Protein identifier
        protein_to_taxon: Protein → taxon mapping
        taxon_to_idx: Taxon → index mapping
        
    Returns:
        Index of the taxon (0 if not found)
    """
    taxon = protein_to_taxon.get(protein_id)
    if taxon is None:
        return 0
    return taxon_to_idx.get(taxon, 0)


class TaxonomyEmbedding(nn.Module):
    """
    Learnable embeddings for taxonomy IDs.
    
    Maps taxonomy IDs to dense vectors that can be fused with protein embeddings.
    Index 0 is reserved for unknown/missing taxonomy (padding).
    """
    
    def __init__(
        self,
        num_taxa: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize taxonomy embedding.
        
        Args:
            num_taxa: Number of unique taxonomy IDs (excluding padding)
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_taxa = num_taxa
        self.embed_dim = embed_dim
        
        # +1 for padding index 0
        self.embedding = nn.Embedding(
            num_taxa + 1,
            embed_dim,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        # Zero out padding embedding
        self.embedding.weight.data[0].zero_()
    
    def forward(self, taxon_ids: torch.Tensor) -> torch.Tensor:
        """
        Get taxonomy embeddings.
        
        Args:
            taxon_ids: (B,) tensor of taxonomy indices
            
        Returns:
            (B, embed_dim) tensor of embeddings
        """
        embeds = self.embedding(taxon_ids)
        return self.dropout(embeds)


class TaxonomyFusion(nn.Module):
    """
    Fuse taxonomy embeddings with protein embeddings.
    
    Supports multiple fusion strategies:
    - "concat": Concatenate and project
    - "add": Add (requires same dimensions)
    - "gate": Gated fusion
    """
    
    def __init__(
        self,
        protein_dim: int,
        taxon_dim: int,
        output_dim: Optional[int] = None,
        fusion_type: str = "concat",
        dropout: float = 0.1,
    ):
        """
        Initialize taxonomy fusion.
        
        Args:
            protein_dim: Protein embedding dimension
            taxon_dim: Taxonomy embedding dimension
            output_dim: Output dimension (defaults to protein_dim)
            fusion_type: "concat", "add", or "gate"
            dropout: Dropout rate
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.taxon_dim = taxon_dim
        self.output_dim = output_dim or protein_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.projection = nn.Sequential(
                nn.Linear(protein_dim + taxon_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "add":
            if taxon_dim != protein_dim:
                self.taxon_proj = nn.Linear(taxon_dim, protein_dim)
            else:
                self.taxon_proj = nn.Identity()
            self.layer_norm = nn.LayerNorm(protein_dim)
        elif fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(protein_dim + taxon_dim, protein_dim),
                nn.Sigmoid(),
            )
            self.value = nn.Linear(taxon_dim, protein_dim)
            self.layer_norm = nn.LayerNorm(protein_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        protein_embed: torch.Tensor,
        taxon_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse protein and taxonomy embeddings.
        
        Args:
            protein_embed: (B, protein_dim) protein embeddings
            taxon_embed: (B, taxon_dim) taxonomy embeddings
            
        Returns:
            (B, output_dim) fused embeddings
        """
        if self.fusion_type == "concat":
            combined = torch.cat([protein_embed, taxon_embed], dim=-1)
            return self.projection(combined)
        elif self.fusion_type == "add":
            taxon_proj = self.taxon_proj(taxon_embed)
            return self.layer_norm(protein_embed + taxon_proj)
        elif self.fusion_type == "gate":
            combined = torch.cat([protein_embed, taxon_embed], dim=-1)
            gate = self.gate(combined)
            value = self.value(taxon_embed)
            return self.layer_norm(protein_embed + gate * value)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


# =========================================================================
# EDA utilities
# =========================================================================

def taxonomy_to_dataframe(
    protein_to_taxon: Dict[str, int],
    taxon_to_idx: Dict[int, int],
) -> pd.DataFrame:
    """
    Export taxonomy data as a pandas DataFrame.
    
    Returns:
        DataFrame with columns: taxon_id, index, count
    """
    taxon_counts = Counter(protein_to_taxon.values())
    
    data = []
    for taxon, idx in sorted(taxon_to_idx.items(), key=lambda x: -taxon_counts.get(x[0], 0)):
        data.append({
            "taxon_id": taxon,
            "index": idx,
            "count": taxon_counts.get(taxon, 0),
        })
    
    return pd.DataFrame(data)


def get_taxonomy_stats(tsv_path: Path | str) -> pd.DataFrame:
    """
    Get taxonomy statistics from the TSV file.
    
    Returns:
        DataFrame with taxon_id, count, and percentage columns
    """
    tsv_path = Path(tsv_path)
    
    taxon_counts: Counter = Counter()
    total = 0
    
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                taxon_id = int(parts[1])
                taxon_counts[taxon_id] += 1
                total += 1
    
    data = []
    for taxon, count in taxon_counts.most_common():
        data.append({
            "taxon_id": taxon,
            "count": count,
            "percentage": 100 * count / total if total > 0 else 0,
        })
    
    return pd.DataFrame(data)


# =========================================================================
# NCBI Taxonomy integration (optional)
# =========================================================================

# Common taxonomy IDs for reference
COMMON_TAXA = {
    9606: "Homo sapiens (Human)",
    10090: "Mus musculus (Mouse)",
    10116: "Rattus norvegicus (Rat)",
    7955: "Danio rerio (Zebrafish)",
    7227: "Drosophila melanogaster (Fruit fly)",
    6239: "Caenorhabditis elegans (Roundworm)",
    3702: "Arabidopsis thaliana (Thale cress)",
    559292: "Saccharomyces cerevisiae S288C (Yeast)",
    284812: "Schizosaccharomyces pombe (Fission yeast)",
    83333: "Escherichia coli K-12",
    9031: "Gallus gallus (Chicken)",
    9913: "Bos taurus (Cow)",
}


def get_taxon_name(taxon_id: int) -> Optional[str]:
    """Get human-readable name for common taxa."""
    return COMMON_TAXA.get(taxon_id)
