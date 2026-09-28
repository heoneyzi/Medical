"""
Dataset utilities for JEPA-based GO term prediction.

Supports:
- Ontology-based label propagation (child → parent)
- Taxonomy embedding for species-aware predictions
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from src.common.fasta import read_fasta

if TYPE_CHECKING:
    from src.jepa_go.ontology import GOntology


def load_terms(
    terms_tsv: Path,
    namespace: Optional[str] = None,
    min_count: int = 5,
    max_labels: int = 8192,
    ontology: Optional["GOntology"] = None,
    propagate_labels: bool = True,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
    """
    Load GO term annotations from TSV file.
    
    Args:
        terms_tsv: Path to train_terms.tsv (EntryID, term, aspect)
        namespace: Optional filter by namespace (MF, BP, CC)
        min_count: Minimum occurrences for a term to be included
        max_labels: Maximum number of labels to keep
        ontology: Optional GOntology for label propagation
        propagate_labels: If True and ontology provided, propagate labels to ancestors
    
    Returns:
        protein_to_terms: Dict mapping protein ID to list of GO terms
        label_list: Ordered list of GO term labels
        label_to_idx: Dict mapping GO term to index
    """
    protein_to_terms: Dict[str, List[str]] = defaultdict(list)
    term_counts: Dict[str, int] = defaultdict(int)
    
    with open(terms_tsv) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            entry_id, term, aspect = parts[:3]
            
            if namespace and aspect != namespace:
                continue
            
            protein_to_terms[entry_id].append(term)
            term_counts[term] += 1
    
    # Propagate labels using ontology (true path rule)
    if ontology is not None and propagate_labels:
        propagated_protein_to_terms: Dict[str, List[str]] = {}
        for pid, terms in protein_to_terms.items():
            original_set = set(terms)
            propagated_set = ontology.propagate_labels(original_set)
            propagated_protein_to_terms[pid] = list(propagated_set)
            # Update term counts
            for term in propagated_set - original_set:
                term_counts[term] += 1
        protein_to_terms = propagated_protein_to_terms
    
    # Filter by min count
    valid_terms = {t for t, c in term_counts.items() if c >= min_count}
    
    # Sort by frequency and take top max_labels
    sorted_terms = sorted(valid_terms, key=lambda t: -term_counts[t])[:max_labels]
    label_list = sorted_terms
    label_to_idx = {t: i for i, t in enumerate(label_list)}
    
    # Filter protein_to_terms
    label_set = set(label_list)
    filtered_protein_to_terms = {}
    for pid, terms in protein_to_terms.items():
        filtered_terms = [t for t in terms if t in label_set]
        if filtered_terms:
            filtered_protein_to_terms[pid] = filtered_terms
    
    return filtered_protein_to_terms, label_list, label_to_idx


def load_taxonomy_mapping(
    taxonomy_tsv: Path,
) -> Tuple[Dict[str, int], Dict[int, int], List[int]]:
    """
    Load protein → taxonomy ID mapping.
    
    Convenience wrapper around taxonomy module.
    
    Args:
        taxonomy_tsv: Path to train_taxonomy.tsv
        
    Returns:
        protein_to_taxon: Dict mapping protein ID → original taxon ID
        taxon_to_idx: Dict mapping original taxon ID → integer index [1, N]
        taxon_list: List of original taxon IDs in index order
    """
    from src.jepa_go.taxonomy import load_taxonomy
    return load_taxonomy(taxonomy_tsv)


def load_ia_weights(ia_tsv: Path, label_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Load Information Accretion weights for GO terms.
    
    Args:
        ia_tsv: Path to IA.tsv file (term, IA value)
        label_to_idx: Dict mapping GO term to index
    
    Returns:
        Tensor of shape (num_labels,) with IA weights (default 1.0 if missing)
    """
    num_labels = len(label_to_idx)
    ia_weights = torch.ones(num_labels)
    
    with open(ia_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                term, ia_value = parts[0], float(parts[1])
                if term in label_to_idx:
                    ia_weights[label_to_idx[term]] = ia_value
    
    return ia_weights


class JepaGoDataset(Dataset):
    """
    Dataset for JEPA encoder-based GO prediction.
    """
    
    def __init__(
        self,
        sequences: Dict[str, str],
        protein_to_terms: Dict[str, List[str]],
        label_to_idx: Dict[str, int],
        tokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)
        
        # Only keep proteins that have both sequence and terms
        self.protein_ids = [
            pid for pid in protein_to_terms.keys()
            if pid in sequences
        ]
        self.sequences = {pid: sequences[pid] for pid in self.protein_ids}
        self.protein_to_terms = {pid: protein_to_terms[pid] for pid in self.protein_ids}
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int):
        pid = self.protein_ids[idx]
        seq = self.sequences[pid]
        terms = self.protein_to_terms[pid]
        
        # Tokenize
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create label tensor
        labels = torch.zeros(self.num_labels)
        for term in terms:
            if term in self.label_to_idx:
                labels[self.label_to_idx[term]] = 1.0
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "protein_id": pid,
        }


class LabelJepaDataset(Dataset):
    """
    Dataset for Label-Space JEPA training.
    
    For each protein, provides:
    - Sequence tokens
    - Context labels (subset of positive labels)
    - Target labels (remaining positive labels to predict)
    """
    
    def __init__(
        self,
        sequences: Dict[str, str],
        protein_to_terms: Dict[str, List[str]],
        label_to_idx: Dict[str, int],
        tokenizer,
        max_length: int = 1024,
        context_ratio: float = 0.6,
        min_context: int = 1,
        min_target: int = 1,
        protein_to_taxon: Optional[Dict[str, int]] = None,
        taxon_to_idx: Optional[Dict[int, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)
        self.context_ratio = context_ratio
        self.min_context = min_context
        self.min_target = min_target
        
        # Taxonomy support
        self.protein_to_taxon = protein_to_taxon or {}
        self.taxon_to_idx = taxon_to_idx or {}
        self.use_taxonomy = bool(protein_to_taxon and taxon_to_idx)
        
        # Only keep proteins with enough labels
        self.protein_ids = []
        for pid in protein_to_terms.keys():
            if pid in sequences:
                terms = protein_to_terms[pid]
                valid_terms = [t for t in terms if t in label_to_idx]
                # Need at least min_context + min_target labels
                if len(valid_terms) >= min_context + min_target:
                    self.protein_ids.append(pid)
        
        self.sequences = {pid: sequences[pid] for pid in self.protein_ids}
        self.protein_to_terms = {pid: protein_to_terms[pid] for pid in self.protein_ids}
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int):
        pid = self.protein_ids[idx]
        seq = self.sequences[pid]
        terms = self.protein_to_terms[pid]
        
        # Filter to valid terms
        valid_terms = [t for t in terms if t in self.label_to_idx]
        
        # Tokenize
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Split labels into context and target
        random.shuffle(valid_terms)
        n_context = max(self.min_context, int(len(valid_terms) * self.context_ratio))
        n_context = min(n_context, len(valid_terms) - self.min_target)
        
        context_terms = valid_terms[:n_context]
        target_terms = valid_terms[n_context:]
        
        # Create tensors
        all_labels = torch.zeros(self.num_labels)
        context_mask = torch.zeros(self.num_labels)
        target_mask = torch.zeros(self.num_labels)
        
        for term in context_terms:
            idx_t = self.label_to_idx[term]
            all_labels[idx_t] = 1.0
            context_mask[idx_t] = 1.0
        
        for term in target_terms:
            idx_t = self.label_to_idx[term]
            all_labels[idx_t] = 1.0
            target_mask[idx_t] = 1.0
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "all_labels": all_labels,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "protein_id": pid,
        }
        
        # Add taxonomy if available
        if self.use_taxonomy:
            taxon_id = self.protein_to_taxon.get(pid)
            taxon_idx = self.taxon_to_idx.get(taxon_id, 0) if taxon_id else 0
            result["taxon_id"] = taxon_idx
        
        return result


class InferenceDataset(Dataset):
    """
    Dataset for inference (no labels needed).
    """
    
    def __init__(
        self,
        sequences: Dict[str, str],
        tokenizer,
        max_length: int = 1024,
        protein_to_taxon: Optional[Dict[str, int]] = None,
        taxon_to_idx: Optional[Dict[int, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.protein_ids = list(sequences.keys())
        self.sequences = sequences
        
        # Taxonomy support
        self.protein_to_taxon = protein_to_taxon or {}
        self.taxon_to_idx = taxon_to_idx or {}
        self.use_taxonomy = bool(protein_to_taxon and taxon_to_idx)
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int):
        pid = self.protein_ids[idx]
        seq = self.sequences[pid]
        
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "protein_id": pid,
        }
        
        # Add taxonomy if available
        if self.use_taxonomy:
            taxon_id = self.protein_to_taxon.get(pid)
            taxon_idx = self.taxon_to_idx.get(taxon_id, 0) if taxon_id else 0
            result["taxon_id"] = taxon_idx
        
        return result


def train_val_split(
    protein_to_terms: Dict[str, List[str]],
    val_fraction: float = 0.03,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Split proteins into training and validation sets.
    
    Args:
        protein_to_terms: Dict mapping protein ID to list of terms
        val_fraction: Fraction for validation (default 3%)
        seed: Random seed
    
    Returns:
        train_ids: Set of training protein IDs
        val_ids: Set of validation protein IDs
    """
    random.seed(seed)
    all_ids = list(protein_to_terms.keys())
    random.shuffle(all_ids)
    
    n_val = max(1, int(len(all_ids) * val_fraction))
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    
    return train_ids, val_ids
