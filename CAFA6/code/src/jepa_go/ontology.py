"""
Gene Ontology (GO) handling module.

Provides utilities for:
- Parsing go-basic.obo files
- Building GO hierarchy graph
- Label propagation (child → parent)
- Hierarchical consistency enforcement (P(parent) >= P(child))
- Export to NetworkX for EDA

Usage:
    from src.jepa_go.ontology import GOntology
    go = GOntology.from_obo("data/Train/go-basic.obo")
    propagated = go.propagate_labels({"GO:0000002"})
    consistent_scores = go.enforce_consistency({"GO:0000002": 0.8, "GO:0007005": 0.5})
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd


class GOntology:
    """
    Gene Ontology graph for hierarchy operations.
    
    Builds a DAG from go-basic.obo file and provides:
    - Ancestor/descendant traversal
    - Label propagation for training
    - Hierarchical consistency enforcement for inference
    """
    
    # Root terms for the three GO namespaces
    ROOT_TERMS = {
        "biological_process": "GO:0008150",
        "molecular_function": "GO:0003674",
        "cellular_component": "GO:0005575",
    }
    
    def __init__(
        self,
        term_to_parents: Dict[str, Set[str]],
        term_to_children: Dict[str, Set[str]],
        term_to_name: Dict[str, str],
        term_to_namespace: Dict[str, str],
    ):
        """
        Initialize with pre-built graph structures.
        
        Use `GOntology.from_obo()` to create from an OBO file.
        """
        self.term_to_parents = term_to_parents
        self.term_to_children = term_to_children
        self.term_to_name = term_to_name
        self.term_to_namespace = term_to_namespace
        
        # Cache for transitive closures
        self._ancestors_cache: Dict[str, FrozenSet[str]] = {}
        self._descendants_cache: Dict[str, FrozenSet[str]] = {}
    
    @classmethod
    def from_obo(cls, obo_path: Path | str) -> "GOntology":
        """
        Parse a GO OBO file and build the ontology graph.
        
        Args:
            obo_path: Path to go-basic.obo file
            
        Returns:
            GOntology instance with parsed hierarchy
        """
        obo_path = Path(obo_path)
        
        term_to_parents: Dict[str, Set[str]] = defaultdict(set)
        term_to_children: Dict[str, Set[str]] = defaultdict(set)
        term_to_name: Dict[str, str] = {}
        term_to_namespace: Dict[str, str] = {}
        
        # Regex patterns
        id_pattern = re.compile(r"^id:\s*(GO:\d+)")
        name_pattern = re.compile(r"^name:\s*(.+)")
        namespace_pattern = re.compile(r"^namespace:\s*(.+)")
        is_a_pattern = re.compile(r"^is_a:\s*(GO:\d+)")
        obsolete_pattern = re.compile(r"^is_obsolete:\s*true")
        
        current_id = None
        current_name = None
        current_namespace = None
        current_parents: List[str] = []
        is_obsolete = False
        in_term = False
        
        with open(obo_path, "r") as f:
            for line in f:
                line = line.strip()
                
                if line == "[Term]":
                    # Save previous term
                    if current_id and not is_obsolete:
                        term_to_name[current_id] = current_name or ""
                        term_to_namespace[current_id] = current_namespace or ""
                        for parent in current_parents:
                            term_to_parents[current_id].add(parent)
                            term_to_children[parent].add(current_id)
                    
                    # Reset for new term
                    current_id = None
                    current_name = None
                    current_namespace = None
                    current_parents = []
                    is_obsolete = False
                    in_term = True
                    continue
                
                if line.startswith("[") and line.endswith("]"):
                    # Different stanza (Typedef, etc.) - save current term first
                    if current_id and not is_obsolete:
                        term_to_name[current_id] = current_name or ""
                        term_to_namespace[current_id] = current_namespace or ""
                        for parent in current_parents:
                            term_to_parents[current_id].add(parent)
                            term_to_children[parent].add(current_id)
                    in_term = False
                    current_id = None
                    continue
                
                if not in_term:
                    continue
                
                # Parse term attributes
                if match := id_pattern.match(line):
                    current_id = match.group(1)
                elif match := name_pattern.match(line):
                    current_name = match.group(1)
                elif match := namespace_pattern.match(line):
                    current_namespace = match.group(1)
                elif match := is_a_pattern.match(line):
                    current_parents.append(match.group(1))
                elif obsolete_pattern.match(line):
                    is_obsolete = True
        
        # Save last term
        if current_id and not is_obsolete:
            term_to_name[current_id] = current_name or ""
            term_to_namespace[current_id] = current_namespace or ""
            for parent in current_parents:
                term_to_parents[current_id].add(parent)
                term_to_children[parent].add(current_id)
        
        return cls(
            term_to_parents=dict(term_to_parents),
            term_to_children=dict(term_to_children),
            term_to_name=term_to_name,
            term_to_namespace=term_to_namespace,
        )
    
    def __len__(self) -> int:
        """Number of terms in the ontology."""
        return len(self.term_to_name)
    
    def __contains__(self, term: str) -> bool:
        """Check if term exists in ontology."""
        return term in self.term_to_name
    
    def get_parents(self, term: str) -> Set[str]:
        """Get direct parents of a term."""
        return self.term_to_parents.get(term, set())
    
    def get_children(self, term: str) -> Set[str]:
        """Get direct children of a term."""
        return self.term_to_children.get(term, set())
    
    def get_ancestors(self, term: str) -> FrozenSet[str]:
        """
        Get all ancestors (transitive parents) of a term.
        
        Uses caching for efficiency.
        """
        if term in self._ancestors_cache:
            return self._ancestors_cache[term]
        
        ancestors: Set[str] = set()
        stack = list(self.get_parents(term))
        
        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                stack.extend(self.get_parents(parent))
        
        result = frozenset(ancestors)
        self._ancestors_cache[term] = result
        return result
    
    def get_descendants(self, term: str) -> FrozenSet[str]:
        """
        Get all descendants (transitive children) of a term.
        
        Uses caching for efficiency.
        """
        if term in self._descendants_cache:
            return self._descendants_cache[term]
        
        descendants: Set[str] = set()
        stack = list(self.get_children(term))
        
        while stack:
            child = stack.pop()
            if child not in descendants:
                descendants.add(child)
                stack.extend(self.get_children(child))
        
        result = frozenset(descendants)
        self._descendants_cache[term] = result
        return result
    
    def propagate_labels(self, terms: Set[str]) -> Set[str]:
        """
        Propagate labels up the hierarchy.
        
        If a protein has term T, it implicitly has all ancestors of T.
        This is the "true path rule" in GO.
        
        Args:
            terms: Set of GO term IDs
            
        Returns:
            Set including original terms and all their ancestors
        """
        propagated = set(terms)
        for term in terms:
            propagated.update(self.get_ancestors(term))
        return propagated
    
    def enforce_consistency(
        self,
        scores: Dict[str, float],
        label_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Enforce hierarchical consistency: P(parent) >= max(P(children)).
        
        For each term, update its score to be at least the max of its children's scores.
        This ensures predictions respect the GO hierarchy.
        
        Args:
            scores: Dict mapping GO term ID to prediction score
            label_list: Optional list of terms to consider (speeds up for known label set)
            
        Returns:
            Dict with consistent scores
        """
        if label_list is None:
            label_list = list(scores.keys())
        
        # Create a working copy
        consistent = dict(scores)
        
        # Sort terms by depth (deeper terms first) to propagate from leaves to root
        # We approximate depth by number of ancestors
        terms_with_depth = [
            (term, len(self.get_ancestors(term)))
            for term in label_list
            if term in scores
        ]
        # Sort by depth descending (process leaves first)
        terms_with_depth.sort(key=lambda x: -x[1])
        
        # Propagate max scores upward
        for term, _ in terms_with_depth:
            term_score = consistent.get(term, 0.0)
            for parent in self.get_parents(term):
                if parent in consistent:
                    consistent[parent] = max(consistent[parent], term_score)
        
        return consistent
    
    def get_depth(self, term: str) -> int:
        """Get the depth of a term (shortest path to root)."""
        return len(self.get_ancestors(term))
    
    # =========================================================================
    # Export methods for EDA
    # =========================================================================
    
    def to_networkx(self, terms: Optional[Set[str]] = None):
        """
        Export ontology as a NetworkX DiGraph.
        
        Args:
            terms: Optional subset of terms to include. If None, exports all.
            
        Returns:
            networkx.DiGraph with GO terms as nodes and is_a relations as edges
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for graph export. Install with: pip install networkx")
        
        G = nx.DiGraph()
        
        # Determine which terms to include
        if terms is None:
            terms_to_add = set(self.term_to_name.keys())
        else:
            # Include specified terms and their ancestors
            terms_to_add = set()
            for term in terms:
                terms_to_add.add(term)
                terms_to_add.update(self.get_ancestors(term))
        
        # Add nodes with attributes
        for term in terms_to_add:
            G.add_node(
                term,
                name=self.term_to_name.get(term, ""),
                namespace=self.term_to_namespace.get(term, ""),
                depth=self.get_depth(term),
            )
        
        # Add edges (child -> parent for is_a relationship)
        for term in terms_to_add:
            for parent in self.get_parents(term):
                if parent in terms_to_add:
                    G.add_edge(term, parent, relation="is_a")
        
        return G
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export ontology as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: term_id, name, namespace, num_parents, num_children, depth
        """
        data = []
        for term_id in self.term_to_name:
            data.append({
                "term_id": term_id,
                "name": self.term_to_name[term_id],
                "namespace": self.term_to_namespace.get(term_id, ""),
                "num_parents": len(self.get_parents(term_id)),
                "num_children": len(self.get_children(term_id)),
                "num_ancestors": len(self.get_ancestors(term_id)),
                "num_descendants": len(self.get_descendants(term_id)),
                "depth": self.get_depth(term_id),
            })
        return pd.DataFrame(data)
    
    def edges_to_dataframe(self) -> pd.DataFrame:
        """
        Export edges as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: child, parent, relation
        """
        edges = []
        for child, parents in self.term_to_parents.items():
            for parent in parents:
                edges.append({
                    "child": child,
                    "parent": parent,
                    "child_name": self.term_to_name.get(child, ""),
                    "parent_name": self.term_to_name.get(parent, ""),
                    "relation": "is_a",
                })
        return pd.DataFrame(edges)
    
    def get_namespace_stats(self) -> pd.DataFrame:
        """Get statistics per namespace."""
        stats = defaultdict(lambda: {"count": 0, "max_depth": 0, "total_edges": 0})
        
        for term_id in self.term_to_name:
            ns = self.term_to_namespace.get(term_id, "unknown")
            stats[ns]["count"] += 1
            stats[ns]["max_depth"] = max(stats[ns]["max_depth"], self.get_depth(term_id))
            stats[ns]["total_edges"] += len(self.get_parents(term_id))
        
        return pd.DataFrame([
            {"namespace": ns, **s} for ns, s in stats.items()
        ])


# =========================================================================
# Utility functions
# =========================================================================

def load_ontology(obo_path: Path | str = "data/Train/go-basic.obo") -> GOntology:
    """Convenience function to load the default GO ontology."""
    return GOntology.from_obo(obo_path)


# =========================================================================
# Optional: Integration with goatools/pronto
# =========================================================================

def load_with_goatools(obo_path: Path | str) -> "GOntology":
    """
    Alternative loader using goatools library.
    
    Advantages:
    - More robust OBO parsing
    - Handles alternative IDs and synonyms
    - Built-in support for GO enrichment analysis
    
    Requires: pip install goatools
    """
    try:
        from goatools.obo_parser import GODag
    except ImportError:
        raise ImportError("goatools is required. Install with: pip install goatools")
    
    obo_path = str(obo_path)
    godag = GODag(obo_path, optional_attrs=["relationship"])
    
    term_to_parents: Dict[str, Set[str]] = defaultdict(set)
    term_to_children: Dict[str, Set[str]] = defaultdict(set)
    term_to_name: Dict[str, str] = {}
    term_to_namespace: Dict[str, str] = {}
    
    for term_id, term in godag.items():
        if term.is_obsolete:
            continue
        
        term_to_name[term_id] = term.name
        term_to_namespace[term_id] = term.namespace
        
        for parent in term.parents:
            parent_id = parent.id
            term_to_parents[term_id].add(parent_id)
            term_to_children[parent_id].add(term_id)
    
    return GOntology(
        term_to_parents=dict(term_to_parents),
        term_to_children=dict(term_to_children),
        term_to_name=term_to_name,
        term_to_namespace=term_to_namespace,
    )


def load_with_pronto(obo_path: Path | str) -> "GOntology":
    """
    Alternative loader using pronto library.
    
    Advantages:
    - Faster parsing (Rust-based fastobo backend)
    - General ontology support (not GO-specific)
    - Modern API
    
    Requires: pip install pronto
    """
    try:
        import pronto
    except ImportError:
        raise ImportError("pronto is required. Install with: pip install pronto")
    
    ont = pronto.Ontology(str(obo_path))
    
    term_to_parents: Dict[str, Set[str]] = defaultdict(set)
    term_to_children: Dict[str, Set[str]] = defaultdict(set)
    term_to_name: Dict[str, str] = {}
    term_to_namespace: Dict[str, str] = {}
    
    for term in ont.terms():
        if term.obsolete:
            continue
        
        term_id = term.id
        term_to_name[term_id] = term.name or ""
        term_to_namespace[term_id] = term.namespace or ""
        
        for parent in term.superclasses(distance=1, with_self=False):
            parent_id = parent.id
            term_to_parents[term_id].add(parent_id)
            term_to_children[parent_id].add(term_id)
    
    return GOntology(
        term_to_parents=dict(term_to_parents),
        term_to_children=dict(term_to_children),
        term_to_name=term_to_name,
        term_to_namespace=term_to_namespace,
    )
