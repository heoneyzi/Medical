"""Lightweight structure-based stand-in for :class:`phenocompass.PhenoCompass`.

Lets the Direction-A (structure) pipeline run end-to-end without checkpoints,
PyTorch or RDKit, so you can smoke-test wiring and preview the deliverable.

⚠️  Embeddings/scores are SYNTHETIC. Any number or figure produced in mock mode
is a placeholder — re-run with the real checkpoints (drop ``--mock``) to get real
results. To make the preview look plausible, per-compound structure embeddings
are drawn near a per-MoA prototype (via :meth:`prime_true_moa`); this injected
structure is artificial.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class MockPhenoCompass:
    N_MODELS = 4
    LATENT_DIM = 64

    def __init__(self, moas: Sequence[str], n_anchor: int = 6, seed: int = 0):
        self._moas = list(moas)
        self._rng = np.random.default_rng(seed)
        self.latent_dim = self.LATENT_DIM
        self._proto = {m: 0.42 * self._rng.standard_normal(self.LATENT_DIM) for m in self._moas}

        class _Ens:
            pass

        self.ensemble = _Ens()
        self.ensemble.model_class_list = [f"mock_{i}" for i in range(self.N_MODELS)]

        # anchor embeddings clustered around each MoA prototype
        self.anchor_embeddings_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        for m in self._moas:
            self.anchor_embeddings_dict[m] = {}
            for mkey in self.ensemble.model_class_list:
                emb = self._proto[m][None, :] + 0.35 * self._rng.standard_normal((n_anchor, self.LATENT_DIM))
                self.anchor_embeddings_dict[m][mkey] = pd.DataFrame(
                    emb, index=[f"{m}_anchor_{j}" for j in range(n_anchor)])

        self._smiles_moa: Dict[str, Optional[str]] = {}

    # -- interface parity with the real class --------------------------------
    def get_anchor_moas(self) -> List[str]:
        return list(self._moas)

    def prime_true_moa(self, smiles: Sequence[str], true_moa: Sequence[Optional[str]]) -> None:
        """Remember each SMILES' MoA so synthetic embeddings carry structure."""
        for s, m in zip(smiles, true_moa):
            self._smiles_moa[s] = m

    def _embed_one(self, smi: str) -> np.ndarray:
        m = self._smiles_moa.get(smi)
        base = self._proto.get(m) if m else None
        if base is None:
            # deterministic per-SMILES random vector
            r = np.random.default_rng(abs(hash(smi)) % (2**32))
            return r.standard_normal(self.LATENT_DIM)
        r = np.random.default_rng(abs(hash(smi)) % (2**32))
        return base + 1.8 * r.standard_normal(self.LATENT_DIM)

    def compute_struct_embeddings(self, smiles: Sequence[str], batch_size: int = 512) -> np.ndarray:
        base = np.stack([self._embed_one(s) for s in smiles])       # (n, d)
        out = np.zeros((self.N_MODELS, len(smiles), self.LATENT_DIM))
        for i in range(self.N_MODELS):
            out[i] = base + 0.25 * self._rng.standard_normal(base.shape)
        return out

    # scoring API mirrors phenocompass.inference.scorer.PhenoCompass ----------
    def score_against_anchors(self, query_mols, moa_list=None, metric="cosine", mode="mean"):
        from sklearn.metrics.pairwise import cosine_similarity
        if moa_list is None:
            moa_list = self._moas
        q = self.compute_struct_embeddings(query_mols)
        df = pd.DataFrame(index=list(query_mols))
        for moa in moa_list:
            per_model = []
            for i, mkey in enumerate(self.ensemble.model_class_list):
                a = self.anchor_embeddings_dict[moa][mkey].values
                per_model.append(cosine_similarity(a, q[i]).mean(axis=0))
            df[f"{moa}_score"] = np.mean(per_model, axis=0)
        return df

    def score_against_reference_groups(self, query_smiles, group_to_smiles,
                                       reference_morph_df=None, metric="cosine",
                                       aggregation="mean", batch_size=512):
        from sklearn.metrics.pairwise import cosine_similarity
        uniq = list(dict.fromkeys(s for v in group_to_smiles.values() for s in v))
        ref = self.compute_struct_embeddings(uniq)
        q = self.compute_struct_embeddings(query_smiles)
        idx = {s: i for i, s in enumerate(uniq)}
        groups = list(group_to_smiles)
        per_model = np.zeros((self.N_MODELS, len(query_smiles), len(groups)))
        for mi in range(self.N_MODELS):
            for gi, g in enumerate(groups):
                ids = [idx[s] for s in group_to_smiles[g] if s in idx]
                if ids:
                    per_model[mi, :, gi] = cosine_similarity(ref[mi][ids], q[mi]).mean(axis=0)
        return pd.DataFrame(per_model.mean(axis=0), index=list(query_smiles), columns=groups)
