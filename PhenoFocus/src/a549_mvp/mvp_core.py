"""Core scoring + evaluation logic for the A549 cross-line MVP.

Kept separate from the CLI (``run_mvp.py``) and free of any hard PyTorch
dependency so the numeric pipeline can be unit-tested with a mock model.

Two demonstrations, both driven by the U2OS-trained PhenoCompass ensemble:

1. **Cross-line MoA recovery (cross-modal).** A549 Cell Painting profiles are
   embedded with the morphology encoder, then scored (cosine, in the shared
   latent space) against the six frozen MoA structure/joint anchor sets.
   If the U2OS model generalises to A549, a compound's *own* MoA anchor scores
   highest -> high one-vs-rest AUROC and strong top-k enrichment.

2. **Customer-anchor few-shot retrieval (the product loop).** For each MoA we
   treat a few A549 actives as the "customer anchors", then rank *all other*
   A549 compounds by morphological similarity to them. Same-MoA compounds
   should rank at the top -> this is exactly "give us a few actives, we return
   the truly related ones", now shown to hold in a cell line the model never
   trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    roc_auc_score = None
    cosine_similarity = None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _fewshot_mean_cosine(anchor_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Mean cosine similarity of each query to a set of anchors.

    anchor_emb: (n_anchor, d); query_emb: (n_query, d) -> (n_query,).
    Mirrors ``phenocompass.evaluation.metrics.compute_fewshot_similarities``
    (metric='cosine', mode='mean') without importing torch.
    """
    sims = cosine_similarity(anchor_emb, query_emb)  # (n_anchor, n_query)
    return sims.mean(axis=0)


def score_morph_against_anchors(
    pc,
    morph_emb: np.ndarray,
    moa_list: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Score A549 morphology embeddings against the frozen MoA anchors.

    Parameters
    ----------
    pc : PhenoCompass-like
        Must expose ``ensemble.model_class_list`` and
        ``anchor_embeddings_dict[moa][model_key]`` (DataFrame, latent columns).
    morph_emb : np.ndarray
        A549 morphology embeddings, shape ``(n_models, n_samples, latent_dim)``.
    moa_list : sequence of str, optional
        MoAs to score against (default: all available anchors).

    Returns
    -------
    pd.DataFrame
        ``(n_samples, n_moa)`` ensemble-averaged cosine score per MoA.
    """
    if moa_list is None:
        moa_list = pc.get_anchor_moas()
    model_keys = list(pc.ensemble.model_class_list)
    n_models, n_samples, _ = morph_emb.shape

    scores = {}
    for moa in moa_list:
        per_model = np.zeros((n_models, n_samples))
        for i, mkey in enumerate(model_keys):
            anchors = np.asarray(pc.anchor_embeddings_dict[moa][mkey].values, dtype=np.float64)
            per_model[i] = _fewshot_mean_cosine(anchors, morph_emb[i])
        scores[moa] = per_model.mean(axis=0)
    return pd.DataFrame(scores)


# ---------------------------------------------------------------------------
# Demonstration 1: cross-line MoA recovery
# ---------------------------------------------------------------------------

@dataclass
class CrossLineResult:
    score_df: pd.DataFrame            # per-compound anchor scores (+ true/pred MoA)
    auroc_per_moa: pd.DataFrame       # one-vs-rest AUROC + enrichment per MoA
    confusion: pd.DataFrame           # true MoA x predicted MoA counts
    macro_auroc: float
    topk_enrichment: float
    accuracy: float


def evaluate_cross_line(
    anchor_scores: pd.DataFrame,
    true_moa: Sequence[Optional[str]],
    moa_list: Sequence[str],
    top_k: int = 1,
) -> CrossLineResult:
    """Evaluate how well A549 morphology recovers each compound's MoA.

    anchor_scores: (n, n_moa) from :func:`score_morph_against_anchors`.
    true_moa: length-n cluster labels (or None for unmapped rows, which are
    excluded from metrics but kept in the score table).
    """
    df = anchor_scores.copy()
    df.insert(0, "true_moa", list(true_moa))
    df["pred_moa"] = df[list(moa_list)].idxmax(axis=1)

    labelled = df[df["true_moa"].notna()].copy()
    y_true = labelled["true_moa"].to_numpy()

    # one-vs-rest AUROC + fold-enrichment of the true class at rank-1
    rows = []
    aurocs = []
    for moa in moa_list:
        pos = (y_true == moa).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            auc = np.nan
        else:
            auc = roc_auc_score(pos, labelled[moa].to_numpy())
            aurocs.append(auc)
        # enrichment: P(pred==moa | true==moa) / P(pred==moa)
        base_rate = (df["pred_moa"] == moa).mean()
        grp = labelled[labelled["true_moa"] == moa]
        hit_rate = (grp["pred_moa"] == moa).mean() if len(grp) else np.nan
        enr = (hit_rate / base_rate) if base_rate > 0 else np.nan
        rows.append(
            {
                "moa": moa,
                "n_compounds": int((y_true == moa).sum()),
                "ovr_auroc": auc,
                "rank1_hit_rate": hit_rate,
                "baseline_rate": base_rate,
                "fold_enrichment": enr,
            }
        )
    auroc_per_moa = pd.DataFrame(rows)

    # top-k MoA recovery enrichment across all labelled compounds
    ranked = np.argsort(-labelled[list(moa_list)].to_numpy(), axis=1)
    moa_arr = np.array(list(moa_list))
    topk_pred = moa_arr[ranked[:, :top_k]]
    in_topk = np.array(
        [y_true[i] in set(topk_pred[i]) for i in range(len(y_true))]
    )
    observed = in_topk.mean()
    expected = top_k / len(moa_list)
    topk_enr = observed / expected if expected > 0 else np.nan

    confusion = pd.crosstab(labelled["true_moa"], labelled["pred_moa"])
    confusion = confusion.reindex(index=list(moa_list), columns=list(moa_list), fill_value=0)

    accuracy = float((labelled["true_moa"] == labelled["pred_moa"]).mean())
    macro = float(np.nanmean(aurocs)) if aurocs else float("nan")

    return CrossLineResult(
        score_df=df,
        auroc_per_moa=auroc_per_moa,
        confusion=confusion,
        macro_auroc=macro,
        topk_enrichment=float(topk_enr),
        accuracy=accuracy,
    )


# ---------------------------------------------------------------------------
# Demonstration 2: customer-anchor few-shot retrieval within A549
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    per_moa: pd.DataFrame            # AUROC / precision@k per MoA
    macro_auroc: float
    mean_precision_at_k: float


def evaluate_fewshot_retrieval(
    morph_latent: np.ndarray,
    true_moa: Sequence[Optional[str]],
    moa_list: Sequence[str],
    n_anchor: int = 3,
    k: int = 10,
    n_repeats: int = 20,
    seed: int = 0,
) -> RetrievalResult:
    """Few-shot retrieval: a few A549 actives -> rank the rest by similarity.

    morph_latent: (n, d) ensemble-averaged A549 morphology embeddings.
    For each MoA and repeat, sample ``n_anchor`` actives as customer anchors,
    score every *other* compound by mean cosine to them, and measure whether
    same-MoA compounds rank at the top (one-vs-rest AUROC and precision@k).
    """
    rng = np.random.default_rng(seed)
    true = np.array([t if t is not None else "__none__" for t in true_moa])
    X = np.asarray(morph_latent, dtype=np.float64)
    n = len(true)

    rows = []
    for moa in moa_list:
        idx_pos = np.where(true == moa)[0]
        if len(idx_pos) <= n_anchor:
            rows.append({"moa": moa, "n_compounds": int(len(idx_pos)),
                         "auroc": np.nan, "precision_at_k": np.nan})
            continue
        aucs, precs = [], []
        for _ in range(n_repeats):
            anchors = rng.choice(idx_pos, size=n_anchor, replace=False)
            rest = np.setdiff1d(np.arange(n), anchors)
            sims = cosine_similarity(X[anchors], X[rest]).mean(axis=0)
            y = (true[rest] == moa).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            aucs.append(roc_auc_score(y, sims))
            topk = rest[np.argsort(-sims)[:k]]
            precs.append(float((true[topk] == moa).mean()))
        rows.append(
            {
                "moa": moa,
                "n_compounds": int(len(idx_pos)),
                "auroc": float(np.mean(aucs)) if aucs else np.nan,
                "precision_at_k": float(np.mean(precs)) if precs else np.nan,
            }
        )
    per_moa = pd.DataFrame(rows)
    macro = float(np.nanmean(per_moa["auroc"]))
    mprec = float(np.nanmean(per_moa["precision_at_k"]))
    return RetrievalResult(per_moa=per_moa, macro_auroc=macro, mean_precision_at_k=mprec)


def ensemble_average(emb: np.ndarray) -> np.ndarray:
    """Collapse (n_models, n, d) -> (n, d) by L2-normalising then averaging."""
    e = np.asarray(emb, dtype=np.float64)
    norm = np.linalg.norm(e, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return (e / norm).mean(axis=0)


def strip_score_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``<moa>_score`` columns (as returned by the real
    ``score_against_anchors``) to plain ``<moa>``."""
    return df.rename(columns={c: c[:-6] for c in df.columns if c.endswith("_score")})


# ---------------------------------------------------------------------------
# Structural independence (the "independent chemotype" claim)
# ---------------------------------------------------------------------------

def max_tanimoto_to_refs(
    query_smiles: Sequence[str],
    ref_smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Max ECFP Tanimoto of each query to any reference SMILES.

    Returns an array of length ``len(query_smiles)`` (NaN where RDKit is
    unavailable or a SMILES can't be parsed). A *low* value alongside a *high*
    anchor score is the evidence for a structurally independent chemotype.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:  # pragma: no cover
        return np.full(len(query_smiles), np.nan)

    def fp(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

    ref_fps = [f for f in (fp(s) for s in ref_smiles) if f is not None]
    out = np.full(len(query_smiles), np.nan)
    if not ref_fps:
        return out
    for i, s in enumerate(query_smiles):
        f = fp(s)
        if f is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, ref_fps))
    return out


def tanimoto_to_one(one_smiles: str, cand_smiles: Sequence[str],
                    radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP Tanimoto of each candidate to a single query SMILES (NaN if unavailable)."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:  # pragma: no cover
        return np.full(len(cand_smiles), np.nan)

    def fp(smi):
        m = Chem.MolFromSmiles(smi)
        return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)

    q = fp(one_smiles)
    out = np.full(len(cand_smiles), np.nan)
    if q is None:
        return out
    for i, s in enumerate(cand_smiles):
        f = fp(s)
        if f is not None:
            out[i] = DataStructs.TanimotoSimilarity(q, f)
    return out


def rank_by_similarity(query_latent: np.ndarray, lib_latent: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query embedding to each library embedding."""
    q = np.asarray(query_latent, dtype=np.float64).reshape(1, -1)
    return cosine_similarity(q, np.asarray(lib_latent, dtype=np.float64))[0]


def pairwise_tanimoto_matrix(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048):
    """Full pairwise ECFP Tanimoto matrix for a list of SMILES.

    Returns an ``(n, n)`` array (NaN where a SMILES can't be parsed / RDKit
    missing). Off-diagonal values near 0 = structurally diverse (independent
    chemotypes); near 1 = look-alikes.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:  # pragma: no cover
        return None

    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
    n = len(smiles)
    M = np.full((n, n), np.nan)
    for i in range(n):
        if fps[i] is None:
            continue
        M[i, i] = 1.0
        for j in range(i + 1, n):
            if fps[j] is None:
                continue
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            M[i, j] = M[j, i] = t
    return M
