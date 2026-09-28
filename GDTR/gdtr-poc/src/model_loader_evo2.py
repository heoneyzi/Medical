"""Evo 2 7B model loader for Phase 1 (lazy + cached, with TE fallback).

Returns an Evo2Bundle exposing the underlying StripedHyena (`sh`), the final
RMSNorm (`norm`), the unembedding callable (`unembed`), the storage-tied
embedding weight, and a tokenizer helper. Falls back to ``evo2_7b_base``
(8K context) when Transformer Engine FP8 ops are unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .constants_evo2 import MODEL_NAME, MODEL_NAME_FALLBACK

log = logging.getLogger(__name__)

# Lazy module cache
_BUNDLE: Optional["Evo2Bundle"] = None


@dataclass
class Evo2Bundle:
    """Loaded Evo 2 artefacts.

    Attributes:
        model: Evo2 wrapper exposing forward(input_ids, return_embeddings, layer_names).
        sh: vortex.model.model.StripedHyena (low-level model).
        tokenizer: CharLevelTokenizer (returns list[uint8]).
        norm: sh.norm — final RMSNorm before unembed.
        unembed: sh.unembed — Lambda wrapping VocabParallelEmbedding.unembed.
        embedding_weight: sh.embedding_layer.weight (storage-tied with unembed).
        loaded_variant: which Evo2 variant was actually loaded.
    """

    model: object
    sh: nn.Module
    tokenizer: object
    norm: nn.Module
    unembed: nn.Module
    embedding_weight: torch.Tensor
    loaded_variant: str


def _patch_safe_globals() -> None:
    """Vortex checkpoints contain `_codecs.encode` -> add to safe globals."""
    import _codecs
    import torch.serialization as _ts
    try:
        _ts.add_safe_globals([_codecs.encode])
    except Exception:
        pass


def load_evo2(force_reload: bool = False) -> Evo2Bundle:
    """Load Evo 2 7B (with FP8 fallback to base 8K). Cached at module level.

    Args:
        force_reload: drop the cache and re-instantiate.

    Returns:
        Evo2Bundle.
    """
    global _BUNDLE
    if _BUNDLE is not None and not force_reload:
        return _BUNDLE

    _patch_safe_globals()
    from evo2 import Evo2  # noqa: E402

    log.info("Loading Evo 2 (try %s, fallback %s) ...", MODEL_NAME, MODEL_NAME_FALLBACK)
    try:
        m = Evo2(MODEL_NAME)
        loaded = MODEL_NAME
    except ImportError as e:
        log.warning("Evo2(%s) failed (%s); falling back to %s.", MODEL_NAME, e, MODEL_NAME_FALLBACK)
        m = Evo2(MODEL_NAME_FALLBACK)
        loaded = f"{MODEL_NAME_FALLBACK} (8K context, no FP8)"

    sh = m.model
    tokenizer = m.tokenizer
    norm = sh.norm
    unembed = sh.unembed
    emb_w = sh.embedding_layer.weight

    _BUNDLE = Evo2Bundle(
        model=m, sh=sh, tokenizer=tokenizer, norm=norm,
        unembed=unembed, embedding_weight=emb_w, loaded_variant=loaded,
    )
    log.info("Evo 2 ready (%s).", loaded)
    return _BUNDLE


def tokenize(seq: str, bundle: Evo2Bundle, device: str = "cuda") -> torch.Tensor:
    """Tokenize a DNA string to a [1, T] int64 tensor (no BOS).

    Args:
        seq: DNA string (uppercase A/C/G/T/N preferred).
        bundle: Evo2Bundle instance.
        device: target device.

    Returns:
        torch.LongTensor of shape [1, T].
    """
    ids = bundle.tokenizer.tokenize(seq)
    arr = np.asarray(ids, dtype=np.int64)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def lm_head_apply(h: torch.Tensor, bundle: Evo2Bundle) -> torch.Tensor:
    """Apply final norm + unembed and slice to VOCAB_SIZE.

    Args:
        h: hidden state [B, T, H].
        bundle: Evo2Bundle.

    Returns:
        logits [B, T, VOCAB_SIZE] (bfloat16).
    """
    from .constants_evo2 import VOCAB_SIZE
    h_cast = h.to(bundle.embedding_weight.dtype)
    out = bundle.unembed(bundle.norm(h_cast))
    return out[..., :VOCAB_SIZE]


def free() -> None:
    """Drop the cache and free GPU memory."""
    global _BUNDLE
    _BUNDLE = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
