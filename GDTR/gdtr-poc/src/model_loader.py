"""HyenaDNA model loader (Phase 0, locked HF revision).

Implements the Appendix C reference forward-pass code path:
- transformers AutoModelForCausalLM with revision pin and trust_remote_code=True
- Returns the locatable final norm and lm_head modules so that lens code does
  not need to import internal HF paths.

The function returns a NamedTuple-like dataclass to make downstream code
explicit about which artefact it consumes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .constants import BOS_OFFSET, HF_REVISION, MODEL_ID, VOCAB_REAL

log = logging.getLogger(__name__)


@dataclass
class HyenaDNABundle:
    """Loaded HyenaDNA artefacts.

    Attributes:
        model: HyenaDNAForCausalLM in eval mode on `device` with `dtype`.
        tokenizer: HyenaDNATokenizer (character-level, BOS auto-prepended).
        ln_f: model.hyena.backbone.ln_f — final LayerNorm before lm_head.
        lm_head: model.lm_head — Linear(256 -> 16); mask logits[..., :12].
        vocab_real: 12 (real vocabulary size; lm_head out is padded to 16).
    """

    model: nn.Module
    tokenizer: object
    ln_f: nn.Module
    lm_head: nn.Module
    vocab_real: int


def load_hyenadna(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HyenaDNABundle:
    """Load HyenaDNA-medium-160k at locked HF revision.

    Args:
        device: torch device string.
        dtype:  parameter dtype (bfloat16 fits comfortably in 24 GB).

    Returns:
        HyenaDNABundle with model, tokenizer, ln_f, lm_head, vocab_real.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer revision=%s", HF_REVISION[:8])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=HF_REVISION, trust_remote_code=True
    )

    log.info("Loading model %s dtype=%s device=%s", MODEL_ID, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=HF_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    ln_f = model.hyena.backbone.ln_f
    lm_head = model.lm_head

    # Sanity: vocabulary not tied (Appendix C.1) and out_features=16 (C.2)
    assert lm_head.out_features == 16, (
        f"lm_head.out_features expected 16, got {lm_head.out_features}"
    )
    log.info(
        "loaded n_layer=%d hidden=%d vocab_real=%d (lm_head=%d)",
        len(model.hyena.backbone.layers),
        model.hyena.backbone.embeddings.word_embeddings.embedding_dim,
        VOCAB_REAL,
        lm_head.out_features,
    )
    return HyenaDNABundle(
        model=model,
        tokenizer=tokenizer,
        ln_f=ln_f,
        lm_head=lm_head,
        vocab_real=VOCAB_REAL,
    )


def tokenize_sequence(
    seq: str,
    tokenizer,
    device: str = "cuda",
) -> Tuple[torch.Tensor, int]:
    """Tokenize a nucleotide string for HyenaDNA.

    The HuggingFace HyenaDNA tokenizer auto-prepends BOS (id=2) and does NOT
    append EOS. Output shape is [1, len(seq)+1].

    Args:
        seq: nucleotide sequence, e.g. "ACGTACGT...".
        tokenizer: HyenaDNATokenizer instance.
        device: torch device for returned tensor.

    Returns:
        (input_ids of shape [1, T+1], bos_offset=1).
    """
    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] != len(seq) + BOS_OFFSET:
        log.warning(
            "tokenizer produced T=%d for seq_len=%d (expected %d)",
            input_ids.shape[1], len(seq), len(seq) + BOS_OFFSET,
        )
    return input_ids, BOS_OFFSET
