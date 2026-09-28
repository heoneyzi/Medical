"""Lightweight logit-lens consistency test using actual HyenaDNA forward.

Loads the locked HF revision once (module-scoped fixture). Tests:
- jsd_lens output shape and finiteness
- p_L from lens matches softmax of out.logits within tight tolerance
- top1_predictions returns ids in [0, vocab_real)
- cosine_lens is in [0, 2] and zero at final layer

This will use ~1.5 GB GPU. Skipped if no CUDA.
"""
import math
import pytest
import torch
import torch.nn.functional as F

from src.constants import LOG_VOCAB_REAL, VOCAB_REAL
from src.logit_lens import jsd_lens, top1_predictions
from src.model_loader import load_hyenadna, tokenize_sequence
from src.ur_gdtr import cosine_lens


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA required for logit_lens test")


@pytest.fixture(scope="module")
def bundle():
    return load_hyenadna(device="cuda", dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def forward(bundle):
    seq = "ACGT" * 25  # 100 bp
    input_ids, _ = tokenize_sequence(seq, bundle.tokenizer, device="cuda")
    with torch.no_grad():
        out = bundle.model(input_ids, output_hidden_states=True)
    return seq, input_ids, out


def test_hidden_states_layout(forward):
    _, _, out = forward
    # embed + 8 blocks + post-ln_f = 10
    assert len(out.hidden_states) == 10


def test_jsd_lens_shape_and_finiteness(bundle, forward):
    seq, _, out = forward
    D = jsd_lens(out.hidden_states, bundle.ln_f, bundle.lm_head)
    L = len(out.hidden_states) - 2
    assert D.shape == (L, len(seq))
    assert torch.isfinite(D).all()
    # Final layer self-distance is exactly zero
    assert torch.equal(D[L - 1], torch.zeros(len(seq), dtype=torch.float32))
    # JSD normalized into [0, 1] (may equal 1 in degenerate cases)
    assert (D >= 0).all() and (D <= 1.0 + 1e-4).all()


def test_final_layer_p_matches_out_logits(bundle, forward):
    """The lens at the final layer should reproduce the softmax of out.logits[..., :12]."""
    seq, _, out = forward
    final_logits_full = out.logits.float()
    final_logits = final_logits_full[..., :VOCAB_REAL]
    p_final_direct = F.softmax(final_logits, dim=-1)

    # post-ln_f hidden state -> lm_head -> mask -> softmax (cast to lm_head dtype)
    hL = out.hidden_states[-1].to(bundle.lm_head.weight.dtype)
    p_final_lens = F.softmax(bundle.lm_head(hL).float()[..., :VOCAB_REAL], dim=-1)

    diff = (p_final_direct - p_final_lens).abs().max().item()
    assert diff < 1e-3, f"final-layer probs diverge by {diff}"


def test_top1_predictions_in_range(bundle, forward):
    _, _, out = forward
    top1 = top1_predictions(out.hidden_states, bundle.ln_f, bundle.lm_head)
    assert top1.min() >= 0
    assert top1.max() < VOCAB_REAL


def test_cosine_lens_in_range(forward):
    _, _, out = forward
    Dc = cosine_lens(out.hidden_states)
    L = len(out.hidden_states) - 2
    assert Dc.shape[0] == L
    assert (Dc >= 0).all()
    # Final layer self-distance == 0
    assert (Dc[L - 1] == 0).all()


def test_log_vocab_real_constant():
    assert math.isclose(LOG_VOCAB_REAL, math.log(12), abs_tol=1e-9)
