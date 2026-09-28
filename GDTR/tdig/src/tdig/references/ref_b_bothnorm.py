"""Reference Variant B — both-norm (DTR-style).

Apply the model's final RMSNorm (gamma vector + epsilon) to every layer's
hidden state, then compare normalized vectors symmetrically.

RMSNorm(h) = gamma_rms * h / sqrt(mean(h**2) + eps_rms)

This puts h_ell and h_{29} in the same post-normalization space and is the
closest analogue to the original NLP DTR pipeline (which projects every
layer through the same lm_head).
"""

from __future__ import annotations


def apply_rmsnorm(h, gamma_rms, eps_rms):
    """Apply Evo 2's final RMSNorm to an arbitrary hidden-state vector.

    Args:
        h: tensor of shape [..., d_model].
        gamma_rms: learned per-dimension scale, shape [d_model].
        eps_rms: epsilon scalar from the trained model.

    Returns:
        RMSNorm(h), same shape as h.
    """
    raise NotImplementedError("S1 implementation")


def get_reference_b(hidden_states, gamma_rms, eps_rms):
    """Return RMSNorm(h_{29}) as Variant B reference.

    Args:
        hidden_states: cache indexed by layer; layer 29 must be present.
        gamma_rms: Evo 2 final RMSNorm gamma.
        eps_rms: Evo 2 final RMSNorm epsilon.

    Returns:
        RMSNorm(h_{29}), shape [n_tokens, d_model].

    Note:
        Callers comparing h_ell to this reference must also apply
        apply_rmsnorm() to h_ell — Variant B is symmetric.
    """
    raise NotImplementedError("S1 implementation")
