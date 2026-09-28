"""Reference Variant A — no-norm.

reference = h_{29} raw (the output of Evo 2 block 29, before any RMSNorm).
h_ell stays raw too. Fully symmetric, no gamma anywhere.
"""

from __future__ import annotations


def get_reference_a(hidden_states):
    """Extract h_{29} raw from the hidden-state cache.

    Args:
        hidden_states: dict-like or array indexed by layer; layer 29 must be present.

    Returns:
        h_{29} as a tensor of shape [n_tokens, d_model].

    Notes:
        Evo 2's canonical interpretively-distinct tap is L* = 29 (see
        gDTR-PoC Phase 1, L31-idle finding). Variant A uses this raw vector
        directly as reference for all reference-dependent metrics.
    """
    raise NotImplementedError("S1 implementation")
