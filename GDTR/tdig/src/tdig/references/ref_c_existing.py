"""Reference Variant C — current gDTR baseline.

reference = h_norm, Evo 2's post-final-norm output.
h_ell stays raw (no RMSNorm applied). Asymmetric — gamma is applied to the
reference only. This is the existing gDTR formulation; included here as the
baseline against which Variants A and B are compared.
"""

from __future__ import annotations


def get_reference_c(hidden_states):
    """Extract h_norm from the cache.

    Args:
        hidden_states: cache containing 'h_norm' (post-final-RMSNorm output).
            Equivalent to RMSNorm(h_{30}) since blocks 30 and 31 are
            saturated / idle in Evo 2 (see gDTR-PoC Phase 1).

    Returns:
        h_norm, shape [n_tokens, d_model].
    """
    raise NotImplementedError("S1 implementation")
