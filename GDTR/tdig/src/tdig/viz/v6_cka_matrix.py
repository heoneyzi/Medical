"""V6 — CKA layer x layer matrix. Diagnostic for M3 c_geo's skip-bias control.

Tokens where c_geo declares 'settled at ell' should show high CKA between
layer ell and layers ell+1..L-1. Mismatches flag skip-connection bias.
"""

from __future__ import annotations


def cka_matrix(hidden_states):
    """L x L Centered Kernel Alignment matrix across layers."""
    raise NotImplementedError("S5 implementation")


def overlay_c_geo(cka, c_geo, savepath=None):
    """Render CKA with c_geo overlay markers."""
    raise NotImplementedError("S5 implementation")
