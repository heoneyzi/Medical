"""Reference variant helpers.

Three reference variants for all reference-dependent metrics:
- Ref A: h_{29} raw, no RMSNorm anywhere (symmetric, gamma-free)
- Ref B: RMSNorm(h_ell) vs RMSNorm(h_{29}), DTR-style symmetric
- Ref C: h_ell raw vs h_norm (= RMSNorm(h_{30})), the existing gDTR baseline (asymmetric)

See docs/reference_variants.md for full definitions and the cell compatibility matrix.
"""

from .ref_a_nonorm import get_reference_a
from .ref_b_bothnorm import apply_rmsnorm, get_reference_b
from .ref_c_existing import get_reference_c

__all__ = [
    "get_reference_a",
    "get_reference_b",
    "get_reference_c",
    "apply_rmsnorm",
]
