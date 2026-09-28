"""Canonical tap names — the plan's §"block과 hook의 고정 표기".

    x_l   = block input
    m_l   = Mixer_l(RMSNorm_pre(x_l))          mixer / attention residual update
    r_l   = x_l + m_l                          post-mixer, pre-MLP residual
    g_l   = MLP_l(RMSNorm_post(r_l))           MLP residual update
    x_l+1 = r_l + g_l                          block output

Legacy h27/h28/... names are ALIASES ONLY and are resolved through the
architecture manifest exactly once. New result tables must never carry a
bare `h28`.
"""
from __future__ import annotations


def x(ell: int) -> str:
    return f"x{ell}"


def m(ell: int) -> str:
    return f"m{ell}"


def r(ell: int) -> str:
    return f"r{ell}"


def g(ell: int) -> str:
    return f"g{ell}"


def mlp_part(ell: int, part: str) -> str:
    """part in {'z','a','b','p','out'} — the exact bilinear accounting of Step 7-1."""
    assert part in {"z", "a", "b", "p", "out"}, part
    return f"mlp{ell}.{part}"


def hcl_stage(ell: int, stage: str) -> str:
    """HCL internal taps of Step 8-1.

    Stage names are placeholders until the runtime graph is read; the
    manifest records the *runtime* names and `taps.py` asserts the
    reconstruction identity, so a mismatch fails loudly instead of
    silently mislabelling a stage.
    """
    return f"hcl{ell}.{stage}"


HCL_STAGES = (
    "in_proj_x2",
    "in_proj_x1",
    "in_proj_v",
    "fir_x2",
    "fir_x1",
    "fir_v",
    "pregate_q",       # q = x1 * v
    "long_conv",       # K_long * q
    "direct",          # D * q
    "path_sum",        # long_conv + direct
    "postgate",        # x2 * path_sum
    "out_proj",        # == m_l
)

FINAL_PRE_NORM = "x_final"     # state entering the final RMSNorm
FINAL_LOGITS = "logits"


def legacy_alias(name: str, manifest) -> str:
    """Resolve a legacy h-name once, through the manifest. Never guess."""
    try:
        return manifest.legacy_alias[name]
    except KeyError as e:  # pragma: no cover
        raise KeyError(
            f"legacy tap {name!r} has no manifest alias; add it to "
            f"architecture_manifest.legacy_alias before use"
        ) from e
