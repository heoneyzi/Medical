"""Bridge the existing GDTR/EXP2 loader into the EXP3 adapter contract.

This module does not replace or modify EXP1 imports.  The caller constructs
the same ``exp1.modelio.Evo2Runner`` it already uses, then passes it to
``from_exp1_runner``.  All module paths are read from the existing ModelSpec;
ambiguous norm or MLP attributes fail loudly and can be pinned explicitly.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from torch import nn

from .vortex import VortexAdapter, _resolve


PRE_NORM_CANDIDATES = ("pre_norm", "norm1", "input_layernorm")
POST_NORM_CANDIDATES = ("post_norm", "norm2", "post_attention_layernorm")
MLP_WEIGHT_CANDIDATES = (
    ("l1", "l2", "l3"),
    ("w1", "w2", "w3"),
    ("gate_proj", "up_proj", "down_proj"),
)


def _relative_to_inner(path: str, inner: nn.Module) -> str:
    """EXP1 specs often say ``model.blocks`` while ``torch_model`` is .model."""
    if path.startswith("model.") and not hasattr(inner, "model"):
        return path[len("model."):]
    return path


def _unique_attr(module: nn.Module, candidates: Sequence[str], label: str) -> str:
    found = [name for name in candidates if hasattr(module, name)
             and isinstance(getattr(module, name), nn.Module)]
    if len(found) != 1:
        raise RuntimeError(
            f"could not uniquely resolve {label}: candidates present={found}; "
            f"pin the attribute explicitly"
        )
    return found[0]


def _weight_attrs(mlp: nn.Module) -> tuple[str, str, str]:
    found = [triple for triple in MLP_WEIGHT_CANDIDATES
             if all(hasattr(mlp, name) for name in triple)]
    if len(found) != 1:
        raise RuntimeError(
            f"could not uniquely resolve bilinear MLP weights: candidates={found}; "
            "pass mlp_weight_attrs explicitly"
        )
    return found[0]


def from_exp1_runner(
    exp1_runner,
    *,
    operator_types: Mapping[int, str] | Sequence[str] | None = None,
    runtime_config=None,
    operator_config_field: str | None = None,
    pre_norm_attr: str | None = None,
    post_norm_attr: str | None = None,
    mlp_weight_attrs: tuple[str, str, str] | None = None,
) -> VortexAdapter:
    """Return an EXP3 adapter around an already-loaded EXP1 runner."""
    required = ("torch_model", "spec", "anatomy")
    missing = [name for name in required if not hasattr(exp1_runner, name)]
    if missing:
        raise TypeError(f"not an EXP1 Evo2Runner; missing {missing}")
    root = exp1_runner.torch_model
    spec = exp1_runner.spec
    block_path = _relative_to_inner(spec.block_module_path, root)
    final_norm_path = _relative_to_inner(spec.norm_module_path, root)
    unembed_path = _relative_to_inner(spec.unembed_module_path, root)
    blocks = _resolve(root, block_path)
    if not len(blocks):
        raise RuntimeError("EXP1 block list is empty")
    block0 = blocks[0]
    mixer_attr = exp1_runner.anatomy.mixer_attr
    mlp_attr = exp1_runner.anatomy.mlp_attr
    if not mixer_attr or not mlp_attr:
        raise RuntimeError("EXP1 did not resolve mixer/mlp attributes")
    pre = pre_norm_attr or _unique_attr(block0, PRE_NORM_CANDIDATES, "pre-norm")
    post = post_norm_attr or _unique_attr(block0, POST_NORM_CANDIDATES, "post-norm")
    weights = mlp_weight_attrs or _weight_attrs(getattr(block0, mlp_attr))
    return VortexAdapter(
        root,
        block_path=block_path,
        mixer_attr=mixer_attr,
        mlp_attr=mlp_attr,
        pre_norm_attr=pre,
        post_norm_attr=post,
        mlp_weight_attrs=weights,
        final_norm_path=final_norm_path,
        unembed_path=unembed_path,
        operator_types=operator_types,
        runtime_config=runtime_config,
        operator_config_field=operator_config_field,
    )


def load_existing_exp1(model_key: str, *, device: str,
                       operator_types: Mapping[int, str] | Sequence[str],
                       **adapter_kwargs):
    """Convenience loader; requires the user's existing EXP1 on PYTHONPATH."""
    try:
        from exp1.config import load_models
        from exp1.modelio import Evo2Runner as Exp1Runner
    except ImportError as exc:
        raise RuntimeError(
            "EXP1 is not importable. Keep its current imports and add its repository "
            "root to PYTHONPATH before calling load_existing_exp1."
        ) from exc
    models = load_models()
    if model_key not in models:
        raise KeyError(f"unknown EXP1 model {model_key!r}; available={sorted(models)}")
    old = Exp1Runner(models[model_key], device=device)
    adapter = from_exp1_runner(old, operator_types=operator_types, **adapter_kwargs)
    return old, adapter
