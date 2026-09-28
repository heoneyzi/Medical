"""Adapter template for the EXP1 environment's Evo 2 / Vortex loader.

This is the ONLY file you edit to plug exp2 into EXP1. Fill in the five
resolution helpers at the top; everything else is generic.

Nothing here guesses the architecture. `read_architecture` walks the LOADED
model and reads block types, attention indices, the RMSNorm form and the MLP
expression off the runtime modules -- Step 0 forbids inferring the 7B map
from the 1B pattern, and the Step 1 gate will fail loudly if this file wires
a hook to the wrong module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from ..manifest import ArchitectureManifest, BlockType, sha256_path


# --------------------------------------------------------------------------
# EDIT THESE FIVE
# --------------------------------------------------------------------------

BLOCK_PATH = "backbone.blocks"          # module list of transformer/hyena blocks
MIXER_ATTR = "mixer"                    # attr on a block giving the mixer/attention
MLP_ATTR = "mlp"                        # attr on a block giving the MLP
PRE_NORM_ATTR = "pre_norm"              # RMSNorm applied to x_l before the mixer
POST_NORM_ATTR = "post_norm"            # RMSNorm applied to r_l before the MLP

MLP_W1_ATTR, MLP_W2_ATTR, MLP_W3_ATTR = "l1", "l2", "l3"
FINAL_NORM_PATH = "backbone.norm"
UNEMBED_PATH = "backbone.unembed"


def _resolve(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class VortexAdapter:
    def __init__(
        self,
        model: nn.Module,
        *,
        rms_eps: float | None = None,
        operator_types: Mapping[int, str] | Sequence[str] | None = None,
        operator_classifier: Callable[[int, nn.Module, Any], str] | None = None,
        operator_class_map: Mapping[type | str, str] | None = None,
        runtime_config: Any = None,
        operator_config_field: str | None = None,
    ):
        """Wrap a loaded Evo/Vortex model without guessing its block map.

        Operator labels must be supplied by, or verifiably read from, the
        actual runtime.  Accepted evidence is an explicit per-layer sequence,
        a callback, an exact class map, a declared module attribute, or an
        explicit field in the loaded config.  Class-name substring heuristics
        are deliberately not part of this contract.
        """
        self.model = model
        self._blocks = _resolve(model, BLOCK_PATH)
        self._rms_eps = rms_eps
        self._operator_types = operator_types
        self._operator_classifier = operator_classifier
        self._operator_class_map = dict(operator_class_map or {})
        self._runtime_config = runtime_config
        self._operator_config_field = operator_config_field

    # ---- structure -----------------------------------------------------

    def n_layers(self) -> int:
        return len(self._blocks)

    def d_model(self) -> int:
        return int(self.final_gamma().shape[-1])

    def vocab_size(self) -> int:
        return int(self.unembedding().shape[0])

    def block(self, ell: int) -> nn.Module:
        return self._blocks[ell]

    def mixer(self, ell: int) -> nn.Module:
        return getattr(self.block(ell), MIXER_ATTR)

    def mlp(self, ell: int) -> nn.Module:
        return getattr(self.block(ell), MLP_ATTR)

    def pre_norm(self, ell: int) -> nn.Module:
        return getattr(self.block(ell), PRE_NORM_ATTR)

    def post_norm(self, ell: int) -> nn.Module:
        return getattr(self.block(ell), POST_NORM_ATTR)

    def mlp_weights(self, ell: int) -> tuple[Tensor, Tensor, Tensor]:
        m = self.mlp(ell)
        def W(attr: str) -> Tensor:
            sub = getattr(m, attr)
            return sub.weight if hasattr(sub, "weight") else sub
        return W(MLP_W1_ATTR), W(MLP_W2_ATTR), W(MLP_W3_ATTR)

    def hcl_submodules(self, ell: int) -> dict[str, nn.Module]:
        """Return the runtime-named internal stages of an HCL mixer.

        Discover the real names once with `dict(self.mixer(ell).named_children())`
        and record them in the manifest. `step8_hcl.reconstruct_reference_path`
        asserts that composing them reproduces m_l, so a wrong mapping fails
        rather than mislabelling a stage.
        """
        return dict(self.mixer(ell).named_children())

    def final_norm(self) -> nn.Module:
        return _resolve(self.model, FINAL_NORM_PATH)

    def final_gamma(self) -> Tensor:
        n = self.final_norm()
        for attr in ("weight", "scale", "gamma", "g"):
            if hasattr(n, attr):
                return getattr(n, attr).detach()
        raise AttributeError("final RMSNorm has no recognisable gain parameter")

    def unembedding(self) -> Tensor:
        u = _resolve(self.model, UNEMBED_PATH)
        if isinstance(u, Tensor):
            return u.detach()
        for attr in ("weight", "unembed", "proj"):
            if hasattr(u, attr):
                w = getattr(u, attr)
                return (w.weight if hasattr(w, "weight") else w).detach()
        raise AttributeError("cannot locate the unembedding matrix")

    def unembedding_bias(self) -> Optional[Tensor]:
        u = _resolve(self.model, UNEMBED_PATH)
        b = getattr(u, "bias", None)
        return None if b is None else b.detach()

    # ---- forward -------------------------------------------------------

    @torch.no_grad()
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        out = self.model(input_ids)
        if isinstance(out, Tensor):
            return out
        for attr in ("logits", "last_hidden_state"):
            if hasattr(out, attr):
                return getattr(out, attr)
        if isinstance(out, (tuple, list)):
            return out[0]
        raise TypeError(f"cannot find logits in {type(out)}")

    # ---- architecture manifest ----------------------------------------

    @staticmethod
    def _operator_label(value: Any, *, source: str, ell: int) -> BlockType:
        label = str(value).strip().lower()
        aliases = {
            "attention": "attn",
            "mha": "attn",
            "short": "hcs",
            "medium": "hcm",
            "long": "hcl",
        }
        label = aliases.get(label, label)
        if label not in {"hcs", "hcm", "hcl", "attn"}:
            raise RuntimeError(
                f"block {ell}: {source} declared unsupported operator {value!r}; "
                "expected one of hcs/hcm/hcl/attn"
            )
        return label  # type: ignore[return-value]

    @staticmethod
    def _get_field(obj: Any, field: str) -> Any:
        cur = obj
        for part in field.split("."):
            if isinstance(cur, Mapping):
                if part not in cur:
                    raise KeyError(field)
                cur = cur[part]
            else:
                if not hasattr(cur, part):
                    raise KeyError(field)
                cur = getattr(cur, part)
        return cur

    def _config_operator_types(self) -> tuple[Any, str] | tuple[None, None]:
        if self._runtime_config is None:
            return None, None
        fields = ([self._operator_config_field] if self._operator_config_field else
                  ["operator_types", "block_types", "mixer_types"])
        found: list[tuple[Any, str]] = []
        for field in fields:
            if field is None:
                continue
            try:
                found.append((self._get_field(self._runtime_config, field), field))
            except KeyError:
                continue
        if len(found) > 1:
            values = [list(v) if not isinstance(v, Mapping) else dict(v) for v, _ in found]
            if any(v != values[0] for v in values[1:]):
                raise RuntimeError(
                    f"runtime config contains conflicting operator maps in "
                    f"{[field for _, field in found]}"
                )
        return found[0] if found else (None, None)

    @staticmethod
    def _per_layer(value: Mapping[int, Any] | Sequence[Any], ell: int, source: str) -> Any:
        if isinstance(value, Mapping):
            if ell in value:
                return value[ell]
            if str(ell) in value:
                return value[str(ell)]
            raise RuntimeError(f"{source} has no entry for block {ell}")
        if isinstance(value, (str, bytes)) or ell >= len(value):
            raise RuntimeError(f"{source} has no entry for block {ell}")
        return value[ell]

    def _classify_mixer(self, ell: int) -> tuple[BlockType, str]:
        mixer = self.mixer(ell)
        evidence: list[tuple[str, BlockType]] = []

        if self._operator_types is not None:
            value = self._per_layer(self._operator_types, ell, "operator_types")
            evidence.append(("explicit", self._operator_label(value, source="operator_types", ell=ell)))

        config_types, config_field = self._config_operator_types()
        if config_types is not None:
            value = self._per_layer(config_types, ell, f"runtime_config.{config_field}")
            evidence.append((
                f"config.{config_field}",
                self._operator_label(value, source=f"runtime_config.{config_field}", ell=ell),
            ))

        if self._operator_classifier is not None:
            value = self._operator_classifier(ell, mixer, self._runtime_config)
            evidence.append(("callback", self._operator_label(value, source="operator_classifier", ell=ell)))

        # Exact runtime declarations are evidence; no substring matching.
        for attr in ("operator_type", "block_type", "mixer_type", "layer_type"):
            if hasattr(mixer, attr):
                value = getattr(mixer, attr)
                if value is not None:
                    evidence.append((
                        f"module.{attr}",
                        self._operator_label(value, source=f"mixer.{attr}", ell=ell),
                    ))

        fqcn = f"{type(mixer).__module__}.{type(mixer).__qualname__}"
        for key in (type(mixer), fqcn):
            if key in self._operator_class_map:
                evidence.append((
                    f"class_map:{fqcn}",
                    self._operator_label(
                        self._operator_class_map[key], source="operator_class_map", ell=ell
                    ),
                ))

        if not evidence:
            raise RuntimeError(
                f"block {ell}: no runtime/config operator classification for {fqcn}. "
                "Pass operator_types, operator_classifier, operator_class_map, "
                "or the loaded runtime_config; class-name guessing is forbidden."
            )
        labels = {label for _, label in evidence}
        if len(labels) != 1:
            raise RuntimeError(f"block {ell}: conflicting operator evidence {evidence}")
        label = evidence[0][1]
        return label, ";".join(f"{source}={value}" for source, value in evidence)

    def _config_sha256(self) -> str:
        if self._runtime_config is None:
            return ""
        try:
            if hasattr(self._runtime_config, "to_dict"):
                value = self._runtime_config.to_dict()
            elif hasattr(self._runtime_config, "__dict__"):
                value = vars(self._runtime_config)
            else:
                value = self._runtime_config
            payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        except Exception as exc:
            raise RuntimeError("cannot deterministically fingerprint runtime_config") from exc
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def read_architecture(
        self,
        checkpoint: str,
        *,
        probe_ids: Tensor,
        checkpoint_sha256: str | None = None,
        checkpoint_path: str | Path | None = None,
        vortex_commit: str = "",
    ) -> ArchitectureManifest:
        n = self.n_layers()
        types, is_bilinear, acts = [], [], []
        attn_idx = []
        operator_evidence: dict[int, str] = {}
        for l in range(n):
            t, evidence = self._classify_mixer(l)
            operator_evidence[l] = evidence
            if t == "attn":
                attn_idx.append(l)
            types.append(t)
            m = self.mlp(l)
            act = getattr(m, "activation", None) or getattr(m, "act", None)
            acts.append(type(act).__name__ if act is not None else "identity")
            try:
                W1, W2, W3 = self.mlp_weights(l)
                is_bilinear.append(acts[-1].lower() in ("identity", "nonetype", "none"))
            except Exception:
                is_bilinear.append(False)

        rms_form, eps = self._detect_rms(probe_ids)
        try:
            runtime_dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except StopIteration:
            runtime_dtype = str(self.final_gamma().dtype).removeprefix("torch.")
        observed_checkpoint_sha256 = checkpoint_sha256 or ""
        if checkpoint_path is not None:
            observed = sha256_path(checkpoint_path)
            if checkpoint_sha256 is not None and observed != checkpoint_sha256:
                raise RuntimeError(
                    f"provided checkpoint SHA-256 {checkpoint_sha256} does not match "
                    f"{checkpoint_path}: {observed}"
                )
            observed_checkpoint_sha256 = observed
        return ArchitectureManifest(
            checkpoint=checkpoint,
            checkpoint_sha256=observed_checkpoint_sha256,
            vortex_commit=vortex_commit,
            n_layers=n, d_model=self.d_model(), vocab_size=self.vocab_size(),
            attn_layer_idxs=tuple(attn_idx), block_types=tuple(types),
            rms_form=rms_form, rms_eps=eps,
            has_unembed_bias=self.unembedding_bias() is not None,
            mlp_is_bilinear=tuple(is_bilinear), mlp_activation=tuple(acts),
            hcl_stage_names={l: list(self.hcl_submodules(l).keys())
                             for l in range(n) if types[l] == "hcl"},
            # Legacy h_l is the OUTPUT of block l, hence x_{l+1}.
            legacy_alias={f"h{l}": f"x{l + 1}" for l in range(n)},
            core_blocks=(28, 29, 30, 31) if n > 31 else tuple(range(max(0, n - 4), n)),
            homologous_pairs=((25, 28), (26, 29), (27, 30)) if n > 31 else (),
            dtype=runtime_dtype,
            config_sha256=self._config_sha256(),
            operator_evidence=operator_evidence,
        )

    @torch.no_grad()
    def validate_architecture(
        self,
        frozen: ArchitectureManifest,
        *,
        probe_ids: Tensor,
        checkpoint_path: str | Path | None = None,
        vortex_commit: str | None = None,
    ) -> ArchitectureManifest:
        """Re-read the loaded graph and compare it with a frozen manifest."""
        observed = self.read_architecture(
            frozen.checkpoint,
            probe_ids=probe_ids,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=frozen.checkpoint_sha256 or None,
            vortex_commit=vortex_commit or frozen.vortex_commit,
        )
        frozen.assert_runtime_match(observed)
        frozen.validate_provenance(
            checkpoint_path=checkpoint_path,
            vortex_commit=vortex_commit,
            require_pinned=checkpoint_path is not None or vortex_commit is not None,
        )
        return observed

    @torch.no_grad()
    def _detect_rms(self, probe_ids: Tensor) -> tuple[str, float]:
        """Read the RMSNorm form off the runtime tensors, as Step 1 requires.

        The two forms give different s' under a coordinate change and only one
        makes the A/B identity exact, so this is decided by measurement.
        """
        n = self.final_norm()
        eps = float(self._rms_eps if self._rms_eps is not None
                    else getattr(n, "eps", getattr(n, "variance_epsilon", 1e-6)))
        gamma_runtime = self.final_gamma()
        x = torch.randn(
            1, 8, self.d_model(), dtype=torch.float64, device=gamma_runtime.device
        )
        gamma = gamma_runtime.double()
        y = n(x.to(gamma_runtime.dtype)).double()
        d = x.shape[-1]
        cand = {
            "norm_over_sqrtd_plus_eps": x.norm(dim=-1, keepdim=True) / d ** 0.5 + eps,
            "sqrt_mean_sq_plus_eps": torch.sqrt((x * x).mean(-1, keepdim=True) + eps),
        }
        best, err = None, float("inf")
        for name, s in cand.items():
            e = float((gamma * x / s - y).norm() / y.norm().clamp_min(1e-30))
            if e < err:
                best, err = name, e
        if err > 1e-3:
            raise RuntimeError(
                f"neither RMSNorm form reproduces the runtime output (best rel err "
                f"{err:.2e}); read the actual expression out of the runtime code and "
                f"extend endpoints.rms_denominator before continuing"
            )
        return best, eps
