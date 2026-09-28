"""Tap + intervention layer. This is the ONLY place that touches the model.

Two things make the rest of the program safe:

1. **Reconstruction identities.** Every captured set must satisfy
   x_l + m_l = r_l and r_l + g_l = x_{l+1}, and the bilinear accounting must
   satisfy g_l = sum_k W3[:,k] p_k. If the adapter is wired to the wrong
   module, these fail loudly instead of producing a plausible wrong number.

2. **Update freeze, not state clamp** (§2.2). Freezing means replacing the
   residual UPDATE tensor u_k with the paired baseline update while the
   perturbed stream s^F_k keeps flowing:

       s^F_{k+1} = s^F_k + u_k(s^{alpha=1}_k)

   Overwriting the post-block state instead would erase the upstream
   perturbation, which is the third defect the plan calls out.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

import torch
from torch import Tensor, nn

from . import naming as N
from .manifest import ArchitectureManifest


# --------------------------------------------------------------------------
# adapter contract
# --------------------------------------------------------------------------


class ModelAdapter(Protocol):
    """Implement this once against the EXP1 environment's Evo 2 loader.

    Nothing else in exp2/ imports vortex or evo2 directly.
    """

    model: nn.Module

    def n_layers(self) -> int: ...
    def d_model(self) -> int: ...
    def vocab_size(self) -> int: ...

    def block(self, ell: int) -> nn.Module: ...
    def mixer(self, ell: int) -> nn.Module: ...
    def mlp(self, ell: int) -> nn.Module: ...
    def pre_norm(self, ell: int) -> nn.Module: ...
    def post_norm(self, ell: int) -> nn.Module: ...

    def mlp_weights(self, ell: int) -> tuple[Tensor, Tensor, Tensor]:
        """(W1, W2, W3) of g = W3[(W1 z) * (W2 z)]. Raise if not bilinear."""

    def hcl_submodules(self, ell: int) -> dict[str, nn.Module]:
        """Runtime-named internal stages of an HCL mixer (Step 8-1)."""

    def final_norm(self) -> nn.Module: ...
    def final_gamma(self) -> Tensor: ...
    def unembedding(self) -> Tensor: ...          # [V, d]
    def unembedding_bias(self) -> Optional[Tensor]: ...

    def forward_logits(self, input_ids: Tensor) -> Tensor: ...

    def read_architecture(self, checkpoint: str, **kwargs) -> ArchitectureManifest:
        """Read block types / attn idxs / rms form OFF THE LOADED MODEL.

        Step 0 forbids inferring the 7B map from the 1B pattern.
        """


def _first_tensor(out):
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, Tensor):
                return o
    raise TypeError(f"cannot find a tensor in module output of type {type(out)}")


def _rewrap(out, new: Tensor):
    if isinstance(out, Tensor):
        return new
    if isinstance(out, tuple):
        rest = list(out)
        for i, o in enumerate(rest):
            if isinstance(o, Tensor):
                rest[i] = new
                return tuple(rest)
    if isinstance(out, list):
        rest = list(out)
        for i, o in enumerate(rest):
            if isinstance(o, Tensor):
                rest[i] = new
                return rest
    raise TypeError(type(out))


def _checked_edit_result(edit: "Edit", before: Tensor, after) -> Tensor:
    """Fail at the intervention site instead of much later in the model.

    A silently broadcast or CPU-returning edit is especially dangerous here:
    the forward can remain numerically plausible while no longer representing
    the declared causal intervention.
    """
    if not isinstance(after, Tensor):
        raise TypeError(
            f"edit {edit.tap!r}/{edit.kind!r} returned {type(after).__name__}, "
            "expected torch.Tensor"
        )
    if after.shape != before.shape:
        raise ValueError(
            f"edit {edit.tap!r}/{edit.kind!r} changed shape "
            f"{tuple(before.shape)} -> {tuple(after.shape)}"
        )
    if after.device != before.device:
        raise ValueError(
            f"edit {edit.tap!r}/{edit.kind!r} moved the tensor from "
            f"{before.device} to {after.device}"
        )
    if after.dtype != before.dtype:
        raise ValueError(
            f"edit {edit.tap!r}/{edit.kind!r} changed dtype "
            f"{before.dtype} -> {after.dtype}; cast explicitly in the edit"
        )
    return after


# --------------------------------------------------------------------------
# edits
# --------------------------------------------------------------------------


@dataclass
class Edit:
    """One intervention, applied at one named tap.

    kind:
      "state"  -- replace the residual-stream state at a block boundary
                  (x_l or r_l); the suffix then recomputes freely.
      "update" -- replace the residual UPDATE produced by a module
                  (m_l or g_l). alpha-scaling and update-freeze are both this.
      "component" -- replace an additive component inside a mixer
                  (HCL stage tensor a_s), preserving everything else.
    """
    tap: str
    kind: str
    fn: Callable[[Tensor], Tensor]
    note: str = ""


def scale_update(alpha: float) -> Callable[[Tensor], Tensor]:
    """Appendix B's intervention: x_{29} = r_{28} + alpha * g_{28}."""
    return lambda t: t * alpha


def replace_with(cached: Tensor) -> Callable[[Tensor], Tensor]:
    """Update freeze: substitute the paired baseline update tensor."""
    return lambda t: cached.to(t.dtype).to(t.device).expand_as(t).clone() if cached.shape != t.shape else cached.to(t.dtype).to(t.device)


def project_out(P: Tensor) -> Callable[[Tensor], Tensor]:
    """a -> (I - P) a, with P an orthogonal projector frozen in discovery.

    Step 8-2: this preserves the complement. Zero-ablating a whole serial tap
    would cut every downstream signal and make t_need trivially the first tap.
    """
    def f(t: Tensor) -> Tensor:
        Pm = P.to(t.dtype).to(t.device)
        return t - t @ Pm.T @ Pm if t.dim() >= 2 else t - Pm.T @ (Pm @ t)
    return f


def set_coordinate(j: int, value_fn: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    def f(t: Tensor) -> Tensor:
        out = t.clone()
        out[..., j] = value_fn(t[..., j])
        return out
    return f


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


@dataclass
class TapSet:
    tensors: dict[str, Tensor] = field(default_factory=dict)
    logits: Optional[Tensor] = None
    # Independent algebraic reconstructions.  These must never overwrite the
    # tensor actually observed at the next block boundary.
    reconstructed: dict[str, Tensor] = field(default_factory=dict)
    edit_application_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __getitem__(self, k: str) -> Tensor:
        return self.tensors[k]

    def get(self, k: str, default=None):
        return self.tensors.get(k, default)

    def get_reconstructed(self, k: str, default=None):
        return self.reconstructed.get(k, default)


class Evo2Runner:
    def __init__(self, adapter: ModelAdapter, arch: ArchitectureManifest):
        self.a = adapter
        self.arch = arch
        self._handles: list = []
        self._capture: dict[str, Tensor] = {}
        self._edits: dict[tuple[str, str], Edit] = {}
        self._want: set[str] = set()
        self._requested_to_canonical: dict[str, str] = {}
        self._edit_application_counts: dict[tuple[str, str], int] = {}
        self._current_x: dict[int, Tensor] = {}

    # ---- contract / validation ---------------------------------------

    def _canonical_tap(self, name: str) -> str:
        """Resolve legacy names and the two names for the final state.

        `x_{n_layers}` is the actual state entering the final RMSNorm.  The
        historical `x_final` spelling remains a public alias so existing Step
        code and in-flight runs do not need import or call-site changes.
        """
        name = self.arch.legacy_alias.get(name, name)
        if name == N.FINAL_PRE_NORM:
            return N.x(self.arch.n_layers)
        return name

    def _valid_taps(self) -> dict[str, Optional[str]]:
        """Canonical tap -> allowed edit kind (None means capture-only)."""
        valid: dict[str, Optional[str]] = {
            N.x(l): "state" for l in range(self.arch.n_layers + 1)
        }
        for l in range(self.arch.n_layers):
            valid[N.m(l)] = "update"
            valid[N.r(l)] = "state"
            valid[N.g(l)] = "update"
            valid[N.mlp_part(l, "z")] = None
        for l, stages in self.arch.hcl_stage_names.items():
            for stage in stages:
                valid[N.hcl_stage(int(l), stage)] = "component"
        return valid

    def _prepare_contract(
        self, taps: Iterable[str], edits: Iterable[Edit]
    ) -> tuple[set[str], dict[tuple[str, str], Edit]]:
        valid = self._valid_taps()
        requested = list(taps)
        mapping = {name: self._canonical_tap(name) for name in requested}
        unknown_taps = sorted({canon for canon in mapping.values() if canon not in valid})
        if unknown_taps:
            raise KeyError(f"unknown/unhookable taps: {unknown_taps}")

        prepared: dict[tuple[str, str], Edit] = {}
        for edit in edits:
            canon = self._canonical_tap(edit.tap)
            if canon not in valid:
                raise KeyError(
                    f"unknown/unhookable edit tap {edit.tap!r} "
                    f"(resolved to {canon!r})"
                )
            allowed = valid[canon]
            if edit.kind != allowed:
                if allowed is None:
                    raise ValueError(f"tap {edit.tap!r} is capture-only")
                raise ValueError(
                    f"edit {edit.tap!r} has kind {edit.kind!r}; "
                    f"the runtime contract requires {allowed!r}"
                )
            key = (canon, edit.kind)
            if key in prepared:
                raise ValueError(
                    f"multiple edits resolve to the same intervention site {key}; "
                    "compose them explicitly in one Edit.fn"
                )
            prepared[key] = Edit(canon, edit.kind, edit.fn, edit.note)
        self._requested_to_canonical = mapping
        return set(mapping.values()), prepared

    def _apply_edit(self, key: tuple[str, str], t: Tensor) -> Tensor:
        edit = self._edits.get(key)
        if edit is None:
            return t
        out = _checked_edit_result(edit, t, edit.fn(t))
        self._edit_application_counts[key] = self._edit_application_counts.get(key, 0) + 1
        return out

    def _validate_observation_contract(self) -> None:
        missing_taps = sorted(self._want - set(self._capture))
        if missing_taps:
            raise RuntimeError(
                "declared taps were never observed; adapter hook is absent or wired "
                f"outside the forward graph: {missing_taps}"
            )
        unapplied = [
            key for key, count in self._edit_application_counts.items() if count == 0
        ]
        if unapplied:
            raise RuntimeError(
                "declared edits were never applied; adapter hook is absent or wired "
                f"to a module outside the forward graph: {unapplied}"
            )

    # ---- capture -------------------------------------------------------

    def _hook_mixer(self, ell: int):
        def hook(_mod, inputs, out):
            t = _first_tensor(out)
            t = self._apply_edit((N.m(ell), "update"), t)
            # Preserve the declared mixer update for analysis.  An r-state
            # intervention is an exogenous overwrite *after* this update; the
            # implementation below changes the returned tensor only to make
            # the surrounding fused block consume r'.
            mixer_update = t

            # r_l is a residual-stream state, not merely the input to the
            # post-MLP norm.  Implement r -> r' by returning m' = r' - x from
            # the mixer hook.  Then the block's own `r = x + m'`, residual
            # addition, post-norm and MLP all consume the edited state.  This
            # is both non-reentrant and semantically stronger than re-running
            # the norm inside its own forward hook.
            r_key = (N.r(ell), "state")
            if r_key in self._edits:
                if ell not in self._current_x:
                    raise RuntimeError(
                        f"cannot apply r{ell} edit: block-input hook did not capture x{ell}"
                    )
                x = self._current_x[ell]
                r = x + t
                r_new = self._apply_edit(r_key, r)
                t = r_new - x
            if N.m(ell) in self._want:
                self._capture[N.m(ell)] = mixer_update.detach().clone()
            return _rewrap(out, t)
        return hook

    def _hook_mlp(self, ell: int):
        def hook(_mod, inputs, out):
            t = _first_tensor(out)
            t = self._apply_edit((N.g(ell), "update"), t)
            if N.g(ell) in self._want:
                self._capture[N.g(ell)] = t.detach().clone()
            return _rewrap(out, t)
        return hook

    def _hook_block_pre(self, ell: int):
        def hook(_mod, inputs):
            t = _first_tensor(inputs)
            key = (N.x(ell), "state")
            edited = key in self._edits
            t = self._apply_edit(key, t)
            self._current_x[ell] = t
            if N.x(ell) in self._want:
                self._capture[N.x(ell)] = t.detach().clone()
            return _rewrap(inputs, t) if edited else None
        return hook

    def _hook_post_norm_pre(self, ell: int):
        """Captures z_l = RMSNorm_post(r_l) and also lets us read r_l via the
        module INPUT -- which is the only place r_l exists as a tensor in a
        fused implementation."""
        def hook(_mod, inputs):
            r = _first_tensor(inputs)
            if N.r(ell) in self._want:
                self._capture[N.r(ell)] = r.detach().clone()
            return None
        return hook

    def _hook_post_norm_out(self, ell: int):
        def hook(_mod, inputs, out):
            z = _first_tensor(out)
            if N.mlp_part(ell, "z") in self._want:
                self._capture[N.mlp_part(ell, "z")] = z.detach().clone()
            return None
        return hook

    def _hook_component(self, name: str):
        def hook(_mod, inputs, out):
            t = _first_tensor(out)
            t = self._apply_edit((name, "component"), t)
            if name in self._want:
                self._capture[name] = t.detach().clone()
            return _rewrap(out, t)
        return hook

    def _hook_final_norm_pre(self):
        final = N.x(self.arch.n_layers)

        def hook(_mod, inputs):
            h = _first_tensor(inputs)
            key = (final, "state")
            edited = key in self._edits
            h = self._apply_edit(key, h)
            if final in self._want:
                self._capture[final] = h.detach().clone()
            return _rewrap(inputs, h) if edited else None
        return hook

    @contextmanager
    def instrumented(self, taps: Iterable[str], edits: Iterable[Edit] = ()):
        if self._handles:
            raise RuntimeError("Evo2Runner instrumentation is not re-entrant")
        self._want, self._edits = self._prepare_contract(taps, edits)
        self._capture = {}
        self._edit_application_counts = {key: 0 for key in self._edits}
        self._current_x = {}
        try:
            for ell in range(self.arch.n_layers):
                needs = any(
                    n in self._want or (n, k) in self._edits
                    for n in (N.x(ell), N.m(ell), N.r(ell), N.g(ell), N.mlp_part(ell, "z"))
                    for k in ("state", "update")
                )
                if not needs and not any(
                    t.startswith(f"hcl{ell}.") for t in set(self._want) | {a for a, _ in self._edits}
                ):
                    continue
                self._handles.append(self.a.block(ell).register_forward_pre_hook(self._hook_block_pre(ell)))
                self._handles.append(self.a.mixer(ell).register_forward_hook(self._hook_mixer(ell)))
                self._handles.append(self.a.mlp(ell).register_forward_hook(self._hook_mlp(ell)))
                self._handles.append(self.a.post_norm(ell).register_forward_pre_hook(self._hook_post_norm_pre(ell)))
                self._handles.append(self.a.post_norm(ell).register_forward_hook(self._hook_post_norm_out(ell)))
                if self.arch.block_type(ell) == "hcl":
                    for stage, mod in self.a.hcl_submodules(ell).items():
                        nm = N.hcl_stage(ell, stage)
                        if nm in self._want or (nm, "component") in self._edits:
                            self._handles.append(mod.register_forward_hook(self._hook_component(nm)))
            final = N.x(self.arch.n_layers)
            if final in self._want or (final, "state") in self._edits:
                self._handles.append(
                    self.a.final_norm().register_forward_pre_hook(self._hook_final_norm_pre())
                )
            yield self
            self._validate_observation_contract()
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []

    @torch.no_grad()
    def run(self, input_ids: Tensor, taps: Iterable[str] = (), edits: Iterable[Edit] = ()) -> TapSet:
        requested_taps = list(taps)
        requested_edits = list(edits)
        with self.instrumented(requested_taps, requested_edits) as _:
            logits = self.a.forward_logits(input_ids)

        tensors = dict(self._capture)
        # Preserve public spellings (legacy h_l and x_final) without changing
        # the canonical capture used by the intervention machinery.
        for requested, canonical in self._requested_to_canonical.items():
            if canonical in tensors:
                tensors[requested] = tensors[canonical]

        ts = TapSet(
            tensors=tensors,
            logits=logits.detach(),
            edit_application_counts=dict(self._edit_application_counts),
        )
        # Keep algebraic reconstruction in a separate namespace.  In
        # particular, never overwrite the independently observed x_{l+1}; a
        # wrong block hook must make the identity test fail.
        for ell in range(self.arch.n_layers):
            if N.r(ell) in ts.tensors and N.g(ell) in ts.tensors:
                ts.reconstructed[N.x(ell + 1)] = ts[N.r(ell)] + ts[N.g(ell)]
        return ts

    # ---- bilinear accounting (Step 7-1) --------------------------------

    @torch.no_grad()
    def bilinear_parts(self, ell: int, z: Tensor) -> dict[str, Tensor]:
        """a = W1 z, b = W2 z, p = a*b, out = W3 p, plus per-channel v_k norms.

        The caller MUST check `out` against the captured g_l; `verify.py`
        does. If the runtime MLP has an extra activation or gate, the
        manifest records it and this function refuses rather than forcing the
        bilinear form (the plan forbids gate language for these blocks).
        """
        if not self.arch.mlp_is_bilinear[ell]:
            raise RuntimeError(
                f"block {ell} MLP is not bilinear in the manifest "
                f"(activation={self.arch.mlp_activation[ell]!r}); use the "
                f"manifest's expression, do not force g = W3[(W1 z)*(W2 z)]"
            )
        W1, W2, W3 = self.a.mlp_weights(ell)
        z = z.to(W1.dtype)
        a = z @ W1.T
        b = z @ W2.T
        p = a * b
        out = p @ W3.T
        return {"a": a, "b": b, "p": p, "out": out, "W3": W3}

    @staticmethod
    def channel_contributions(p: Tensor, W3: Tensor) -> Tensor:
        """||v_k|| where v_k = W3[:, k] * p_k  (Step 7-1).

        |p_k| alone misses the projection direction, which is exactly the
        distinction between a big activation channel and a causal one.
        """
        return p.abs() * W3.norm(dim=0).unsqueeze(0)
