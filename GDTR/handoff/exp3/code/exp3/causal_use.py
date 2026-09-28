"""EXP3 Phase 2 -- factorial use of the characterized late-stack mechanism.

The experiments here do not introduce a new representation lens.  They use
the raw causal objects established in Steps 3--11:

* scale and direction are edited at ``g28``;
* a frozen, partial b30 subspace is blocked at ``m30``;
* the carrier coefficient is edited at ``x31`` while its orthogonal
  complement is left untouched;
* exact projected-component rescue is labelled as a hook/decomposition
  diagnostic, whereas a held-out predicted delta is the sufficiency test.

The module supplies (i) a scale x content x carrier cube and (ii) reciprocal
donor-direction transfer with norm-only, wrong-layer, wrong-pair and matched
random-direction controls.  All comparisons remain within the same recipient
and are bootstrapped by dependency cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .exp2_api import (
    N, Edit, Evo2Runner, Manifest, Row, cluster_bootstrap, paired_contrast,
    replace_with, solve_beta_shape,
)
from .mechanism import (
    DeltaPrediction, FrozenSubspace, project_parallel, validate_subspace,
)


@dataclass(frozen=True)
class FrozenCarrierAxis:
    vector: Tensor
    sha256: str
    source_split: str
    source_role: str = "discovery"
    semantic_role: str = "x31_carrier_axis"


def _carrier_hash(vector: Tensor) -> str:
    value = vector.detach().double().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_carrier_axis(vector: Tensor, *, source_split: str,
                        source_role: str = "discovery") -> FrozenCarrierAxis:
    value = vector.detach().double().clone()
    if (value.ndim != 1 or not torch.isfinite(value).all()
            or float(value.norm()) == 0):
        raise ValueError("carrier axis must be a finite non-zero vector")
    if source_role != "discovery" or not source_split.strip():
        raise ValueError("carrier axis must be selected and frozen on discovery data")
    return FrozenCarrierAxis(
        value, _carrier_hash(value), source_split, source_role)


def validate_carrier_axis(axis: FrozenCarrierAxis, width: int, *, locked: bool) -> Tensor:
    if not isinstance(axis, FrozenCarrierAxis):
        raise TypeError("carrier_axis must be a discovery-frozen FrozenCarrierAxis")
    if axis.vector.ndim != 1 or axis.vector.numel() != width:
        raise ValueError("carrier axis width does not match x31")
    if _carrier_hash(axis.vector) != axis.sha256:
        raise RuntimeError("carrier axis changed after freezing")
    if axis.semantic_role != "x31_carrier_axis":
        raise RuntimeError("carrier artifact has the wrong semantic role")
    if locked and axis.source_role != "discovery":
        raise RuntimeError("locked carrier analysis requires a discovery-frozen axis")
    return axis.vector


def _last(x: Tensor) -> Tensor:
    return x.reshape(-1, x.shape[-1])[-1]


def _centre(x: Tensor) -> Tensor:
    x = x.double()
    return x - x.mean()


def _unit(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _cosine(a: Tensor, b: Tensor) -> Tensor:
    return (_unit(a.double()) * _unit(b.double())).sum(-1).clamp(-1.0, 1.0)


def _scope_mask(x: Tensor, scope: str,
                context_window: Optional[tuple[int, int]] = None) -> Tensor:
    """Boolean mask over all dimensions except the final feature axis."""
    if x.ndim < 2:
        raise ValueError("activation must have a sequence and feature axis")
    seq = x.shape[-2]
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    flat = mask.reshape(-1, seq)
    if scope == "target":
        flat[:, -1] = True
    elif scope == "context":
        if seq > 1:
            flat[:, :-1] = True
    elif scope == "all":
        flat[:] = True
    elif scope == "window":
        if context_window is None:
            raise ValueError("scope='window' requires context_window")
        lo, hi = context_window
        lo = lo if lo >= 0 else seq + lo
        hi = hi if hi >= 0 else seq + hi
        if not 0 <= lo < hi <= seq:
            raise ValueError(f"invalid window {context_window} for sequence length {seq}")
        flat[:, lo:hi] = True
    else:
        raise ValueError(f"unknown position scope {scope!r}")
    return mask


def _replace_scoped(native: Tensor, desired: Tensor, scope: str,
                    context_window: Optional[tuple[int, int]] = None) -> Tensor:
    if native.shape != desired.shape:
        raise ValueError(f"activation shapes differ: {native.shape} vs {desired.shape}")
    out = native.clone()
    mask = _scope_mask(native, scope, context_window)
    out[mask] = desired.to(out.dtype).to(out.device)[mask]
    return out


def _branch_state(native: Tensor, direction: Tensor, scale: float, *,
                  scope: str, context_window: Optional[tuple[int, int]]) -> Tensor:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("branch scale must be finite and positive")
    if native.shape != direction.shape:
        raise ValueError("content donor must be position-aligned with recipient")
    desired = _unit(direction.to(native.dtype).to(native.device)) \
        * native.norm(dim=-1, keepdim=True) * float(scale)
    return _replace_scoped(native, desired, scope, context_window)


def _carrier_state(dynamic: Tensor, coefficient_source: Tensor, axis: Tensor, *,
                   scope: str, context_window: Optional[tuple[int, int]]) -> Tensor:
    """Set only <x31,u>; preserve the trajectory's carrier-orthogonal state."""
    if dynamic.shape != coefficient_source.shape:
        raise ValueError("carrier source must be position-aligned")
    u = axis.to(dynamic.dtype).to(dynamic.device)
    if u.ndim != 1 or u.numel() != dynamic.shape[-1] or float(u.norm()) == 0:
        raise ValueError("carrier axis must be a non-zero vector of activation width")
    u = u / u.norm()
    current = (dynamic * u).sum(-1, keepdim=True)
    target = (coefficient_source.to(dynamic) * u).sum(-1, keepdim=True)
    desired = dynamic + (target - current) * u
    return _replace_scoped(dynamic, desired, scope, context_window)


# ---------------------------------------------------------------------------
# b30 block/rescue contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B30Condition:
    name: str
    edits: tuple[Edit, ...]
    rescue_kind: str
    strong_sufficiency: bool
    held_out: bool
    target_derived: bool
    subspace_hash: str


def b30_block_rescue_conditions(
    reference_update: Tensor,
    *,
    baseline_update: Optional[Tensor] = None,
    space: FrozenSubspace,
    prediction: Optional[DeltaPrediction] = None,
    tap: str = "m30",
    locked: bool = False,
    off_subspace_tolerance: float = 0.05,
    locked_input: Optional[Tensor] = None,
    locked_unit_id: str = "",
) -> dict[str, B30Condition]:
    """Create one-edit b30 conditions without a full-state rescue loophole.

    ``reference_update`` is the intervened m30 update and ``baseline_update``
    is the same recipient before the upstream intervention.  The blocked
    object is therefore ``P(m30_intervened-m30_baseline)``, not all of
    ``P(m30_intervened)``.  This distinction prevents an apparent mediation
    result caused by deleting b30's ordinary baseline computation.

    ``projected_rescue`` restores that measured delta; it verifies the
    subspace and hook algebra, but is marked diagnostic. ``predicted_rescue``
    is strong sufficiency only when the same delta was predicted out of sample
    without access to the target response.
    """
    if tap != N.m(30):
        raise ValueError("the standard b30 contract edits m30 only")
    B = validate_subspace(space, reference_update.shape[-1], locked=locked)
    if baseline_update is None:
        if locked:
            raise ValueError("locked b30 delta rescue requires baseline_update")
        baseline_update = torch.zeros_like(reference_update)
    if baseline_update.shape != reference_update.shape:
        raise ValueError("baseline/intervened m30 updates are not shape-aligned")
    measured_delta = reference_update - baseline_update.to(reference_update)
    natural_component = project_parallel(measured_delta, B)

    def block_fn(t: Tensor) -> Tensor:
        return t - natural_component.to(t.dtype).to(t.device)

    def restore_fn(t: Tensor, d: Tensor) -> Tensor:
        return (t - natural_component.to(t.dtype).to(t.device)
                + d.to(t.dtype).to(t.device))

    out = {
        "free": B30Condition("free", (), "none", False, False, False,
                             space.sha256),
        "block": B30Condition(
            "block", (Edit(tap, "update", block_fn,
                            "remove projected intervention-induced b30 delta"),),
            "none", False, False, False, space.sha256),
        "projected_rescue": B30Condition(
            "projected_rescue",
            (Edit(tap, "update",
                  lambda t, d=natural_component: restore_fn(t, d),
                  "restore measured projected intervention delta"),),
            "measured_projected_delta", False, False, True,
            space.sha256),
    }
    if prediction is not None:
        d = prediction.validate(
            reference_update, locked=locked, locked_input=locked_input,
            locked_unit_id=locked_unit_id)
        dp = project_parallel(d, B)
        off = float((d.double() - dp.double()).norm()
                    / d.double().norm().clamp_min(1e-30))
        if off > off_subspace_tolerance:
            raise ValueError(
                f"predicted b30 delta is {off:.3f} off the frozen subspace; "
                f"tolerance={off_subspace_tolerance:.3f}")
        out["predicted_rescue"] = B30Condition(
            "predicted_rescue",
            (Edit(tap, "update", lambda t, dd=dp: restore_fn(t, dd),
                  f"held-out predicted b30 rescue:{prediction.method}"),),
            prediction.method,
            bool(prediction.held_out and not prediction.target_derived),
            prediction.held_out, prediction.target_derived, space.sha256)
    elif locked:
        raise ValueError("locked b30 panel requires a held-out predicted delta")
    return out


def validate_b30_condition(condition: B30Condition, *, locked: bool) -> None:
    if condition.name == "free":
        if condition.edits:
            raise ValueError("free b30 condition may not contain edits")
        return
    if len(condition.edits) != 1:
        raise ValueError("b30 block/rescue must be composed into one edit")
    edit = condition.edits[0]
    if edit.tap != N.m(30) or edit.kind != "update":
        raise ValueError("b30 rescue may not overwrite x31/full downstream state")
    if condition.name == "predicted_rescue" and locked:
        if not condition.strong_sufficiency or not condition.held_out \
                or condition.target_derived:
            raise RuntimeError("locked predicted rescue violates provenance contract")
    if condition.name == "projected_rescue" and condition.strong_sufficiency:
        raise ValueError("same-run projected rescue is fidelity, not strong sufficiency")


# ---------------------------------------------------------------------------
# Scale x content x carrier cube
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeContext:
    pair_id: str
    scale_level: str
    scale: float
    content_level: str
    recipient_ids: Tensor
    donor_ids: Tensor
    recipient_g28: Tensor
    donor_g28: Tensor
    baseline_m30: Tensor
    upstream_m30: Tensor


@dataclass
class CausalCubeRow:
    pair_id: str
    scale_level: str
    scale: float
    content_level: str
    carrier_level: str
    b30_condition: str
    d_shape: float
    log_beta: float
    signed_donor_following: float
    branch_dose: float
    carrier_shift: float
    b30_strong_sufficiency: bool
    b30_rescue_held_out: bool
    b30_target_derived: bool
    b30_subspace_hash: str = ""
    dependency_keys: tuple[str, ...] = ()


def _unpack_pair(item) -> tuple[str, Tensor, Tensor, dict]:
    if len(item) == 3:
        pid, rec, donor = item
        return str(pid), rec, donor, {}
    if len(item) == 4:
        pid, rec, donor, metadata = item
        return str(pid), rec, donor, dict(metadata)
    raise ValueError("pairs must be (pair_id, recipient_ids, donor_ids[, metadata])")


@torch.no_grad()
def scale_content_carrier_cube(
    runner: Evo2Runner,
    pairs: Sequence[tuple],
    man: Manifest,
    *,
    carrier_axis: FrozenCarrierAxis,
    scale_levels: Mapping[str, float],
    b30_space: Optional[FrozenSubspace] = None,
    b30_prediction: Optional[Callable[[CubeContext], DeltaPrediction]] = None,
    position_scope: str = "target",
    context_window: Optional[tuple[int, int]] = None,
    locked: bool = False,
) -> list[CausalCubeRow]:
    """Run the orthogonalised 2 x 2 x 2 intervention cube.

    Exactly two scale levels are required for an interpretable factorial.
    Content is self versus donor direction at matched recipient branch norm;
    carrier is recipient versus donor coefficient at x31.  Thus none of the
    three factors is inferred from another one's natural correlation.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    if len(scale_levels) != 2:
        raise ValueError("the primary causal cube requires exactly two scale levels")
    if len(set(float(v) for v in scale_levels.values())) != 2:
        raise ValueError("scale levels must have distinct positive values")
    taps = {N.g(28), N.m(30), N.x(31)}
    rows: list[CausalCubeRow] = []

    for item in pairs:
        pid, rec_ids, donor_ids, md = _unpack_pair(item)
        if rec_ids.shape == donor_ids.shape and torch.equal(rec_ids, donor_ids):
            raise ValueError(f"{pid}: recipient and donor are identical")
        rec = runner.run(rec_ids, taps=taps)
        donor = runner.run(donor_ids, taps=taps)
        carrier_vector = validate_carrier_axis(
            carrier_axis, rec[N.x(31)].shape[-1], locked=locked)
        z0, zD = _last(rec.logits), _last(donor.logits)
        donor_dir = _centre(zD) - _centre(z0)
        donor_n2 = donor_dir.norm().square().clamp_min(1e-30)
        keys = tuple(md.get("keys", ())) or (f"pair:{pid}",)

        for scale_name, scale in scale_levels.items():
            for content in ("self", "donor"):
                source = rec[N.g(28)] if content == "self" else donor[N.g(28)]
                desired_g = _branch_state(
                    rec[N.g(28)], source, float(scale), scope=position_scope,
                    context_window=context_window)
                branch_edit = Edit(N.g(28), "update", replace_with(desired_g),
                                   f"cube scale={scale_name} content={content}")
                upstream = runner.run(rec_ids, taps=taps, edits=[branch_edit])
                ctx = CubeContext(
                    pid, str(scale_name), float(scale), content,
                    rec_ids, donor_ids, rec[N.g(28)], donor[N.g(28)],
                    rec[N.m(30)], upstream[N.m(30)])

                if b30_space is None:
                    b30_conditions = {
                        "free": B30Condition("free", (), "none", False,
                                             False, False, "")}
                else:
                    prediction = (b30_prediction(ctx)
                                  if b30_prediction is not None else None)
                    b30_conditions = b30_block_rescue_conditions(
                        upstream[N.m(30)], baseline_update=rec[N.m(30)],
                        space=b30_space,
                        prediction=prediction, locked=locked,
                        locked_input=rec_ids, locked_unit_id=pid)

                for carrier in ("self", "donor"):
                    coeff_source = rec[N.x(31)] if carrier == "self" else donor[N.x(31)]

                    def carrier_edit_fn(t: Tensor, src=coeff_source) -> Tensor:
                        return _carrier_state(
                            t, src, carrier_vector, scope=position_scope,
                            context_window=context_window)

                    carrier_edit = Edit(N.x(31), "state", carrier_edit_fn,
                                        f"cube carrier={carrier}")
                    for bname, bcond in b30_conditions.items():
                        validate_b30_condition(bcond, locked=locked)
                        trace = runner.run(
                            rec_ids, taps={N.g(28), N.m(30), N.x(31)},
                            edits=[branch_edit, *bcond.edits, carrier_edit])
                        z = _last(trace.logits)
                        shape = solve_beta_shape(
                            z, z0, beta_min=man.margins.beta_min,
                            beta_max=man.margins.beta_max,
                            flat_logit_threshold=man.margins.flat_logit_threshold)
                        dz = _centre(z) - _centre(z0)
                        branch_dose = float(
                            (trace[N.g(28)].double() - rec[N.g(28)].double()).norm()
                            / rec[N.g(28)].double().norm().clamp_min(1e-30))
                        u = carrier_vector.double().to(trace[N.x(31)].device)
                        u = u / u.norm().clamp_min(1e-30)
                        dx = trace[N.x(31)].double() - rec[N.x(31)].double()
                        carrier_shift = float((dx @ u).norm()
                                              / rec[N.x(31)].double().norm().clamp_min(1e-30))
                        rows.append(CausalCubeRow(
                            pair_id=pid, scale_level=str(scale_name), scale=float(scale),
                            content_level=content, carrier_level=carrier,
                            b30_condition=bname, d_shape=shape.d_shape,
                            log_beta=shape.log_beta,
                            signed_donor_following=float(torch.dot(dz, donor_dir)
                                                         / donor_n2),
                            branch_dose=branch_dose, carrier_shift=carrier_shift,
                            b30_strong_sufficiency=bcond.strong_sufficiency,
                            b30_rescue_held_out=bcond.held_out,
                            b30_target_derived=bcond.target_derived,
                            b30_subspace_hash=bcond.subspace_hash,
                            dependency_keys=keys))
    return rows


def _cube_effect(cells: Mapping[tuple[int, int, int], float], effect: str) -> float:
    """Balanced 2^3 factorial contrasts, in natural high-minus-low units."""
    y = cells
    if effect == "scale":
        return sum(y[1, c, k] - y[0, c, k] for c in (0, 1) for k in (0, 1)) / 4
    if effect == "content":
        return sum(y[s, 1, k] - y[s, 0, k] for s in (0, 1) for k in (0, 1)) / 4
    if effect == "carrier":
        return sum(y[s, c, 1] - y[s, c, 0] for s in (0, 1) for c in (0, 1)) / 4
    if effect == "scale:content":
        return sum((y[1, 1, k] - y[1, 0, k])
                   - (y[0, 1, k] - y[0, 0, k]) for k in (0, 1)) / 2
    if effect == "scale:carrier":
        return sum((y[1, c, 1] - y[1, c, 0])
                   - (y[0, c, 1] - y[0, c, 0]) for c in (0, 1)) / 2
    if effect == "content:carrier":
        return sum((y[s, 1, 1] - y[s, 1, 0])
                   - (y[s, 0, 1] - y[s, 0, 0]) for s in (0, 1)) / 2
    if effect == "scale:content:carrier":
        return ((y[1, 1, 1] - y[1, 1, 0] - y[1, 0, 1] + y[1, 0, 0])
                - (y[0, 1, 1] - y[0, 1, 0] - y[0, 0, 1] + y[0, 0, 0]))
    raise KeyError(effect)


def cube_factorial_effects(
    rows: Sequence[CausalCubeRow],
    *,
    endpoint: str = "signed_donor_following",
    b30_condition: str = "free",
    low_scale: Optional[str] = None,
    high_scale: Optional[str] = None,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Estimate all seven balanced factorial effects with cluster intervals."""
    selected = [r for r in rows if r.b30_condition == b30_condition]
    levels = sorted({r.scale_level for r in selected})
    if low_scale is None or high_scale is None:
        by_name = {r.scale_level: r.scale for r in selected}
        if len(by_name) != 2:
            raise ValueError("cannot infer two scale levels")
        low_scale, high_scale = sorted(by_name, key=by_name.get)
    if low_scale == high_scale:
        raise ValueError("low_scale and high_scale must differ")
    enc_s = {low_scale: 0, high_scale: 1}
    enc_c = {"self": 0, "donor": 1}
    enc_k = {"self": 0, "donor": 1}

    by_pair: dict[str, dict[tuple[int, int, int], CausalCubeRow]] = {}
    for r in selected:
        if r.scale_level not in enc_s:
            continue
        key = (enc_s[r.scale_level], enc_c[r.content_level], enc_k[r.carrier_level])
        if key in by_pair.setdefault(r.pair_id, {}):
            raise ValueError(f"duplicate cube cell {r.pair_id}:{key}:{b30_condition}")
        by_pair[r.pair_id][key] = r
    required = {(s, c, k) for s in (0, 1) for c in (0, 1) for k in (0, 1)}
    effects = ("scale", "content", "carrier", "scale:content",
               "scale:carrier", "content:carrier", "scale:content:carrier")
    effect_rows = {name: [] for name in effects}
    per_pair = {}
    for pid, cells0 in by_pair.items():
        if set(cells0) != required:
            raise RuntimeError(f"{pid}: incomplete cube; missing={sorted(required-set(cells0))}")
        cells = {k: float(getattr(v, endpoint)) for k, v in cells0.items()}
        keys = tuple(next(iter(cells0.values())).dependency_keys) or (f"pair:{pid}",)
        per_pair[pid] = {}
        for name in effects:
            value = _cube_effect(cells, name)
            effect_rows[name].append(Row(value, keys=keys))
            per_pair[pid][name] = value
    estimates = {
        name: cluster_bootstrap(vals, min_clusters=min_clusters,
                                n_boot=n_boot, alpha=alpha).as_row()
        for name, vals in effect_rows.items()
    }
    return {
        "endpoint": endpoint, "b30_condition": b30_condition,
        "scale_levels": {"low": low_scale, "high": high_scale},
        "estimates": estimates, "per_pair": per_pair,
        "interpretation": {
            "content": "g28 direction changes relative token preference",
            "scale": "g28 amplitude has an effect beyond direction",
            "carrier": "x31 carrier changes calibration/output at fixed content trajectory",
            "interactions": "one axis gates another; do not collapse them into one difficulty axis",
        },
        "sampling_used": {
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
    }


def cube_factorization_verdict(
    summaries: Mapping[str, Mapping[str, object]],
    *,
    axis_sources: Mapping[str, tuple[str, str]],
    effect_margins: Mapping[str, float],
    equivalence_margin: float,
) -> dict:
    """Turn preregistered cube endpoints into one axis-level decision.

    ``summaries`` should contain independently computed
    :func:`cube_factorial_effects` outputs, typically donor-following for the
    content axis and ``log_beta`` for scale/carrier.  ``axis_sources`` binds
    each of ``scale``, ``content`` and ``carrier`` to
    ``(summary_name, factorial_effect_name)`` *before* locked evaluation.
    This avoids declaring all three axes from whichever endpoint happened to
    look strongest.
    """
    required = {"scale", "content", "carrier"}
    if set(axis_sources) != required or set(effect_margins) != required:
        raise ValueError("cube verdict requires exact scale/content/carrier bindings")
    if equivalence_margin <= 0 or any(
            not np.isfinite(float(v)) or float(v) <= 0
            for v in effect_margins.values()):
        raise ValueError("cube effect/equivalence margins must be positive")

    checks: dict[str, bool] = {}
    estimates: dict[str, object] = {}
    equivalent: dict[str, bool] = {}
    sampling_contracts = {
        tuple(sorted(dict(summary.get("sampling_used", {})).items()))
        for summary in summaries.values()
    }
    if len(sampling_contracts) != 1 or not next(iter(sampling_contracts)):
        raise ValueError("all cube endpoints must use one explicit sampling contract")
    sampling_used = dict(next(iter(sampling_contracts)))
    for axis in sorted(required):
        summary_name, effect_name = axis_sources[axis]
        if summary_name not in summaries:
            raise KeyError(f"missing preregistered cube summary {summary_name!r}")
        summary = summaries[summary_name]
        effect_rows = summary.get("estimates")
        if not isinstance(effect_rows, Mapping) or effect_name not in effect_rows:
            raise KeyError(f"{summary_name!r} lacks factorial effect {effect_name!r}")
        row = effect_rows[effect_name]
        if not isinstance(row, Mapping):
            raise TypeError("cube interval rows must be mappings")
        lo_key = "ci_lo" if "ci_lo" in row else "lo"
        hi_key = "ci_hi" if "ci_hi" in row else "hi"
        lo, hi = float(row[lo_key]), float(row[hi_key])
        margin = float(effect_margins[axis])
        checks[f"{axis}_effect_identified"] = lo > margin or hi < -margin
        equivalent[axis] = lo > -equivalence_margin and hi < equivalence_margin
        estimates[axis] = dict(row)

    interaction_names = (
        "scale:content", "scale:carrier", "content:carrier",
        "scale:content:carrier",
    )
    interaction_estimates: dict[str, object] = {}
    interactions_equivalent = True
    for summary_name, summary in sorted(summaries.items()):
        effect_rows = summary.get("estimates")
        if not isinstance(effect_rows, Mapping):
            raise TypeError("cube summary lacks interval estimates")
        for effect_name in interaction_names:
            if effect_name not in effect_rows:
                raise KeyError(
                    f"{summary_name!r} lacks interaction {effect_name!r}")
            row = effect_rows[effect_name]
            if not isinstance(row, Mapping):
                raise TypeError("cube interaction rows must be mappings")
            lo_key = "ci_lo" if "ci_lo" in row else "lo"
            hi_key = "ci_hi" if "ci_hi" in row else "hi"
            lo, hi = float(row[lo_key]), float(row[hi_key])
            interactions_equivalent &= (
                lo > -equivalence_margin and hi < equivalence_margin)
            interaction_estimates[f"{summary_name}:{effect_name}"] = dict(row)

    n_identified = sum(checks.values())
    if n_identified == 3 and interactions_equivalent:
        status, code = "supported", "axes_causally_separable"
    elif n_identified >= 1:
        status, code = "mixed", "axes_conditionally_separable"
    elif all(equivalent.values()):
        status, code = "equivalent", "cube_axis_effects_bounded_null"
    elif any(equivalent.values()):
        status, code = "mixed", "axes_conditionally_separable"
    else:
        status, code = "unresolved", "cube_factorization_unresolved"
    return {
        "status": status,
        "conclusion_code": code,
        "checks": checks,
        "equivalent_to_zero": equivalent,
        "axis_sources": {k: list(v) for k, v in axis_sources.items()},
        "estimates": estimates,
        "interaction_estimates": interaction_estimates,
        "interactions_equivalent_to_zero": interactions_equivalent,
        "margins_used": {
            "effect_margins": dict(effect_margins),
            "equivalence_margin": equivalence_margin,
            **sampling_used,
        },
        "claim": (
            "scale, content and carrier have preregistered causal readouts"
            if status == "supported" else
            "the causal axes require a conditional or bounded account"),
    }


def cube_b30_mediation(
    rows: Sequence[CausalCubeRow],
    *,
    endpoint: str = "signed_donor_following",
    scale_level: str,
    carrier_level: str = "self",
    effect_margin: float,
    rescue_margin: float,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
) -> dict:
    """Does b30 block and held-out rescue the cube's g28 content effect?"""
    def effect_map(condition: str) -> dict[str, Row]:
        grouped: dict[str, dict[str, CausalCubeRow]] = {}
        for r in rows:
            if (r.b30_condition == condition and r.scale_level == scale_level
                    and r.carrier_level == carrier_level):
                grouped.setdefault(r.pair_id, {})[r.content_level] = r
        out = {}
        for pid, cells in grouped.items():
            if set(cells) != {"self", "donor"}:
                continue
            keys = cells["self"].dependency_keys or (f"pair:{pid}",)
            out[pid] = Row(float(getattr(cells["donor"], endpoint)
                                 - getattr(cells["self"], endpoint)), keys=keys)
        return out

    free, block = effect_map("free"), effect_map("block")
    projected, predicted = effect_map("projected_rescue"), effect_map("predicted_rescue")
    e_free = _paired_map_zero(free, min_clusters=min_clusters,
                              n_boot=n_boot, alpha=alpha)
    e_block_loss = _paired_maps(free, block, min_clusters=min_clusters,
                                n_boot=n_boot, alpha=alpha)
    e_block_residual = _paired_map_zero(
        block, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
    e_projected = (_paired_maps(projected, block, min_clusters=min_clusters,
                                n_boot=n_boot, alpha=alpha) if projected else None)
    if not predicted:
        raise RuntimeError("b30 mediation requires the predicted-rescue cube cells")
    e_predicted = _paired_maps(predicted, block, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    e_predicted_residual = _paired_maps(
        free, predicted, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
    pred_rows = [r for r in rows if r.b30_condition == "predicted_rescue"]
    provenance = all(r.b30_strong_sufficiency and r.b30_rescue_held_out
                     and not r.b30_target_derived for r in pred_rows)
    checks = {
        "content_effect_exists": e_free.lo > effect_margin,
        "b30_block_removes_content_effect": e_block_loss.lo > effect_margin,
        "blocked_content_effect_is_equivalent_to_zero": (
            e_block_residual.lo > -equivalence_margin
            and e_block_residual.hi < equivalence_margin),
        "heldout_predicted_delta_rescues": e_predicted.lo > rescue_margin,
        "predicted_rescue_matches_free_effect": (
            e_predicted_residual.lo > -equivalence_margin
            and e_predicted_residual.hi < equivalence_margin),
        "predicted_rescue_provenance": provenance,
    }
    free_equivalent = (
        e_free.lo > -equivalence_margin and e_free.hi < equivalence_margin)
    if all(checks.values()):
        status, code = "supported", "b30_mediates_content"
    elif free_equivalent:
        status, code = "equivalent", "b30_mediation_bounded_null"
    elif checks["content_effect_exists"] and not checks["b30_block_removes_content_effect"]:
        status, code = "refuted", "b30_mediation_refuted"
    elif any(checks.values()):
        status, code = "mixed", "partial_b30_mediation"
    else:
        status, code = "unresolved", "b30_mediation_unresolved"
    return {
        "status": status,
        "conclusion_code": code,
        "checks": checks,
        "claim": ("g28 content effect is transported through the frozen b30 component"
                  if all(checks.values()) else "b30 mediation incomplete"),
        "estimates": {
            "free_content_effect": e_free.as_row(),
            "block_loss": e_block_loss.as_row(),
            "blocked_content_residual": e_block_residual.as_row(),
            "projected_component_recovery_diagnostic": (
                None if e_projected is None else e_projected.as_row()),
            "predicted_delta_recovery": e_predicted.as_row(),
            "predicted_delta_residual_to_free": e_predicted_residual.as_row(),
        },
        "margins_used": {
            "effect_margin": effect_margin,
            "rescue_margin": rescue_margin,
            "equivalence_margin": equivalence_margin,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
    }


# ---------------------------------------------------------------------------
# Reciprocal direction transfer and matched controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferContext:
    pair_id: str
    direction_id: str
    condition: str
    recipient_ids: Tensor
    donor_ids: Tensor
    recipient_g28: Tensor
    replacement_g28: Tensor
    baseline_m30: Tensor
    upstream_m30: Tensor


@dataclass
class DirectionTransferRow:
    pair_id: str
    direction_id: str
    condition: str
    b30_condition: str
    d_shape: float
    log_beta: float
    signed_donor_following: float
    branch_dose: float
    angular_dose: float
    b30_strong_sufficiency: bool
    b30_rescue_held_out: bool
    b30_target_derived: bool
    b30_subspace_hash: str = ""
    dependency_keys: tuple[str, ...] = ()


def _rematch_angle(origin: Tensor, candidate: Tensor, target: Tensor) -> Tensor:
    """Put candidate in origin's tangent plane at origin->target angle."""
    o, c, t = _unit(origin.double()), _unit(candidate.double()), _unit(target.double())
    dot_t = (o * t).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.arccos(dot_t)
    tangent = c - (c * o).sum(-1, keepdim=True) * o
    bad = tangent.norm(dim=-1, keepdim=True) < 1e-10
    # Deterministic fallback: choose a coordinate least aligned with origin.
    idx = o.abs().argmin(-1, keepdim=True)
    fallback = torch.zeros_like(o).scatter_(-1, idx, 1.0)
    fallback = fallback - (fallback * o).sum(-1, keepdim=True) * o
    tangent = torch.where(bad, fallback, tangent)
    tangent = _unit(tangent)
    return (torch.cos(angle) * o + torch.sin(angle) * tangent).to(origin.dtype)


def _deterministic_random_direction(origin: Tensor, target: Tensor, seed_text: str) -> Tensor:
    digest = hashlib.sha256(seed_text.encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rnd = torch.randn(origin.shape, generator=gen, dtype=torch.float64)
    return _rematch_angle(origin, rnd.to(origin.device), target)


@torch.no_grad()
def bidirectional_direction_transfer(
    runner: Evo2Runner,
    pairs: Sequence[tuple],
    man: Manifest,
    *,
    wrong_pair_ids: Callable[[str, str], Tensor],
    random_direction: Optional[Callable[[str, str, Tensor, Tensor], Tensor]] = None,
    wrong_layer: int = 27,
    b30_space: Optional[FrozenSubspace] = None,
    b30_prediction: Optional[Callable[[TransferContext], DeltaPrediction]] = None,
    position_scope: str = "target",
    context_window: Optional[tuple[int, int]] = None,
    locked: bool = False,
) -> list[DirectionTransferRow]:
    """Transfer g28 direction both ways with four explicit control families."""
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    if not 0 <= wrong_layer < runner.arch.n_layers or wrong_layer == 28:
        raise ValueError("wrong_layer must be a real layer other than b28")
    if locked and random_direction is None:
        raise ValueError("locked transfer requires a discovery-frozen local-covariance control")
    taps = {N.g(28), N.g(wrong_layer), N.m(30)}
    rows: list[DirectionTransferRow] = []

    for item in pairs:
        pid, left_ids, right_ids, md = _unpack_pair(item)
        keys = tuple(md.get("keys", ())) or (f"pair:{pid}",)
        for rec_label, donor_label, rec_ids, donor_ids in (
            ("A", "B", left_ids, right_ids),
            ("B", "A", right_ids, left_ids),
        ):
            did = f"{pid}:{rec_label}<-{donor_label}"
            rec = runner.run(rec_ids, taps=taps)
            donor = runner.run(donor_ids, taps=taps)
            wrong_ids = wrong_pair_ids(pid, rec_label)
            wrong = runner.run(wrong_ids, taps={N.g(28)})
            z0, zD = _last(rec.logits), _last(donor.logits)
            out_dir = _centre(zD) - _centre(z0)
            out_n2 = out_dir.norm().square().clamp_min(1e-30)
            gR, gD = rec[N.g(28)], donor[N.g(28)]

            correct = _branch_state(gR, gD, 1.0, scope=position_scope,
                                    context_window=context_window)
            delta_norm = (correct - gR).norm(dim=-1, keepdim=True)
            norm_only = gR + _unit(gR) * delta_norm
            norm_only = _replace_scoped(gR, norm_only, position_scope, context_window)
            wrong_layer_dir = _rematch_angle(gR, donor[N.g(wrong_layer)], gD)
            wrong_layer_state = _branch_state(
                gR, wrong_layer_dir, 1.0, scope=position_scope,
                context_window=context_window)
            wrong_pair_dir = _rematch_angle(gR, wrong[N.g(28)], gD)
            wrong_pair_state = _branch_state(
                gR, wrong_pair_dir, 1.0, scope=position_scope,
                context_window=context_window)
            raw_random = (random_direction(pid, rec_label, gR, gD)
                          if random_direction is not None else
                          _deterministic_random_direction(gR, gD, did))
            random_dir = _rematch_angle(gR, raw_random, gD)
            random_state = _branch_state(
                gR, random_dir, 1.0, scope=position_scope,
                context_window=context_window)

            replacements = {
                "self_patch": gR,
                "direction_transfer": correct,
                "norm_only": norm_only,
                "wrong_layer": wrong_layer_state,
                "wrong_pair": wrong_pair_state,
                "random_matched": random_state,
            }
            for condition, replacement in replacements.items():
                branch = Edit(N.g(28), "update", replace_with(replacement),
                              f"{did}:{condition}")
                upstream = runner.run(rec_ids, taps=taps, edits=[branch])
                ctx = TransferContext(
                    pid, did, condition, rec_ids, donor_ids, gR,
                    replacement, rec[N.m(30)], upstream[N.m(30)])
                if b30_space is None:
                    bconds = {"free": B30Condition(
                        "free", (), "none", False, False, False, "")}
                else:
                    pred = (b30_prediction(ctx) if b30_prediction is not None else None)
                    bconds = b30_block_rescue_conditions(
                        upstream[N.m(30)], baseline_update=rec[N.m(30)],
                        space=b30_space,
                        prediction=pred, locked=locked,
                        locked_input=rec_ids, locked_unit_id=did)
                for bname, bcond in bconds.items():
                    validate_b30_condition(bcond, locked=locked)
                    trace = runner.run(rec_ids, taps={N.g(28), N.m(30)},
                                       edits=[branch, *bcond.edits])
                    z = _last(trace.logits)
                    shape = solve_beta_shape(
                        z, z0, beta_min=man.margins.beta_min,
                        beta_max=man.margins.beta_max,
                        flat_logit_threshold=man.margins.flat_logit_threshold)
                    dz = _centre(z) - _centre(z0)
                    dose = float((trace[N.g(28)].double() - gR.double()).norm()
                                 / gR.double().norm().clamp_min(1e-30))
                    angular = float(torch.arccos(
                        _cosine(_last(trace[N.g(28)]), _last(gR))))
                    rows.append(DirectionTransferRow(
                        pair_id=pid, direction_id=did, condition=condition,
                        b30_condition=bname, d_shape=shape.d_shape,
                        log_beta=shape.log_beta,
                        signed_donor_following=float(torch.dot(dz, out_dir) / out_n2),
                        branch_dose=dose, angular_dose=angular,
                        b30_strong_sufficiency=bcond.strong_sufficiency,
                        b30_rescue_held_out=bcond.held_out,
                        b30_target_derived=bcond.target_derived,
                        b30_subspace_hash=bcond.subspace_hash,
                        dependency_keys=keys))
    return rows


def _transfer_map(rows: Sequence[DirectionTransferRow], condition: str,
                  b30: str) -> dict[str, Row]:
    out = {}
    for r in rows:
        if r.condition == condition and r.b30_condition == b30:
            if r.direction_id in out:
                raise ValueError(f"duplicate transfer cell {r.direction_id}:{condition}:{b30}")
            out[r.direction_id] = Row(
                float(r.signed_donor_following),
                keys=(r.dependency_keys or (f"pair:{r.pair_id}",)),
                unit_id=r.direction_id)
    return out


def _paired_maps(a: Mapping[str, Row], b: Mapping[str, Row], *, min_clusters: int,
                 n_boot: int, alpha: float):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise RuntimeError("no aligned intervention rows")
    return paired_contrast([a[k] for k in keys], [b[k] for k in keys],
                           min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)


def _paired_map_zero(a: Mapping[str, Row], *, min_clusters: int,
                     n_boot: int, alpha: float):
    return _paired_maps(a, {k: Row(0.0, keys=v.keys) for k, v in a.items()},
                        min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)


def direction_transfer_verdict(
    rows: Sequence[DirectionTransferRow],
    *,
    effect_margin: float,
    specificity_margins: Mapping[str, float],
    equivalence_margin: float,
    rescue_margin: float,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Conjunction for reciprocal, specific, b30-mediated direction transfer."""
    control_names = ("norm_only", "wrong_layer", "wrong_pair", "random_matched")
    missing = sorted(set(control_names) - set(specificity_margins))
    if missing:
        raise ValueError(f"specificity margins missing {missing}")
    correct = _transfer_map(rows, "direction_transfer", "free")
    zero = {k: Row(0.0, keys=v.keys, unit_id=v.unit_id) for k, v in correct.items()}
    e_effect = _paired_maps(correct, zero, min_clusters=min_clusters,
                            n_boot=n_boot, alpha=alpha)

    specificity = {}
    specificity_ok = True
    norm_null = False
    for name in control_names:
        control = _transfer_map(rows, name, "free")
        est = _paired_maps(correct, control, min_clusters=min_clusters,
                           n_boot=n_boot, alpha=alpha / len(control_names))
        specificity[name] = est.as_row()
        specificity_ok &= est.lo > float(specificity_margins[name])
        if name == "norm_only":
            e_norm = _paired_maps(control, zero, min_clusters=min_clusters,
                                  n_boot=n_boot, alpha=alpha)
            norm_null = e_norm.lo > -equivalence_margin and e_norm.hi < equivalence_margin

    block = _transfer_map(rows, "direction_transfer", "block")
    projected = _transfer_map(rows, "direction_transfer", "projected_rescue")
    predicted = _transfer_map(rows, "direction_transfer", "predicted_rescue")
    if not block or not predicted:
        raise RuntimeError("direction transfer verdict requires b30 block and predicted rescue")
    e_block = _paired_maps(correct, block, min_clusters=min_clusters,
                           n_boot=n_boot, alpha=alpha)
    e_block_residual = _paired_maps(
        block, zero, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
    e_pred = _paired_maps(predicted, block, min_clusters=min_clusters,
                          n_boot=n_boot, alpha=alpha)
    e_pred_residual = _paired_maps(
        correct, predicted, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    e_proj = (_paired_maps(projected, block, min_clusters=min_clusters,
                           n_boot=n_boot, alpha=alpha) if projected else None)
    pred_rows = [r for r in rows if r.condition == "direction_transfer"
                 and r.b30_condition == "predicted_rescue"]
    provenance = all(r.b30_strong_sufficiency and r.b30_rescue_held_out
                     and not r.b30_target_derived for r in pred_rows)

    # Both reciprocal directions must exist for every pair.  They share a
    # dependency key and therefore never count as two independent loci.
    by_pair: dict[str, set[str]] = {}
    for r in rows:
        if r.condition == "direction_transfer" and r.b30_condition == "free":
            by_pair.setdefault(r.pair_id, set()).add(r.direction_id)
    reciprocal_complete = bool(by_pair) and all(len(v) == 2 for v in by_pair.values())
    checks = {
        "bidirectional_panel_complete": reciprocal_complete,
        "direction_transfer_has_output_effect": e_effect.lo > effect_margin,
        "beats_every_matched_control": specificity_ok,
        "norm_only_is_equivalent_to_null": norm_null,
        "b30_component_is_necessary": e_block.lo > effect_margin,
        "blocked_transfer_is_equivalent_to_zero": (
            e_block_residual.lo > -equivalence_margin
            and e_block_residual.hi < equivalence_margin),
        "heldout_b30_delta_rescues": e_pred.lo > rescue_margin,
        "predicted_rescue_matches_free_transfer": (
            e_pred_residual.lo > -equivalence_margin
            and e_pred_residual.hi < equivalence_margin),
        "predicted_rescue_provenance": provenance,
    }
    effect_equivalent = (
        e_effect.lo > -equivalence_margin and e_effect.hi < equivalence_margin)
    if all(checks.values()):
        status, code = "supported", "specific_bidirectional_transfer"
    elif effect_equivalent:
        status, code = "equivalent", "direction_transfer_bounded_null"
    elif checks["direction_transfer_has_output_effect"] and not (
            checks["beats_every_matched_control"]
            and checks["b30_component_is_necessary"]):
        status, code = "refuted", "direction_transfer_nonspecific"
    elif any(checks.values()):
        status, code = "mixed", "partial_direction_transfer"
    else:
        status, code = "unresolved", "direction_transfer_unresolved"
    return {
        "status": status,
        "conclusion_code": code,
        "checks": checks,
        "claim": ("specific reciprocal g28-direction transfer through b30"
                  if all(checks.values()) else "direction transfer evidence incomplete"),
        "estimates": {
            "direction_effect": e_effect.as_row(),
            "specificity": specificity,
            "b30_block_loss": e_block.as_row(),
            "b30_blocked_residual": e_block_residual.as_row(),
            "projected_rescue_diagnostic": None if e_proj is None else e_proj.as_row(),
            "predicted_rescue": e_pred.as_row(),
            "predicted_rescue_residual_to_free": e_pred_residual.as_row(),
        },
        "multiplicity": {
            "specificity_method": "Bonferroni over four pre-registered controls",
            "family_alpha": alpha,
        },
        "margins_used": {
            "effect_margin": effect_margin,
            "specificity_margins": dict(specificity_margins),
            "equivalence_margin": equivalence_margin,
            "rescue_margin": rescue_margin,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
    }


def causal_factorization_synthesis(
    cube_verdict: Mapping[str, object],
    mediation_verdict: Mapping[str, object],
    transfer_verdict: Mapping[str, object],
) -> dict:
    """Combine the three Phase-2 decisions without reopening EXP2 claims."""
    components = {
        "cube": str(cube_verdict.get("status", "")),
        "b30_mediation": str(mediation_verdict.get("status", "")),
        "direction_transfer": str(transfer_verdict.get("status", "")),
    }
    allowed = {"supported", "equivalent", "mixed", "unresolved", "refuted"}
    if any(value not in allowed for value in components.values()):
        raise ValueError("every Phase-2 component needs a canonical status")
    if all(value == "supported" for value in components.values()):
        status, code = "supported", "scale_carrier_content_factorization"
    elif all(value == "equivalent" for value in components.values()):
        status, code = "equivalent", "factorization_bounded_null"
    elif "unresolved" in components.values():
        status, code = "unresolved", "factorization_unresolved"
    elif "refuted" in components.values():
        status, code = "refuted", "alternative_transport_architecture"
    else:
        status, code = "mixed", "conditional_scale_carrier_content_factorization"
    return {
        "status": status,
        "conclusion_code": code,
        "component_statuses": components,
        "component_codes": {
            "cube": cube_verdict.get("conclusion_code"),
            "b30_mediation": mediation_verdict.get("conclusion_code"),
            "direction_transfer": transfer_verdict.get("conclusion_code"),
        },
        "claim": (
            "EXP2's fixed stages support an operational scale/carrier/content factorization"
            if status == "supported" else
            "EXP2's fixed stages require a conditional EXP3 transport overlay"),
        "fixed_exp2_foundation_retested": False,
    }
