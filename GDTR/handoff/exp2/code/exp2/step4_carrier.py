"""Step 4 — coordinate 3756 exact causal audit.

Two things the plan insists on and this module enforces:

* The A/B identity and the numerator x denominator 2x2 are EXACT ONLY at the
  final pre-RMSNorm tap. x28 / x30 interventions run through nonlinear suffix
  computation, so they are evaluated by the observed scale/shape endpoints and
  the downstream state trajectory -- never by pushing them through the identity.

* The denominator effect A is, for a matched s -> s', independent of WHICH
  coordinate moved. So coordinate specificity is NOT tested by comparing
  ||A|| or the magnitude index across axes. It is tested by
  `upstream_use_contrast` -- whether the model naturally writes, modulates and
  transports this axis more than matched axes do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import ABDecomposition, centre, decompose_coordinate_change, rms_denominator, solve_beta_shape
from .manifest import Manifest
from .stats import (
    Row, equivalence, family_cluster_bootstrap, paired_difference_rows,
    positive, required_cluster_count,
)
from .taps import Edit, Evo2Runner, set_coordinate


def _require_exp1_alias(man: Manifest) -> None:
    expected = {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)}
    bad = {k: (man.arch.legacy_alias.get(k), v) for k, v in expected.items()
           if man.arch.legacy_alias.get(k) != v}
    if bad:
        raise RuntimeError(f"EXP1 legacy aliases are missing/off-by-one: {bad}")


@dataclass
class CarrierRow:
    op: str                # raw_clamp | sign_flip | fixed_radius_angular | radius_only
    tap: str
    j: int
    delta: float
    d_shape: float
    log_beta: float
    beta_identified: bool
    ab: Optional[dict] = None          # only at the final tap
    note: str = ""


def _nd_2x2(
    h: Tensor, j: int, delta: float, *, gamma: Tensor, W_U: Tensor, eps: float,
    rms_form: str, bias: Optional[Tensor],
) -> dict:
    """numerator x denominator exact 2x2 (final tap only).

    Reports main effects and the interaction, as the plan requires; a single
    contribution number is only ever a Shapley-style auxiliary, never primary.
    """
    h = h.double()
    hp = h.clone(); hp[j] += delta
    s0 = rms_denominator(h, eps=eps, form=rms_form)
    s1 = rms_denominator(hp, eps=eps, form=rms_form)

    def logits(num: Tensor, den: Tensor) -> Tensor:
        z = W_U.double() @ (gamma.double() * num / den)
        return z + bias.double() if bias is not None else z

    cells = {
        "base":        logits(h,  s0),
        "num_only":    logits(hp, s0),
        "den_only":    logits(h,  s1),
        "total":       logits(hp, s1),
    }
    C = {k: centre(v) for k, v in cells.items()}
    num_main = float((C["num_only"] - C["base"]).norm())
    den_main = float((C["den_only"] - C["base"]).norm())
    total = float((C["total"] - C["base"]).norm())
    inter = float((C["total"] - C["num_only"] - C["den_only"] + C["base"]).norm())
    return {"numerator_main": num_main, "denominator_main": den_main,
            "total": total, "interaction": inter}


@torch.no_grad()
def audit_coordinate(
    runner: Evo2Runner,
    windows: Sequence[Tensor],
    man: Manifest,
    *,
    j: int,
    deltas: Sequence[float],
    taps: Sequence[str] = (N.FINAL_PRE_NORM,),
) -> list[CarrierRow]:
    """Operations 1-5 of Step 4.

    `fixed_radius_angular` is deliberately NOT called 'direct-coordinate-only':
    it is a denominator-CONTROLLED angular intervention.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    _require_exp1_alias(man)
    gamma = runner.a.final_gamma()
    W_U = runner.a.unembedding()
    bias = runner.a.unembedding_bias()
    rows: list[CarrierRow] = []

    for ids in windows:
        base = runner.run(ids, taps=set(taps))
        z0 = base.logits.reshape(-1, base.logits.shape[-1])[-1]

        for tap in taps:
            if tap not in base.tensors:
                continue
            h_full = base[tap]
            h = h_full.reshape(-1, h_full.shape[-1])[-1]

            for delta in deltas:
                ops: dict[str, Callable[[Tensor], Tensor]] = {
                    "raw_clamp": set_coordinate(j, lambda v, d=delta: v + d),
                    "sign_flip": set_coordinate(j, lambda v: -v),
                    "radius_only": (lambda t, d=delta: t * float(
                        (t.reshape(-1, t.shape[-1])[-1].norm() + d)
                        / t.reshape(-1, t.shape[-1])[-1].norm().clamp_min(1e-30))),
                }

                def fixed_radius(t: Tensor, d=delta) -> Tensor:
                    out = t.clone()
                    v = out.reshape(-1, out.shape[-1])[-1]
                    r0 = v.norm()
                    v2 = v.clone(); v2[j] = v2[j] + d
                    v2 = v2 * (r0 / v2.norm().clamp_min(1e-30))
                    out.reshape(-1, out.shape[-1])[-1] = v2
                    return out
                ops["fixed_radius_angular"] = fixed_radius

                for name, fn in ops.items():
                    e = Edit(tap, "state", fn, f"{name} j={j} d={delta}")
                    ts = runner.run(ids, taps=(), edits=[e])
                    z1 = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
                    r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                         beta_max=man.margins.beta_max,
                                         flat_logit_threshold=man.margins.flat_logit_threshold)
                    ab = None
                    if tap == N.FINAL_PRE_NORM and name == "raw_clamp":
                        dec: ABDecomposition = decompose_coordinate_change(
                            h, j, delta, gamma=gamma, W_U=W_U, eps=runner.arch.rms_eps,
                            rms_form=runner.arch.rms_form, bias=bias,
                            observed_logits_pair=(z0, z1))
                        ab = dec.as_row()
                        ab.update(_nd_2x2(h, j, delta, gamma=gamma, W_U=W_U,
                                          eps=runner.arch.rms_eps,
                                          rms_form=runner.arch.rms_form, bias=bias))
                    rows.append(CarrierRow(
                        op=name, tap=tap, j=j, delta=delta, d_shape=r.d_shape,
                        log_beta=r.log_beta, beta_identified=r.beta_identified, ab=ab,
                        note="" if tap == N.FINAL_PRE_NORM
                             else "upstream tap: A/B identity NOT applicable"))
    return rows


# --------------------------------------------------------------------------
# coordinate specificity -- the four upstream-use contrasts of Step 4
# --------------------------------------------------------------------------


@dataclass
class UpstreamUseResult:
    natural_writing: dict
    block_effect: dict
    axis_superiority: dict
    endpoint_selectivity: dict
    verdict: str
    rescue_effect: dict = field(default_factory=dict)
    control_iut: dict = field(default_factory=dict)
    exp1_bridge: dict = field(default_factory=dict)


@torch.no_grad()
def upstream_use_contrast(
    runner: Evo2Runner,
    paired_edits: Sequence[tuple[Tensor, Tensor]],   # (ref_ids, alt_ids) with the edit STRICTLY upstream
    man: Manifest,
    *,
    j: int,
    axis_pool: Sequence[int],
    late_taps: Sequence[str],
    block_fn: Callable[[int], Callable[[Tensor], Tensor]],
    rescue_component: Optional[Callable[[int, Tensor], Tensor]] = None,
    axis_families: Optional[dict[str, Sequence[int]]] = None,
    seed: int = 42,
    n_boot: int = 10_000,
) -> UpstreamUseResult:
    """Step 4's specificity test, and the simplest first test of natural
    input modulation.

    The paired design is the point: target position t, its base, crop and
    prefix length are all held FIXED, and only a natural ref<->alt edit at a
    strictly upstream position s < t is changed. Coordinate j is then compared
    at the SAME target t. An s == t self-token edit measures the trivial
    current-token embedding effect and belongs in a separate positive control,
    not here. Poly-A and text-LM comparisons are not primary at this step.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold",
                        "delta_shape", "delta_spec")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step4_carrier")
    if rescue_component is None:
        # Measurement can still be collected, but a carrier verdict requires
        # an independently supplied, discovery-frozen component for rescue.
        rescue_available = False
    else:
        rescue_available = True

    families = axis_families or {f"axis_{ax}": (ax,) for ax in axis_pool}
    flat_axes = sorted({int(ax) for axes in families.values() for ax in axes})
    if j in flat_axes:
        raise ValueError("the true carrier axis cannot also be a control axis")
    if not families:
        raise ValueError("at least one pre-registered matched-axis family is required")

    def unpack(item, k: int):
        if len(item) == 2:
            ref, alt = item
            return f"pair:{k}", ref, alt, {}
        if len(item) == 3:
            pid, ref, alt = item
            return str(pid), ref, alt, {}
        if len(item) == 4:
            pid, ref, alt, md = item
            return str(pid), ref, alt, dict(md)
        raise ValueError("paired_edits entries must be (ref,alt), (id,ref,alt), or (id,ref,alt,metadata)")

    def make_row(value: float, pid: str, tap: str, md: dict) -> Row:
        uid = f"{pid}|{tap}"
        return Row(
            float(value), tuple(md.get("keys", (f"pair:{pid}",))),
            windows=tuple(md.get("windows", ())), stratum=str(md.get("stratum", "all")),
            unit_id=uid, meta={"pair_id": pid, "tap": tap},
        )

    def endpoint(z: Tensor, z_ref: Tensor):
        return solve_beta_shape(
            z.reshape(-1, z.shape[-1])[-1], z_ref,
            beta_min=man.margins.beta_min, beta_max=man.margins.beta_max,
            flat_logit_threshold=man.margins.flat_logit_threshold,
        )

    true: dict[str, list[Row]] = {k: [] for k in (
        "writing", "block_beta", "block_shape", "rescue_beta", "rescue_shape")}
    controls: dict[str, dict[str, list[Row]]] = {
        fam: {k: [] for k in true} for fam in families
    }

    for k, item in enumerate(paired_edits):
        pid, ref, alt, md = unpack(item, k)
        ref_ts = runner.run(ref, taps=set(late_taps))
        alt_ts = runner.run(alt, taps=set(late_taps))
        z_ref = ref_ts.logits.reshape(-1, ref_ts.logits.shape[-1])[-1]
        free = endpoint(alt_ts.logits, z_ref)

        for tap in late_taps:
            if tap not in ref_ts.tensors or tap not in alt_ts.tensors:
                raise RuntimeError(f"carrier audit requested uncaptured tap {tap!r}")
            va = ref_ts[tap].reshape(-1, ref_ts[tap].shape[-1])[-1]
            vb = alt_ts[tap].reshape(-1, alt_ts[tap].shape[-1])[-1]

            def writing(axis: int) -> float:
                return abs(float(vb[axis] / vb.norm().clamp_min(1e-30)
                                 - va[axis] / va.norm().clamp_min(1e-30)))

            def one_axis(axis: int) -> dict[str, float]:
                blocker = block_fn(axis)
                edit = Edit(tap, "state", blocker, f"block axis {axis} at {tap}")
                blocked = endpoint(runner.run(alt, taps=(), edits=[edit]).logits, z_ref)
                vals = {
                    "writing": writing(axis),
                    "block_beta": abs(free.log_beta) - abs(blocked.log_beta),
                    "block_shape": blocked.d_shape - free.d_shape,
                    "rescue_beta": float("nan"),
                    "rescue_shape": float("nan"),
                }
                if rescue_component is not None:
                    # The callback returns an ADDITIVE held-out/matched
                    # component learned outside this locked pair.  Re-adding
                    # the exact removed same-run value would be hook fidelity,
                    # not causal rescue.
                    comp = rescue_component(axis, alt_ts[tap])

                    def block_then_rescue(t: Tensor, b=blocker, c=comp) -> Tensor:
                        bt = b(t)
                        return bt + c.to(bt.dtype).to(bt.device)

                    er = Edit(tap, "state", block_then_rescue,
                              f"block+heldout-rescue axis {axis} at {tap}")
                    rescued = endpoint(runner.run(alt, taps=(), edits=[er]).logits, z_ref)
                    vals["rescue_beta"] = abs(rescued.log_beta) - abs(blocked.log_beta)
                    vals["rescue_shape"] = rescued.d_shape - free.d_shape
                return vals

            vals = one_axis(j)
            for name, value in vals.items():
                if np.isfinite(value):
                    true[name].append(make_row(value, pid, tap, md))

            for fam, axes in families.items():
                per_axis = [one_axis(int(ax)) for ax in axes]
                for name in true:
                    values = [v[name] for v in per_axis if np.isfinite(v[name])]
                    if values:
                        controls[fam][name].append(make_row(float(np.mean(values)), pid, tap, md))

    # One simultaneous family contains the direct estimates and every
    # true-vs-control contrast.  Thus a carrier label cannot be assembled
    # from separately favourable uncorrected intervals.
    fam_rows: dict[str, list[Row]] = {
        f"true::{name}": rs for name, rs in true.items() if rs
    }
    for fam in families:
        for name in ("writing", "block_beta", "rescue_beta"):
            if true[name] and controls[fam][name]:
                fam_rows[f"specificity::{name}::{fam}"] = paired_difference_rows(
                    true[name], controls[fam][name]
                )
    estimates = family_cluster_bootstrap(
        fam_rows, seed=seed, n_boot=n_boot, required=required
    )

    required_margins = (
        "matched_axis", "carrier_writing", "carrier_log_beta", "carrier_rescue",
    )
    missing_margins = [k for k in required_margins
                       if k not in man.margins.delta_spec]
    if missing_margins:
        raise RuntimeError(
            "Step 4 is missing development-calibrated delta_spec margins: "
            f"{missing_margins}"
        )
    margin = float(man.margins.delta_spec["matched_axis"])
    writing_margin = float(man.margins.delta_spec["carrier_writing"])
    beta_margin = float(man.margins.delta_spec["carrier_log_beta"])
    rescue_margin = float(man.margins.delta_spec["carrier_rescue"])
    spec_checks = {
        name: positive(est, margin) for name, est in estimates.items()
        if name.startswith("specificity::")
    }
    writing_ok = positive(estimates["true::writing"], writing_margin)
    block_ok = positive(estimates["true::block_beta"], beta_margin)
    rescue_ok = rescue_available and "true::rescue_beta" in estimates and positive(
        estimates["true::rescue_beta"], rescue_margin
    )
    block_shape_eq = equivalence(estimates["true::block_shape"], man.margins.delta_shape)
    rescue_shape_eq = (
        equivalence(estimates["true::rescue_shape"], man.margins.delta_shape)
        if "true::rescue_shape" in estimates else None
    )
    selectivity_ok = (
        block_shape_eq.verdict == "equivalent" and
        rescue_shape_eq is not None and rescue_shape_eq.verdict == "equivalent"
    )
    all_specific = bool(spec_checks) and all(spec_checks.values())

    if writing_ok and block_ok and rescue_ok and selectivity_ok and all_specific:
        verdict = "specific causal scale carrier"
    elif block_ok:
        verdict = "mixed or non-specific output feature"
    else:
        verdict = "natural correlate only; causal carrier criteria failed"

    def row(name: str) -> dict:
        return estimates[name].as_row() if name in estimates else {}

    return UpstreamUseResult(
        natural_writing={
            "true": row("true::writing"),
            "positive": writing_ok,
            "controls": {fam: row(f"specificity::writing::{fam}") for fam in families},
        },
        block_effect={"true_log_beta_damage": row("true::block_beta"),
                      "true_d_shape_change": row("true::block_shape")},
        rescue_effect={"true_log_beta_recovery": row("true::rescue_beta"),
                       "true_d_shape_change_from_free": row("true::rescue_shape"),
                       "independent_component_supplied": rescue_available},
        axis_superiority={k: v.as_row() for k, v in estimates.items()
                          if k.startswith("specificity::")},
        endpoint_selectivity={
            "block_shape_equivalence": block_shape_eq.verdict,
            "rescue_shape_equivalence": rescue_shape_eq.verdict if rescue_shape_eq else "not-run",
            "passed": selectivity_ok,
            "rule": "log-beta damage/recovery positive while D_shape change is practically equivalent to zero",
        },
        control_iut={"per_control": spec_checks, "passed": all_specific},
        exp1_bridge={
            "legacy_alias": {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)},
            "keyed_rows": {
                name: [{"unit_id": r.unit_id, "value": r.value,
                        "keys": list(r.keys), "windows": [list(w) for w in r.all_windows()],
                        "stratum": r.stratum} for r in rs]
                for name, rs in true.items()
            },
            "use": "directly join natural writing/block/rescue to EXP1 locus/task labels",
        },
        verdict=verdict,
    )
