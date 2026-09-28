"""Step 8 — inside the b30 HCL: t_amp vs t_need, and the mediator candidates.

The two localisation points are NOT assumed to coincide:

    t_amp  : the first stage whose norm / effective rank departs from the
             matched earlier HCL
    t_need : the first stage where removing the FIXED causal component
             damages the endpoint and the next stage's natural component
             rescues it

Zero-ablating a whole tap on a serial path cuts every downstream signal and
makes t_need trivially the first tap. So the ablation is
a_s -> (I - P_s) a_s with P_s frozen in discovery, preserving the complement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import solve_beta_shape
from .manifest import Manifest
from .stats import Row, paired_contrast, required_cluster_count
from .taps import Edit, Evo2Runner, project_out


@dataclass
class StageRow:
    locus: str
    stage: str
    energy: float
    reference_energy: float
    effective_rank: float
    carrier_occupancy: float
    recoverability_x28: Optional[float]
    causal_component_strength: float
    d_shape_on_component_removal: float
    d_shape_after_next_stage_rescue: Optional[float]
    d_shape_random_Ps: float
    missing_delta_norm: Optional[float]
    m30_energy: float
    g30_energy: float
    m30_carrier_occupancy: float
    g30_carrier_occupancy: float
    reference_effective_rank: Optional[float] = None
    energy_log_ratio: Optional[float] = None
    rank_log_ratio: Optional[float] = None
    direction_cosine_reference: Optional[float] = None


@dataclass(frozen=True)
class HCLAlgebraIdentity:
    """One runtime-checked equality in the stateless HCL reference graph."""
    name: str
    observed_stage: str
    reconstruct: Callable[[dict[str, Tensor]], Tensor]


@dataclass
class HCLReferenceTrace:
    tensors: dict[str, Tensor]
    relative_residuals: dict[str, float]


def standard_hcl_algebra() -> tuple[HCLAlgebraIdentity, ...]:
    """The three weight-free identities of the canonical HCL dataflow.

    Projection/FIR/convolution/output-projection identities require the
    loaded runtime weights and can be appended by the adapter.  These three
    must already hold whenever the canonical stage names are declared.
    """
    return (
        HCLAlgebraIdentity(
            "pregate_q=x1*v", "pregate_q",
            lambda t: t["fir_x1"] * t["fir_v"]),
        HCLAlgebraIdentity(
            "path_sum=long_conv+direct", "path_sum",
            lambda t: t["long_conv"] + t["direct"]),
        HCLAlgebraIdentity(
            "postgate=x2*path_sum", "postgate",
            lambda t: t["fir_x2"] * t["path_sum"]),
    )


def _relative_error(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).norm()
                 / a.double().norm().clamp_min(1e-30))


def _axis_occupancy(t: Tensor, axis: Optional[Tensor]) -> float:
    if axis is None:
        return float("nan")
    u = axis.double()
    u = u / u.norm().clamp_min(1e-30)
    x = t.double().reshape(-1, t.shape[-1])
    return float((x @ u).norm() / x.norm().clamp_min(1e-30))


def _assert_projector(P: Tensor, width: int, *, label: str) -> tuple[Tensor, int]:
    if P is None:
        raise ValueError(f"{label}: projector is missing")
    Pd = P.double()
    if Pd.shape != (width, width):
        raise ValueError(f"{label}: expected [{width},{width}] projector, got {tuple(Pd.shape)}")
    if not torch.isfinite(Pd).all():
        raise ValueError(f"{label}: non-finite projector")
    sym = float((Pd - Pd.T).norm())
    idem = float((Pd @ Pd - Pd).norm())
    rank = int(torch.linalg.matrix_rank(Pd))
    if sym >= 1e-7 or idem >= 1e-6 or not (0 < rank < width):
        raise ValueError(f"{label}: invalid non-trivial orthogonal projector "
                         f"(sym={sym:.2e}, idem={idem:.2e}, rank={rank})")
    return Pd, rank


def _project_out_axis(P: Tensor, feature_axis: int) -> Callable[[Tensor], Tensor]:
    """Project along an adapter-declared feature axis, not an assumed layout."""
    def fn(t: Tensor) -> Tensor:
        axis = feature_axis if feature_axis >= 0 else t.ndim + feature_axis
        if not (0 <= axis < t.ndim):
            raise ValueError(f"feature axis {feature_axis} invalid for shape {tuple(t.shape)}")
        x = t.movedim(axis, -1)
        Pd = P.to(x.dtype).to(x.device)
        if x.shape[-1] != Pd.shape[0]:
            raise ValueError(f"projector width {Pd.shape[0]} != feature width {x.shape[-1]}")
        y = x - x @ Pd.T @ Pd
        return y.movedim(-1, axis)
    return fn


def _eff_rank(X: Tensor) -> float:
    Xc = X.double() - X.double().mean(0, keepdim=True)
    S = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(S).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-300)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def trace_reference_path(
    runner: Evo2Runner,
    ids: Tensor,
    ell: int,
    stage_names: Sequence[str],
    *,
    identities: Sequence[HCLAlgebraIdentity] = (),
    tolerance: Optional[float] = None,
    locked: bool = False,
) -> HCLReferenceTrace:
    """Capture a production trace and verify an explicit reference algebra.

    The symbols in the plan are used only if they match the loaded Vortex
    implementation; otherwise the manifest's runtime names are used. The
    stateful cache path and the stateless/reference path must agree on the
    same input -- that check belongs in the Step 1 gate and is re-asserted here.
    """
    if tolerance is None:
        if locked:
            raise RuntimeError(
                "locked HCL algebra validation requires the Step-1 numerical "
                "identity tolerance"
            )
        tolerance = 1e-3  # discovery wiring diagnostic only; never a locked margin
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("HCL algebra tolerance must be finite and positive")
    taps = {N.hcl_stage(ell, s) for s in stage_names} | {
        N.m(ell), N.g(ell), N.x(ell), N.r(ell), N.x(ell + 1)}
    ts = runner.run(ids, taps=taps)
    got = {s: ts[N.hcl_stage(ell, s)] for s in stage_names if N.hcl_stage(ell, s) in ts.tensors}
    missing = sorted(set(stage_names) - set(got))
    if missing:
        raise AssertionError(f"HCL runtime did not emit required stages: {missing}")
    residuals = {}
    if "out_proj" in got and N.m(ell) in ts.tensors:
        err = _relative_error(ts[N.m(ell)], got["out_proj"])
        residuals["out_proj_equals_m"] = err
        if not err < tolerance:
            raise AssertionError(
                f"HCL stage reconstruction failed for block {ell}: "
                f"out_proj vs m_{ell} relative error {err:.3e}. The stage map in the "
                f"manifest does not match the runtime graph -- fix the manifest, do "
                f"not relabel the stages."
            )
    if all(k in ts.tensors for k in (N.x(ell), N.m(ell), N.r(ell))):
        err = _relative_error(ts[N.r(ell)], ts[N.x(ell)] + ts[N.m(ell)])
        residuals["x_plus_m_equals_r"] = err
        if err >= tolerance:
            raise AssertionError(f"x{ell}+m{ell}=r{ell} failed: {err:.3e}")
    if all(k in ts.tensors for k in (N.r(ell), N.g(ell), N.x(ell + 1))):
        err = _relative_error(ts[N.x(ell + 1)], ts[N.r(ell)] + ts[N.g(ell)])
        residuals["r_plus_g_equals_xnext"] = err
        if err >= tolerance:
            raise AssertionError(f"r{ell}+g{ell}=x{ell+1} failed: {err:.3e}")

    for identity in identities:
        if identity.observed_stage not in got:
            raise AssertionError(
                f"identity {identity.name!r} names absent stage {identity.observed_stage!r}")
        predicted = identity.reconstruct(got)
        err = _relative_error(got[identity.observed_stage], predicted)
        residuals[identity.name] = err
        if err >= tolerance:
            raise AssertionError(
                f"HCL algebra identity {identity.name!r} failed: relative error {err:.3e}")

    for key in (N.x(ell), N.m(ell), N.r(ell), N.g(ell), N.x(ell + 1)):
        if key in ts.tensors:
            got[key] = ts[key]
    return HCLReferenceTrace(got, residuals)


@torch.no_grad()
def reconstruct_reference_path(
    runner: Evo2Runner, ids: Tensor, ell: int, stage_names: Sequence[str]
) -> dict[str, Tensor]:
    """Backward-compatible tensor-only wrapper around ``trace_reference_path``."""
    return trace_reference_path(runner, ids, ell, stage_names).tensors


@torch.no_grad()
def localize_stages(
    runner: Evo2Runner,
    loci: Sequence[tuple[str, Tensor]],
    man: Manifest,
    *,
    ell: int,
    stage_names: Sequence[str],
    P_by_stage: dict[str, Tensor],              # frozen in discovery
    random_P_by_stage: dict[str, Tensor],       # rank- and dose-matched
    reference_ell: int,
    next_stage_component: Optional[Callable[[str, Tensor], Optional[Tensor]]] = None,
    carrier_axis: Optional[Tensor] = None,
    carrier_axis_by_stage: Optional[dict[str, Tensor]] = None,
    recoverability_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
    projector_rank_by_stage: Optional[dict[str, int]] = None,
    reference_stage_by_stage: Optional[dict[str, str]] = None,
    feature_axis_by_stage: Optional[dict[str, int]] = None,
    reference_feature_axis_by_stage: Optional[dict[str, int]] = None,
    identity_tolerance: Optional[float] = None,
) -> list[StageRow]:
    """Ablate only the frozen matched-dose component at each stage, then
    rescue with the NEXT stage's *missing delta*.

    ``next_stage_component`` is retained as a compatibility/audit hook.  If
    supplied, it must return the missing delta, not the full natural next-stage
    tensor, and is checked against the automatically measured delta.  The
    actual rescue always uses ``natural_next - ablated_next`` and therefore
    cannot accidentally double-dose the downstream stage.
    """
    man.margins.require(
        "beta_min", "beta_max", "flat_logit_threshold", "r_num_identity_residual"
    )
    identity_tolerance = (
        float(man.margins.r_num_identity_residual)
        if identity_tolerance is None else float(identity_tolerance)
    )
    if not np.isfinite(identity_tolerance) or identity_tolerance <= 0:
        raise ValueError("identity_tolerance must be finite and positive")
    rows: list[StageRow] = []

    for locus, ids in loci:
        base_taps = {N.hcl_stage(ell, s) for s in stage_names} | {
            N.m(ell), N.g(ell), N.x(28)}
        base = runner.run(ids, taps=base_taps)
        z0 = base.logits.reshape(-1, base.logits.shape[-1])[-1]
        ref_names = reference_stage_by_stage or {s: s for s in stage_names}
        missing_ref_names = sorted(set(stage_names) - set(ref_names))
        if missing_ref_names:
            raise ValueError(f"reference stage map missing {missing_ref_names}")
        ref_taps = {N.m(reference_ell)} | {
            N.hcl_stage(reference_ell, ref_names[s]) for s in stage_names}
        ref = runner.run(ids, taps=ref_taps)
        m30 = base[N.m(ell)]
        g30 = base[N.g(ell)]
        x28 = base.get(N.x(28))

        for s in stage_names:
            nm = N.hcl_stage(ell, s)
            if nm not in base.tensors:
                continue
            a_s = base[nm]
            axis = (feature_axis_by_stage or {}).get(s, -1)
            axis = axis if axis >= 0 else a_s.ndim + axis
            if not (0 <= axis < a_s.ndim):
                raise ValueError(f"{s}: feature axis invalid for shape {tuple(a_s.shape)}")
            a_view = a_s.movedim(axis, -1)
            flat = a_view.reshape(-1, a_view.shape[-1])

            def measure(edits: Sequence[Edit]) -> float:
                z = runner.run(ids, taps=(), edits=list(edits)).logits
                z = z.reshape(-1, z.shape[-1])[-1]
                return solve_beta_shape(z, z0, beta_min=man.margins.beta_min,
                                        beta_max=man.margins.beta_max,
                                        flat_logit_threshold=man.margins.flat_logit_threshold).d_shape

            P = P_by_stage.get(s)
            Pr = random_P_by_stage.get(s)
            if P is None or Pr is None:
                raise ValueError(f"{s}: both causal and random projectors are required")
            Pd, rank = _assert_projector(P, flat.shape[-1], label=f"{s}:P")
            Prd, random_rank = _assert_projector(Pr, flat.shape[-1], label=f"{s}:P_random")
            expected_rank = ((projector_rank_by_stage or {}).get(s, rank))
            if rank != expected_rank or random_rank != rank:
                raise ValueError(f"{s}: rank mismatch causal={rank}, random={random_rank}, "
                                 f"frozen={expected_rank}")
            ablate_edit = Edit(nm, "component", _project_out_axis(Pd, axis), f"(I-P) at {s}")
            d_true = measure([ablate_edit])
            d_rand = measure([Edit(nm, "component", _project_out_axis(Prd, axis),
                                   f"random P at {s}")])

            d_rescue = None
            nxt = stage_names[stage_names.index(s) + 1] if s != stage_names[-1] else None
            missing_norm = None
            if nxt is not None:
                nxt_nm = N.hcl_stage(ell, nxt)
                ablated_trace = runner.run(ids, taps={nxt_nm}, edits=[ablate_edit])
                missing = base[nxt_nm] - ablated_trace[nxt_nm]
                missing_norm = float(missing.double().norm())
                if next_stage_component is not None:
                    declared = next_stage_component(nxt, ids)
                    if declared is not None:
                        audit_err = _relative_error(missing, declared)
                        if audit_err >= identity_tolerance:
                            raise AssertionError(
                                f"{s}->{nxt}: callback is not the measured missing delta "
                                f"(relative error {audit_err:.3e}); a full natural component "
                                f"must never be used for rescue")
                d_rescue = measure([
                    ablate_edit,
                    Edit(nxt_nm, "component",
                         (lambda t, d=missing: t + d.to(t.dtype).to(t.device)),
                         f"missing-delta rescue at {nxt}"),
                ])

            stage_carrier = (carrier_axis_by_stage or {}).get(s, carrier_axis)
            if stage_carrier is not None and stage_carrier.numel() != a_view.shape[-1]:
                raise ValueError(
                    f"{s}: carrier axis width {stage_carrier.numel()} != stage feature "
                    f"width {a_view.shape[-1]}; supply a frozen mapped stage axis")
            car = _axis_occupancy(a_view, stage_carrier)
            recover = (float(recoverability_fn(a_s, x28))
                       if recoverability_fn is not None and x28 is not None else None)
            ref_stage = ref.get(N.hcl_stage(reference_ell, ref_names[s]))
            ref_energy = (float(ref_stage.double().norm()) if ref_stage is not None
                          else float("nan"))
            ref_view = None
            if ref_stage is not None:
                ref_axis = (reference_feature_axis_by_stage or {}).get(
                    s, (feature_axis_by_stage or {}).get(s, -1))
                ref_axis = ref_axis if ref_axis >= 0 else ref_stage.ndim + ref_axis
                if not (0 <= ref_axis < ref_stage.ndim):
                    raise ValueError(f"{s}: reference feature axis invalid")
                ref_view = ref_stage.movedim(ref_axis, -1)
            ref_rank = (_eff_rank(ref_view.reshape(-1, ref_view.shape[-1]))
                        if ref_view is not None else None)
            ref_cos = None
            if ref_view is not None and ref_view.shape == a_view.shape:
                aa, bb = a_view.double().flatten(), ref_view.double().flatten()
                ref_cos = float(torch.dot(aa, bb)
                                / (aa.norm() * bb.norm()).clamp_min(1e-30))
            rank_now = _eff_rank(flat)
            rows.append(StageRow(
                locus=locus, stage=s, energy=float(a_s.double().norm()),
                reference_energy=ref_energy,
                effective_rank=rank_now,
                carrier_occupancy=car, recoverability_x28=recover,
                causal_component_strength=float((flat @ Pd.T.to(flat.dtype)).norm()),
                d_shape_on_component_removal=d_true,
                d_shape_after_next_stage_rescue=d_rescue,
                d_shape_random_Ps=d_rand,
                missing_delta_norm=missing_norm,
                m30_energy=float(m30.double().norm()),
                g30_energy=float(g30.double().norm()),
                m30_carrier_occupancy=_axis_occupancy(m30, carrier_axis),
                g30_carrier_occupancy=_axis_occupancy(g30, carrier_axis),
                reference_effective_rank=ref_rank,
                energy_log_ratio=(float(np.log(max(float(a_s.double().norm()), 1e-300)
                                                 / max(ref_energy, 1e-300)))
                                  if np.isfinite(ref_energy) else None),
                rank_log_ratio=(float(np.log(max(rank_now, 1e-300)
                                               / max(ref_rank, 1e-300)))
                                if ref_rank is not None else None),
                direction_cosine_reference=ref_cos))
    return rows


def amp_and_need(rows: Sequence[StageRow], stage_names: Sequence[str],
                 reference_energy: dict[str, float], *,
                 man: Optional[Manifest] = None,
                 amp_log_margin: Optional[float] = None,
                 need_margin: Optional[float] = None,
                 rescue_margin: Optional[float] = None,
                 random_specificity_margin: Optional[float] = None,
                 bottleneck_margin: Optional[float] = None,
                 rotation_margin: Optional[float] = None,
                 rank_equivalence_margin: Optional[float] = None,
                 min_clusters: Optional[int] = None,
                 n_boot: int = 10_000,
                 alpha: float = 0.05) -> dict:
    """t_amp and t_need, reported as two separate stages with the evidence
    for each. The plan forbids assuming they coincide, and forbids dropping
    g_30 just because EXP1 found the b30 MLP small -- that is a prior, not a
    result."""
    names = {
        "amp_log_margin": "hcl_amp_log",
        "need_margin": "hcl_need",
        "rescue_margin": "hcl_rescue",
        "random_specificity_margin": "hcl_random_specificity",
        "bottleneck_margin": "hcl_bottleneck",
        "rotation_margin": "hcl_rotation",
        "rank_equivalence_margin": "hcl_rank_equivalence",
    }
    supplied = {
        "amp_log_margin": amp_log_margin,
        "need_margin": need_margin,
        "rescue_margin": rescue_margin,
        "random_specificity_margin": random_specificity_margin,
        "bottleneck_margin": bottleneck_margin,
        "rotation_margin": rotation_margin,
        "rank_equivalence_margin": rank_equivalence_margin,
    }
    if man is not None:
        man.margins.require("delta_spec")
    resolved: dict[str, float] = {}
    for arg_name, spec_name in names.items():
        value = supplied[arg_name]
        if value is None and man is not None:
            value = man.margins.delta_spec.get(spec_name)
        if value is None:
            raise RuntimeError(
                f"Step 8 needs a development-calibrated {arg_name} (delta_spec[{spec_name!r}])"
            )
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{arg_name} must be finite and non-negative")
        resolved[arg_name] = value
    amp_log_margin = resolved["amp_log_margin"]
    need_margin = resolved["need_margin"]
    rescue_margin = resolved["rescue_margin"]
    random_specificity_margin = resolved["random_specificity_margin"]
    bottleneck_margin = resolved["bottleneck_margin"]
    rotation_margin = resolved["rotation_margin"]
    rank_equivalence_margin = resolved["rank_equivalence_margin"]
    if min_clusters is None:
        if man is None:
            raise RuntimeError(
                "Step 8 needs min_clusters explicitly or a calibrated manifest "
                "power key 'step8_hcl'"
            )
        min_clusters = required_cluster_count(man, "step8_hcl")

    by_stage: dict[str, list[StageRow]] = {}
    for r in rows:
        by_stage.setdefault(r.stage, []).append(r)

    if not rows:
        raise RuntimeError("no stage rows")
    # Family-wise stage localisation: every per-stage interval uses a
    # Bonferroni-adjusted alpha.  This is conservative and transparent; it
    # prevents the earliest noisy stage from winning merely because many
    # serial taps were inspected.
    adjusted_alpha = alpha / max(len(stage_names), 1)
    evidence = {}
    t_amp = None
    for s in stage_names:
        rs = by_stage.get(s, [])
        if not rs:
            continue
        ref = reference_energy.get(s)
        if ref is None or ref <= 0:
            raise ValueError(f"missing positive matched reference energy for stage {s}")
        amp_rows = [Row(
            np.log(max(r.energy, 1e-300)
                   / max(r.reference_energy
                         if np.isfinite(r.reference_energy) else ref, 1e-300)),
            keys=(f"locus:{r.locus}",)) for r in rs]
        amp_zero = [Row(0.0, keys=x.keys) for x in amp_rows]
        amp_est = paired_contrast(amp_rows, amp_zero, min_clusters=min_clusters,
                                  n_boot=n_boot, alpha=adjusted_alpha)
        evidence.setdefault(s, {})["amplification"] = amp_est.as_row()
        rank_rows = [Row(-float(r.rank_log_ratio), keys=(f"locus:{r.locus}",))
                     for r in rs if r.rank_log_ratio is not None
                     and np.isfinite(r.rank_log_ratio)]
        rot_rows = [Row(1.0 - abs(float(r.direction_cosine_reference)),
                        keys=(f"locus:{r.locus}",))
                    for r in rs if r.direction_cosine_reference is not None
                    and np.isfinite(r.direction_cosine_reference)]
        bottleneck_est = rotation_est = None
        if rank_rows:
            bottleneck_est = paired_contrast(
                rank_rows, [Row(0.0, keys=x.keys) for x in rank_rows],
                min_clusters=min_clusters, n_boot=n_boot, alpha=adjusted_alpha)
            evidence[s]["rank_bottleneck"] = bottleneck_est.as_row()
        if rot_rows:
            rotation_est = paired_contrast(
                rot_rows, [Row(0.0, keys=x.keys) for x in rot_rows],
                min_clusters=min_clusters, n_boot=n_boot, alpha=adjusted_alpha)
            evidence[s]["direction_rotation"] = rotation_est.as_row()
        bottleneck = bottleneck_est is not None and bottleneck_est.lo > bottleneck_margin
        rotation = rotation_est is not None and rotation_est.lo > rotation_margin
        rank_equiv = (bottleneck_est is not None
                      and bottleneck_est.lo > -rank_equivalence_margin
                      and bottleneck_est.hi < rank_equivalence_margin)
        evidence[s]["transformation_class"] = (
            "mixed bottleneck+rotation" if bottleneck and rotation else
            "bottleneck" if bottleneck else
            "rotation" if rotation and rank_equiv else
            "undetermined")
        if amp_est.lo > amp_log_margin and t_amp is None:
            t_amp = s

    t_need = None
    for s in stage_names:
        rs = by_stage.get(s, [])
        if not rs:
            continue
        usable = [r for r in rs if r.d_shape_after_next_stage_rescue is not None
                  and np.isfinite(r.d_shape_on_component_removal)
                  and np.isfinite(r.d_shape_random_Ps)]
        if not usable:
            evidence.setdefault(s, {})["need"] = {"passed": False,
                                                    "reason": "no rescuable rows"}
            continue
        removal = [Row(r.d_shape_on_component_removal,
                       keys=(f"locus:{r.locus}",)) for r in usable]
        zero = [Row(0.0, keys=x.keys) for x in removal]
        random = [Row(r.d_shape_random_Ps,
                      keys=(f"locus:{r.locus}",)) for r in usable]
        rescued = [Row(float(r.d_shape_after_next_stage_rescue),
                       keys=(f"locus:{r.locus}",)) for r in usable]
        e_need = paired_contrast(removal, zero, min_clusters=min_clusters,
                                 n_boot=n_boot, alpha=adjusted_alpha)
        e_spec = paired_contrast(removal, random, min_clusters=min_clusters,
                                 n_boot=n_boot, alpha=adjusted_alpha)
        e_resc = paired_contrast(removal, rescued, min_clusters=min_clusters,
                                 n_boot=n_boot, alpha=adjusted_alpha)
        passed = (e_need.lo > need_margin
                  and e_spec.lo > random_specificity_margin
                  and e_resc.lo > rescue_margin)
        evidence.setdefault(s, {})["need"] = {
            "removal_vs_zero": e_need.as_row(),
            "removal_vs_random": e_spec.as_row(),
            "removal_vs_missing_delta_rescue": e_resc.as_row(),
            "passed": passed,
        }
        if passed and t_need is None:
            t_need = s

    per_unit = [{
        "locus": r.locus, "stage": r.stage,
        "energy": r.energy, "reference_energy": r.reference_energy,
        "effective_rank": r.effective_rank,
        "reference_effective_rank": r.reference_effective_rank,
        "direction_cosine_reference": r.direction_cosine_reference,
        "recoverability_x28": r.recoverability_x28,
        "component_removal_d_shape": r.d_shape_on_component_removal,
        "missing_delta_rescue_d_shape": r.d_shape_after_next_stage_rescue,
        "random_subspace_d_shape": r.d_shape_random_Ps,
        "m30_energy": r.m30_energy, "g30_energy": r.g30_energy,
    } for r in rows]
    return {"t_amp": t_amp, "t_need": t_need,
            "coincide": (t_amp is not None and t_amp == t_need),
            "stage_evidence": evidence,
            "per_unit": per_unit,
            "multiplicity": {"method": "Bonferroni over ordered HCL stages",
                             "family_alpha": alpha,
                             "per_stage_alpha": adjusted_alpha},
            "margins": {"amp_log": amp_log_margin, "need": need_margin,
                        "rescue": rescue_margin,
                        "random_specificity": random_specificity_margin,
                        "bottleneck": bottleneck_margin,
                        "rotation": rotation_margin,
                        "rank_equivalence": rank_equivalence_margin},
            "reading": {
                "same": "causal amplification candidate",
                "amp before need": "scaffold, then readout coupling",
                "need without amp": "selective re-encoding",
                "no ablation matters": "output-null transformation",
            },
            "note": "m30 and g30 are both measured; EXP1's small b30 MLP is a prior"}


@torch.no_grad()
def input_vs_module_2x2(runner: Evo2Runner, ids: Tensor, *, pair: tuple[int, int],
                        common_radius: bool = True) -> dict:
    """Step 8-3 secondary diagnostic: b27 HCL input/module x b30 input/module.

    Off the production trajectory, so never a primary proof -- it only narrows
    the origin to unusual input, learned module, or interaction.
    """
    from .step7_bilinear import _factorial_effects
    e, l = pair
    ts = runner.run(ids, taps={N.x(e), N.x(l)})
    xe, xl = ts[N.x(e)], ts[N.x(l)]
    if common_radius:
        rad = 0.5 * (xe.norm(dim=-1, keepdim=True) + xl.norm(dim=-1, keepdim=True))
        xe = xe / xe.norm(dim=-1, keepdim=True).clamp_min(1e-30) * rad
        xl = xl / xl.norm(dim=-1, keepdim=True).clamp_min(1e-30) * rad
    cells = {}
    for iname, xin in (("early", xe), ("late", xl)):
        for mname, mell in (("early", e), ("late", l)):
            y = runner.a.mixer(mell)(runner.a.pre_norm(mell)(xin))
            y = y[0] if isinstance(y, (tuple, list)) else y
            cells[f"input={iname},module={mname}"] = float(y.double().norm())
    out = _factorial_effects(cells)
    out["caveat"] = ("secondary diagnostic only; leaves the production trajectory. "
                     "Natural-mechanism evidence is the in-run stage ablation-rescue.")
    return out


def lag_band_plan(first_causal_stage: Optional[str]) -> dict:
    """Step 8-4: expand into lag bands ONLY if the long-convolution / direct
    path is where the causal effect first appears. No up-front spectral sweep.
    """
    gate = first_causal_stage in {"long_conv", "direct", "path_sum"}
    return {
        "expand": gate,
        "bands_bp": [(0, 2), (3, 31), (32, 127), (128, None)] if gate else [],
        "note": "finer log-spaced bins / frequency / phase response only if needed",
    }


# --------------------------------------------------------------------------
# 8-5 mediator candidates -- the two kinds are never mixed
# --------------------------------------------------------------------------


def make_x31_projector(basis: Tensor) -> Tensor:
    """Type A. Orthonormalise a discovery-fixed span in x31 space and verify
    P = P^T = P^2 numerically."""
    Q, _ = torch.linalg.qr(basis.double())
    P = Q @ Q.T
    sym = float((P - P.T).norm())
    idem = float((P @ P - P).norm())
    if not (sym < 1e-8 and idem < 1e-6):
        raise AssertionError(f"P is not an orthogonal projector: ||P-P^T||={sym:.2e}, "
                             f"||P^2-P||={idem:.2e}")
    if float(torch.linalg.matrix_rank(P)) >= P.shape[0]:
        raise AssertionError("P = I is forbidden (Step 9-1)")
    return P


def internal_component_mediator_note() -> str:
    """Type B. HCL internal stage tensors and lag-band sums are NOT x31
    projectors. They are blocked / patched as additive components a_s at their
    own tap, with the rest of the computation re-run. To be promoted to a
    type-A projector, the component's contribution to x31 must first be mapped
    and orthonormalised in discovery and frozen before locked analysis.
    """
    return internal_component_mediator_note.__doc__
