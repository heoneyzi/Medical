"""Step 3 — final scale sanity and the x24–x31 radial sensitivity map.

3A asks whether the FINAL readout discards whole-state scale.
3B asks where in the late stack scale is still being used.
Those are different questions and the plan keeps them apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import solve_beta_shape
from .manifest import Manifest
from .stats import (
    Row, equivalence, family_cluster_bootstrap, paired_contrast,
    paired_difference_rows, positive, required_cluster_count,
    simultaneous_equivalence,
)
from .taps import Edit, Evo2Runner


@dataclass
class RadialPoint:
    tap: str
    dlog_r: float
    d_shape: float
    log_beta: float
    nll: Optional[float]
    branch_over_residual: dict[str, float] = field(default_factory=dict)
    downstream_direction_change: dict[str, float] = field(default_factory=dict)
    natural_range: bool = True
    unit_id: str = ""
    dose_id: str = ""
    keys: tuple[str, ...] = ()
    windows: tuple[tuple[str, int, int], ...] = ()
    stratum: str = "all"
    beta_identified: bool = True
    boundary_censored: bool = False


def _scale_state(factor: float):
    return lambda t: t * factor


@torch.no_grad()
def final_scale_sanity(
    runner: Evo2Runner,
    windows: Sequence[Tensor],
    man: Manifest,
    *,
    natural_alphas: Sequence[float],
    stress_alphas: Sequence[float] = (0.01, 100.0),
    unit_metadata: Optional[Sequence[dict]] = None,
) -> dict:
    """3A. h' = alpha h immediately before the final RMSNorm.

    Primary is the SIMULTANEOUS upper CI of max D_shape over the natural
    range; the extreme alphas are stress tests and are reported separately.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    out = {"natural": [], "stress": []}
    if unit_metadata is not None and len(unit_metadata) != len(windows):
        raise ValueError("unit_metadata must align one-to-one with windows")
    metadata = unit_metadata or [{} for _ in windows]
    for wi, ids in enumerate(windows):
        md = metadata[wi]
        uid = str(md.get("unit_id", f"window:{wi}"))
        base = runner.run(ids, taps=()).logits
        z0 = base.reshape(-1, base.shape[-1])[-1]
        for kind, alphas in (("natural", natural_alphas), ("stress", stress_alphas)):
            for a in alphas:
                e = Edit(N.FINAL_PRE_NORM, "state", _scale_state(a), f"final scale {a}")
                z1 = runner.run(ids, taps=(), edits=[e]).logits
                z1 = z1.reshape(-1, z1.shape[-1])[-1]
                r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                     beta_max=man.margins.beta_max,
                                     flat_logit_threshold=man.margins.flat_logit_threshold)
                out[kind].append({"unit_id": uid, "alpha": a,
                                  "keys": tuple(md.get("keys", (f"unit:{uid}",))),
                                  "windows": tuple(md.get("windows", ())),
                                  "stratum": str(md.get("stratum", "all")),
                                  "d_shape": r.d_shape, "log_beta": r.log_beta,
                                  "censored": r.boundary_censored,
                                  "beta_identified": r.beta_identified})
    return out


def final_scale_verdict(records: dict, man: Manifest, *, seed: int = 42,
                        n_boot: int = 10_000) -> dict:
    """Locked, simultaneous test of final-RMSNorm scale invariance.

    Every natural alpha must be equivalent on D_shape.  Stress doses are
    diagnostics only and cannot rescue or defeat the locked claim.
    """
    man.margins.require("delta_shape")
    required = required_cluster_count(man, "step3_final_scale")
    fam: dict[str, list[Row]] = {}
    natural = records.get("natural", [])
    for r in natural:
        if not r.get("beta_identified", True) or r.get("censored", False):
            continue
        uid = str(r["unit_id"])
        fam.setdefault(f"alpha={r['alpha']}", []).append(Row(
            float(r["d_shape"]), tuple(r.get("keys", (f"unit:{uid}",))),
            windows=tuple(r.get("windows", ())), stratum=r.get("stratum", "all"),
            unit_id=uid,
        ))
    ests = family_cluster_bootstrap(fam, seed=seed, n_boot=n_boot, required=required)
    eq = simultaneous_equivalence(ests, man.margins.delta_shape)
    return {
        "simultaneous": {k: v.estimate.as_row() for k, v in eq.items()},
        "per_alpha_equivalent": {k: v.verdict == "equivalent" for k, v in eq.items()},
        "final_scale_invariant_in_natural_range": bool(eq) and all(
            v.verdict == "equivalent" for v in eq.values()),
        "stress": records.get("stress", []),
        "diagnostics": {
            "beta_unidentified": sum(not r.get("beta_identified", True) for r in natural),
            "boundary_censored": sum(bool(r.get("censored", False)) for r in natural),
            "n_natural_rows": len(natural),
        },
    }


@torch.no_grad()
def late_layer_map(
    runner: Evo2Runner,
    windows: Sequence[Tensor],
    man: Manifest,
    *,
    coarse_taps: Sequence[str],
    fine_taps: Sequence[str],
    natural_radius_quantiles: dict[str, dict[str, float]],
    downstream_blocks: Sequence[int],
    unit_metadata: Optional[Sequence[dict]] = None,
) -> list[RadialPoint]:
    """3B. Direction held fixed, radius set to the tap's own natural q25/q50/q75.

    Using the tap's natural quantiles -- rather than arbitrary multipliers --
    is what keeps this a natural-range intervention. Coarse taps are block
    inputs x24..x31 with x0/x6/x12/x18 as early sentinels; the fine map runs
    x28 -> r28 -> x29 -> r29 -> x30 -> r30 -> x31 -> r31 -> x32, and ONLY on
    that fine map may one say "after the b28 mixer" or "after the b30 HCL".
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    pts: list[RadialPoint] = []
    all_taps = list(dict.fromkeys(list(coarse_taps) + list(fine_taps)))

    metadata = unit_metadata or [{} for _ in windows]
    if len(metadata) != len(windows):
        raise ValueError("unit_metadata must align one-to-one with windows")
    for wi, ids in enumerate(windows):
        md = metadata[wi]
        uid = str(md.get("unit_id", f"window:{wi}"))
        want = set(all_taps) | {N.m(l) for l in downstream_blocks} | {N.g(l) for l in downstream_blocks} \
               | {N.x(l) for l in downstream_blocks} | {N.r(l) for l in downstream_blocks}
        base = runner.run(ids, taps=want)
        z0 = base.logits.reshape(-1, base.logits.shape[-1])[-1]

        for tap in all_taps:
            if tap not in base.tensors:
                continue
            h = base[tap]
            cur = float(h.reshape(-1, h.shape[-1])[-1].norm())
            qs = natural_radius_quantiles.get(tap)
            if qs is None:
                continue
            for qname, target in qs.items():
                factor = target / max(cur, 1e-30)
                if tap.startswith("m") or tap.startswith("g"):
                    kind = "update"
                elif tap.startswith("hcl"):
                    kind = "component"
                else:
                    kind = "state"
                e = Edit(tap, kind, _scale_state(factor), f"{tap} radius -> {qname}")
                ts = runner.run(ids, taps=want, edits=[e])
                z1 = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
                r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                     beta_max=man.margins.beta_max,
                                     flat_logit_threshold=man.margins.flat_logit_threshold)
                bor, dirchg = {}, {}
                for l in downstream_blocks:
                    if N.m(l) in ts.tensors and N.x(l) in ts.tensors:
                        bor[f"m{l}/x{l}"] = float(
                            ts[N.m(l)].norm() / ts[N.x(l)].norm().clamp_min(1e-30))
                    if N.g(l) in ts.tensors and N.r(l) in ts.tensors:
                        bor[f"g{l}/r{l}"] = float(
                            ts[N.g(l)].norm() / ts[N.r(l)].norm().clamp_min(1e-30))
                    if N.x(l) in ts.tensors and N.x(l) in base.tensors:
                        a_, b_ = ts[N.x(l)].flatten().double(), base[N.x(l)].flatten().double()
                        dirchg[f"x{l}"] = float(
                            1 - torch.dot(a_, b_) / (a_.norm() * b_.norm()).clamp_min(1e-30))
                pts.append(RadialPoint(
                    tap=tap, dlog_r=float(np.log(max(factor, 1e-30))), d_shape=r.d_shape,
                    log_beta=r.log_beta, nll=None,
                    branch_over_residual=bor, downstream_direction_change=dirchg,
                    unit_id=uid, dose_id=str(qname),
                    keys=tuple(md.get("keys", (f"unit:{uid}",))),
                    windows=tuple(md.get("windows", ())),
                    stratum=str(md.get("stratum", "all")),
                    beta_identified=r.beta_identified,
                    boundary_censored=r.boundary_censored))
    return pts


def radial_sensitivity(points: Sequence[RadialPoint], tap: str) -> float:
    """S_tap = slope of D_shape against |dlog r| (plan's 'sensitivity').

    Reported with its CI by `primary_contrast`; the change-point over taps is
    SECONDARY (plan Step 3, "분석").
    """
    sel = [p for p in points if p.tap == tap]
    if len(sel) < 2:
        return float("nan")
    x = np.abs([p.dlog_r for p in sel])
    y = np.array([p.d_shape for p in sel])
    if x.std() < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def primary_contrast(
    points: Sequence[RadialPoint],
    *,
    tap_a: str = N.x(28),
    tap_b: str = N.x(31),
    seed: int = 42,
    man: Optional[Manifest] = None,
    n_boot: int = 10_000,
) -> dict:
    """The plan's primary: radial_sensitivity(x28) - radial_sensitivity(x31).

    x28 is immediately before b28; x31 is immediately after the whole b30
    block and immediately before b31. This contrast is scoped to the b28-b30
    handoff ONLY. x31 vs x32 (b31's extra scale use) and x32 vs final readout
    (terminal RMSNorm redundancy) are SEPARATE contrasts, computed the same
    way, and must be reported as such.
    """
    if man is None:
        raise RuntimeError("primary_contrast is confirmatory and requires the sealed Manifest")
    required = required_cluster_count(man, "step3_primary")
    slopes = per_unit_radial_slopes(points)
    ra, rb = slopes.get(tap_a, []), slopes.get(tap_b, [])
    est = paired_contrast(ra, rb, seed=seed, n_boot=n_boot, required=required)
    return {"contrast": f"S({tap_a}) - S({tap_b})", "estimate": est.as_row(),
            "scope": "b28-b30 handoff only; x31/x32 and x32/final are separate contrasts"}


def per_unit_radial_slopes(
    points: Sequence[RadialPoint], *, endpoint: str = "d_shape",
) -> dict[str, list[Row]]:
    """Fit dose response within unit first; loci, not doses, are replicates."""
    grouped: dict[tuple[str, str], list[RadialPoint]] = {}
    for p in points:
        if p.natural_range and p.beta_identified:
            grouped.setdefault((p.tap, p.unit_id), []).append(p)
    out: dict[str, list[Row]] = {}
    for (tap, uid), ps in grouped.items():
        x = np.abs(np.asarray([p.dlog_r for p in ps], dtype=float))
        y = np.asarray([getattr(p, endpoint) for p in ps], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2 or x[ok].std() < 1e-12:
            continue
        slope = float(np.polyfit(x[ok], y[ok], 1)[0])
        p0 = ps[0]
        out.setdefault(tap, []).append(Row(
            slope, keys=p0.keys or (f"unit:{uid}",), windows=p0.windows,
            stratum=p0.stratum, unit_id=uid,
            meta={"tap": tap, "endpoint": endpoint, "n_doses": int(ok.sum())},
        ))
    return out


def _margin(man: Manifest, name: str, fallback: Optional[float] = None) -> float:
    value = man.margins.delta_spec.get(name, fallback)
    if value is None:
        raise RuntimeError(
            f"missing development-calibrated delta_spec[{name!r}] for Step 3 onset"
        )
    return float(value)


def _row_payload(r: Row) -> dict:
    return {"unit_id": r.unit_id, "value": r.value, "keys": list(r.keys),
            "windows": [list(w) for w in r.all_windows()],
            "stratum": r.stratum, "meta": r.meta}


def onset_verdict(
    points: Sequence[RadialPoint],
    man: Manifest,
    *,
    # Legacy EXP1 h24..h27 are canonical x25..x28.  Therefore the legacy
    # h27->h28 transition is exactly x28->x29, not x27->x28.
    pre_taps: Sequence[str] = (N.x(25), N.x(26), N.x(27), N.x(28)),
    boundary: tuple[str, str] = (N.x(28), N.x(29)),
    b28_state_transition: tuple[str, str] = (N.x(28), N.x(29)),
    serial_fine_taps: Sequence[str] = (
        N.x(28), N.r(28), N.x(29), N.r(29), N.x(30), N.r(30),
        N.x(31), N.r(31), N.x(32),
    ),
    seed: int = 42,
    n_boot: int = 10_000,
) -> dict:
    """Locked b24--b31 onset decision with exact within-block localisation.

    The legacy layer-boundary question and the exact block question are both
    returned.  EXP1 h27->h28 is resolved before analysis to x28->x29; its
    m28/r28/g28 fine map then says where inside b28 the change is written.
    """
    man.margins.require("delta_shape", "delta_spec")
    expected_alias = {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)}
    bad_alias = {k: (man.arch.legacy_alias.get(k), v) for k, v in expected_alias.items()
                 if man.arch.legacy_alias.get(k) != v}
    if bad_alias:
        raise RuntimeError(
            f"EXP1 legacy aliases are missing/off-by-one: {bad_alias}; expected "
            "h27=x28, h28=x29, h30=x31 before any bridge claim"
        )
    required = required_cluster_count(man, "step3_onset")
    eq_margin = _margin(man, "radial_sensitivity_equivalence", man.margins.delta_shape)
    pos_margin = _margin(man, "onset_positive", 0.0)
    hom_margin = _margin(man, "homologous_superiority", 0.0)
    slopes = per_unit_radial_slopes(points)

    missing = [t for t in set(pre_taps) | set(boundary) | set(b28_state_transition)
               if t not in slopes]
    if missing:
        raise RuntimeError(f"Step 3 onset map is missing per-unit slopes for {sorted(missing)}")

    contrast_rows: dict[str, list[Row]] = {}
    pre_names: list[str] = []
    for a, b in zip(pre_taps[:-1], pre_taps[1:]):
        name = f"pre::{b}-{a}"
        contrast_rows[name] = paired_difference_rows(slopes[b], slopes[a])
        pre_names.append(name)
    bname = f"boundary::{boundary[1]}-{boundary[0]}"
    contrast_rows[bname] = paired_difference_rows(slopes[boundary[1]], slopes[boundary[0]])
    wname = f"b28_write::{b28_state_transition[1]}-{b28_state_transition[0]}"
    contrast_rows[wname] = paired_difference_rows(
        slopes[b28_state_transition[1]], slopes[b28_state_transition[0]])

    # Same-phase earlier blocks control whether b28 is exceptional rather
    # than merely another periodic HCS block.  Compare whole-block radial
    # change on matched units, one control at a time (IUT logic).
    hom_names: list[str] = []
    b28_effect = contrast_rows[wname]
    for earlier, later in man.arch.homologous_pairs:
        if later != 28:
            continue
        a, b = N.x(earlier), N.x(earlier + 1)
        if a not in slopes or b not in slopes:
            raise RuntimeError(f"homologous control requires {a} and {b} in the radial map")
        ctrl = paired_difference_rows(slopes[b], slopes[a])
        name = f"homolog::{wname}-({b}-{a})"
        contrast_rows[name] = paired_difference_rows(b28_effect, ctrl)
        hom_names.append(name)

    # Branch taps are reported explicitly.  They are not inserted into the
    # serial change-point order because m_l/g_l are side branches, not states.
    branch_names: list[str] = []
    for tap in (N.m(28), N.g(28), N.m(29), N.g(29),
                N.m(30), N.g(30), N.m(31), N.g(31)):
        if tap in slopes:
            name = f"branch::{tap}-{N.x(int(tap[1:]))}"
            base = N.x(int(tap[1:]))
            if base in slopes:
                contrast_rows[name] = paired_difference_rows(slopes[tap], slopes[base])
                branch_names.append(name)

    estimates = family_cluster_bootstrap(
        contrast_rows, seed=seed, n_boot=n_boot, required=required)
    pre_eq = simultaneous_equivalence(
        {k: estimates[k] for k in pre_names}, eq_margin)
    pre_pass = bool(pre_eq) and all(v.verdict == "equivalent" for v in pre_eq.values())
    boundary_pass = positive(estimates[bname], pos_margin)
    write_pass = positive(estimates[wname], pos_margin)
    hom_pass = bool(hom_names) and all(positive(estimates[k], hom_margin) for k in hom_names)

    # Earliest serial state transition under the same simultaneous family.
    serial_candidates: dict[str, list[Row]] = {}
    serial_order = list(pre_taps[:-1]) + list(serial_fine_taps)
    # De-duplicate while preserving the pre-registered causal order.
    serial_order = list(dict.fromkeys(serial_order))
    available = [t for t in serial_order if t in slopes]
    for a, b in zip(available[:-1], available[1:]):
        serial_candidates[f"serial::{b}-{a}"] = paired_difference_rows(slopes[b], slopes[a])
    serial_est = family_cluster_bootstrap(
        serial_candidates, seed=seed + 1000, n_boot=n_boot, required=required
    ) if serial_candidates else {}
    earliest = next((name for name, est in serial_est.items() if positive(est, pos_margin)), None)

    exact_attr = {
        name: {"estimate": estimates[name].as_row(),
               "positive": positive(estimates[name], pos_margin)}
        for name in branch_names
    }
    b28_branch = [v["positive"] for k, v in exact_attr.items()
                  if k.startswith("branch::m28-") or k.startswith("branch::g28-")]
    branch_localized = bool(b28_branch) and any(b28_branch)
    return {
        "per_unit_slope_counts": {k: len(v) for k, v in slopes.items()},
        "simultaneous_estimates": {k: v.as_row() for k, v in estimates.items()},
        "pre_b28_equivalence": {k: v.verdict for k, v in pre_eq.items()},
        "legacy_boundary_h27_to_h28__canonical_x28_to_x29": boundary_pass,
        "exact_b28_state_write_x28_to_x29": write_pass,
        "exact_b28_branch_localized_m28_or_g28": branch_localized,
        "homologous_controls_all_pass": hom_pass,
        "branch_attribution": exact_attr,
        "earliest_serial_change": earliest,
        "verdict": (
            "b28-onset-with-exact-write" if pre_pass and boundary_pass and write_pass
            and branch_localized and hom_pass
            else "no-locked-b28-onset"
        ),
        "interpretation_guard": (
            "EXP1 h27=x28 and h28=x29. Never substitute x27->x28 for the legacy "
            "h27->h28 contrast; m28/r28/g28 localise the within-b28 write."
        ),
        "exp1_bridge": {
            "legacy_alias": expected_alias,
            "estimand": "per-unit natural-radius D_shape slope",
            "keyed_rows": {tap: [_row_payload(r) for r in rs]
                           for tap, rs in slopes.items()},
            "use": "join to EXP1 facts by unit_id/dependency keys; never by row order",
        },
    }
