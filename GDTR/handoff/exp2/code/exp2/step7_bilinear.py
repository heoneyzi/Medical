"""Step 7 — the exact operation that makes the b28/b29 branch explosion.

The distinction this module exists to draw: a BIG activation channel is not
the same thing as a channel the OUTPUT needs. |p_k| alone misses the
projection direction, so every channel is scored by ||v_k|| = ||W3[:,k] p_k||
together with its carrier / content projections and its downstream output
effect.

Selection happens in discovery only (Step 7-2). Development and locked never
re-choose channels; a per-locus rule is allowed only if the rule itself was
frozen in discovery and is cross-fitted.
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
from .stats import (
    Row, cluster_bootstrap, intersection_union, paired_contrast,
    required_cluster_count,
)
from .taps import Edit, Evo2Runner


# --------------------------------------------------------------------------
# 7-1 exact accounting
# --------------------------------------------------------------------------


@dataclass
class BilinearAudit:
    block: int
    norm_a: float
    norm_b: float
    norm_p: float
    norm_out: float
    reconstruction_err: float
    coactivation_excess: float          # ||a*b|| vs channel-permuted b
    projection_gain_G3: float           # ||W3 p|| / ||p||
    projection_gain_G3_permuted: float
    channel_norms: Tensor               # ||v_k||
    carrier_projection: Tensor          # per channel
    content_projection: Tensor
    first_anomaly: str                  # factor_a | factor_b | product | projection | unidentified
    anomaly_ratios: dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def bilinear_audit(
    runner: Evo2Runner,
    ell: int,
    z: Tensor,
    g_observed: Tensor,
    *,
    carrier_axis: Tensor,
    reference_block: Optional[int] = None,
    reference_z: Optional[Tensor] = None,
    anomaly_ratio_threshold: Optional[float] = None,
    seed: int = 42,
) -> BilinearAudit:
    """Factor -> product -> projection waterfall, with the two separations of
    Step 7-4A/B computed on the spot:

    A. co-activation excess: permute `b` across channels, keeping each
       factor's marginal distribution but destroying the joint tail.
    B. projection gain G3 = ||W3 p|| / ||p||, compared against a permuted p
       with the same norm and sparsity, to tell "the product was already big"
       apart from "W3 selectively amplifies this product direction".
    """
    parts = runner.bilinear_parts(ell, z)
    a, b, p, out, W3 = parts["a"], parts["b"], parts["p"], parts["out"], parts["W3"]
    v = Evo2Runner.channel_contributions(p, W3)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(b.shape[-1], generator=gen).to(b.device)
    p_perm = a * b[..., perm]
    coact = float(p.double().norm() / p_perm.double().norm().clamp_min(1e-30))

    G3 = float(out.double().norm() / p.double().norm().clamp_min(1e-30))
    pp = p.clone().reshape(-1, p.shape[-1])
    for i in range(pp.shape[0]):
        pp[i] = pp[i][torch.randperm(pp.shape[-1], generator=gen)]
    G3p = float((pp @ W3.T.to(pp.dtype)).double().norm() / pp.double().norm().clamp_min(1e-30))

    u = carrier_axis.double() / carrier_axis.double().norm().clamp_min(1e-30)
    W3d = W3.double()
    carrier_proj = (W3d.T @ u).abs() * p.double().reshape(-1, p.shape[-1])[-1].abs()
    content_proj = (v.double().reshape(-1, v.shape[-1])[-1] ** 2 - carrier_proj ** 2).clamp_min(0).sqrt()

    first = "unidentified"
    ratios: dict[str, float] = {}
    if reference_block is not None and reference_z is not None:
        ref = runner.bilinear_parts(reference_block, reference_z)
        ratios = {
            "factor_a": float(a.double().norm() / ref["a"].double().norm().clamp_min(1e-30)),
            "factor_b": float(b.double().norm() / ref["b"].double().norm().clamp_min(1e-30)),
            "product": float(p.double().norm() / ref["p"].double().norm().clamp_min(1e-30)),
            "projection": float(out.double().norm() / ref["out"].double().norm().clamp_min(1e-30)),
        }
        if anomaly_ratio_threshold is not None:
            if not np.isfinite(anomaly_ratio_threshold) or anomaly_ratio_threshold <= 1:
                raise ValueError(
                    "anomaly_ratio_threshold must be a development-calibrated value > 1"
                )
            order = ["factor_a", "factor_b", "product", "projection"]
            prev = 1.0
            first = "none"
            for k in order:
                if ratios[k] / max(prev, 1e-30) > anomaly_ratio_threshold:
                    first = k
                    break
                prev = ratios[k]

    return BilinearAudit(
        block=ell,
        norm_a=float(a.double().norm()), norm_b=float(b.double().norm()),
        norm_p=float(p.double().norm()), norm_out=float(out.double().norm()),
        reconstruction_err=float((out.double() - g_observed.double()).norm()
                                 / g_observed.double().norm().clamp_min(1e-30)),
        coactivation_excess=coact, projection_gain_G3=G3, projection_gain_G3_permuted=G3p,
        channel_norms=v.detach(), carrier_projection=carrier_proj.detach(),
        content_projection=content_proj.detach(), first_anomaly=first,
        anomaly_ratios=ratios,
    )


# --------------------------------------------------------------------------
# 7-2 selection (discovery only)
# --------------------------------------------------------------------------


def select_channels_discovery(
    audits: Sequence[BilinearAudit],
    *,
    K_ladder: Sequence[int],
    incremental_effect: Callable[[Sequence[int]], float],
    pool_size: int,
    calibrated_min_effect: Optional[float] = None,
    calibrated_increment_tolerance: Optional[float] = None,
    plateau_steps: int = 1,
) -> dict:
    """Build a broad candidate pool by contribution norm, profile it against
    carrier / content / nuisance, then take the minimal set by the incremental
    output effect of grouped ablation. K, tie-breaking and group ordering are
    all fixed HERE and never revisited.
    """
    if not audits:
        raise ValueError("channel selection needs at least one discovery audit")
    ladder = sorted({int(k) for k in K_ladder})
    if not ladder or ladder[0] <= 0 or ladder[-1] > pool_size:
        raise ValueError("K_ladder must be positive, increasing, and no larger than pool_size")
    if plateau_steps < 1:
        raise ValueError("plateau_steps must be >= 1")

    # torch.topk does not promise a deterministic tie order.  The explicit
    # lexicographic sort makes the frozen rule reproducible across devices.
    v = torch.stack([
        a.channel_norms.reshape(-1, a.channel_norms.shape[-1])[-1].double()
        for a in audits
    ]).mean(0)
    if ladder[-1] > v.numel():
        raise ValueError(f"K_ladder reaches {ladder[-1]} but bilinear width is {v.numel()}")
    order = sorted(range(v.numel()), key=lambda j: (-float(v[j]), j))
    pool = order[:min(pool_size, len(order))]

    effects = [float(incremental_effect(pool[:K])) for K in ladder]
    curve = []
    previous = 0.0
    for K, effect in zip(ladder, effects):
        curve.append({"K": K, "effect": effect,
                      "incremental_effect": effect - previous})
        previous = effect

    # Locked selection is permitted only with thresholds calibrated before
    # this discovery curve is evaluated.  In particular, the final/largest K
    # is never silently selected just because the loop ended there.
    calibrated = (calibrated_min_effect is not None
                  and calibrated_increment_tolerance is not None)
    chosen_K = None
    if calibrated:
        for i, (K, effect) in enumerate(zip(ladder, effects)):
            lookahead = effects[i + 1:i + 1 + plateau_steps]
            if len(lookahead) < plateau_steps:
                continue
            saturated = all(abs(nxt - effect) <= calibrated_increment_tolerance
                            for nxt in lookahead)
            if effect >= calibrated_min_effect and saturated:
                chosen_K = K
                break

    return {
        "pool": pool,
        "K_ladder": ladder,
        "incremental_curve": curve,
        "selected": pool[:chosen_K] if chosen_K is not None else [],
        "selected_K": chosen_K,
        "selected_fraction_of_width": (chosen_K / v.numel()
                                       if chosen_K is not None else None),
        "calibration": {
            "min_effect": calibrated_min_effect,
            "increment_tolerance": calibrated_increment_tolerance,
            "plateau_steps": plateau_steps,
        },
        "locked_ready": chosen_K is not None,
        "failure_reason": (None if chosen_K is not None else
                           "no K met the pre-calibrated effect-plus-plateau rule"),
        "tie_break": "descending mean ||v_k||, index ascending",
        "frozen": chosen_K is not None,
    }


# --------------------------------------------------------------------------
# 7-3 necessity ladder
# --------------------------------------------------------------------------


CONTROL_FAMILIES = (
    "energy_matched_random",      # same sum ||v_k||^2
    "carrier_projection_matched",
    "occupancy_sign_kurtosis_matched",
    "homologous_earlier_block",
    "wrong_token_or_shifted_position",
)


@dataclass
class NecessityRow:
    locus: str
    K: int
    family: str                   # "true" or one of CONTROL_FAMILIES
    d_shape: float
    donor_following: Optional[float]
    plateau_shift: Optional[float]
    branch_norm_drop: float       # reported, but NOT the primary endpoint
    position_scope: str = "all"   # target | context | all
    applied_position_scope: str = "all"
    control_match_passed: bool = True
    control_match: dict = field(default_factory=dict)


def _validate_channel_indices(idx: Sequence[int], *, K: int, width: int,
                              family: str) -> list[int]:
    out = [int(j) for j in idx]
    if len(out) != K:
        raise ValueError(f"{family}: requested K={K}, got {len(out)} channels")
    if len(set(out)) != len(out):
        raise ValueError(f"{family}: duplicate channel indices are forbidden")
    if any(j < 0 or j >= width for j in out):
        raise IndexError(f"{family}: channel index outside [0,{width})")
    return out


def validate_control_completeness(
    control_sets: dict[str, Callable[[int], Sequence[int]]],
    *,
    K_ladder: Sequence[int],
    width: int,
    true_channels: Sequence[int],
    match_validator: Optional[
        Callable[[str, int, Sequence[int], Sequence[int]], dict]
    ] = None,
    locked: bool = False,
) -> dict:
    """Validate the control panel before any model call.

    In locked mode a matching report is mandatory for every family.  The
    callback returns a dictionary containing at least ``passed: bool`` and
    may record energy/carrier/occupancy/sign/kurtosis/position diagnostics.
    This keeps a nominally present but badly matched control from counting as
    specificity evidence.
    """
    missing = sorted(set(CONTROL_FAMILIES) - set(control_sets))
    extra = sorted(set(control_sets) - set(CONTROL_FAMILIES))
    if missing or extra:
        raise ValueError(f"control panel mismatch; missing={missing}, extra={extra}")
    if locked and match_validator is None:
        raise ValueError("locked necessity analysis requires a frozen control match validator")
    report = {}
    for K in sorted({int(k) for k in K_ladder}):
        truth = _validate_channel_indices(true_channels[:K], K=K, width=width,
                                          family="true")
        for fam in CONTROL_FAMILIES:
            ctrl = _validate_channel_indices(control_sets[fam](K), K=K,
                                              width=width, family=fam)
            mr = ({"passed": True, "mode": "structural-only"}
                  if match_validator is None
                  else dict(match_validator(fam, K, truth, ctrl)))
            if "passed" not in mr:
                raise ValueError(f"{fam}: match report must contain 'passed'")
            if not bool(mr["passed"]):
                raise ValueError(f"{fam}: frozen matching constraints failed: {mr}")
            report[(fam, K)] = mr
    return report


def _ablate_positions(p: Tensor, idx: Sequence[int], scope: str) -> Tensor:
    """Remove selected channels at the target, context, or all positions."""
    if scope not in {"target", "context", "all"}:
        raise ValueError(f"unknown position scope {scope!r}")
    out = p.clone()
    if scope == "target":
        out.reshape(-1, out.shape[-2], out.shape[-1])[:, -1, list(idx)] = 0
    elif scope == "context":
        if out.shape[-2] > 1:
            out.reshape(-1, out.shape[-2], out.shape[-1])[:, :-1, list(idx)] = 0
    else:
        out[..., list(idx)] = 0
    return out


@torch.no_grad()
def necessity_ladder(
    runner: Evo2Runner,
    loci: Sequence[tuple[str, Tensor]],
    man: Manifest,
    *,
    ell: int,
    channels: Sequence[int],
    K_ladder: Sequence[int],
    control_sets: dict[str, Callable[[int], Sequence[int]]],
    position_scopes: Sequence[str] = ("target", "context", "all"),
    endpoint_callbacks: Optional[dict[str, Callable[[object, object], float]]] = None,
    control_match_validator: Optional[
        Callable[[str, int, Sequence[int], Sequence[int]], dict]
    ] = None,
    control_scope_resolver: Optional[Callable[[str, str], str]] = None,
    locked: bool = False,
) -> list[NecessityRow]:
    """Cumulative ablation of the frozen top-K, against all five control
    families. Primary endpoints are D_shape / donor-following / the Step 6
    plateau shift -- a drop in branch norm is reported but never the claim.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    rows: list[NecessityRow] = []
    want = {N.mlp_part(ell, "z"), N.g(ell)}

    if not loci:
        return rows
    # Width is learned from a real discovery/analysis trace rather than from
    # a hand-written architecture constant.
    probe = runner.run(loci[0][1], taps=want)
    probe_parts = runner.bilinear_parts(ell, probe[N.mlp_part(ell, "z")])
    match_report = validate_control_completeness(
        control_sets, K_ladder=K_ladder, width=probe_parts["p"].shape[-1],
        true_channels=channels, match_validator=control_match_validator,
        locked=locked)
    if locked and control_scope_resolver is None:
        raise ValueError(
            "locked necessity analysis requires a frozen control_scope_resolver "
            "so the wrong-token/shifted-position control is actually shifted")

    for locus, ids in loci:
        base = runner.run(ids, taps=want)
        z0 = base.logits.reshape(-1, base.logits.shape[-1])[-1]
        parts = runner.bilinear_parts(ell, base[N.mlp_part(ell, "z")])
        W3 = parts["W3"]
        p0 = parts["p"]

        def ablate(idx: Sequence[int], scope: str) -> Tensor:
            p = _ablate_positions(p0, idx, scope)
            return (p @ W3.T.to(p.dtype)).to(base[N.g(ell)].dtype)

        for K in K_ladder:
            variants = {"true": _validate_channel_indices(
                channels[:K], K=K, width=p0.shape[-1], family="true")}
            for fam in CONTROL_FAMILIES:
                variants[fam] = _validate_channel_indices(
                    control_sets[fam](K), K=K, width=p0.shape[-1], family=fam)
            for scope in position_scopes:
                for fam, idx in variants.items():
                    applied_scope = (scope if fam == "true" or control_scope_resolver is None
                                     else control_scope_resolver(fam, scope))
                    if (locked and fam == "wrong_token_or_shifted_position"
                            and applied_scope == scope):
                        raise ValueError(
                            "wrong_token_or_shifted_position must use a different "
                            "pre-registered position scope in locked analysis")
                    repl = ablate(idx, applied_scope)
                    e = Edit(N.g(ell), "update",
                             (lambda t, r=repl: r.to(t.dtype).to(t.device)),
                             f"ablate {fam} K={K} scope={scope}")
                    ts = runner.run(ids, taps=(), edits=[e])
                    z1 = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
                    r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                         beta_max=man.margins.beta_max,
                                         flat_logit_threshold=man.margins.flat_logit_threshold)
                    drop = float(1 - repl.double().norm()
                                 / base[N.g(ell)].double().norm().clamp_min(1e-30))
                    callbacks = endpoint_callbacks or {}
                    donor = (float(callbacks["donor_following"](base, ts))
                             if "donor_following" in callbacks else None)
                    plateau = (float(callbacks["plateau_shift"](base, ts))
                               if "plateau_shift" in callbacks else None)
                    mr = ({"passed": True, "mode": "true"} if fam == "true"
                          else match_report[(fam, K)])
                    rows.append(NecessityRow(
                        locus, K, fam, r.d_shape, donor, plateau, drop,
                        position_scope=scope, applied_position_scope=applied_scope,
                        control_match_passed=bool(mr["passed"]),
                        control_match=mr))
    return rows


# --------------------------------------------------------------------------
# 7-4C input x homologous module factorial
# --------------------------------------------------------------------------


@torch.no_grad()
def mixer_factorial(
    runner: Evo2Runner,
    ids: Tensor,
    *,
    pair: tuple[int, int],                 # (earlier, later) e.g. (25, 28)
    common_radius: bool = True,
) -> dict:
    """Cross RAW block inputs with homologous MIXER functions.

    The pre-RMSNorm lives INSIDE the module and is applied exactly once; this
    function therefore feeds raw (or direction-preserving common-radius) block
    inputs, never pre-normalised ones. Feeding a normalised state into a whole
    block would double-normalise and change the skip scale.
    """
    e, l = pair
    ts = runner.run(ids, taps={N.x(e), N.x(l)})
    xe, xl = ts[N.x(e)], ts[N.x(l)]
    if common_radius:
        rad = 0.5 * (xe.norm(dim=-1, keepdim=True) + xl.norm(dim=-1, keepdim=True))
        xe = xe / xe.norm(dim=-1, keepdim=True).clamp_min(1e-30) * rad
        xl = xl / xl.norm(dim=-1, keepdim=True).clamp_min(1e-30) * rad
    out = {}
    for iname, xin in (("early", xe), ("late", xl)):
        for mname, mell in (("early", e), ("late", l)):
            y = runner.a.mixer(mell)(runner.a.pre_norm(mell)(xin))
            y = y[0] if isinstance(y, (tuple, list)) else y
            out[f"input={iname},module={mname}"] = float(y.double().norm())
    return _factorial_effects(out)


@torch.no_grad()
def mlp_factorial(
    runner: Evo2Runner,
    ids: Tensor,
    *,
    pair: tuple[int, int],
) -> dict:
    """Cross ALREADY-NORMALISED z with homologous bilinear CORE weights.

    z_e and z_l are the natural post-RMSNorm values, so normalisation is NOT
    applied again here; the crossing happens on (W1, W2, W3) only.
    """
    e, l = pair
    ts = runner.run(ids, taps={N.mlp_part(e, "z"), N.mlp_part(l, "z")})
    ze, zl = ts[N.mlp_part(e, "z")], ts[N.mlp_part(l, "z")]
    out = {}
    for iname, z in (("early", ze), ("late", zl)):
        for mname, mell in (("early", e), ("late", l)):
            W1, W2, W3 = runner.a.mlp_weights(mell)
            zz = z.to(W1.dtype)
            y = ((zz @ W1.T) * (zz @ W2.T)) @ W3.T
            out[f"input={iname},module={mname}"] = float(y.double().norm())
    return _factorial_effects(out)


def _factorial_effects(cells: dict[str, float]) -> dict:
    """Main effects and interaction on log branch norm."""
    g = {k: np.log(max(v, 1e-300)) for k, v in cells.items()}
    ee = g["input=early,module=early"]; le = g["input=late,module=early"]
    el = g["input=early,module=late"];  ll = g["input=late,module=late"]
    return {
        "cells": cells,
        "input_main_effect": 0.5 * ((le - ee) + (ll - el)),
        "module_main_effect": 0.5 * ((el - ee) + (ll - le)),
        "interaction": (ll - el) - (le - ee),
        "reading": {
            "input main": "late incoming state is the cause",
            "module main": "learned late weights are the cause",
            "interaction": "a specific alignment of late state and late module",
        },
        "caveat": "synthetic states: this is an ORIGIN diagnostic. Production-path "
                  "necessity is Step 7-3 and Step 9.",
    }


# --------------------------------------------------------------------------
# 7-5 fidelity vs sufficiency
# --------------------------------------------------------------------------


def sufficiency_protocol() -> dict:
    """Re-inserting the SAME ablated values and recovering the output is a
    fidelity test of the hooks and the decomposition -- nothing more."""
    return {
        "fidelity_only": "exact re-add of the ablated channel values",
        "required_for_sufficiency": [
            "patch a HELD-OUT natural donor's channel vector into the recipient",
            "matched donor reproduces the donor-direction output effect",
            "wrong-pair and wrong-position donors fail",
            "after erasing upstream content, a matched channel patch partially "
            "restores the pre-specified effect",
            "reciprocal donor reverses the effect sign",
        ],
    }


NATURAL_DONOR_CONTROLS = ("wrong_pair_donor", "shifted_position_donor")


@dataclass
class NaturalDonorRow:
    pair_id: str
    locus: str
    condition: str
    donor_family: str
    position_scope: str
    d_shape: float
    log_beta: float
    signed_donor_following: float
    component_dose: float
    held_out: bool


def _selected_component(p: Tensor, W3: Tensor, channels: Sequence[int],
                        scope: str) -> Tensor:
    """Additive W3-projected contribution of selected product channels."""
    keep = torch.zeros_like(p)
    if scope == "target":
        keep.reshape(-1, keep.shape[-2], keep.shape[-1])[:, -1, list(channels)] = \
            p.reshape(-1, p.shape[-2], p.shape[-1])[:, -1, list(channels)]
    elif scope == "context":
        if keep.shape[-2] > 1:
            keep.reshape(-1, keep.shape[-2], keep.shape[-1])[:, :-1, list(channels)] = \
                p.reshape(-1, p.shape[-2], p.shape[-1])[:, :-1, list(channels)]
    elif scope == "all":
        keep[..., list(channels)] = p[..., list(channels)]
    else:
        raise ValueError(f"unknown position scope {scope!r}")
    return keep @ W3.T.to(keep.dtype)


@torch.no_grad()
def natural_donor_test(
    runner: Evo2Runner,
    pairs: Sequence[tuple[str, Tensor, Tensor]],
    man: Manifest,
    *,
    ell: int,
    channels: Sequence[int],
    wrong_donors: dict[str, Callable[[str], Tensor]],
    position_scopes: Sequence[str] = ("target", "context", "all"),
    upstream_ablation: Optional[Callable[[Tensor], list[Edit]]] = None,
    locus_of_pair: Optional[Callable[[str], str]] = None,
    held_out: bool = True,
) -> list[NaturalDonorRow]:
    """Executable held-out natural-donor sufficiency and rescue panel.

    A donor patch *replaces* only the selected channel contribution
    ``C_R`` with ``C_D``.  It never adds ``C_D`` on top of ``C_R``.  When an
    upstream-ablation factory is supplied, the ablated trajectory is first
    re-run, its actually remaining contribution ``C_A`` is measured, and the
    rescue is ``g_A - C_A + C_D``.  Wrong-pair and shifted-position donors are
    executed by the same code path and therefore share the causal endpoint.
    """
    if not held_out:
        raise ValueError("natural donor sufficiency must be evaluated on held-out pairs")
    missing = sorted(set(NATURAL_DONOR_CONTROLS) - set(wrong_donors))
    extra = sorted(set(wrong_donors) - set(NATURAL_DONOR_CONTROLS))
    if missing or extra:
        raise ValueError(f"natural-donor controls incomplete; missing={missing}, extra={extra}")
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    tapz, tapg = N.mlp_part(ell, "z"), N.g(ell)
    rows: list[NaturalDonorRow] = []

    def centred(v: Tensor) -> Tensor:
        v = v.double()
        return v - v.mean()

    for pair_id, rec_ids, donor_ids in pairs:
        if rec_ids.shape == donor_ids.shape and torch.equal(rec_ids, donor_ids):
            raise ValueError(f"{pair_id}: recipient and donor are identical")
        loc = locus_of_pair(pair_id) if locus_of_pair else pair_id
        rec = runner.run(rec_ids, taps={tapz, tapg})
        donor = runner.run(donor_ids, taps={tapz, tapg})
        z0 = rec.logits.reshape(-1, rec.logits.shape[-1])[-1]
        zD = donor.logits.reshape(-1, donor.logits.shape[-1])[-1]
        donor_dir = centred(zD) - centred(z0)
        donor_den = donor_dir.norm().square().clamp_min(1e-30)
        rp = runner.bilinear_parts(ell, rec[tapz])
        dp = runner.bilinear_parts(ell, donor[tapz])
        W3 = rp["W3"]
        idx = _validate_channel_indices(channels, K=len(channels),
                                        width=rp["p"].shape[-1], family="true")

        upstream_edits = upstream_ablation(rec_ids) if upstream_ablation else []
        if any(e.tap == tapg and e.kind == "update" for e in upstream_edits):
            raise ValueError(
                f"{pair_id}: upstream_ablation edits {tapg} itself; it must act strictly upstream")
        abl_trace = runner.run(rec_ids, taps={tapz, tapg}, edits=upstream_edits)
        ap = runner.bilinear_parts(ell, abl_trace[tapz])

        wrong_parts = {}
        for fam in NATURAL_DONOR_CONTROLS:
            wid = wrong_donors[fam](pair_id)
            if wid.shape == rec_ids.shape and torch.equal(wid, donor_ids):
                raise ValueError(f"{pair_id}: {fam} returned the matched donor")
            wt = runner.run(wid, taps={tapz})
            wrong_parts[fam] = runner.bilinear_parts(ell, wt[tapz])

        if dp["p"].shape != rp["p"].shape or ap["p"].shape != rp["p"].shape:
            raise ValueError(f"{pair_id}: donor/recipient channel tensors are not position-aligned")
        for fam, wp in wrong_parts.items():
            if wp["p"].shape != rp["p"].shape:
                raise ValueError(f"{pair_id}:{fam}: wrong donor is not position-aligned")

        for scope in position_scopes:
            CR = _selected_component(rp["p"], W3, idx, scope).to(rec[tapg].dtype)
            CD = _selected_component(dp["p"], W3, idx, scope).to(rec[tapg].dtype)
            CA = _selected_component(ap["p"], W3, idx, scope).to(rec[tapg].dtype)

            variants: list[tuple[str, str, list[Edit], Tensor]] = [
                ("donor_patch", "matched", [], CD - CR),
                ("channel_ablation", "matched", [], -CR),
                ("upstream_ablation", "matched", upstream_edits, -CA),
                ("ablation_rescue", "matched", upstream_edits, CD - CA),
            ]
            for fam in NATURAL_DONOR_CONTROLS:
                CW = _selected_component(wrong_parts[fam]["p"], W3, idx, scope).to(rec[tapg].dtype)
                variants.append(("donor_patch", fam, [], CW - CR))
                variants.append(("ablation_rescue", fam, upstream_edits, CW - CA))

            for cond, fam, prior_edits, delta in variants:
                edit = Edit(tapg, "update",
                            (lambda t, d=delta: t + d.to(t.dtype).to(t.device)),
                            f"{cond}:{fam}:{scope}")
                ts = runner.run(rec_ids, taps=(), edits=list(prior_edits) + [edit])
                z = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
                shape = solve_beta_shape(
                    z, z0, beta_min=man.margins.beta_min,
                    beta_max=man.margins.beta_max,
                    flat_logit_threshold=man.margins.flat_logit_threshold)
                dz = centred(z) - centred(z0)
                rows.append(NaturalDonorRow(
                    pair_id=pair_id, locus=loc, condition=cond,
                    donor_family=fam, position_scope=scope,
                    d_shape=shape.d_shape, log_beta=shape.log_beta,
                    signed_donor_following=float(torch.dot(dz, donor_dir) / donor_den),
                    component_dose=float(delta.double().norm()
                                         / rec[tapg].double().norm().clamp_min(1e-30)),
                    held_out=True))
    return rows


def _rows_by_key(rows, *, value: Callable[[object], float],
                 predicate: Callable[[object], bool]) -> dict[str, Row]:
    out = {}
    for r in rows:
        if predicate(r):
            # A locus may participate in several held-out donor pairs.  The
            # pair is the repeated-measure row; the locus remains its
            # dependency key so the cluster bootstrap does not treat those
            # pairs as independent.
            unit = getattr(r, "pair_id", getattr(r, "locus", ""))
            locus = getattr(r, "locus", unit)
            if unit in out:
                raise ValueError(f"duplicate causal row for unit {unit!r}")
            out[unit] = Row(float(value(r)), keys=(f"locus:{locus}",))
    return out


def _paired_maps(a: dict[str, Row], b: dict[str, Row], *, min_clusters: int,
                 n_boot: int) -> object:
    keys = sorted(set(a) & set(b))
    if not keys:
        raise RuntimeError("no aligned loci for causal contrast")
    return paired_contrast([a[k] for k in keys], [b[k] for k in keys],
                           min_clusters=min_clusters, n_boot=n_boot)


def causal_channel_verdict(
    necessity: Sequence[NecessityRow],
    sufficiency: Sequence[NaturalDonorRow],
    *,
    K: int,
    position_scope: str,
    effect_margin: float,
    specificity_margins: dict[str, float],
    man: Optional[Manifest] = None,
    min_clusters: Optional[int] = None,
    n_boot: int = 10_000,
) -> dict:
    """Necessity + natural sufficiency + rescue + per-family specificity.

    The label ``causal branch channel set`` is returned only if every member
    of this conjunction passes.  Large channel energy by itself is never a
    positive verdict.
    """
    required_margin_names = set(CONTROL_FAMILIES) | set(NATURAL_DONOR_CONTROLS)
    missing_margin_names = sorted(required_margin_names - set(specificity_margins))
    if missing_margin_names:
        raise RuntimeError(
            "Step 7 causal verdict is missing development-calibrated specificity "
            f"margins: {missing_margin_names}"
        )
    if min_clusters is None:
        if man is None:
            raise RuntimeError(
                "Step 7 causal verdict requires min_clusters explicitly or a calibrated "
                "manifest power key 'step7_causal_channel'"
            )
        min_clusters = required_cluster_count(man, "step7_causal_channel")

    nt = _rows_by_key(
        necessity, value=lambda r: r.d_shape,
        predicate=lambda r: r.K == K and r.position_scope == position_scope
        and r.family == "true")
    spec_est = {}
    spec_ok = True
    for fam in CONTROL_FAMILIES:
        nc = _rows_by_key(
            necessity, value=lambda r: r.d_shape,
            predicate=lambda r, fam=fam: r.K == K
            and r.position_scope == position_scope and r.family == fam)
        est = _paired_maps(nt, nc, min_clusters=min_clusters, n_boot=n_boot)
        margin = specificity_margins[fam]
        spec_est[f"necessity::{fam}"] = est.as_row()
        spec_ok &= est.lo > margin

    matched = _rows_by_key(
        sufficiency, value=lambda r: r.signed_donor_following,
        predicate=lambda r: r.position_scope == position_scope
        and r.condition == "donor_patch" and r.donor_family == "matched")
    zero = {k: Row(0.0, keys=v.keys) for k, v in matched.items()}
    suff_est = _paired_maps(matched, zero, min_clusters=min_clusters, n_boot=n_boot)

    rescue = _rows_by_key(
        sufficiency, value=lambda r: r.signed_donor_following,
        predicate=lambda r: r.position_scope == position_scope
        and r.condition == "ablation_rescue" and r.donor_family == "matched")
    ablated = _rows_by_key(
        sufficiency, value=lambda r: r.signed_donor_following,
        predicate=lambda r: r.position_scope == position_scope
        and r.condition == "upstream_ablation" and r.donor_family == "matched")
    rescue_est = _paired_maps(rescue, ablated, min_clusters=min_clusters, n_boot=n_boot)

    donor_spec = {}
    donor_spec_ok = True
    for fam in NATURAL_DONOR_CONTROLS:
        wrong = _rows_by_key(
            sufficiency, value=lambda r: r.signed_donor_following,
            predicate=lambda r, fam=fam: r.position_scope == position_scope
            and r.condition == "donor_patch" and r.donor_family == fam)
        est = _paired_maps(matched, wrong, min_clusters=min_clusters, n_boot=n_boot)
        margin = specificity_margins[fam]
        donor_spec[fam] = est.as_row()
        donor_spec_ok &= est.lo > margin

    checks = {
        "necessity_beats_every_control": spec_ok,
        "heldout_natural_sufficiency": suff_est.lo > effect_margin,
        "upstream_ablation_rescue": rescue_est.lo > effect_margin,
        "wrong_donor_specificity": donor_spec_ok,
        "control_matching_complete": all(r.control_match_passed for r in necessity),
        "heldout_only": all(r.held_out for r in sufficiency),
    }
    per_unit = {
        "necessity_by_locus": {
            key: {"K": K, "position_scope": position_scope,
                  "d_shape": row.value} for key, row in nt.items()},
        "sufficiency_by_pair": {
            key: {"position_scope": position_scope,
                  "natural_donor_following": matched[key].value,
                  "ablation_following": ablated[key].value,
                  "rescue_following": rescue[key].value,
                  "rescue_gain": rescue[key].value - ablated[key].value}
            for key in sorted(set(matched) & set(rescue) & set(ablated))},
    }
    return {
        "checks": checks,
        "necessity_specificity": spec_est,
        "sufficiency": suff_est.as_row(),
        "rescue": rescue_est.as_row(),
        "donor_specificity": donor_spec,
        "per_unit": per_unit,
        "claim": ("causal branch channel set" if all(checks.values())
                  else "association/partial causal evidence only"),
    }


@dataclass
class FactorialEndpointRow:
    locus: str
    input_level: str              # early | late
    module_level: str             # early | late
    endpoint: str
    value: float
    keys: tuple[str, ...] = ()


def cross_locus_factorial(
    loci: Sequence[tuple[str, Tensor]],
    *,
    endpoint: str,
    cell_measure: Callable[[str, Tensor, str, str], float],
    min_clusters: Optional[int] = None,
    n_boot: int = 10_000,
) -> dict:
    """Cross-locus 2x2 using one *causal* endpoint in all four cells.

    ``cell_measure`` executes the frozen cell intervention and returns the
    same endpoint (normally signed donor-following or D_shape) for all cells.
    The function refuses missing/duplicate cells and bootstraps locus-level
    main effects and interaction rather than reporting a single norm table.
    """
    if min_clusters is None:
        raise RuntimeError(
            "cross-locus factorial requires a development power-calibrated "
            "min_clusters value"
        )
    rows = []
    for locus, ids in loci:
        for il in ("early", "late"):
            for ml in ("early", "late"):
                rows.append(FactorialEndpointRow(
                    locus, il, ml, endpoint,
                    float(cell_measure(locus, ids, il, ml)),
                    keys=(f"locus:{locus}",)))

    by_locus: dict[str, dict[tuple[str, str], FactorialEndpointRow]] = {}
    for r in rows:
        if r.endpoint != endpoint:
            raise ValueError("factorial cells do not share one endpoint")
        key = (r.input_level, r.module_level)
        if key in by_locus.setdefault(r.locus, {}):
            raise ValueError(f"duplicate factorial cell {r.locus}:{key}")
        by_locus[r.locus][key] = r

    required = {("early", "early"), ("late", "early"),
                ("early", "late"), ("late", "late")}
    effects = {"input_main": [], "module_main": [], "interaction": []}
    for locus, cells in by_locus.items():
        if set(cells) != required:
            raise ValueError(f"{locus}: incomplete factorial cells {sorted(cells)}")
        ee = cells[("early", "early")].value
        le = cells[("late", "early")].value
        el = cells[("early", "late")].value
        ll = cells[("late", "late")].value
        k = (f"locus:{locus}",)
        effects["input_main"].append(Row(0.5 * ((le - ee) + (ll - el)), keys=k))
        effects["module_main"].append(Row(0.5 * ((el - ee) + (ll - le)), keys=k))
        effects["interaction"].append(Row((ll - el) - (le - ee), keys=k))
    estimates = {
        name: cluster_bootstrap(vals, min_clusters=min_clusters, n_boot=n_boot).as_row()
        for name, vals in effects.items()
    }
    return {"endpoint": endpoint, "rows": rows, "estimates": estimates,
            "n_loci": len(by_locus),
            "caveat": "causal only if cell_measure executes the pre-registered intervention"}


def role_table(b28: BilinearAudit, b29: BilinearAudit, necessity: Sequence[NecessityRow]) -> dict:
    """Step 7-6: b28 and b29 are judged SEPARATELY, and for carrier and
    content separately. Names come after the result."""
    return {
        "b28": {"first_anomaly": b28.first_anomaly, "coactivation_excess": b28.coactivation_excess,
                "G3_vs_permuted": b28.projection_gain_G3 / max(b28.projection_gain_G3_permuted, 1e-30)},
        "b29": {"first_anomaly": b29.first_anomaly, "coactivation_excess": b29.coactivation_excess,
                "G3_vs_permuted": b29.projection_gain_G3 / max(b29.projection_gain_G3_permuted, 1e-30)},
        "admissible_readings": [
            "b28 creates a new causal mode, b29 grows the same mode -> writer -> amplifier",
            "b28 and b29 create different content -> sequential writers",
            "norm grows but ablation is output-inert -> output-null amplification",
            "indistinguishable from matched axes -> generic bilinear energy concentration",
            "moves the carrier only -> carrier writer/amplifier",
            "moves relative preference selectively -> content writer/amplifier",
        ],
    }
