"""Step 6 — decompose the Appendix B alpha plateau into four causes.

Four mechanisms can produce the same accuracy curve: a small directional
seed that is enough to move the residual ray; an amplitude threshold; argmax
saturation while the full distribution keeps moving; and downstream
compensation. Accuracy is therefore demoted to an auxiliary variable here.

Two design points the plan is emphatic about and this module enforces:

* **Primary is GLOBAL alpha**, the same value at every locus, exactly as in
  Appendix B. Per-token normalisation by alpha_eq would align the crossovers
  artificially and change what the intervention means; alpha_eq and the
  dimensionless q are computed and reported, but only for geometry and
  heterogeneity plots.

* **Compensation is tested by freezing the UPDATE, not the state.** Writing a
  clean post-block state over a perturbed stream can erase the upstream
  perturbation itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import BranchGeometry, branch_geometry, solve_beta_shape
from .manifest import Manifest
from .stats import (
    Row, equivalence, family_cluster_bootstrap, paired_contrast,
    paired_difference_rows, positive, required_cluster_count,
)
from .taps import Edit, Evo2Runner, replace_with, scale_update


def _require_exp1_alias(man: Manifest) -> None:
    expected = {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)}
    bad = {k: (man.arch.legacy_alias.get(k), v) for k, v in expected.items()
           if man.arch.legacy_alias.get(k) != v}
    if bad:
        raise RuntimeError(f"EXP1 legacy aliases are missing/off-by-one: {bad}")


LEGACY_ALPHA_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0)

FREEZE_SETS = {
    "free": (),
    "joint": ((29, "m"), (29, "g"), (30, "m"), (30, "g"), (31, "m"), (31, "g")),
    "b29": ((29, "m"), (29, "g")),
    "b30": ((30, "m"), (30, "g")),
    "b31": ((31, "m"), (31, "g")),
    "b31_attn_only": ((31, "m"),),
    "b31_mlp_only": ((31, "g"),),
}


@dataclass
class AlphaRow:
    locus: str
    alpha: float
    direction: str                 # "natural" | "orthogonal_control" | "interp:<f>"
    freeze: str
    top1_correct: Optional[bool]
    nll: Optional[float]
    d_shape: float
    log_beta: float
    entropy: float
    top2_margin: float
    geom: Optional[dict] = None    # alpha_eq, theta(alpha), cos(r,g), cancellation, q
    updates: dict = field(default_factory=dict)   # ||m_l||, ||g_l|| and direction change
    carrier_coeff: Optional[float] = None
    final_radius: Optional[float] = None
    final_rms: Optional[float] = None
    beta_identified: bool = True
    boundary_censored: bool = False
    centered_logit_norm: Optional[float] = None
    response_alignment: Optional[float] = None
    heldout_response_effect: Optional[float] = None
    random_response_effect: Optional[float] = None
    exact_reinsert_fidelity: Optional[float] = None
    keys: tuple[str, ...] = ()
    windows: tuple[tuple[str, int, int], ...] = ()
    stratum: str = "all"


def _entropy_margin(z: Tensor) -> tuple[float, float]:
    p = torch.softmax(z.double(), -1)
    ent = float(-(p * p.clamp_min(1e-30).log()).sum())
    top2 = torch.topk(z.double(), 2).values
    return ent, float(top2[0] - top2[1])


def _freeze_edits(freeze: str, baseline_updates: dict[str, Tensor]) -> list[Edit]:
    """s^F_{k+1} = s^F_k + u_k(s^{alpha=1}_k): substitute the paired baseline
    update tensor while the perturbed stream keeps flowing."""
    edits = []
    for (ell, which) in FREEZE_SETS[freeze]:
        tap = N.m(ell) if which == "m" else N.g(ell)
        if tap in baseline_updates:
            edits.append(Edit(tap, "update", replace_with(baseline_updates[tap]),
                              f"freeze {tap} at alpha=1"))
    return edits


@torch.no_grad()
def alpha_sweep(
    runner: Evo2Runner,
    loci: Sequence[tuple[str, Tensor, Optional[int]]],   # (locus_id, ids, true_next_token)
    man: Manifest,
    *,
    alphas: Sequence[float] = LEGACY_ALPHA_GRID,
    freezes: Sequence[str] = ("free", "joint"),
    direction_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
    direction_name: str = "natural",
    carrier_j: Optional[int] = None,
) -> list[AlphaRow]:
    """Step 6-1 + 6-2 + 6-4.

    x_29^(alpha) = r_28 + alpha * g_28   (Appendix B's legacy h_28 alias)

    `direction_fn(g28, alpha) -> replacement branch output` injects the
    Step 6-3 direction conditions; the default is plain scaling of the natural
    branch.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    _require_exp1_alias(man)
    core = [28, 29, 30, 31]
    want = set()
    for l in core:
        want |= {N.x(l), N.m(l), N.r(l), N.g(l)}
    want |= {N.FINAL_PRE_NORM}

    rows: list[AlphaRow] = []
    for item in loci:
        if len(item) == 3:
            locus, ids, true_tok = item
            md = {}
        elif len(item) == 4:
            locus, ids, true_tok, md = item
            md = dict(md)
        else:
            raise ValueError("loci entries must be (locus,ids,true_token[,metadata])")
        locus = str(locus)
        base = runner.run(ids, taps=want)
        z0 = base.logits.reshape(-1, base.logits.shape[-1])[-1]
        baseline_updates = {N.m(l): base[N.m(l)] for l in core if N.m(l) in base.tensors}
        baseline_updates |= {N.g(l): base[N.g(l)] for l in core if N.g(l) in base.tensors}

        r28 = base[N.r(28)].reshape(-1, base[N.r(28)].shape[-1])[-1]
        g28 = base[N.g(28)].reshape(-1, base[N.g(28)].shape[-1])[-1]

        for alpha in alphas:
            geom: BranchGeometry = branch_geometry(r28, g28, alpha)
            if direction_fn is None:
                fn = scale_update(alpha)
            else:
                repl = direction_fn(base[N.g(28)], alpha)
                fn = replace_with(repl)
            for freeze in freezes:
                edits = [Edit(N.g(28), "update", fn, f"alpha={alpha} dir={direction_name}")]
                edits += _freeze_edits(freeze, baseline_updates)
                ts = runner.run(ids, taps=want, edits=edits)
                z1 = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
                r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                     beta_max=man.margins.beta_max,
                                     flat_logit_threshold=man.margins.flat_logit_threshold)
                ent, marg = _entropy_margin(z1)
                nll = None
                correct = None
                if true_tok is not None:
                    nll = float(-torch.log_softmax(z1.double(), -1)[true_tok])
                    correct = bool(int(z1.argmax()) == true_tok)
                upd = {}
                response_sum = None
                for l in core:
                    for tapname in (N.m(l), N.g(l)):
                        if tapname in ts.tensors and tapname in base.tensors:
                            a_ = ts[tapname].flatten().double()
                            b_ = base[tapname].flatten().double()
                            upd[f"{tapname}.norm"] = float(a_.norm())
                            upd[f"{tapname}.dircos"] = float(
                                torch.dot(a_, b_) / (a_.norm() * b_.norm()).clamp_min(1e-30))
                            if l >= 29:
                                dt = (ts[tapname] - base[tapname]).reshape(
                                    -1, ts[tapname].shape[-1])[-1].double()
                                response_sum = dt if response_sum is None else response_sum + dt
                response_alignment = None
                if response_sum is not None and N.g(28) in ts.tensors:
                    dg = (ts[N.g(28)] - base[N.g(28)]).reshape(
                        -1, ts[N.g(28)].shape[-1])[-1].double()
                    if float(dg.norm()) > 0 and float(response_sum.norm()) > 0:
                        response_alignment = float(
                            torch.dot(response_sum, dg) /
                            (response_sum.norm() * dg.norm()).clamp_min(1e-30))
                h_fin = ts.get(N.FINAL_PRE_NORM)
                cc = fr = fs = None
                if h_fin is not None:
                    v = h_fin.reshape(-1, h_fin.shape[-1])[-1]
                    fr = float(v.norm())
                    fs = float(v.double().pow(2).mean().sqrt())
                    if carrier_j is not None:
                        cc = float(v[carrier_j])
                rows.append(AlphaRow(
                    locus=locus, alpha=alpha, direction=direction_name, freeze=freeze,
                    top1_correct=correct, nll=nll, d_shape=r.d_shape, log_beta=r.log_beta,
                    entropy=ent, top2_margin=marg,
                    geom={"alpha_eq": geom.alpha_eq, "theta": geom.theta, "q": geom.q,
                          "cos_rg": geom.cos_rg, "cancellation": geom.cancellation,
                          "sign_crossing": geom.sign_crossing},
                    updates=upd, carrier_coeff=cc, final_radius=fr, final_rms=fs,
                    beta_identified=r.beta_identified,
                    boundary_censored=r.boundary_censored,
                    centered_logit_norm=r.centered_norm,
                    response_alignment=response_alignment,
                    keys=tuple(md.get("keys", (f"locus:{locus}",))),
                    windows=tuple(md.get("windows", ())),
                    stratum=str(md.get("stratum", "all"))))
    return rows


# --------------------------------------------------------------------------
# Step 6-3 direction controls
# --------------------------------------------------------------------------


def orthogonalized_control(
    g28: Tensor, r28: Tensor, *, seed: int, local_cov_half: Optional[Tensor] = None
) -> Tensor:
    """A. Sample a control direction in the orthogonal complement of the real
    g28 FIRST, then rematch norm and local-covariance dose.

    Orthogonalisation can break the matching, so the caller must re-check
    <r28, q> and the realised theta(alpha) curve afterwards -- `rematch_report`
    below computes exactly those. If angle matching at the same global alpha is
    impossible, compare at the dose that reaches the same theta instead.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(g28.shape[-1], generator=gen, dtype=torch.float64)
    if local_cov_half is not None:
        v = local_cov_half.double() @ v
    g = g28.reshape(-1, g28.shape[-1])[-1].double()
    v = v - torch.dot(v, g) / g.norm().clamp_min(1e-30) ** 2 * g
    v = v / v.norm().clamp_min(1e-30) * g.norm()
    out = g28.clone()
    out.reshape(-1, out.shape[-1])[-1] = v.to(g28.dtype)
    return out


def rematch_report(r28: Tensor, g_nat: Tensor, g_ctrl: Tensor, alphas: Sequence[float]) -> dict:
    """Post-orthogonalisation matching audit required by Step 6-3A."""
    r = r28.reshape(-1, r28.shape[-1])[-1]
    gn = g_nat.reshape(-1, g_nat.shape[-1])[-1]
    gc = g_ctrl.reshape(-1, g_ctrl.shape[-1])[-1]
    return {
        "norm_ratio": float(gc.double().norm() / gn.double().norm().clamp_min(1e-30)),
        "inner_r_gnat": float(torch.dot(r.double(), gn.double())),
        "inner_r_gctrl": float(torch.dot(r.double(), gc.double())),
        "theta_natural": [branch_geometry(r, gn, a).theta for a in alphas],
        "theta_control": [branch_geometry(r, gc, a).theta for a in alphas],
    }


def natural_interpolation(g_recipient: Tensor, g_donor: Tensor, f: float) -> Tensor:
    """B. Slerp between two REAL branch outputs. This is a biologically
    reachable direction change and is not the same experiment as an
    orthogonal-axis rotation; the two are never pooled."""
    from .endpoints import slerp
    a = g_recipient.reshape(-1, g_recipient.shape[-1])[-1].double()
    b = g_donor.reshape(-1, g_donor.shape[-1])[-1].double()
    w = slerp(a / a.norm().clamp_min(1e-30), b / b.norm().clamp_min(1e-30), f)
    out = g_recipient.clone()
    out.reshape(-1, out.shape[-1])[-1] = (w * a.norm()).to(g_recipient.dtype)
    return out


# --------------------------------------------------------------------------
# contrasts
# --------------------------------------------------------------------------


def _condition_safe(
    rows: Sequence[AlphaRow], *, direction: Optional[str] = None,
    freeze: Optional[str] = None,
) -> list[AlphaRow]:
    out = [r for r in rows if (direction is None or r.direction == direction) and
           (freeze is None or r.freeze == freeze)]
    conditions = {(r.direction, r.freeze) for r in out}
    if len(conditions) > 1:
        raise ValueError(
            f"mixed alpha conditions {sorted(conditions)}; specify direction/freeze. "
            "A locus/alpha from one condition must never overwrite another."
        )
    return out


def _assert_same_alpha_grid(a: Sequence[AlphaRow], b: Sequence[AlphaRow], label: str) -> None:
    def grids(rows: Sequence[AlphaRow]) -> dict[str, tuple[float, ...]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            out.setdefault(r.locus, []).append(float(r.alpha))
        final = {}
        for loc, vals in out.items():
            if len(vals) != len(set(vals)):
                raise ValueError(f"duplicate alpha cell for locus={loc} in {label}")
            final[loc] = tuple(sorted(vals))
        return final
    ga, gb = grids(a), grids(b)
    if set(ga) != set(gb):
        raise ValueError(f"{label}: condition arms contain different locus keys")
    bad = [loc for loc in ga if ga[loc] != gb[loc]]
    if bad:
        raise ValueError(f"{label}: global-alpha grids differ for loci {bad[:5]}")


def _slope_over(
    rows: Sequence[AlphaRow], endpoint: str,
    q_lo: Optional[float] = None, q_hi: Optional[float] = None,
    *, dose: str = "alpha", alpha_lo: Optional[float] = None,
    alpha_hi: Optional[float] = None,
) -> dict[str, float]:
    """Per-locus slope; GLOBAL alpha is primary and q is secondary only."""
    if dose not in {"alpha", "q"}:
        raise ValueError("dose must be 'alpha' or 'q'")
    by_locus: dict[str, list[AlphaRow]] = {}
    for r in rows:
        if not r.beta_identified:
            continue
        if dose == "alpha" and ((alpha_lo is not None and r.alpha < alpha_lo) or
                                (alpha_hi is not None and r.alpha > alpha_hi)):
            continue
        if dose == "q" and (not r.geom or q_lo is None or q_hi is None or
                            not (q_lo <= r.geom["q"] <= q_hi)):
            continue
        if dose == "alpha" or r.geom:
            by_locus.setdefault(r.locus, []).append(r)
    out = {}
    for loc, rs in by_locus.items():
        x = np.array([r.alpha if dose == "alpha" else r.geom["q"] for r in rs])
        y = np.array([getattr(r, endpoint) for r in rs], dtype=float)
        uncensored = np.array([not r.boundary_censored for r in rs]) \
            if endpoint == "log_beta" else np.ones(len(rs), dtype=bool)
        m = np.isfinite(x) & np.isfinite(y) & uncensored
        out[loc] = float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 2 and x[m].std() > 1e-12 else np.nan
    return out


def _slope_rows(rows: Sequence[AlphaRow], endpoint: str, *, dose: str,
                q_lo: Optional[float] = None, q_hi: Optional[float] = None,
                alpha_lo: Optional[float] = None,
                alpha_hi: Optional[float] = None) -> list[Row]:
    vals = _slope_over(rows, endpoint, q_lo, q_hi, dose=dose,
                       alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    first = {r.locus: r for r in rows}
    out = []
    for loc, value in vals.items():
        if not np.isfinite(value):
            continue
        r = first[loc]
        out.append(Row(value, r.keys or (f"locus:{loc}",), windows=r.windows,
                       stratum=r.stratum, unit_id=loc,
                       meta={"locus": loc, "endpoint": endpoint, "dose": dose}))
    return out


def _censor_report(rows: Sequence[AlphaRow]) -> dict:
    if not rows:
        return {"n": 0, "beta_unidentified_rate": float("nan"),
                "boundary_censored_rate": float("nan")}
    return {
        "n": len(rows),
        "beta_unidentified_rate": float(np.mean([not r.beta_identified for r in rows])),
        "boundary_censored_rate": float(np.mean([r.boundary_censored for r in rows])),
    }


def _bridge_rows(rows: Sequence[Row]) -> list[dict]:
    return [{"unit_id": r.unit_id, "value": r.value, "keys": list(r.keys),
             "windows": [list(w) for w in r.all_windows()],
             "stratum": r.stratum, "meta": r.meta} for r in rows]


def c_dir(nat: Sequence[AlphaRow], ctrl: Sequence[AlphaRow], man: Manifest,
          endpoint: str = "d_shape", *, freeze: str = "free",
          natural_direction: str = "natural",
          control_direction: str = "orthogonal_control",
          seed: int = 42, n_boot: int = 10_000,
          primary_family_alpha: float = 0.05 / 3) -> dict:
    """C_dir on global alpha; q-normalised geometry is secondary."""
    man.margins.require("q_lo", "q_hi")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step6_alpha")
    nsel = _condition_safe(nat, direction=natural_direction, freeze=freeze)
    csel = _condition_safe(ctrl, direction=control_direction, freeze=freeze)
    _assert_same_alpha_grid(nsel, csel, "C_dir")
    a = _slope_rows(nsel, endpoint, dose="alpha")
    b = _slope_rows(csel, endpoint, dose="alpha")
    est = paired_contrast(a, b, required=required, seed=seed, n_boot=n_boot,
                          alpha=primary_family_alpha)
    aq = _slope_rows(nsel, endpoint, dose="q", q_lo=man.margins.q_lo, q_hi=man.margins.q_hi)
    bq = _slope_rows(csel, endpoint, dose="q", q_lo=man.margins.q_lo, q_hi=man.margins.q_hi)
    q_est = paired_contrast(aq, bq, required=required, seed=seed + 1, n_boot=n_boot)
    return {"contrast": "C_dir = global-alpha slope(natural) - slope(orthogonal control)",
            "endpoint": endpoint, "estimate": est.as_row(),
            "primary_global_alpha": est.as_row(),
            "secondary_q": q_est.as_row(), "n_loci": est.n_rows,
            "condition": {"freeze": freeze, "natural": natural_direction,
                          "control": control_direction},
            "primary_family_alpha": primary_family_alpha,
            "censoring": {"natural": _censor_report(nsel), "control": _censor_report(csel)},
            "exp1_bridge": {"legacy_alias": {"h28": N.x(29)},
                            "natural_global_alpha_slopes": _bridge_rows(a),
                            "control_global_alpha_slopes": _bridge_rows(b)}}


def c_acc(rows: Sequence[AlphaRow], man: Manifest, endpoint: str = "d_shape", *,
          direction: str = "natural", freeze: str = "free",
          seed: int = 42, n_boot: int = 10_000,
          primary_family_alpha: float = 0.05 / 3) -> dict:
    """C_acc: the full-distribution change BETWEEN alpha_sat and alpha = 1.

    alpha_sat is fixed in development (first dose whose accuracy is within
    delta_acc of alpha=1 and stays so at the next two grid points) and applied
    unchanged; the locked accuracy curve is never re-read to pick it.
    """
    man.margins.require("alpha_sat", "delta_acc")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step6_alpha")
    sat = man.margins.alpha_sat
    selected = _condition_safe(rows, direction=direction, freeze=freeze)
    by_locus: dict[str, dict[float, AlphaRow]] = {}
    for r in selected:
        if r.alpha in by_locus.setdefault(r.locus, {}):
            raise ValueError(f"duplicate alpha={r.alpha} for locus={r.locus} condition")
        by_locus[r.locus][r.alpha] = r
    at_sat, at_one = [], []
    for loc, m in by_locus.items():
        if sat in m and 1.0 in m:
            a, b = m[sat], m[1.0]
            if not (a.beta_identified and b.beta_identified):
                continue
            at_sat.append(Row(getattr(a, endpoint), a.keys or (f"locus:{loc}",),
                               windows=a.windows, stratum=a.stratum, unit_id=loc))
            at_one.append(Row(getattr(b, endpoint), b.keys or (f"locus:{loc}",),
                              windows=b.windows, stratum=b.stratum, unit_id=loc))
    est = paired_contrast(at_sat, at_one, required=required, seed=seed, n_boot=n_boot,
                          alpha=primary_family_alpha)
    return {"contrast": f"C_acc = {endpoint}(alpha_sat) - {endpoint}(alpha=1)",
            "alpha_sat": sat, "estimate": est.as_row(), "n_loci": len(at_sat),
            "condition": {"direction": direction, "freeze": freeze},
            "primary_family_alpha": primary_family_alpha,
            "censoring": _censor_report(selected),
            "exp1_bridge": {"legacy_alias": {"h28": N.x(29)},
                            "alpha_sat_rows": _bridge_rows(at_sat),
                            "alpha_one_rows": _bridge_rows(at_one)}}


def c_comp(free: Sequence[AlphaRow], freeze: Sequence[AlphaRow], man: Manifest,
           endpoint: str = "d_shape", *, direction: str = "natural",
           free_name: str = "free", freeze_name: str = "joint",
           seed: int = 42, n_boot: int = 10_000,
           primary_family_alpha: float = 0.05 / 3) -> dict:
    """C_comp = slope_freeze - slope_free, as a paired condition x dose
    interaction -- NOT two separate significance tests."""
    man.margins.require("q_lo", "q_hi")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step6_alpha")
    fsel = _condition_safe(free, direction=direction, freeze=free_name)
    zsel = _condition_safe(freeze, direction=direction, freeze=freeze_name)
    _assert_same_alpha_grid(fsel, zsel, "C_comp")
    a = _slope_rows(zsel, endpoint, dose="alpha")
    b = _slope_rows(fsel, endpoint, dose="alpha")
    est = paired_contrast(a, b, required=required, seed=seed, n_boot=n_boot,
                          alpha=primary_family_alpha)
    aq = _slope_rows(zsel, endpoint, dose="q", q_lo=man.margins.q_lo, q_hi=man.margins.q_hi)
    bq = _slope_rows(fsel, endpoint, dose="q", q_lo=man.margins.q_lo, q_hi=man.margins.q_hi)
    q_est = paired_contrast(aq, bq, required=required, seed=seed + 1, n_boot=n_boot)
    return {"contrast": "C_comp = slope(update-freeze) - slope(free)",
            "estimate": est.as_row(), "primary_global_alpha": est.as_row(),
            "secondary_q": q_est.as_row(),
            "n_loci": est.n_rows,
            "condition": {"direction": direction, "free": free_name, "freeze": freeze_name},
            "primary_family_alpha": primary_family_alpha,
            "censoring": {"free": _censor_report(fsel), "freeze": _censor_report(zsel)},
            "exp1_bridge": {"legacy_alias": {"h28": N.x(29)},
                            "free_global_alpha_slopes": _bridge_rows(b),
                            "freeze_global_alpha_slopes": _bridge_rows(a)}}


def compensation_verdict(
    *, free_plateau: bool, freeze_restores: bool, response_antialigned: bool,
    heldout_response_beats_random: bool, exact_reinsert_fidelity: float,
) -> dict:
    """All FOUR conditions are required. The exact re-insertion of the same
    run's full response is a hook-fidelity test and is deliberately excluded
    from the count -- as is 'the b29/b30/b31 update simply got smaller'.
    """
    conds = {
        "1_free_curve_flat_or_plateau": free_plateau,
        "2_update_freeze_restores_sensitivity": freeze_restores,
        "3_downstream_response_antialigned": response_antialigned,
        "4_heldout_response_beats_random_or_wrong": heldout_response_beats_random,
    }
    return {"conditions": conds,
            "compensation": all(conds.values()),
            "exact_reinsert_fidelity": exact_reinsert_fidelity,
            "note": "exact re-insertion is fidelity accounting only; it is NOT "
                    "independent evidence of compensation"}


def _mean_scalar_by_locus(
    rows: Sequence[AlphaRow], attr: str, *, negate: bool = False,
    difference_attr: Optional[str] = None,
) -> list[Row]:
    grouped: dict[str, list[AlphaRow]] = {}
    for r in rows:
        a = getattr(r, attr)
        b = getattr(r, difference_attr) if difference_attr else None
        if a is None or not np.isfinite(a) or (difference_attr and (b is None or not np.isfinite(b))):
            continue
        grouped.setdefault(r.locus, []).append(r)
    out = []
    for loc, rs in grouped.items():
        vals = [float(getattr(r, attr)) - (float(getattr(r, difference_attr))
                if difference_attr else 0.0) for r in rs]
        value = float(np.mean(vals))
        if negate:
            value = -value
        r0 = rs[0]
        out.append(Row(value, r0.keys or (f"locus:{loc}",), windows=r0.windows,
                       stratum=r0.stratum, unit_id=loc,
                       meta={"locus": loc, "estimand": attr}))
    return out


def compensation_estimands(
    free: Sequence[AlphaRow],
    freeze: Sequence[AlphaRow],
    man: Manifest,
    *,
    direction: str = "natural",
    free_name: str = "free",
    freeze_name: str = "joint",
    endpoint: str = "d_shape",
    seed: int = 42,
    n_boot: int = 10_000,
) -> dict:
    """Derive all four compensation criteria from locked per-locus data.

    Unlike `compensation_verdict` (kept as a low-level compatibility helper),
    callers cannot pass favourable booleans.  The plateau, update-freeze
    interaction, downstream anti-alignment, and held-out-vs-random response
    advantage are estimated here in one simultaneous family.
    """
    man.margins.require("delta_spec", "r_num_d_shape", "alpha_sat")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step6_alpha")
    fsel = _condition_safe(free, direction=direction, freeze=free_name)
    zsel = _condition_safe(freeze, direction=direction, freeze=freeze_name)
    _assert_same_alpha_grid(fsel, zsel, "compensation")
    # Compensation concerns the Appendix-B *plateau*, so estimate both
    # curves on the development-frozen [alpha_sat, 1] region.  Using the full
    # 0..2 sweep would mix the initial response into the plateau estimand.
    free_slopes = _slope_rows(
        fsel, endpoint, dose="alpha", alpha_lo=man.margins.alpha_sat, alpha_hi=1.0)
    freeze_slopes = _slope_rows(
        zsel, endpoint, dose="alpha", alpha_lo=man.margins.alpha_sat, alpha_hi=1.0)
    restore = paired_difference_rows(freeze_slopes, free_slopes)
    response_rows = [r for r in fsel if man.margins.alpha_sat <= r.alpha < 1.0]
    antialigned = _mean_scalar_by_locus(response_rows, "response_alignment", negate=True)
    heldout_adv = _mean_scalar_by_locus(
        response_rows, "heldout_response_effect", difference_attr="random_response_effect")
    if not antialigned:
        raise RuntimeError("response_alignment was not measured; compensation is not identifiable")
    if not heldout_adv:
        raise RuntimeError(
            "heldout_response_effect/random_response_effect were not measured; "
            "same-run exact reinsertion cannot substitute for this criterion"
        )

    fam = {
        "free_global_alpha_slope": free_slopes,
        "freeze_minus_free_slope": restore,
        "negative_downstream_alignment": antialigned,
        "heldout_minus_random_response": heldout_adv,
    }
    est = family_cluster_bootstrap(fam, seed=seed, n_boot=n_boot, required=required)

    def calibrated(name: str) -> float:
        if name not in man.margins.delta_spec:
            raise RuntimeError(f"missing development-calibrated delta_spec[{name!r}]")
        return float(man.margins.delta_spec[name])

    plateau_eq = equivalence(
        est["free_global_alpha_slope"], calibrated("alpha_plateau_slope_equivalence"))
    checks = {
        "1_free_curve_practically_flat": plateau_eq.verdict == "equivalent",
        "2_update_freeze_restores_sensitivity": positive(
            est["freeze_minus_free_slope"], calibrated("compensation_restore")),
        "3_downstream_response_antialigned": positive(
            est["negative_downstream_alignment"], calibrated("compensation_antialignment")),
        "4_heldout_response_beats_random": positive(
            est["heldout_minus_random_response"],
            calibrated("compensation_heldout_advantage")),
    }
    fidelity = [r.exact_reinsert_fidelity for r in fsel
                if r.exact_reinsert_fidelity is not None
                and np.isfinite(r.exact_reinsert_fidelity)]
    return {
        "estimates": {k: v.as_row() for k, v in est.items()},
        "conditions": checks,
        "compensation": all(checks.values()),
        "exact_reinsert_fidelity": {
            "max": max(fidelity) if fidelity else None,
            "passes_numerical_gate": (
                max(fidelity) <= man.margins.r_num_d_shape if fidelity else None
            ),
            "evidence_role": "hook fidelity only; excluded from the four causal criteria",
        },
        "censoring": {"free": _censor_report(fsel), "freeze": _censor_report(zsel)},
        "primary_dose": (
            "one shared global alpha grid; q appears only in secondary geometry panels"
        ),
        "plateau_interval": [man.margins.alpha_sat, 1.0],
        "exp1_bridge": {
            "legacy_alias": {"h28": N.x(29)},
            "free_global_alpha_slopes": _bridge_rows(free_slopes),
            "freeze_minus_free_slopes": _bridge_rows(restore),
            "use": "link Appendix-B alpha loci to EXP1 recovery/benchmark labels by locus key",
        },
    }


@torch.no_grad()
def measure_compensation_responses(
    runner: Evo2Runner,
    loci: Sequence[tuple],
    rows: Sequence[AlphaRow],
    man: Manifest,
    *,
    heldout_component: Callable[[str, float, Tensor, Tensor], Tensor],
    random_component: Callable[[str, float, Tensor, Tensor], Tensor],
    response_tap: str = N.x(31),
    alphas: Sequence[float] = LEGACY_ALPHA_GRID,
    freeze: str = "joint",
) -> list[AlphaRow]:
    """Run held-out/random response rescue and attach its measured effects.

    The two callbacks receive `(locus, alpha, frozen_state, free_state)` and
    return an ADDITIVE response component.  `heldout_component` must be fixed
    from other loci/splits; passing `free_state-frozen_state` there would leak
    the answer.  That same-run delta is evaluated separately as hook fidelity.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    _require_exp1_alias(man)
    if freeze == "free":
        raise ValueError("response rescue needs an update-frozen comparison condition")
    by_key = {(r.locus, r.alpha, r.direction, r.freeze): r for r in rows}
    out = list(rows)

    for item in loci:
        locus, ids = str(item[0]), item[1]
        want = {N.g(28), response_tap}
        for l in (29, 30, 31):
            want |= {N.m(l), N.g(l)}
        base = runner.run(ids, taps=want)
        baseline_updates = {name: base[name] for name in want
                            if (name.startswith("m") or name.startswith("g")) and
                            name in base.tensors}

        for alpha in alphas:
            alpha_edit = Edit(N.g(28), "update", scale_update(alpha), f"alpha={alpha}")
            free_ts = runner.run(ids, taps={response_tap}, edits=[alpha_edit])
            freeze_edits = [alpha_edit] + _freeze_edits(freeze, baseline_updates)
            frozen_ts = runner.run(ids, taps={response_tap}, edits=freeze_edits)
            if response_tap not in free_ts.tensors or response_tap not in frozen_ts.tensors:
                raise RuntimeError(f"response tap {response_tap!r} was not captured")
            hf, hz = free_ts[response_tap], frozen_ts[response_tap]
            z_free = free_ts.logits.reshape(-1, free_ts.logits.shape[-1])[-1]

            def run_component(comp: Tensor, label: str) -> float:
                def add(t: Tensor, c=comp) -> Tensor:
                    return t + c.to(t.dtype).to(t.device)
                z = runner.run(
                    ids, taps=(), edits=freeze_edits +
                    [Edit(response_tap, "state", add, label)]).logits
                z = z.reshape(-1, z.shape[-1])[-1]
                return solve_beta_shape(
                    z, z_free, beta_min=man.margins.beta_min,
                    beta_max=man.margins.beta_max,
                    flat_logit_threshold=man.margins.flat_logit_threshold).d_shape

            frozen_z = frozen_ts.logits.reshape(-1, frozen_ts.logits.shape[-1])[-1]
            damage = solve_beta_shape(
                frozen_z, z_free, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold).d_shape
            hcomp = heldout_component(locus, float(alpha), hz, hf)
            rcomp = random_component(locus, float(alpha), hz, hf)
            hdist = run_component(hcomp, "heldout compensation response")
            rdist = run_component(rcomp, "rank/dose-matched random response")
            exact_dist = run_component(hf - hz, "same-run exact response fidelity")
            target = by_key.get((locus, float(alpha), "natural", "free"))
            if target is None:
                raise RuntimeError(
                    f"missing natural/free AlphaRow for locus={locus}, alpha={alpha}"
                )
            target.heldout_response_effect = float(damage - hdist)
            target.random_response_effect = float(damage - rdist)
            target.exact_reinsert_fidelity = float(exact_dist)
    return out


def read_surface(rows: Sequence[AlphaRow], *, direction: str = "natural",
                 freeze: str = "free") -> dict:
    """Step 6-5 reading rules, as a structured summary rather than a verdict.

    Locus heterogeneity is reported explicitly: if no single explanation holds
    across loci, the plan says to withdraw the averaged plateau claim.
    """
    selected = _condition_safe(rows, direction=direction, freeze=freeze)
    by_locus: dict[str, list[AlphaRow]] = {}
    for r in selected:
        by_locus.setdefault(r.locus, []).append(r)
    sats, thetas, cancels = [], [], []
    for loc, rs in by_locus.items():
        rs = sorted(rs, key=lambda r: r.alpha)
        acc_rows = [r for r in rs if r.top1_correct is not None]
        if acc_rows:
            sats.append(next((r.alpha for r in acc_rows if r.top1_correct), np.nan))
        thetas.append([(r.alpha, r.geom["theta"]) for r in rs if r.geom])
        cancels.append(min((r.geom["cancellation"] for r in rs if r.geom), default=np.nan))
    return {
        "n_loci": len(by_locus),
        "accuracy_recovery_alpha_median": float(np.nanmedian(sats)) if sats else float("nan"),
        "accuracy_recovery_alpha_iqr": float(np.nanpercentile(sats, 75) - np.nanpercentile(sats, 25))
                                        if sats else float("nan"),
        "min_cancellation_median": float(np.nanmedian(cancels)) if cancels else float("nan"),
        "theta_curves": thetas[:5],
        "condition": {"direction": direction, "freeze": freeze},
        "censoring": _censor_report(selected),
        "heterogeneity_note": "large IQR in the recovery dose means the plateau is "
                              "locus-heterogeneous; withdraw the averaged claim rather "
                              "than reporting a mean plateau",
    }
