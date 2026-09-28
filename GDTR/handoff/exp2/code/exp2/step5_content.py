"""Step 5 — carrier-excluded content, moved by a NATURAL-ANGLE dose.

The dose is an absolute geodesic angle drawn from the development natural-pair
angle distribution, and theta <= phi is enforced so no pair is ever
extrapolated past its own donor. Pair-relative f in {0.25,0.5,0.75} is
secondary. A global random sphere direction is NOT a primary control -- in
4095 dimensions it is near-orthogonal to everything and would flatter the
result; the primary control is a local-covariance direction at equal distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import content_dose_state, solve_beta_shape
from .manifest import Manifest
from .stats import (
    Row, equivalence, family_cluster_bootstrap, paired_difference_rows,
    positive, required_cluster_count,
)
from .taps import Edit, Evo2Runner


def _require_exp1_alias(man: Manifest) -> None:
    expected = {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)}
    bad = {k: (man.arch.legacy_alias.get(k), v) for k, v in expected.items()
           if man.arch.legacy_alias.get(k) != v}
    if bad:
        raise RuntimeError(f"EXP1 legacy aliases are missing/off-by-one: {bad}")


@dataclass
class ContentRow:
    pair_id: str
    tap: str
    control: str              # "natural_donor" | "self" | "local_cov" | "full_state" | "wrong_pair"
    theta: float
    d_shape: float
    log_beta: float
    donor_following: float    # signed movement toward the donor's output direction
    ood_score: float
    excluded: Optional[str]
    diag: dict = field(default_factory=dict)
    path: str = "free_suffix"
    reciprocal: bool = False
    beta_identified: bool = True
    boundary_censored: bool = False
    keys: tuple[str, ...] = ()
    windows: tuple[tuple[str, int, int], ...] = ()
    stratum: str = "all"


def _donor_following(z_int: Tensor, z_base: Tensor, z_donor: Tensor) -> float:
    """Signed projection of the induced centred-logit change onto the
    baseline->donor centred-logit direction."""
    a = (z_int.double() - z_int.double().mean()) - (z_base.double() - z_base.double().mean())
    b = (z_donor.double() - z_donor.double().mean()) - (z_base.double() - z_base.double().mean())
    return float(torch.dot(a, b) / b.norm().clamp_min(1e-30) ** 2)


def _following_fixed_direction(z_int: Tensor, z_base: Tensor, direction: Tensor) -> float:
    """Projection onto one fixed forward R->D direction.

    This lets the D->R reciprocal intervention have the expected negative
    sign instead of redefining its axis to be positive by construction.
    """
    a = (z_int.double() - z_int.double().mean()) - (z_base.double() - z_base.double().mean())
    b = direction.double() - direction.double().mean()
    return float(torch.dot(a, b) / b.norm().clamp_min(1e-30) ** 2)


@torch.no_grad()
def content_transport(
    runner: Evo2Runner,
    pairs: Sequence[tuple[str, Tensor, Tensor]],     # (pair_id, recipient_ids, donor_ids)
    man: Manifest,
    *,
    tap: str,
    carrier_axis: Tensor,
    thetas: Sequence[float],
    local_cov_direction: Callable[[Tensor, float], Tensor],
    ood_score: Callable[[Tensor], float],
    wrong_pair_donor: Optional[Callable[[str], Tensor]] = None,
    reciprocal: bool = True,
) -> list[ContentRow]:
    """Hold r_R and c_R fixed; rotate the carrier-orthogonal direction toward
    the donor by an absolute angle theta.

    `local_cov_direction(w_R, theta)` must return a unit carrier-orthogonal
    direction at geodesic distance theta from w_R, drawn from the LOCAL
    natural covariance -- that is the matched control, not a random sphere point.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold",
                        "c_abs_max", "phi_min", "phi_max")
    _require_exp1_alias(man)
    rows: list[ContentRow] = []
    u = carrier_axis

    for item in pairs:
        if len(item) == 3:
            pid, rec_ids, don_ids = item
            md = {}
        elif len(item) == 4:
            pid, rec_ids, don_ids, md = item
            md = dict(md)
        else:
            raise ValueError("pairs entries must be (pair_id,recipient,donor[,metadata])")
        pid = str(pid)
        keys = tuple(md.get("keys", (f"pair:{pid}",)))
        windows = tuple(md.get("windows", ()))
        stratum = str(md.get("stratum", "all"))
        rec = runner.run(rec_ids, taps={tap})
        don = runner.run(don_ids, taps={tap})
        if tap not in rec.tensors or tap not in don.tensors:
            continue
        hR = rec[tap].reshape(-1, rec[tap].shape[-1])[-1]
        hD = don[tap].reshape(-1, don[tap].shape[-1])[-1]
        z0 = rec.logits.reshape(-1, rec.logits.shape[-1])[-1]
        zD = don.logits.reshape(-1, don.logits.shape[-1])[-1]
        forward_direction = zD - z0

        def run_state(base_ids: Tensor, h_new: Tensor) -> Tensor:
            def fn(t: Tensor) -> Tensor:
                out = t.clone()
                out.reshape(-1, out.shape[-1])[-1] = h_new.to(t.dtype)
                return out
            z = runner.run(base_ids, taps=(), edits=[Edit(tap, "state", fn, "content")]).logits
            return z.reshape(-1, z.shape[-1])[-1]

        for theta in thetas:
            h_new, diag = content_dose_state(
                hR, hD, u, theta,
                c_abs_max=man.margins.c_abs_max,
                phi_min=man.margins.phi_min, phi_max=man.margins.phi_max)
            if h_new is None:
                rows.append(ContentRow(pid, tap, "natural_donor", theta, float("nan"),
                                       float("nan"), float("nan"), float("nan"),
                                       diag["excluded"], diag, keys=keys,
                                       windows=windows, stratum=stratum))
                continue
            z1 = run_state(rec_ids, h_new)
            r = solve_beta_shape(z1, z0, beta_min=man.margins.beta_min,
                                 beta_max=man.margins.beta_max,
                                 flat_logit_threshold=man.margins.flat_logit_threshold)
            rows.append(ContentRow(pid, tap, "natural_donor", theta, r.d_shape, r.log_beta,
                                   _following_fixed_direction(z1, z0, forward_direction),
                                   ood_score(h_new), None, diag,
                                   beta_identified=r.beta_identified,
                                   boundary_censored=r.boundary_censored,
                                   keys=keys, windows=windows, stratum=stratum))

            # local-covariance control at the SAME geodesic distance
            from .endpoints import carrier_split
            sR = carrier_split(hR, u, c_abs_max=man.margins.c_abs_max)
            w_ctrl = local_cov_direction(sR.w, theta)
            h_ctrl = sR.radius * (sR.c * (u / u.norm()) + (1 - sR.c ** 2) ** 0.5 * w_ctrl)
            z2 = run_state(rec_ids, h_ctrl)
            rc = solve_beta_shape(z2, z0, beta_min=man.margins.beta_min,
                                  beta_max=man.margins.beta_max,
                                  flat_logit_threshold=man.margins.flat_logit_threshold)
            rows.append(ContentRow(pid, tap, "local_cov", theta, rc.d_shape, rc.log_beta,
                                   _following_fixed_direction(z2, z0, forward_direction),
                                   ood_score(h_ctrl), None, {},
                                   beta_identified=rc.beta_identified,
                                   boundary_censored=rc.boundary_censored,
                                   keys=keys, windows=windows, stratum=stratum))

            if reciprocal:
                h_rev, rev_diag = content_dose_state(
                    hD, hR, u, theta,
                    c_abs_max=man.margins.c_abs_max,
                    phi_min=man.margins.phi_min, phi_max=man.margins.phi_max)
                if h_rev is not None:
                    zr = run_state(don_ids, h_rev)
                    rr = solve_beta_shape(
                        zr, zD, beta_min=man.margins.beta_min,
                        beta_max=man.margins.beta_max,
                        flat_logit_threshold=man.margins.flat_logit_threshold)
                    rows.append(ContentRow(
                        pid, tap, "reciprocal", theta, rr.d_shape, rr.log_beta,
                        _following_fixed_direction(zr, zD, forward_direction),
                        ood_score(h_rev), None, rev_diag, reciprocal=True,
                        beta_identified=rr.beta_identified,
                        boundary_censored=rr.boundary_censored,
                        keys=keys, windows=windows, stratum=stratum))

            if wrong_pair_donor is not None:
                hW = wrong_pair_donor(pid)
                h_wrong, wrong_diag = content_dose_state(
                    hR, hW, u, theta,
                    c_abs_max=man.margins.c_abs_max,
                    phi_min=man.margins.phi_min, phi_max=man.margins.phi_max)
                if h_wrong is not None:
                    zw = run_state(rec_ids, h_wrong)
                    rw = solve_beta_shape(
                        zw, z0, beta_min=man.margins.beta_min,
                        beta_max=man.margins.beta_max,
                        flat_logit_threshold=man.margins.flat_logit_threshold)
                    rows.append(ContentRow(
                        pid, tap, "wrong_pair", theta, rw.d_shape, rw.log_beta,
                        _following_fixed_direction(zw, z0, forward_direction),
                        ood_score(h_wrong), None, wrong_diag,
                        beta_identified=rw.beta_identified,
                        boundary_censored=rw.boundary_censored,
                        keys=keys, windows=windows, stratum=stratum))

        # self donor (null) and full donor state (positive control)
        z_self = run_state(rec_ids, hR)
        rs = solve_beta_shape(z_self, z0, beta_min=man.margins.beta_min,
                              beta_max=man.margins.beta_max,
                              flat_logit_threshold=man.margins.flat_logit_threshold)
        rows.append(ContentRow(pid, tap, "self", 0.0, rs.d_shape, rs.log_beta,
                               _following_fixed_direction(z_self, z0, forward_direction),
                               ood_score(hR), None, {},
                               beta_identified=rs.beta_identified,
                               boundary_censored=rs.boundary_censored,
                               keys=keys, windows=windows, stratum=stratum))
        z_full = run_state(rec_ids, hD)
        rf = solve_beta_shape(z_full, z0, beta_min=man.margins.beta_min,
                              beta_max=man.margins.beta_max,
                              flat_logit_threshold=man.margins.flat_logit_threshold)
        rows.append(ContentRow(pid, tap, "full_state", float("nan"), rf.d_shape, rf.log_beta,
                               _following_fixed_direction(z_full, z0, forward_direction),
                               ood_score(hD), None, {},
                               beta_identified=rf.beta_identified,
                               boundary_censored=rf.boundary_censored,
                               keys=keys, windows=windows, stratum=stratum))
    return rows


def _content_rows_by_pair_slope(
    rows: Sequence[ContentRow], control: str, attr: str,
    *, negate: bool = False,
) -> list[Row]:
    """One slope per biological pair; dose points are repeated measures."""
    grouped: dict[tuple[str, str], list[ContentRow]] = {}
    for r in rows:
        if r.control == control and r.excluded is None:
            grouped.setdefault((r.pair_id, r.tap), []).append(r)
    out: list[Row] = []
    for (pid, tap), rs in grouped.items():
        x = np.asarray([r.theta for r in rs], dtype=float)
        if attr == "abs_log_beta":
            y = np.abs(np.asarray([r.log_beta for r in rs], dtype=float))
            valid = np.asarray([r.beta_identified and not r.boundary_censored for r in rs])
        else:
            y = np.asarray([getattr(r, attr) for r in rs], dtype=float)
            valid = np.ones(len(rs), dtype=bool)
        keep = np.isfinite(x) & np.isfinite(y) & valid
        if keep.sum() < 2 or x[keep].std() < 1e-12:
            continue
        value = float(np.polyfit(x[keep], y[keep], 1)[0])
        if negate:
            value = -value
        r0 = rs[0]
        out.append(Row(
            value, r0.keys or (f"pair:{pid}",), windows=r0.windows,
            stratum=r0.stratum, unit_id=f"{pid}|{tap}",
            meta={"pair_id": pid, "tap": tap, "control": control, "endpoint": attr},
        ))
    return out


def _content_level_rows(rows: Sequence[ContentRow], control: str, attr: str,
                        *, smallest_dose: bool = False) -> list[Row]:
    grouped: dict[tuple[str, str], list[ContentRow]] = {}
    for r in rows:
        if r.control == control and r.excluded is None:
            grouped.setdefault((r.pair_id, r.tap), []).append(r)
    out = []
    for (pid, tap), rs in grouped.items():
        finite = [r for r in rs if np.isfinite(getattr(r, attr)) and
                  (np.isfinite(r.theta) if smallest_dose else True)]
        if not finite:
            continue
        chosen = min(finite, key=lambda r: r.theta) if smallest_dose else finite[0]
        out.append(Row(
            float(getattr(chosen, attr)), chosen.keys or (f"pair:{pid}",),
            windows=chosen.windows, stratum=chosen.stratum,
            unit_id=f"{pid}|{tap}", meta={"pair_id": pid, "tap": tap},
        ))
    return out


def content_verdict(rows: Sequence[ContentRow], man: Manifest, *, tap: str = N.x(28),
                    seed: int = 42, n_boot: int = 10_000) -> dict:
    """Locked five-condition verdict for carrier-excluded natural content.

    Every inferential input is a per-pair slope/level.  Dose rows are never
    treated as independent replicates, reciprocal direction is measured on
    the same forward R->D axis, and every available control family must pass.
    """
    man.margins.require("delta_shape", "delta_path", "delta_spec")
    _require_exp1_alias(man)
    required = required_cluster_count(man, "step5_content")
    selected = [r for r in rows if r.tap == tap]
    if not selected:
        raise RuntimeError(f"no Step 5 rows at the pre-registered tap {tap}")

    nat_shape = _content_rows_by_pair_slope(selected, "natural_donor", "d_shape")
    nat_follow = _content_rows_by_pair_slope(selected, "natural_donor", "donor_following")
    nat_temp = _content_rows_by_pair_slope(selected, "natural_donor", "abs_log_beta")
    smallest = _content_level_rows(
        selected, "natural_donor", "donor_following", smallest_dose=True)
    reciprocal = _content_rows_by_pair_slope(
        selected, "reciprocal", "donor_following", negate=True)
    full = _content_level_rows(selected, "full_state", "donor_following")

    fam_rows: dict[str, list[Row]] = {
        "natural::dshape_slope": nat_shape,
        "natural::following_slope": nat_follow,
        "natural::smallest_dose_following": smallest,
        "natural::abs_logbeta_slope": nat_temp,
    }
    if reciprocal:
        fam_rows["reciprocal::negative_following_slope"] = reciprocal
    if full:
        fam_rows["positive_control::full_state_following"] = full

    controls_present = sorted({r.control for r in selected
                               if r.control in {"local_cov", "wrong_pair"}})
    specificity_names: list[str] = []
    for control in controls_present:
        ctl_shape = _content_rows_by_pair_slope(selected, control, "d_shape")
        ctl_follow = _content_rows_by_pair_slope(selected, control, "donor_following")
        if ctl_shape:
            name = f"specificity::dshape::{control}"
            fam_rows[name] = paired_difference_rows(nat_shape, ctl_shape)
            specificity_names.append(name)
        if ctl_follow:
            name = f"specificity::following::{control}"
            fam_rows[name] = paired_difference_rows(nat_follow, ctl_follow)
            specificity_names.append(name)
    if "local_cov" not in controls_present:
        raise RuntimeError("local-covariance equal-angle control is mandatory")

    estimates = family_cluster_bootstrap(
        fam_rows, seed=seed, n_boot=n_boot, required=required
    )
    required_margins = (
        "content_shape_slope", "content_following", "content_control",
        "content_log_beta_slope_equivalence",
    )
    missing_margins = [k for k in required_margins
                       if k not in man.margins.delta_spec]
    if missing_margins:
        raise RuntimeError(
            "Step 5 is missing development-calibrated delta_spec margins "
            f"{missing_margins}; D_shape, donor-following and log-beta have different "
            "units and cannot share an ad hoc fallback"
        )
    shape_margin = float(man.margins.delta_spec["content_shape_slope"])
    follow_margin = float(man.margins.delta_spec["content_following"])
    spec_margin = float(man.margins.delta_spec["content_control"])
    temp_eq_margin = float(
        man.margins.delta_spec["content_log_beta_slope_equivalence"]
    )

    c1 = (
        positive(estimates["natural::smallest_dose_following"], follow_margin) and
        positive(estimates["natural::following_slope"], follow_margin)
    )
    temp_eq = equivalence(
        estimates["natural::abs_logbeta_slope"], temp_eq_margin)
    c2 = positive(estimates["natural::dshape_slope"], shape_margin) and \
        temp_eq.verdict == "equivalent"
    spec_checks = {name: positive(estimates[name], spec_margin)
                   for name in specificity_names}
    c3 = bool(spec_checks) and all(spec_checks.values())
    c4 = (
        "reciprocal::negative_following_slope" in estimates and
        positive(estimates["reciprocal::negative_following_slope"], follow_margin)
    )
    c5 = (
        tap == N.x(28) and
        all(r.path == "free_suffix" for r in selected if r.control == "natural_donor") and
        positive(estimates["natural::following_slope"], follow_margin) and
        "positive_control::full_state_following" in estimates and
        positive(estimates["positive_control::full_state_following"], follow_margin)
    )

    # OOD model is descriptive only: rows/doses are repeated measures and
    # therefore this coefficient table is never used in the five-condition
    # locked verdict.
    nat = [r for r in selected if r.control == "natural_donor" and
           r.excluded is None and np.isfinite(r.d_shape)]
    X = np.column_stack([
        np.asarray([r.theta for r in nat]), np.asarray([r.ood_score for r in nat]),
        np.asarray([r.theta * r.ood_score for r in nat]), np.ones(len(nat)),
    ]) if nat else np.empty((0, 4))
    y = np.asarray([r.d_shape for r in nat])
    good = np.isfinite(X).all(1) & np.isfinite(y) if nat else np.zeros(0, dtype=bool)
    coef = np.linalg.lstsq(X[good], y[good], rcond=None)[0].tolist() \
        if good.sum() > 4 else [float("nan")] * 4

    conditions = {
        "1_smallest_dose_and_dose_response_follow_donor": c1,
        "2_shape_positive_temperature_slope_equivalent_zero": c2,
        "3_beats_every_matched_control_family": c3,
        "4_reciprocal_reverses_signed_direction": c4,
        "5_x28_free_suffix_reproduces_natural_path": c5,
    }
    beta_rows = [r for r in selected if r.control == "natural_donor" and r.excluded is None]
    return {
        "estimates": {k: v.as_row() for k, v in estimates.items()},
        "temperature_equivalence": temp_eq.verdict,
        "specificity_IUT": {"per_control": spec_checks,
                             "passed": c3, "controls": controls_present},
        "conditions": conditions,
        "content": all(conditions.values()),
        "effect_model_theta_ood_descriptive_only": {
            "theta": coef[0], "ood": coef[1], "interaction": coef[2], "intercept": coef[3]},
        "diagnostics": {
            "excluded_rate": float(np.mean([r.excluded is not None for r in selected])),
            "beta_unidentified_rate": float(np.mean([not r.beta_identified for r in beta_rows]))
                                      if beta_rows else float("nan"),
            "beta_boundary_censored_rate": float(np.mean([r.boundary_censored for r in beta_rows]))
                                           if beta_rows else float("nan"),
        },
        "exp1_bridge": {
            "legacy_alias": {"h27": N.x(28), "h28": N.x(29), "h30": N.x(31)},
            "tap": tap,
            "keyed_per_pair_slopes": {
                "natural_dshape": [{"unit_id": r.unit_id, "value": r.value,
                                    "keys": list(r.keys),
                                    "windows": [list(w) for w in r.all_windows()],
                                    "stratum": r.stratum} for r in nat_shape],
                "natural_following": [{"unit_id": r.unit_id, "value": r.value,
                                       "keys": list(r.keys),
                                       "windows": [list(w) for w in r.all_windows()],
                                       "stratum": r.stratum} for r in nat_follow],
            },
            "use": "join motif/task family labels without treating dose rows as replicates",
        },
    }
