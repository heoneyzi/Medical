"""Step 10 — reinforce and extend EXP1 after the raw-model mechanism is frozen.

This module is deliberately a *bridge*, not a second mechanism-discovery
pipeline.  Steps 0--9 establish the exact Evo 2 path without task labels.
Step 10 then asks three narrower questions:

1. do the original EXP1 observations replicate on independent units;
2. does b30 selectively contract output-irrelevant state variation; and
3. do the frozen scale/carrier/content/branch measurements predict why tasks
   or benchmarks prefer different layers?

No feature, layer or margin is selected here.  Those choices must already be
sealed in the manifest.  In particular, an out-of-fold association in this
module is not promoted to a causal claim unless the corresponding component
has passed its Step 4--9 necessity/rescue gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .manifest import Manifest
from .stats import (
    Estimate,
    Row,
    build_clusters,
    cluster_means,
    equivalence,
    family_cluster_bootstrap,
    paired_difference_rows,
    positive,
    required_cluster_count,
    simultaneous_equivalence,
)
from .taps import Evo2Runner


# EXP1 stored block outputs as h_i.  The exact-tap convention is immutable:
# changing this table would make the old and new experiments incomparable.
EXP1_LEGACY_TAPS: dict[str, str] = {
    "h27": N.x(28),
    "h28": N.x(29),
    "h29": N.x(30),
    "h30": N.x(31),
    "h31": N.x(32),
}


# ---------------------------------------------------------------------------
# A. Independent replication of EXP1 premises
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationClaim:
    claim_id: str
    description: str
    expected_direction: int  # +1, -1, or 0 for an equivalence/null premise
    equivalence_margin: float
    legacy_taps: tuple[str, ...] = ()
    exact_taps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_direction not in (-1, 0, 1):
            raise ValueError("expected_direction must be -1, 0 or 1")
        if not np.isfinite(self.equivalence_margin) or self.equivalence_margin <= 0:
            raise ValueError("equivalence_margin must be a positive calibrated value")
        if len(self.legacy_taps) != len(self.exact_taps):
            raise ValueError("legacy_taps and exact_taps must have the same length")
        for legacy, exact in zip(self.legacy_taps, self.exact_taps):
            if legacy in EXP1_LEGACY_TAPS and EXP1_LEGACY_TAPS[legacy] != exact:
                raise ValueError(
                    f"legacy mapping mismatch for {legacy}: expected "
                    f"{EXP1_LEGACY_TAPS[legacy]}, got {exact}"
                )


@dataclass(frozen=True)
class ReplicationResult:
    claim_id: str
    exp1: Estimate
    exp2: Estimate
    difference: Estimate  # exp2 - exp1
    direction_replicated: bool
    difference_equivalent: bool
    status: str
    note: str

    def as_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "exp1": self.exp1.as_row(),
            "exp2": self.exp2.as_row(),
            "exp2_minus_exp1": self.difference.as_row(),
            "direction_replicated": self.direction_replicated,
            "difference_equivalent": self.difference_equivalent,
            "status": self.status,
            "note": self.note,
        }


def _cluster_values(rows: Sequence[Row], *, required: int) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise RuntimeError("replication arm has no rows")
    ids = build_clusters(rows)
    values, strata = cluster_means(rows, ids)
    if len(values) < required:
        raise RuntimeError(
            f"replication arm has {len(values)} dependency clusters (< {required}); "
            "open only the next pre-registered overflow split"
        )
    if not np.isfinite(values).all():
        raise ValueError("replication rows contain non-finite cluster means")
    return values, strata


def _stratified_draw(
    values: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> float:
    picked = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        picked.append(rng.choice(values[idx], size=len(idx), replace=True))
    return float(np.concatenate(picked).mean())


def independent_replication(
    claim: ReplicationClaim,
    exp1_rows: Sequence[Row],
    exp2_rows: Sequence[Row],
    *,
    required_clusters_per_arm: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ReplicationResult:
    """Compare non-overlapping EXP1 and EXP2 dependency clusters.

    The two arms are bootstrapped independently and subtracted on every
    replicate.  Reusing EXP1's old p-value or treating windows/tokens as
    independent is intentionally impossible through this API.
    """

    if required_clusters_per_arm < 50:
        raise ValueError(
            "confirmatory EXP1 replication requires at least 50 dependency clusters "
            "per arm in addition to the development power result"
        )
    if n_boot < 200:
        raise ValueError("n_boot must be at least 200 for a replication interval")
    old, old_s = _cluster_values(exp1_rows, required=required_clusters_per_arm)
    new, new_s = _cluster_values(exp2_rows, required=required_clusters_per_arm)
    rng = np.random.default_rng(seed)
    old_boot = np.empty(n_boot)
    new_boot = np.empty(n_boot)
    for b in range(n_boot):
        old_boot[b] = _stratified_draw(old, old_s, rng)
        new_boot[b] = _stratified_draw(new, new_s, rng)
    diff_boot = new_boot - old_boot

    def est(point: float, boot: np.ndarray, k: int, n: int) -> Estimate:
        return Estimate(
            point=float(point),
            lo=float(np.quantile(boot, alpha / 2)),
            hi=float(np.quantile(boot, 1 - alpha / 2)),
            n_clusters=k,
            n_rows=n,
            method="independent stratified dependency-cluster bootstrap",
        )

    e_old = est(old.mean(), old_boot, len(old), len(exp1_rows))
    e_new = est(new.mean(), new_boot, len(new), len(exp2_rows))
    e_diff = est(new.mean() - old.mean(), diff_boot,
                 min(len(old), len(new)), len(exp1_rows) + len(exp2_rows))

    if claim.expected_direction > 0:
        direction = e_new.lo > 0
        opposite = e_new.hi < 0
    elif claim.expected_direction < 0:
        direction = e_new.hi < 0
        opposite = e_new.lo > 0
    else:
        direction = equivalence(e_new, claim.equivalence_margin).verdict == "equivalent"
        opposite = False
    diff_eq = equivalence(e_diff, claim.equivalence_margin).verdict == "equivalent"

    if direction and diff_eq:
        status = "reinforced"
        note = "independent direction/equivalence and cross-wave equivalence both pass"
    elif direction:
        status = "replicated-but-shifted"
        note = "direction replicates, but the new effect is not equivalent to EXP1"
    elif opposite:
        status = "contradicted"
        note = "the independent confidence interval supports the opposite direction"
    else:
        status = "unresolved"
        note = "neither replication nor contradiction is identified at calibrated precision"
    return ReplicationResult(
        claim.claim_id, e_old, e_new, e_diff, direction, diff_eq, status, note
    )


# ---------------------------------------------------------------------------
# B. E-PASS: what b30 selectively contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalPair:
    """A pair whose class/matching was frozen before the locked run."""

    pair_id: str
    match_id: str
    pair_class: str  # output_equivalent | output_distinct | composition_null
    left_ids: Tensor
    right_ids: Tensor
    target_position: int = -1
    keys: tuple[str, ...] = ()
    windows: tuple[tuple[str, int, int], ...] = ()
    stratum: str = "all"

    def __post_init__(self) -> None:
        allowed = {"output_equivalent", "output_distinct", "composition_null"}
        if self.pair_class not in allowed:
            raise ValueError(f"unknown pair_class {self.pair_class!r}")


@dataclass(frozen=True)
class PairGeometry:
    pair_id: str
    match_id: str
    pair_class: str
    tap: str
    view: str
    distance: float
    keys: tuple[str, ...] = ()
    windows: tuple[tuple[str, int, int], ...] = ()
    stratum: str = "all"


def _state_views(h: Tensor, carrier_axis: Tensor, eps: float) -> dict[str, Tensor]:
    h = h.double()
    u = carrier_axis.double()
    u = u / u.norm().clamp_min(eps)
    rms = h.square().mean().sqrt().clamp_min(eps)
    norm = h.norm().clamp_min(eps)
    rem = h - torch.dot(h, u) * u
    return {
        "raw": h,
        "rms_normalized": h / rms,
        "direction": h / norm,
        "carrier_removed": rem,
        "carrier_removed_direction": rem / rem.norm().clamp_min(eps),
    }


@torch.no_grad()
def collect_pair_geometry(
    runner: Evo2Runner,
    pairs: Sequence[NaturalPair],
    *,
    taps: Sequence[str],
    carrier_axis: Tensor,
    views: Sequence[str] = (
        "raw", "rms_normalized", "direction", "carrier_removed_direction"
    ),
    eps: float = 1e-12,
) -> list[PairGeometry]:
    """Measure matched natural-pair separation at exact taps.

    Pair membership is supplied, never inferred from locked logits here.  The
    caller must create and seal output-equivalent/distinct matches on the
    development split, including the h27/x28 distance match.
    """

    unknown = set(views) - {
        "raw", "rms_normalized", "direction", "carrier_removed",
        "carrier_removed_direction",
    }
    if unknown:
        raise ValueError(f"unknown state views: {sorted(unknown)}")
    out: list[PairGeometry] = []
    for pair in pairs:
        left = runner.run(pair.left_ids, taps=set(taps))
        right = runner.run(pair.right_ids, taps=set(taps))
        for tap in taps:
            if tap not in left.tensors or tap not in right.tensors:
                raise RuntimeError(f"required E-PASS tap {tap!r} was not captured")
            hl = left[tap].reshape(-1, left[tap].shape[-1])[pair.target_position]
            hr = right[tap].reshape(-1, right[tap].shape[-1])[pair.target_position]
            lv = _state_views(hl, carrier_axis, eps)
            rv = _state_views(hr, carrier_axis, eps)
            for view in views:
                distance = float((lv[view] - rv[view]).norm())
                keys = tuple(sorted(set(pair.keys) | {
                    f"pair:{pair.pair_id}", f"match:{pair.match_id}"
                }))
                out.append(PairGeometry(
                    pair.pair_id, pair.match_id, pair.pair_class, tap, view,
                    distance, keys, pair.windows, pair.stratum,
                ))
    return out


def _index_geometry(records: Sequence[PairGeometry]) -> dict[tuple[str, str, str], PairGeometry]:
    idx: dict[tuple[str, str, str], PairGeometry] = {}
    for r in records:
        key = (r.pair_id, r.tap, r.view)
        if key in idx:
            raise ValueError(f"duplicate pair geometry row {key}")
        if not np.isfinite(r.distance) or r.distance < 0:
            raise ValueError(f"invalid distance for {key}: {r.distance}")
        idx[key] = r
    return idx


def _contractions(
    records: Sequence[PairGeometry], *, before: str, after: str, view: str,
    eps: float = 1e-12,
) -> dict[str, tuple[PairGeometry, float]]:
    idx = _index_geometry(records)
    pairs = sorted({r.pair_id for r in records if r.view == view})
    out: dict[str, tuple[PairGeometry, float]] = {}
    for pid in pairs:
        a = idx.get((pid, before, view))
        b = idx.get((pid, after, view))
        if a is None or b is None:
            raise RuntimeError(f"pair {pid!r} is missing {before}/{after}/{view}")
        # Positive means that the pair became closer across the interval.
        out[pid] = (a, float(np.log((a.distance + eps) / (b.distance + eps))))
    return out


def selective_contraction_rows(
    records: Sequence[PairGeometry], *, before: str, after: str, view: str,
) -> list[Row]:
    """Matched difference: contraction(output-equivalent) - contraction(distinct)."""

    vals = _contractions(records, before=before, after=after, view=view)
    by_match: dict[str, dict[str, tuple[PairGeometry, float]]] = {}
    for rec, value in vals.values():
        if rec.pair_class in {"output_equivalent", "output_distinct"}:
            slot = by_match.setdefault(rec.match_id, {})
            if rec.pair_class in slot:
                raise ValueError(
                    f"match {rec.match_id!r} has duplicate {rec.pair_class} pairs"
                )
            slot[rec.pair_class] = (rec, value)
    rows: list[Row] = []
    for match_id, d in sorted(by_match.items()):
        if set(d) != {"output_equivalent", "output_distinct"}:
            raise RuntimeError(
                f"match {match_id!r} must contain one output-equivalent and one "
                "output-distinct pair"
            )
        eq, ceq = d["output_equivalent"]
        ds, cds = d["output_distinct"]
        rows.append(Row(
            ceq - cds,
            keys=tuple(sorted(set(eq.keys) | set(ds.keys))),
            windows=tuple(sorted(set(eq.windows) | set(ds.windows))),
            stratum="|".join(sorted({eq.stratum, ds.stratum})),
            unit_id=match_id,
            meta={
                "match_id": match_id, "before": before, "after": after,
                "view": view, "equivalent_pair": eq.pair_id,
                "distinct_pair": ds.pair_id,
            },
        ))
    if not rows:
        raise RuntimeError("no complete E-PASS matched sets")
    return rows


def selective_contraction_verdict(
    records: Sequence[PairGeometry],
    man: Manifest,
    *,
    views: Sequence[str] = (
        "raw", "rms_normalized", "direction", "carrier_removed_direction"
    ),
    handoff_start: str = N.x(28),
    pre_b30: str = N.x(30),
    post_b30: str = N.x(31),
    post_b31: str = N.x(32),
    seed: int = 42,
    n_boot: int = 10_000,
) -> dict:
    """Locked E-PASS test and exact b30 localisation.

    The primary effect is selective contraction at b30 (x30 -> x31) in all
    frozen state views.  A second contrast requires it to exceed the earlier
    x28 -> x30 contraction on the same matched sets.  x31 -> x32 is reported
    with an equivalence test but does not get silently assumed transparent.
    """

    man.margins.require("delta_spec")
    required = required_cluster_count(man, "step10_exp1_pass")
    pos_margin = man.margins.delta_spec.get("exp1_selective_contraction")
    loc_margin = man.margins.delta_spec.get("exp1_b30_localization")
    post_margin = man.margins.delta_spec.get("exp1_post_b31_equivalence")
    if pos_margin is None or loc_margin is None or post_margin is None:
        raise RuntimeError(
            "Step 10 needs development-calibrated margins "
            "exp1_selective_contraction, exp1_b30_localization and "
            "exp1_post_b31_equivalence"
        )

    families: dict[str, list[Row]] = {}
    for view in views:
        early = selective_contraction_rows(
            records, before=handoff_start, after=pre_b30, view=view)
        b30 = selective_contraction_rows(
            records, before=pre_b30, after=post_b30, view=view)
        post = selective_contraction_rows(
            records, before=post_b30, after=post_b31, view=view)
        total = selective_contraction_rows(
            records, before=handoff_start, after=post_b30, view=view)
        families[f"b30::{view}"] = b30
        families[f"b30_minus_early::{view}"] = paired_difference_rows(b30, early)
        families[f"handoff_total::{view}"] = total
        families[f"post_b31::{view}"] = post

    estimates = family_cluster_bootstrap(
        families, seed=seed, n_boot=n_boot, required=required
    )
    b30_pass = {
        view: positive(estimates[f"b30::{view}"], float(pos_margin)) for view in views
    }
    localized = {
        view: positive(estimates[f"b30_minus_early::{view}"], float(loc_margin))
        for view in views
    }
    post_eq = simultaneous_equivalence(
        {view: estimates[f"post_b31::{view}"] for view in views},
        float(post_margin),
    )
    return {
        "estimates": {k: v.as_row() for k, v in estimates.items()},
        "b30_selective_contraction_per_view": b30_pass,
        "b30_localized_beyond_earlier_handoff_per_view": localized,
        "post_b31_equivalence_per_view": {
            k: v.verdict for k, v in post_eq.items()
        },
        "exp1_b30_bottleneck_reinforced": bool(views) and all(b30_pass.values())
        and all(localized.values()),
        "b31_transparent_for_this_endpoint": bool(post_eq) and all(
            v.verdict == "equivalent" for v in post_eq.values()
        ),
        "interpretation_guard": (
            "Selective contraction strengthens EXP1 only after pair classes and x28-distance "
            "matches are frozen on development data; it does not by itself identify the HCL "
            "component, which remains a Step 8 necessity/rescue claim."
        ),
    }


# ---------------------------------------------------------------------------
# C. Why tasks/benchmarks differ: frozen-mechanism out-of-family prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskMechanismRow:
    task_id: str
    task_family: str
    target: float
    output_coupling: float
    baseline_accessibility: float
    scale: float
    carrier: float
    content: float
    b28: float
    b29: float
    b30: float
    b31: float
    endpoint_family: str = "all"
    circular: bool = False

    def feature(self, name: str) -> float:
        if not hasattr(self, name):
            raise KeyError(name)
        return float(getattr(self, name))


@dataclass(frozen=True)
class MechanismModel:
    name: str
    features: tuple[str, ...]


DEFAULT_MECHANISM_MODELS: tuple[MechanismModel, ...] = (
    MechanismModel("baseline", ("output_coupling", "baseline_accessibility")),
    MechanismModel("baseline_plus_scale", (
        "output_coupling", "baseline_accessibility", "scale")),
    MechanismModel("baseline_plus_carrier", (
        "output_coupling", "baseline_accessibility", "carrier")),
    MechanismModel("baseline_plus_content", (
        "output_coupling", "baseline_accessibility", "content")),
    MechanismModel("baseline_plus_branches", (
        "output_coupling", "baseline_accessibility", "b28", "b29", "b30", "b31")),
    MechanismModel("full", (
        "output_coupling", "baseline_accessibility", "scale", "carrier", "content",
        "b28", "b29", "b30", "b31")),
    MechanismModel("full_without_scale", (
        "output_coupling", "baseline_accessibility", "carrier", "content",
        "b28", "b29", "b30", "b31")),
    MechanismModel("full_without_carrier", (
        "output_coupling", "baseline_accessibility", "scale", "content",
        "b28", "b29", "b30", "b31")),
    MechanismModel("full_without_content", (
        "output_coupling", "baseline_accessibility", "scale", "carrier",
        "b28", "b29", "b30", "b31")),
    MechanismModel("full_without_branches", (
        "output_coupling", "baseline_accessibility", "scale", "carrier", "content")),
)


@dataclass(frozen=True)
class CrossValidatedTaskModel:
    name: str
    features: tuple[str, ...]
    predictions: dict[str, float]
    squared_error: dict[str, float]
    selected_lambda_by_outer_family: dict[str, float]
    r2: float
    mae: float


def _matrix(rows: Sequence[TaskMechanismRow], features: Sequence[str]) -> np.ndarray:
    X = np.asarray([[r.feature(f) for f in features] for r in rows], dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("task mechanism features contain non-finite values")
    return X


def _ridge_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lam: float
) -> np.ndarray:
    """Train-only standardisation with a centred, SVD-stable ridge fit."""

    mean = X_train.mean(0)
    scale = X_train.std(0)
    scale[scale < 1e-12] = 1.0
    A = (X_train - mean) / scale
    B = (X_test - mean) / scale
    y_mean = float(y_train.mean())
    yc = y_train - y_mean
    # Avoid normal equations: the late-stack predictors can be nearly
    # collinear by construction, exactly the regime where A.T@A squares the
    # condition number and may manufacture numerical "task explanations".
    U, singular, Vh = np.linalg.svd(A, full_matrices=False)
    filt = singular / (singular ** 2 + lam)
    coef = Vh.T @ (filt * (U.T @ yc))
    return y_mean + B @ coef


def _nested_group_predictions(
    rows: Sequence[TaskMechanismRow],
    model: MechanismModel,
    *,
    lambdas: Sequence[float],
) -> CrossValidatedTaskModel:
    families = sorted({r.task_family for r in rows})
    if len(families) < 4:
        raise RuntimeError(
            "nested leave-task-family-out validation needs at least four task families"
        )
    y = np.asarray([r.target for r in rows], dtype=float)
    X = _matrix(rows, model.features)
    pred = np.full(len(rows), np.nan)
    chosen: dict[str, float] = {}

    for outer in families:
        test = np.asarray([r.task_family == outer for r in rows])
        train = ~test
        train_families = sorted({r.task_family for i, r in enumerate(rows) if train[i]})
        losses: dict[float, list[float]] = {float(lam): [] for lam in lambdas}
        for inner in train_families:
            val = np.asarray([train[i] and r.task_family == inner
                              for i, r in enumerate(rows)])
            fit = train & ~val
            if fit.sum() <= len(model.features) or val.sum() == 0:
                # Ridge can fit p >= n, but such a fold carries essentially no
                # evidence for hyperparameter selection.  Fail rather than
                # let the pseudo-inverse manufacture a preferred lambda.
                raise RuntimeError(
                    f"inner fold {inner!r} has insufficient tasks for model {model.name!r}"
                )
            for lam in losses:
                phat = _ridge_predict(X[fit], y[fit], X[val], lam)
                losses[lam].append(float(np.mean((y[val] - phat) ** 2)))
        best = min(losses, key=lambda lam: (np.mean(losses[lam]), lam))
        chosen[outer] = float(best)
        pred[test] = _ridge_predict(X[train], y[train], X[test], best)

    if not np.isfinite(pred).all():
        raise RuntimeError("cross-validation left one or more tasks without a prediction")
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return CrossValidatedTaskModel(
        name=model.name,
        features=model.features,
        predictions={r.task_id: float(p) for r, p in zip(rows, pred)},
        squared_error={r.task_id: float((r.target - p) ** 2) for r, p in zip(rows, pred)},
        selected_lambda_by_outer_family=chosen,
        r2=1.0 - sse / max(sst, 1e-30),
        mae=float(np.mean(np.abs(y - pred))),
    )


def crossvalidated_task_mechanisms(
    rows: Sequence[TaskMechanismRow],
    *,
    models: Sequence[MechanismModel] = DEFAULT_MECHANISM_MODELS,
    lambdas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    required_tasks: int,
    required_families: int,
) -> dict[str, CrossValidatedTaskModel]:
    """Nested leave-task-family-out prediction of EXP1/benchmark variation.

    Circular tasks are rejected rather than silently filtered: the caller
    must explicitly pass the non-circular, pre-registered task panel.  This
    prevents a repeat of EXP1's 13-task correlation whose effective sample
    became ten after auditing three circular endpoints.
    """

    if any(r.circular for r in rows):
        bad = [r.task_id for r in rows if r.circular]
        raise ValueError(f"remove circular task definitions before fitting: {bad}")
    ids = [r.task_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id must be unique")
    if len(rows) < required_tasks:
        raise RuntimeError(
            f"only {len(rows)} non-circular tasks (< power requirement {required_tasks})"
        )
    families = {r.task_family for r in rows}
    if len(families) < required_families:
        raise RuntimeError(
            f"only {len(families)} task families (< requirement {required_families})"
        )
    if len(families) < 4:
        raise RuntimeError("at least four task families are required for nested CV")
    if not np.isfinite([r.target for r in rows]).all():
        raise ValueError("task targets contain non-finite values")
    if np.std([r.target for r in rows]) < 1e-12:
        raise ValueError("task target has no variation")
    if not lambdas or any((not np.isfinite(x)) or x < 0 for x in lambdas):
        raise ValueError("lambdas must be a non-empty sequence of finite non-negative values")

    return {
        model.name: _nested_group_predictions(rows, model, lambdas=lambdas)
        for model in models
    }


def _family_bootstrap_error_improvement(
    rows: Sequence[TaskMechanismRow],
    smaller: CrossValidatedTaskModel,
    larger: CrossValidatedTaskModel,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> Estimate:
    """MSE(smaller)-MSE(larger), resampling whole task families."""

    families = sorted({r.task_family for r in rows})
    by_family = {
        fam: [r.task_id for r in rows if r.task_family == fam] for fam in families
    }
    per_family = np.asarray([
        np.mean([
            smaller.squared_error[t] - larger.squared_error[t] for t in by_family[fam]
        ])
        for fam in families
    ])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(per_family, size=len(per_family), replace=True)
        boot[b] = pick.mean()
    return Estimate(
        float(per_family.mean()), float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)), len(families), len(rows),
        method="task-family cluster bootstrap of nested-CV error improvement",
    )


def task_mechanism_verdict(
    rows: Sequence[TaskMechanismRow],
    models: Mapping[str, CrossValidatedTaskModel],
    *,
    improvement_margin: float,
    unique_component_margin: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Does the frozen mechanism explain held-out task/benchmark differences?"""

    required_names = {
        "baseline", "full", "full_without_scale", "full_without_carrier",
        "full_without_content", "full_without_branches",
    }
    missing = required_names - set(models)
    if missing:
        raise RuntimeError(f"task mechanism verdict is missing models: {sorted(missing)}")
    if improvement_margin < 0 or unique_component_margin < 0:
        raise ValueError("prediction margins must be non-negative and development-calibrated")

    contrasts: dict[str, Estimate] = {}
    contrasts["full_vs_baseline"] = _family_bootstrap_error_improvement(
        rows, models["baseline"], models["full"], n_boot=n_boot, alpha=alpha, seed=seed
    )
    for i, component in enumerate(("scale", "carrier", "content", "branches"), 1):
        contrasts[f"unique_{component}"] = _family_bootstrap_error_improvement(
            rows, models[f"full_without_{component}"], models["full"],
            n_boot=n_boot, alpha=alpha, seed=seed + i,
        )
    full_pass = positive(contrasts["full_vs_baseline"], improvement_margin)
    unique = {
        component: positive(contrasts[f"unique_{component}"], unique_component_margin)
        for component in ("scale", "carrier", "content", "branches")
    }
    return {
        "models": {
            name: {
                "features": list(m.features), "oof_r2": m.r2, "oof_mae": m.mae,
                "selected_lambda_by_outer_family": m.selected_lambda_by_outer_family,
            }
            for name, m in models.items()
        },
        "error_improvement": {k: v.as_row() for k, v in contrasts.items()},
        "full_beats_exp1_baseline_out_of_family": full_pass,
        "unique_component_support": unique,
        "benchmark_heterogeneity_explained": full_pass and any(unique.values()),
        "interpretation_guard": (
            "This is held-out predictive explanation, not causality. A named component may be "
            "called mechanistic only if its independent Step 4--9 intervention gate passed."
        ),
    }


# ---------------------------------------------------------------------------
# D. Claim table: never hide a missing link
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimRule:
    claim: str
    required_pass: tuple[str, ...]
    required_equivalent: tuple[str, ...] = ()


EXP1_EXTENSION_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        "EXP1 late-stack event independently replicates",
        ("runtime_gate", "exp1_headlines_replicated", "exact_b28_onset"),
    ),
    ClaimRule(
        "b30 is a selective information bottleneck, not merely a large rotation",
        ("econd_stable", "b30_selective_contraction", "hcl_necessity_rescue"),
    ),
    ClaimRule(
        "scale, carrier and content are causally distinct transport components",
        ("carrier_gate", "content_gate", "same_recipient_transport"),
    ),
    ClaimRule(
        "frozen mechanisms explain why tasks or benchmarks differ",
        ("task_out_of_family_prediction", "same_recipient_transport"),
    ),
)


def exp1_extension_claim_table(
    evidence: Mapping[str, str],
    *,
    rules: Sequence[ClaimRule] = EXP1_EXTENSION_RULES,
) -> list[dict]:
    """Evaluate explicit pass/equivalent/fail/unidentified evidence states."""

    allowed = {"pass", "equivalent", "fail", "unidentified"}
    bad = {k: v for k, v in evidence.items() if v not in allowed}
    if bad:
        raise ValueError(f"invalid evidence states: {bad}")
    out = []
    for rule in rules:
        missing = [k for k in (*rule.required_pass, *rule.required_equivalent)
                   if k not in evidence]
        failed = [k for k in rule.required_pass if evidence.get(k) == "fail"]
        failed += [k for k in rule.required_equivalent
                   if evidence.get(k) not in {None, "equivalent"}]
        unresolved = [k for k in (*rule.required_pass, *rule.required_equivalent)
                      if evidence.get(k) == "unidentified"]
        pass_ok = all(evidence.get(k) == "pass" for k in rule.required_pass)
        eq_ok = all(evidence.get(k) == "equivalent" for k in rule.required_equivalent)
        if missing or unresolved:
            verdict = "unidentified"
        elif failed:
            verdict = "not-supported"
        elif pass_ok and eq_ok:
            verdict = "supported"
        else:
            verdict = "not-supported"
        out.append({
            "claim": rule.claim,
            "verdict": verdict,
            "missing": missing,
            "failed": failed,
            "unresolved": unresolved,
            "required_pass": list(rule.required_pass),
            "required_equivalent": list(rule.required_equivalent),
        })
    return out
