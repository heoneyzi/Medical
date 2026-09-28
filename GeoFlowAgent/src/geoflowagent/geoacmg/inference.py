"""Estimators to findings.

Thin adapters so every experiment reports its result the same way.  Each one
takes paired measurements and a cluster label per measurement, and returns a
:class:`Finding` that the adjudicator can read.  The cluster label is required,
not optional: forgetting it is how a study ends up reporting intervals that are
too narrow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from geoflowagent.geoacmg import estimators
from geoflowagent.geoacmg.claims import Evidence, Finding, Role


def _check_paired(left: Sequence[float], right: Sequence[float], clusters: Sequence[Any]) -> None:
    if not (len(left) == len(right) == len(clusters)):
        raise ValueError(
            "paired contrast needs one value per side per unit and a cluster label for each; "
            f"got {len(left)}, {len(right)}, {len(clusters)}"
        )
    if not left:
        raise ValueError("paired contrast over an empty sample")


def paired_contrast(
    *,
    claim_id: str,
    name: str,
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence[Any],
    unit: str = "gene",
    role: Role = Role.EXPLORATORY,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
    detail: dict[str, Any] | None = None,
) -> Finding:
    """``left - right`` per unit, with a cluster bootstrap interval.

    Both arms must be evaluated on the *same* units in the same order.  That is
    what makes this paired, and pairing is what removes between-task difficulty
    from the contrast.
    """

    _check_paired(left, right, clusters)
    if role is Role.PRIMARY and len(set(clusters)) < 3:
        # With fewer than three clusters the bootstrap falls back to percentiles,
        # and with one cluster every resample is the same sample: the interval is
        # zero-width, so any difference at all "excludes zero". Some exploratory
        # contrasts are deliberately single-corpus and say so in their detail;
        # none of them may be promoted to decide a claim.
        raise ValueError(
            f"finding {name!r} claims primary role with {len(set(clusters))} cluster(s); "
            "a cluster bootstrap needs at least three independent units, otherwise the "
            "interval is degenerate and excludes zero by construction"
        )
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    interval = estimators.cluster_bootstrap(
        differences, clusters, resamples=resamples, alpha=alpha, seed=seed
    )
    return Finding(
        claim_id=claim_id,
        name=name,
        evidence=Evidence.PAIRED_INTERVAL,
        estimate=interval.estimate,
        ci_low=interval.low,
        ci_high=interval.high,
        unit=unit,
        n_units=interval.units,
        role=role,
        method=f"paired cluster bootstrap ({interval.method}), {resamples} resamples",
        detail={
            "left_mean": sum(map(float, left)) / len(left),
            "right_mean": sum(map(float, right)) / len(right),
            "n_observations": len(differences),
            **(detail or {}),
        },
    )


def null_referenced(
    *,
    claim_id: str,
    name: str,
    statistic: Callable[[Sequence[Any]], float],
    values: Sequence[Any],
    labels: Sequence[Any],
    unit: str,
    n_units: int,
    role: Role = Role.EXPLORATORY,
    draws: int = 200,
    seed: int = 17,
    greater_is_better: bool = True,
    detail: dict[str, Any] | None = None,
) -> Finding:
    """Locate a statistic in its own label-permutation null.

    Use this wherever the alternative would be to assume a chance level.
    """

    result = estimators.permutation_null(
        statistic, values, labels, draws=draws, seed=seed, greater_is_better=greater_is_better
    )
    return Finding(
        claim_id=claim_id,
        name=name,
        evidence=Evidence.NULL_POSITION,
        estimate=result["observed"],
        p_value=result["p_value"],
        null_mean=result["null_mean"],
        null_draws=result["draws"],
        unit=unit,
        n_units=n_units,
        role=role,
        method=f"label-permutation null, {draws} draws, add-one corrected",
        detail={**{k: result[k] for k in ("null_sd", "null_quantiles")}, **(detail or {})},
    )


def equivalence(
    *,
    claim_id: str,
    name: str,
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence[Any],
    margin: float,
    margin_source: str,
    unit: str = "gene",
    role: Role = Role.EXPLORATORY,
    alpha: float = 0.05,
) -> Finding:
    """Two one-sided tests, for when "no difference" is the claim.

    ``margin_source`` is mandatory: an equivalence margin decides an outcome, so
    it must be justified rather than chosen.
    """

    if not margin_source.strip():
        raise ValueError("an equivalence margin must carry its justification")
    _check_paired(left, right, clusters)
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    result = estimators.tost(differences, margin=margin, alpha=alpha)
    return Finding(
        claim_id=claim_id,
        name=name,
        evidence=Evidence.EQUIVALENCE,
        estimate=sum(differences) / len(differences),
        p_value=float(result["p_value"]),
        unit=unit,
        n_units=len(set(clusters)),
        role=role,
        method=f"TOST at margin {margin:g}",
        detail={"margin": margin, "margin_source": margin_source, **result},
    )


def moderator_slope(
    *,
    claim_id: str,
    name: str,
    moderator: Sequence[float],
    benefit: Sequence[float],
    clusters: Sequence[Any],
    moderator_name: str,
    unit: str = "gene",
    role: Role = Role.EXPLORATORY,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Finding:
    """Does the benefit grow with a measured property of the task?

    This is the estimator behind the geometry claim (does an asymmetric distance
    help more where the task graph is more irreversible) and behind the linkage
    claim (does plan generation help more where the representation's geometry is
    better aligned).  A slope with an interval says *when* something helps, which
    is a stronger result than saying that on average it did.
    """

    interval = estimators.cluster_slope(
        moderator, benefit, clusters, resamples=resamples, alpha=alpha, seed=seed
    )
    return Finding(
        claim_id=claim_id,
        name=name,
        evidence=Evidence.PAIRED_INTERVAL,
        estimate=interval.estimate,
        ci_low=interval.low,
        ci_high=interval.high,
        unit=unit,
        n_units=interval.units,
        role=role,
        method=f"OLS slope with cluster bootstrap, {resamples} resamples",
        detail={"moderator": moderator_name, "n_observations": len(benefit)},
    )


def adjust_exploratory(findings: Sequence[Finding], *, method: str = "holm") -> dict[str, float]:
    """Multiplicity adjustment over the exploratory findings that carry p-values.

    Primary findings are pre-registered and are not adjusted; everything else is.
    The adjusted values are reported alongside the raw ones rather than replacing
    them, so a reader can see both.
    """

    targets = [f for f in findings if f.role is Role.EXPLORATORY and f.p_value is not None]
    if not targets:
        return {}
    raw = [float(f.p_value) for f in targets]
    adjusted = estimators.holm(raw) if method == "holm" else estimators.benjamini_hochberg(raw)
    return {f.name: value for f, value in zip(targets, adjusted, strict=True)}
