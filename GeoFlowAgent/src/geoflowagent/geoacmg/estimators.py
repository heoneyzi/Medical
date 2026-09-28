"""Interval and test estimators, carried over from the phase-1 analysis code.

Everything here resamples *clusters*, never rows.  In this study the cluster is
the gene: two variants in BRCA2 are not two independent observations, and an
interval that pretends otherwise is too narrow by a factor that grows with the
number of variants per gene.

These are deliberately plain functions.  The layer that turns them into
:class:`~geoflowagent.geoacmg.claims.Finding` objects lives in
:mod:`geoflowagent.geoacmg.inference`, so that an estimator can be unit-tested
without a claim registry and a claim can be adjudicated without re-running an
estimator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only when SciPy is installed
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


def _z(quantile: float) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.ppf(quantile))
    # Acklam's inverse normal CDF approximation; |error| < 1.15e-9.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if quantile < p_low:
        q = math.sqrt(-2 * math.log(quantile))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if quantile > p_high:
        q = math.sqrt(-2 * math.log(1 - quantile))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = quantile - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    units: int
    method: str

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "excludes_zero": self.excludes_zero}


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[Any] | None = None,
    *,
    statistic: Callable[[np.ndarray], float] = lambda array: float(np.mean(array)),
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 17,
    method: str = "bca",
) -> Interval:
    """Cluster bootstrap confidence interval (BCa by default).

    ``clusters`` defaults to one cluster per observation, i.e. the ordinary
    nonparametric bootstrap over independent tasks.
    """

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    keys = list(clusters) if clusters is not None else list(range(array.size))
    if len(keys) != array.size:
        raise ValueError("clusters must align with values")
    unique = sorted({str(key) for key in keys})
    index_by_cluster = {name: [] for name in unique}
    for position, key in enumerate(keys):
        index_by_cluster[str(key)].append(position)
    groups = [np.asarray(index_by_cluster[name], dtype=int) for name in unique]
    observed = statistic(array)

    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    count = len(groups)
    for draw in range(resamples):
        picks = rng.integers(0, count, size=count)
        indices = np.concatenate([groups[index] for index in picks])
        samples[draw] = statistic(array[indices])

    lower_q, upper_q = alpha / 2, 1 - alpha / 2
    if method == "percentile" or count < 3:
        low, high = np.quantile(samples, [lower_q, upper_q])
        return Interval(observed, float(low), float(high), count, "percentile")

    proportion = float(np.mean(samples < observed))
    proportion = min(max(proportion, 1e-6), 1 - 1e-6)
    bias = _z(proportion)
    jackknife = np.empty(count, dtype=float)
    for position in range(count):
        indices = np.concatenate([groups[other] for other in range(count) if other != position])
        jackknife[position] = statistic(array[indices])
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * (float(np.sum(centered**2)) ** 1.5)
    acceleration = float(np.sum(centered**3)) / denominator if denominator > 0 else 0.0

    def adjust(quantile: float) -> float:
        z_q = _z(quantile)
        adjusted = bias + (bias + z_q) / max(1e-12, 1 - acceleration * (bias + z_q))
        return float(min(max(_normal_cdf(adjusted), 1e-6), 1 - 1e-6))

    low, high = np.quantile(samples, [adjust(lower_q), adjust(upper_q)])
    return Interval(observed, float(low), float(high), count, "bca")


def paired_difference(
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence[Any] | None = None,
    **kwargs: Any,
) -> Interval:
    """Bootstrap CI for the paired mean difference ``left - right``."""

    first = np.asarray(list(left), dtype=float)
    second = np.asarray(list(right), dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired samples must have the same length")
    return cluster_bootstrap(first - second, clusters, **kwargs)


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> Interval:
    if total <= 0:
        raise ValueError("total must be positive")
    z = _z(1 - alpha / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    half /= denominator
    return Interval(proportion, centre - half, centre + half, total, "wilson")


def mcnemar(left: Sequence[bool], right: Sequence[bool], *, exact: bool = True) -> dict[str, Any]:
    """Paired binary comparison; exact binomial by default."""

    first = [bool(value) for value in left]
    second = [bool(value) for value in right]
    if len(first) != len(second):
        raise ValueError("paired samples must have the same length")
    b = sum(1 for x, y in zip(first, second, strict=True) if x and not y)
    c = sum(1 for x, y in zip(first, second, strict=True) if y and not x)
    total = b + c
    if total == 0:
        return {"b": b, "c": c, "p_value": 1.0, "method": "no_discordant_pairs",
                "difference": 0.0, "n": len(first)}
    if exact:
        tail = sum(math.comb(total, k) for k in range(min(b, c) + 1)) * 0.5**total
        p_value = min(1.0, 2 * tail)
        method = "exact_binomial"
    else:
        statistic = (abs(b - c) - 1) ** 2 / total
        p_value = math.erfc(math.sqrt(statistic / 2))
        method = "chi_square_continuity_corrected"
    return {
        "b": b,
        "c": c,
        "p_value": float(p_value),
        "method": method,
        "difference": (b - c) / len(first),
        "n": len(first),
    }


def tost(
    differences: Sequence[float], margin: float, *, alpha: float = 0.05
) -> dict[str, Any]:
    """Two one-sided tests for equivalence within +/- ``margin``."""

    array = np.asarray(list(differences), dtype=float)
    n = array.size
    if n < 2:
        raise ValueError("need at least two observations")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    stderr = sd / math.sqrt(n) if sd > 0 else 1e-12
    t_low = (mean + margin) / stderr
    t_high = (mean - margin) / stderr
    if _scipy_stats is not None:
        p_low = float(_scipy_stats.t.sf(t_low, df=n - 1))
        p_high = float(_scipy_stats.t.cdf(t_high, df=n - 1))
    else:  # normal approximation
        p_low = 1.0 - _normal_cdf(t_low)
        p_high = _normal_cdf(t_high)
    p_value = max(p_low, p_high)
    return {
        "mean_difference": mean,
        "margin": margin,
        "p_value": p_value,
        "equivalent": p_value < alpha,
        "n": n,
        "sd": sd,
        "method": "tost_t" if _scipy_stats is not None else "tost_normal",
    }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    total = len(indexed)
    adjusted = [0.0] * total
    running = 0.0
    for rank, (position, value) in enumerate(indexed):
        candidate = (total - rank) * value
        running = max(running, min(1.0, candidate))
        adjusted[position] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    total = len(indexed)
    adjusted = [0.0] * total
    running = 1.0
    for rank, (position, value) in enumerate(indexed):
        candidate = value * total / (total - rank)
        running = min(running, min(1.0, candidate))
        adjusted[position] = running
    return adjusted


def sample_size_paired_continuous(sd: float, delta: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """n for a paired mean difference (task-macro metrics)."""

    if sd <= 0 or delta <= 0:
        raise ValueError("sd and delta must be positive")
    return math.ceil(((_z(1 - alpha / 2) + _z(power)) * sd / delta) ** 2)


def sample_size_mcnemar(discordance: float, delta: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """n for a paired binary difference (Connor, 1987 approximation)."""

    if not 0 < discordance < 1:
        raise ValueError("discordance must be a proportion")
    if delta <= 0 or delta >= discordance:
        raise ValueError("delta must be positive and smaller than the discordance rate")
    numerator = _z(1 - alpha / 2) * math.sqrt(discordance) + _z(power) * math.sqrt(
        discordance - delta * delta
    )
    return math.ceil(numerator**2 / (delta * delta))


def sample_size_tost(sd: float, margin: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    if sd <= 0 or margin <= 0:
        raise ValueError("sd and margin must be positive")
    return math.ceil(((_z(1 - alpha) + _z((1 + power) / 2)) * sd / margin) ** 2)


def permutation_null(
    statistic: Callable[[Sequence[Any]], float],
    values: Sequence[Any],
    labels: Sequence[Any],
    *,
    draws: int = 200,
    seed: int = 17,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """Locate an observed statistic inside its own label-permutation null.

    Used wherever "chance level" would otherwise have to be assumed.  For a
    multi-class probe, chance is not 0.5 and depends on the class balance; for a
    distance separation, it depends on the pair sampling.  Permuting the labels
    and re-running the *same* estimator gives the null that the observed value
    should be read against, with no assumption at all.

    The p-value is the add-one-corrected tail fraction, so it is never zero and
    never claims more resolution than ``draws`` supports.
    """

    if len(values) != len(labels):
        raise ValueError("values and labels must be the same length")
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    observed = float(statistic(list(zip(values, labels, strict=True))))
    null: list[float] = []
    order = np.arange(len(labels))
    for _ in range(draws):
        rng.shuffle(order)
        shuffled = [labels[index] for index in order]
        null.append(float(statistic(list(zip(values, shuffled, strict=True)))))
    array = np.asarray(null, dtype=float)
    if greater_is_better:
        tail = int(np.sum(array >= observed))
    else:
        tail = int(np.sum(array <= observed))
    p_value = (tail + 1.0) / (draws + 1.0)
    return {
        "observed": observed,
        "null_mean": float(array.mean()),
        "null_sd": float(array.std(ddof=1)) if draws > 1 else 0.0,
        "null_quantiles": {
            "q05": float(np.quantile(array, 0.05)),
            "q50": float(np.quantile(array, 0.50)),
            "q95": float(np.quantile(array, 0.95)),
        },
        "draws": draws,
        "p_value": float(p_value),
        "greater_is_better": greater_is_better,
    }


def cluster_slope(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence[Any],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Interval:
    """OLS slope of ``y`` on ``x`` with a cluster bootstrap interval.

    This is the estimator behind every moderator analysis in the plan: does the
    benefit of an asymmetric geometry grow with a task's measured irreversibility,
    does the benefit of plan generation grow with the measured quality of the
    representation's geometry.  Reporting a slope with an interval is what turns
    a bake-off into an explanation.
    """

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must be the same length")
    if x_array.size < 3:
        raise ValueError("a slope needs at least three points")

    def _slope(indices: np.ndarray) -> float:
        xs, ys = x_array[indices], y_array[indices]
        spread = float(((xs - xs.mean()) ** 2).sum())
        if spread <= 0.0:
            return float("nan")
        return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / spread)

    point = _slope(np.arange(x_array.size))
    groups: dict[Any, list[int]] = {}
    for index, key in enumerate(clusters):
        groups.setdefault(key, []).append(index)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        picked = rng.integers(0, len(keys), size=len(keys))
        indices = np.concatenate([np.asarray(groups[keys[i]], dtype=int) for i in picked])
        value = _slope(indices)
        if not math.isnan(value):
            draws.append(value)
    if len(draws) < 10:
        return Interval(estimate=point, low=float("nan"), high=float("nan"),
                        method="cluster_slope", units=len(keys))
    array = np.sort(np.asarray(draws, dtype=float))
    low = float(np.quantile(array, alpha / 2.0))
    high = float(np.quantile(array, 1.0 - alpha / 2.0))
    return Interval(estimate=point, low=low, high=high, method="cluster_slope", units=len(keys))
