"""Statistical helpers — bootstrap CIs, MWU+effect size, Cohen's d, binomial.

All functions return float (or tuple of floats) and accept numpy arrays /
1-D iterables. Use seeds where applicable for reproducibility.
"""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np

log = logging.getLogger(__name__)


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap mean and percentile CI.

    Args:
        values: iterable of numbers.
        n_boot: number of bootstrap resamples.
        ci: confidence level (e.g. 0.95).
        seed: RNG seed.

    Returns:
        (mean, low, high)
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = arr[idx].mean()
    alpha = 1.0 - ci
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return (float(arr.mean()), low, high)


def bootstrap_proportion_ci(
    successes: int,
    n: int,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for a proportion (Bernoulli draws).

    Args:
        successes: count of successes.
        n: number of trials.
        n_boot, ci, seed: as bootstrap_ci.

    Returns:
        (proportion, low, high)
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    p = successes / n
    samples = rng.binomial(n, p, size=n_boot) / n
    alpha = 1.0 - ci
    return (
        float(p),
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def mwu_with_effect(
    group_a: Iterable[float],
    group_b: Iterable[float],
    alternative: str = "two-sided",
) -> Tuple[float, float, float]:
    """Mann-Whitney U test with rank-biserial effect size.

    rank-biserial r = 1 - 2 * U / (n_a * n_b), where U is the smaller
    Mann-Whitney statistic. Reported with sign indicating direction
    (positive => group_a stochastically larger).

    Args:
        group_a, group_b: iterables of numbers.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        (U_statistic, p_value, rank_biserial_r)
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    res = mannwhitneyu(a, b, alternative=alternative)
    U = float(res.statistic)
    p = float(res.pvalue)
    n_a, n_b = a.size, b.size
    # rank-biserial r in [-1, 1]; positive => a > b
    r = 1.0 - 2.0 * U / (n_a * n_b)
    # Note: scipy U is U_a (count of a > b minus ties); flip sign to match
    # "positive => a larger" convention used here.
    r = -r
    return (U, p, float(r))


def cohens_d(
    a: Iterable[float],
    b: Iterable[float],
) -> float:
    """Pooled-SD Cohen's d for two independent samples.

    Returns:
        scalar (positive => a larger than b).
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    s_p = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                  / (a.size + b.size - 2))
    if s_p == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s_p)


def binomial_test_one_sided(
    successes: int,
    n: int,
    p_null: float,
) -> float:
    """One-sided binomial test, alternative='greater'.

    Args:
        successes: observed successes.
        n: trials.
        p_null: null probability.

    Returns:
        p-value (float).
    """
    from scipy.stats import binomtest

    if n <= 0:
        return float("nan")
    res = binomtest(int(successes), int(n), p_null, alternative="greater")
    return float(res.pvalue)
