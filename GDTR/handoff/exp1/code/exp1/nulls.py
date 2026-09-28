"""Thresholds that come from a null, a gap, or a citation -- never from taste.

The rule this package now enforces
----------------------------------
Every cut-off in this codebase must be one of exactly three kinds, and each
kind carries its own obligation:

**derived**
    The cut is computed from the data's own null distribution and controls a
    *stated error rate*.  What gets reported is the error rate and the null, not
    the number.  ``0.01`` is not a justification; "the threshold at which the
    empirical false-discovery rate against a dinucleotide-preserving shuffle is
    5%" is, because the 5% is a claim someone can disagree with and the shuffle
    is a thing someone can rerun.

**separated**
    The cut sits inside a *gap* between two regimes, and the gap is measured.
    Here the reportable object is not the threshold but the interval of
    thresholds giving the same answer.  The handoff manuscript's onset detector
    is of this kind: it does not depend on T = 10, it depends on every
    T in [6, 200] agreeing, and that agreement is a measurement.

**inherited**
    The value is fixed by a specific published analysis and is kept *only* so
    the new numbers stay comparable with the old ones.  It must be cited, and it
    must be reported with a sensitivity sweep, so that no conclusion rests on
    the particular value.

Anything that is none of these three is removed, and replaced by a continuous
estimate with an interval or by an exact decomposition.  ``exp1.thresholds``
holds the registry and ``python -m exp1 audit-thresholds`` prints it, so the
rule is checkable rather than promised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# kind 1: derived from a null, controlling a stated error rate
# --------------------------------------------------------------------------

@dataclass
class NullCalibration:
    """A cut-off and the error rate it achieves against an explicit null."""

    threshold: float
    target_fdr: float
    achieved_fdr: float
    n_observed_above: int
    n_null_above: int
    n_observed: int
    n_null: int
    null_name: str
    feasible: bool
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        return dict(threshold=self.threshold, target_fdr=self.target_fdr,
                    achieved_fdr=self.achieved_fdr,
                    n_above=self.n_observed_above, null_above=self.n_null_above,
                    n_observed=self.n_observed, n_null=self.n_null,
                    null=self.null_name, feasible=self.feasible, note=self.note)


def fdr_threshold(observed: Sequence[float], null: Sequence[float],
                  target_fdr: float = 0.05, null_name: str = "",
                  grid: int = 400) -> NullCalibration:
    """Smallest cut whose empirical FDR against ``null`` is at most ``target``.

    The estimate is the standard permutation FDR: at a cut ``t``, the expected
    number of false positives is the null rate above ``t`` scaled to the size of
    the observed set, and the FDR is that over the observed count above ``t``.

    ``target_fdr`` is the only free number, and it is a claim about how much
    contamination the analysis tolerates -- the kind of number a reviewer can
    argue with.  If no cut reaches it, this returns ``feasible=False`` with the
    best achievable rate rather than silently returning the most extreme cut:
    a motif that cannot be called at any threshold is a finding, not a tuning
    problem.
    """
    obs = np.asarray(observed, dtype=np.float64)
    nul = np.asarray(null, dtype=np.float64)
    obs = obs[np.isfinite(obs)]
    nul = nul[np.isfinite(nul)]
    if obs.size == 0 or nul.size == 0:
        return NullCalibration(np.nan, target_fdr, np.nan, 0, 0,
                               int(obs.size), int(nul.size), null_name, False,
                               "no finite scores on one side")

    lo, hi = float(min(obs.min(), nul.min())), float(max(obs.max(), nul.max()))
    cuts = np.linspace(lo, hi, grid)
    scale = obs.size / nul.size
    best: Optional[NullCalibration] = None
    best_rate = np.inf
    for t in cuts:
        n_obs = int((obs >= t).sum())
        n_nul = int((nul >= t).sum())
        if n_obs == 0:
            continue
        fdr = min(1.0, (n_nul * scale) / n_obs)
        cal = NullCalibration(float(t), target_fdr, float(fdr), n_obs, n_nul,
                              int(obs.size), int(nul.size), null_name,
                              fdr <= target_fdr)
        if fdr <= target_fdr:
            return cal                      # cuts ascend, so this is the smallest
        if fdr < best_rate:
            best_rate, best = fdr, cal
    if best is None:
        return NullCalibration(np.nan, target_fdr, np.nan, 0, 0,
                               int(obs.size), int(nul.size), null_name, False,
                               "no cut retains any observation")
    best.note = (f"target FDR {target_fdr} is unreachable; the best achievable "
                 f"is {best.achieved_fdr:.3f}. Report this rather than moving "
                 "the target.")
    return best


def null_exceedance(observed: float, null: Sequence[float],
                    alpha: float = 0.05) -> Dict[str, object]:
    """Is one statistic beyond the ``1 - alpha`` quantile of its own null?

    Used where a yes/no is unavoidable -- "did this probe decode anything" --
    so that the boundary is a quantile of a computed null rather than a number
    chosen because it looked reasonable.
    """
    nul = np.asarray(null, dtype=np.float64)
    nul = nul[np.isfinite(nul)]
    if nul.size < 5 or not np.isfinite(observed):
        return dict(exceeds=False, quantile=np.nan, cut=np.nan, n_null=int(nul.size),
                    reason="null too small to calibrate against")
    cut = float(np.quantile(nul, 1.0 - alpha))
    q = float((nul < observed).mean())
    return dict(exceeds=bool(observed > cut), cut=cut, alpha=alpha,
                empirical_quantile=q, n_null=int(nul.size),
                p_one_sided=float((np.sum(nul >= observed) + 1) / (nul.size + 1)))


# --------------------------------------------------------------------------
# kind 2: a cut inside a measured gap
# --------------------------------------------------------------------------

@dataclass
class Separation:
    """Two regimes, the gap between them, and every cut that respects it."""

    answer: Optional[int]
    below_max: float           # largest value on the low side
    at_value: float            # the value that defines the answer
    gap_factor: float          # at_value / below_max
    t_interval: Tuple[float, float]
    separated: bool
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        return dict(answer=self.answer, below_max=self.below_max,
                    at_value=self.at_value, gap_factor=self.gap_factor,
                    t_lo=self.t_interval[0], t_hi=self.t_interval[1],
                    separated=self.separated, note=self.note)


def separating_interval(values: Sequence[float], answer_index: int
                        ) -> Separation:
    """The full set of thresholds that select ``answer_index`` and no earlier one.

    A threshold detector is only honest when its answer does not depend on the
    threshold.  Rather than fixing T and hoping, this measures the interval
    ``(max value before the answer, value at the answer]`` -- every T in it
    returns the same index -- and reports the gap as a factor.

    The handoff manuscript argues exactly this way for its onset rule (largest
    pre-onset ratio 3.50 against onset ratios 214-267, so every T in [6, 200]
    agrees).  This turns that argument into a computation, so that on a new
    model or a new chromosome the claim is re-established rather than assumed.
    """
    v = np.asarray(values, dtype=np.float64)
    if not (0 <= answer_index < v.size):
        return Separation(None, np.nan, np.nan, np.nan, (np.nan, np.nan), False,
                          "index out of range")
    before = v[:answer_index]
    before = before[np.isfinite(before)]
    at = float(v[answer_index])
    below = float(np.nanmax(before)) if before.size else 0.0
    gap = at / below if below > 0 else np.inf
    # Any T strictly above the largest pre-answer value and at most the answer's
    # own value selects this index; the interval is open below and closed above.
    lo = float(np.nextafter(below, np.inf))
    return Separation(int(answer_index), below, at, float(gap), (lo, at),
                      separated=bool(gap > 1.0),
                      note=("the answer is the same for every threshold in "
                            f"({below:.4g}, {at:.4g}]; gap factor {gap:.4g}"))


def variance_separation(curve: np.ndarray, per_position: np.ndarray
                        ) -> Dict[str, object]:
    """Does a curve move more across blocks than it wobbles within a position?

    Replaces a fixed "minimum dynamic range".  The question a range was standing
    in for is whether the block-to-block movement is large against the noise the
    same quantity shows at a fixed block, and that is a variance ratio, which
    needs no constant: an F-like statistic whose reference is 1.

    ``curve``          mean value per block, shape [L]
    ``per_position``   values per position per block, shape [N, L]
    """
    c = np.asarray(curve, dtype=np.float64)
    m = np.asarray(per_position, dtype=np.float64)
    between = float(np.nanvar(c))
    within = float(np.nanmean(np.nanvar(m, axis=0)))
    ratio = between / within if within > 0 else np.inf
    return dict(between_block_variance=between, within_position_variance=within,
                variance_ratio=ratio,
                informative=bool(ratio > 1.0),
                note=("the curve varies across blocks more than the same quantity "
                      "varies across positions at a fixed block"
                      if ratio > 1.0 else
                      "block-to-block movement is smaller than the spread at a "
                      "single block; this curve carries no usable depth signal "
                      "and must not be thresholded"))


# --------------------------------------------------------------------------
# null generators
# --------------------------------------------------------------------------

def dinucleotide_null_scores(seq_upper: np.ndarray, scorer: Callable,
                             n_shuffles: int = 5, seed: int = 42) -> np.ndarray:
    """Scores of the same scorer on dinucleotide-preserving shuffles.

    The right null for a motif score: it holds the composition and the
    dinucleotide structure that a PWM is most easily fooled by, and destroys
    only the arrangement the motif is supposed to detect.
    """
    from .variants import v_shuffle_di

    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for _ in range(max(n_shuffles, 1)):
        out.append(np.asarray(scorer(v_shuffle_di(seq_upper, rng)),
                              dtype=np.float64))
    return np.concatenate(out) if out else np.zeros(0)


def within_window_permutation_null(fit: Callable[[np.ndarray], float],
                                   y: np.ndarray, groups: np.ndarray,
                                   n: int = 20, seed: int = 42) -> np.ndarray:
    """Scores under repeated within-window label permutation.

    ``fit`` takes a permuted label vector and returns a score.  Permuting inside
    the window keeps window identity, which is the thing positions cluster in,
    so the null answers "could this score arise with the labels scrambled but
    the clustering intact".
    """
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(max(n, 1)):
        yp = np.asarray(y).copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = rng.permutation(yp[m])
        out.append(float(fit(yp)))
    return np.asarray(out, dtype=np.float64)
