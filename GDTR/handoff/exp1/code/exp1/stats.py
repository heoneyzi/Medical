"""Window-clustered statistics, matching the protocol of the follow-up manuscript.

Why this module is deliberately rigid
-------------------------------------
Positions inside a 6 kb window are not independent: the crossing rate is
overdispersed against binomial by ~266x, lag-1 autocorrelation is ~0.53, and
window identity alone explains ~10.6% of the variance.  A pooled p-value of
3e-281 became p = 0.055 once positions were compared within the windows they
came from.  So:

* the resampling unit is the **window**, never the position;
* the estimand is the **within-window paired effect size**;
* windows lacking either class are **dropped and counted**, never imputed;
* no position-level p-value is produced by anything in this file.

Keeping the estimator identical to the published one is itself an asset: new
numbers are directly comparable to the ones already in the manuscripts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
MIN_N_PER_CLASS = 30


# --------------------------------------------------------------------------
# effect size
# --------------------------------------------------------------------------

@dataclass
class WindowEffect:
    """Result of one within-window paired contrast."""

    target: str
    baseline: str
    value: str
    mean_d: float
    ci_lo: float
    ci_hi: float
    n_windows_used: int
    n_windows_dropped: int
    per_window: Optional[np.ndarray] = None

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("per_window")
        return d


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    num = (na - 1) * va + (nb - 1) * vb
    den = na + nb - 2
    if den <= 0:
        return float("nan")
    return float(np.sqrt(num / den))


def per_window_d(
    df: pd.DataFrame,
    value: str,
    target: str,
    baseline: str = "intron",
    context_col: str = "context",
    window_col: str = "window_id",
    min_n: int = MIN_N_PER_CLASS,
) -> pd.DataFrame:
    """Standardised difference of ``value`` between ``target`` and ``baseline``.

    One row per usable window.  ``used=False`` rows record why a window dropped
    out, so the drop count can be reported with every estimate.
    """
    rows: List[Dict[str, object]] = []
    for wid, g in df.groupby(window_col, sort=True):
        a = g.loc[g[context_col] == target, value].to_numpy(dtype=np.float64)
        b = g.loc[g[context_col] == baseline, value].to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < min_n or len(b) < min_n:
            rows.append(dict(window_id=wid, d=np.nan, n_target=len(a), n_baseline=len(b),
                             used=False, reason="count"))
            continue
        sd = _pooled_sd(a, b)
        if not np.isfinite(sd) or sd <= 0:
            rows.append(dict(window_id=wid, d=np.nan, n_target=len(a), n_baseline=len(b),
                             used=False, reason="zero_sd"))
            continue
        rows.append(dict(window_id=wid, d=float((a.mean() - b.mean()) / sd),
                         n_target=len(a), n_baseline=len(b), used=True, reason=""))
    return pd.DataFrame(rows)


def cluster_bootstrap_mean(
    d: Sequence[float],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile interval for the mean of per-window effect sizes."""
    x = np.asarray([v for v in d if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return dict(mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(b, x.size))
    boot = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(x.mean()), ci_lo=float(lo), ci_hi=float(hi), n=int(x.size))


def window_effect(
    df: pd.DataFrame,
    value: str,
    target: str,
    baseline: str = "intron",
    **kwargs,
) -> WindowEffect:
    pw = per_window_d(df, value=value, target=target, baseline=baseline, **kwargs)
    used = pw.loc[pw["used"], "d"].to_numpy()
    bs = cluster_bootstrap_mean(used)
    return WindowEffect(
        target=target,
        baseline=baseline,
        value=value,
        mean_d=bs["mean"],
        ci_lo=bs["ci_lo"],
        ci_hi=bs["ci_hi"],
        n_windows_used=int(pw["used"].sum()),
        n_windows_dropped=int((~pw["used"]).sum()),
        per_window=used,
    )


# --------------------------------------------------------------------------
# covariate adjustment
# --------------------------------------------------------------------------

def residualise_within_window(
    df: pd.DataFrame,
    value: str,
    covariates: Sequence[str],
    window_col: str = "window_id",
    out_col: Optional[str] = None,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Regress ``value`` on ``covariates`` **inside each window** and keep the residual.

    Adjustment happens within the window because that is also the comparison
    unit; adjusting globally would let between-window composition differences
    leak back in.  A tiny ridge term keeps the solve stable when a covariate is
    constant inside a window (e.g. repeat flag all-zero).
    """
    out_col = out_col or f"{value}_adj"
    # A covariate that is never finite -- phyloP with no bigwig, or a track with
    # gaps over the whole panel -- would otherwise remove EVERY row from the
    # adjusted estimate through the joint isfinite test, and the result would be
    # an empty fit reported as n_windows = 0 with no reason attached.  Drop such
    # covariates here, by name, and record it on the returned frame.
    covariates = list(dict.fromkeys(covariates))
    present = [c for c in covariates if c in df.columns]
    usable = [c for c in present
              if np.isfinite(pd.to_numeric(df[c], errors="coerce")).any()]
    dropped = [c for c in present if c not in usable]
    covariates = usable
    parts = []
    for wid, g in df.groupby(window_col, sort=False):
        g = g.copy()
        col = g[value]
        if getattr(col, "ndim", 1) > 1:
            raise ValueError(
                f"column {value!r} appears {col.shape[1]} times in the frame; "
                "residualising is undefined against a duplicated column")
        y = col.to_numpy(dtype=np.float64)
        cov = [c for c in covariates if c != value]
        X = (g[cov].to_numpy(dtype=np.float64) if cov
             else np.zeros((len(g), 0), dtype=np.float64))
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        resid = np.full(len(g), np.nan)
        if ok.sum() > len(cov) + 2:
            Xo = np.column_stack([np.ones(ok.sum()), X[ok]])
            yo = y[ok]
            # centre + scale for conditioning, then ridge-regularised least squares
            mu = Xo.mean(axis=0)
            mu[0] = 0.0
            sd = Xo.std(axis=0)
            sd[sd == 0] = 1.0
            sd[0] = 1.0
            Xs = (Xo - mu) / sd
            A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1])
            beta = np.linalg.solve(A, Xs.T @ yo)
            resid[ok] = yo - Xs @ beta
        g[out_col] = resid
        parts.append(g)
    out = pd.concat(parts, axis=0)
    out.attrs["covariates_used"] = list(covariates)
    out.attrs["covariates_dropped_all_missing"] = list(dropped)
    out.attrs["rows_retained"] = float(np.isfinite(out[out_col]).mean())
    return out


def covariate_audit(df: pd.DataFrame, covariates: Sequence[str]) -> Dict[str, object]:
    """What each covariate costs before anything is adjusted for it.

    The protocol already counts dropped *windows*; this counts dropped
    *positions*, which is the same discipline applied to the covariates. A
    column that is 30% missing silently removes 30% of every adjusted estimate,
    and in the rehearsal a fully-missing phyloP removed 100% of them and was
    reported only as ``n_windows: 0``.
    """
    present = [c for c in covariates if c in df.columns]
    missing = [c for c in covariates if c not in df.columns]
    per: Dict[str, float] = {}
    for c in present:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        per[c] = round(float(np.isfinite(v).mean()), 6)
    unusable = [c for c, f in per.items() if f == 0.0]
    usable = [c for c in present if c not in unusable]
    if usable:
        X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
                             for c in usable])
        joint = float(np.all(np.isfinite(X), axis=1).mean())
    else:
        joint = 1.0
    return dict(finite_fraction=per, absent=missing, all_missing=unusable,
                used=usable, joint_retained_fraction=round(joint, 6),
                note=("covariates that are never finite are dropped by name rather "
                      "than allowed to empty the estimate"))


# --------------------------------------------------------------------------
# null models
# --------------------------------------------------------------------------

def permutation_null(
    df: pd.DataFrame,
    value: str,
    target: str,
    baseline: str = "intron",
    context_col: str = "context",
    window_col: str = "window_id",
    n_perm: int = 200,
    seed: int = BOOTSTRAP_SEED,
    **kwargs,
) -> Dict[str, float]:
    """Permute context labels **within** each window and re-run the whole estimate."""
    rng = np.random.default_rng(seed)
    stats = []
    sub = df[df[context_col].isin([target, baseline])].copy()
    for _ in range(n_perm):
        permuted = []
        for _, g in sub.groupby(window_col, sort=False):
            g = g.copy()
            lab = g[context_col].to_numpy().copy()
            rng.shuffle(lab)
            g[context_col] = lab
            permuted.append(g)
        pdf = pd.concat(permuted, axis=0)
        pw = per_window_d(pdf, value=value, target=target, baseline=baseline,
                          context_col=context_col, window_col=window_col, **kwargs)
        used = pw.loc[pw["used"], "d"].to_numpy()
        if used.size:
            stats.append(float(used.mean()))
    if not stats:
        return dict(null_mean=np.nan, null_lo=np.nan, null_hi=np.nan, n=0)
    s = np.asarray(stats)
    lo, hi = np.quantile(s, [0.025, 0.975])
    return dict(null_mean=float(s.mean()), null_lo=float(lo), null_hi=float(hi), n=int(s.size))


# --------------------------------------------------------------------------
# bridge regression (step 8)
# --------------------------------------------------------------------------

@dataclass
class PartialFit:
    predictor: str
    beta: float
    ci_lo: float
    ci_hi: float
    partial_r2: float
    n_windows: int


def bridge_partial_effect(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    controls: Sequence[str],
    window_col: str = "window_id",
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> PartialFit:
    """Does ``predictor`` explain ``outcome`` beyond ``controls``?

    Both sides are residualised within the window first (so the estimate is a
    within-window partial effect), then a single slope is fitted on the pooled
    residuals.  Uncertainty comes from resampling **windows**, so the interval
    reflects the number of windows and not the number of positions.
    """
    # A predictor that also appears in the controls is not a modelling choice
    # to resolve silently: residualising a variable on itself leaves nothing,
    # and selecting the name twice yields a duplicated column that only shows up
    # much later as a broadcasting error.  Drop it and record that it was dropped.
    controls = [c for c in dict.fromkeys(controls)
                if c not in (outcome, predictor)]
    need = list(dict.fromkeys([outcome, predictor, *controls, window_col]))
    d = df[need].copy()
    d = residualise_within_window(d, outcome, controls, window_col, out_col="_y")
    d = residualise_within_window(d, predictor, controls, window_col, out_col="_x")
    d = d[np.isfinite(d["_y"]) & np.isfinite(d["_x"])]
    if d.empty:
        return PartialFit(predictor, np.nan, np.nan, np.nan, np.nan, 0)

    def slope_and_r2(frame: pd.DataFrame):
        x = frame["_x"].to_numpy()
        y = frame["_y"].to_numpy()
        vx = float((x * x).sum())
        if vx <= 0:
            return np.nan, np.nan
        bhat = float((x * y).sum() / vx)
        ss_tot = float((y * y).sum())
        ss_res = float(((y - bhat * x) ** 2).sum())
        r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
        return bhat, r2

    beta, r2 = slope_and_r2(d)

    windows = d[window_col].unique()
    rng = np.random.default_rng(seed)
    groups = {w: g for w, g in d.groupby(window_col, sort=False)}
    boots = []
    for _ in range(b):
        pick = rng.choice(windows, size=len(windows), replace=True)
        frame = pd.concat([groups[w] for w in pick], axis=0)
        bb, _ = slope_and_r2(frame)
        if np.isfinite(bb):
            boots.append(bb)
    if boots:
        lo, hi = np.quantile(np.asarray(boots), [0.025, 0.975])
    else:
        lo = hi = np.nan
    return PartialFit(predictor, beta, float(lo), float(hi), float(r2), len(windows))


def quadrant_assign(
    df: pd.DataFrame,
    x: str,
    y: str,
    window_col: str = "window_id",
    x_label: str = "pre",
    y_label: str = "post",
) -> pd.Series:
    """Assign each position to one of the four pre-registered quadrants.

    Splits are within-window medians, so the assignment is not driven by
    between-window shifts (which is what the overdispersion is made of).
    """
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(window_col, sort=False):
        mx = g[x].median()
        my = g[y].median()
        hi_x = g[x] >= mx
        hi_y = g[y] >= my
        out.loc[g.index] = np.where(
            hi_x & hi_y, f"{x_label}_hi/{y_label}_hi",
            np.where(hi_x & ~hi_y, f"{x_label}_hi/{y_label}_lo",
                     np.where(~hi_x & hi_y, f"{x_label}_lo/{y_label}_hi",
                              f"{x_label}_lo/{y_label}_lo")),
        )
    return out


def overdispersion(df: pd.DataFrame, indicator: str, window_col: str = "window_id") -> Dict[str, float]:
    """chi2/df of a binary indicator across windows -- the sanity number to re-report."""
    g = df.groupby(window_col)[indicator].agg(["sum", "count"])
    g = g[g["count"] >= 10]
    if g.empty:
        return dict(ratio=np.nan, n_windows=0, p_hat=np.nan)
    p_hat = float(g["sum"].sum() / g["count"].sum())
    if p_hat in (0.0, 1.0):
        return dict(ratio=np.nan, n_windows=int(len(g)), p_hat=p_hat)
    exp = g["count"] * p_hat
    var = g["count"] * p_hat * (1 - p_hat)
    chi2 = (((g["sum"] - exp) ** 2) / var).sum()
    dof = max(len(g) - 1, 1)
    return dict(ratio=float(chi2 / dof), n_windows=int(len(g)), p_hat=p_hat)
