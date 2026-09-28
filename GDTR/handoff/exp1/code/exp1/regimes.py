"""Three-regime decomposition: R1 (0..onset-1), R2 (the transition), R3 (rotation..end).

The protocol asks three different questions of three stretches of the stack, and
they need different summaries:

    R1  0 .. onset-1      how biological structure accumulates
    R2  onset-1 -> onset -> rotation    what the handoff does in two steps
    R3  rotation .. norm  how the output state is sharpened

This module turns each position's block profile into a small set of **regime
features**, then provides (a) per-regime context effects using the same
window-clustered estimator as everything else and (b) an explicit **regime
contrast**: does the same context separate differently in R1 than in R3?  That
last one is the comparison the manuscripts currently cannot make, because a
per-block curve is not an estimand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats as S


@dataclass
class Regimes:
    onset: int
    rotation: int
    n_blocks: int

    @property
    def r1(self) -> List[str]:
        return [f"{i:02d}" for i in range(0, self.onset)]

    @property
    def r2(self) -> List[str]:
        return [f"{i:02d}" for i in range(self.onset - 1, self.rotation + 1)]

    @property
    def r3(self) -> List[str]:
        return [f"{i:02d}" for i in range(self.rotation, self.n_blocks)] + ["norm"]

    @property
    def pre_tag(self) -> str:
        return f"{self.onset - 1:02d}"

    @property
    def onset_tag(self) -> str:
        return f"{self.onset:02d}"

    @property
    def rot_tag(self) -> str:
        return f"{self.rotation:02d}"

    def as_dict(self) -> Dict[str, object]:
        return dict(onset=self.onset, rotation=self.rotation, n_blocks=self.n_blocks,
                    r1=self.r1, r2=self.r2, r3=self.r3)


# --------------------------------------------------------------------------
# per-position regime features
# --------------------------------------------------------------------------

def _cols(df: pd.DataFrame, prefix: str, tags: Sequence[str]) -> List[str]:
    return [f"{prefix}_{t}" for t in tags if f"{prefix}_{t}" in df.columns]


def _slope(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares slope per row, ignoring non-finite entries."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    xm = np.where(ok, x[None, :], np.nan)
    n = ok.sum(axis=1)
    out = np.full(v.shape[0], np.nan)
    good = n >= 3
    if good.any():
        xs = np.nanmean(xm[good], axis=1, keepdims=True)
        ys = np.nanmean(np.where(ok, v, np.nan)[good], axis=1, keepdims=True)
        dx = np.where(ok[good], xm[good] - xs, 0.0)
        dy = np.where(ok[good], np.where(ok, v, 0.0)[good] - ys, 0.0)
        denom = (dx * dx).sum(axis=1)
        out[good] = np.where(denom > 0, (dx * dy).sum(axis=1) / np.maximum(denom, 1e-30), np.nan)
    return out


def regime_features(df: pd.DataFrame, reg: Regimes,
                    prefixes: Sequence[str] = ("av", "au", "a", "sctr", "kl"),
                    log_prefixes: Sequence[str] = ("n",)) -> pd.DataFrame:
    """Add per-position regime summaries for each quantity.

    For every prefix the following are produced, where the blocks exist:

        <p>_r1_slope   <p>_r1_peak   <p>_r1_peak_block   <p>_r1_end   <p>_r1_auc
        <p>_r2_delta   <p>_r2_rel
        <p>_r3_slope   <p>_r3_end    <p>_r3_delta

    ``log_prefixes`` are summarised in log10 because they move by orders of
    magnitude; taking a slope of a raw norm across the onset is meaningless.
    """
    out = df.copy()
    for prefix in list(prefixes) + list(log_prefixes):
        use_log = prefix in log_prefixes

        c1 = _cols(out, prefix, reg.r1)
        if len(c1) >= 3:
            v1 = out[c1].to_numpy(dtype=np.float64)
            if use_log:
                v1 = np.log10(np.maximum(v1, 1e-300))
            x1 = np.arange(len(c1), dtype=np.float64)
            out[f"{prefix}_r1_slope"] = _slope(v1, x1)
            out[f"{prefix}_r1_peak"] = np.nanmax(v1, axis=1)
            out[f"{prefix}_r1_peak_block"] = np.nanargmax(
                np.where(np.isfinite(v1), v1, -np.inf), axis=1).astype(float)
            out[f"{prefix}_r1_end"] = v1[:, -1]
            out[f"{prefix}_r1_auc"] = np.nanmean(v1, axis=1)

        pre, ons, rot = f"{prefix}_{reg.pre_tag}", f"{prefix}_{reg.onset_tag}", f"{prefix}_{reg.rot_tag}"
        if pre in out.columns and ons in out.columns:
            a = out[pre].to_numpy(dtype=np.float64)
            b = out[ons].to_numpy(dtype=np.float64)
            if use_log:
                a, b = np.log10(np.maximum(a, 1e-300)), np.log10(np.maximum(b, 1e-300))
            out[f"{prefix}_r2_delta"] = b - a
            out[f"{prefix}_r2_rel"] = (b - a) / np.maximum(np.abs(a), 1e-12)
        if pre in out.columns and rot in out.columns:
            a = out[pre].to_numpy(dtype=np.float64)
            c = out[rot].to_numpy(dtype=np.float64)
            if use_log:
                a, c = np.log10(np.maximum(a, 1e-300)), np.log10(np.maximum(c, 1e-300))
            out[f"{prefix}_r2_delta_to_rotation"] = c - a

        c3 = _cols(out, prefix, reg.r3)
        if len(c3) >= 3:
            v3 = out[c3].to_numpy(dtype=np.float64)
            if use_log:
                v3 = np.log10(np.maximum(v3, 1e-300))
            x3 = np.arange(len(c3), dtype=np.float64)
            out[f"{prefix}_r3_slope"] = _slope(v3, x3)
            out[f"{prefix}_r3_end"] = v3[:, -1]
            out[f"{prefix}_r3_delta"] = v3[:, -1] - v3[:, 0]
    return out


# --------------------------------------------------------------------------
# per-regime context effects and the regime contrast
# --------------------------------------------------------------------------

def regime_effect_table(
    df: pd.DataFrame,
    features: Sequence[str],
    contexts: Sequence[str],
    baseline: str = "intron",
    covariates: Sequence[str] = (),
    group_col: str = "context",
) -> pd.DataFrame:
    """Within-window paired effect for each (feature, context), unadjusted and adjusted."""
    have = [c for c in covariates if c in df.columns]
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        adj = S.residualise_within_window(df, feat, have, out_col=f"{feat}__adj") if have else None
        for ctx in contexts:
            raw = S.window_effect(df, value=feat, target=ctx, baseline=baseline,
                                  context_col=group_col)
            row = dict(feature=feat, group=ctx, d=raw.mean_d, lo=raw.ci_lo, hi=raw.ci_hi,
                       n_windows=raw.n_windows_used, n_dropped=raw.n_windows_dropped)
            if adj is not None:
                a = S.window_effect(adj, value=f"{feat}__adj", target=ctx,
                                    baseline=baseline, context_col=group_col)
                row.update(d_adj=a.mean_d, lo_adj=a.ci_lo, hi_adj=a.ci_hi)
            rows.append(row)
    return pd.DataFrame(rows)


def regime_contrast(
    df: pd.DataFrame,
    feat_a: str,
    feat_b: str,
    target: str,
    baseline: str = "intron",
    context_col: str = "context",
    window_col: str = "window_id",
    b: int = S.BOOTSTRAP_B,
    seed: int = S.BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Is the context effect different in one regime than in another?

    Computes the per-window effect for both features, pairs them **by window**,
    and bootstraps the difference.  Pairing matters: window-level heterogeneity
    is the dominant variance component, and it cancels in the difference.
    """
    pa = S.per_window_d(df, value=feat_a, target=target, baseline=baseline,
                        context_col=context_col, window_col=window_col)
    pb = S.per_window_d(df, value=feat_b, target=target, baseline=baseline,
                        context_col=context_col, window_col=window_col)
    m = pa[pa["used"]].merge(pb[pb["used"]], on="window_id", suffixes=("_a", "_b"))
    if m.empty:
        return dict(feature_a=feat_a, feature_b=feat_b, target=target,
                    diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_windows=0)
    diff = (m["d_a"] - m["d_b"]).to_numpy()
    bs = S.cluster_bootstrap_mean(diff, b=b, seed=seed)
    return dict(feature_a=feat_a, feature_b=feat_b, target=target,
                mean_a=float(m["d_a"].mean()), mean_b=float(m["d_b"].mean()),
                diff=bs["mean"], ci_lo=bs["ci_lo"], ci_hi=bs["ci_hi"],
                n_windows=int(len(m)),
                excludes_zero=bool(np.isfinite(bs["ci_lo"]) and
                                   (bs["ci_lo"] > 0 or bs["ci_hi"] < 0)))


def regime_transfer(
    df: pd.DataFrame,
    r1_feature: str,
    r3_feature: str,
    controls: Sequence[str] = ("entropy_final", "gc", "repeat", "phylop"),
) -> Dict[str, object]:
    """Does what happened in R1 predict what happens in R3, beyond the controls?"""
    have = [c for c in controls if c in df.columns]
    fit = S.bridge_partial_effect(df.dropna(subset=[r1_feature, r3_feature]),
                                  outcome=r3_feature, predictor=r1_feature, controls=have)
    return dict(r1_feature=r1_feature, r3_feature=r3_feature,
                beta=fit.beta, ci_lo=fit.ci_lo, ci_hi=fit.ci_hi,
                partial_r2=fit.partial_r2, n_windows=fit.n_windows, controls=have)


def regime_profile_by_group(
    df: pd.DataFrame,
    prefix: str,
    reg: Regimes,
    group_col: str = "context",
    groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean block profile of one quantity per group, window-weighted.

    This is the picture behind the regime tables: one line per context (or per
    motif class), so R1 / R2 / R3 can be read off the same axis.
    """
    tags = [t for t in (reg.r1 + reg.r2[1:] + reg.r3) if f"{prefix}_{t}" in df.columns]
    seen, ordered = set(), []
    for t in tags:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    groups = groups or sorted(df[group_col].dropna().unique())
    rows = []
    for g in groups:
        sub = df[df[group_col] == g]
        if sub.empty:
            continue
        for t in ordered:
            col = f"{prefix}_{t}"
            per_window = sub.groupby("window_id")[col].mean()
            rows.append(dict(group=g, block=t, mean=float(per_window.mean()),
                             sd_windows=float(per_window.std(ddof=1)) if len(per_window) > 1 else np.nan,
                             n_windows=int(len(per_window)),
                             regime=("R1" if t in reg.r1 else ("R3" if t in reg.r3 else "R2"))))
    return pd.DataFrame(rows)


def plot_regime_profiles(prof: pd.DataFrame, out, reg: Regimes, title: str = "regimes"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for g, sub in prof.groupby("group"):
        sub = sub.copy()
        sub["x"] = [reg.n_blocks if t == "norm" else int(t) for t in sub["block"]]
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["mean"], lw=1.8, label=str(g))
    ax.axvspan(-0.5, reg.onset - 0.5, color="0.92", zorder=0)
    ax.axvspan(reg.onset - 0.5, reg.rotation - 0.5, color="0.82", zorder=0)
    ax.axvline(reg.onset, color="C3", ls=":", lw=1.2)
    ax.axvline(reg.rotation, color="C0", ls=":", lw=1.2)
    ax.text(reg.onset / 2, ax.get_ylim()[1], "R1", ha="center", va="top", fontsize=9)
    ax.text((reg.onset + reg.rotation) / 2, ax.get_ylim()[1], "R2", ha="center", va="top", fontsize=9)
    ax.text((reg.rotation + reg.n_blocks) / 2, ax.get_ylim()[1], "R3", ha="center", va="top", fontsize=9)
    ax.set_xlabel("block")
    ax.set_ylabel("mean")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
