"""Two references, joined at the handoff.

The idea
--------
Every settling readout in this line of work measures one thing: how close the
residual stream at block l is to ``h_norm``, the state the output head reads.
The follow-up manuscript then shows that in Evo 2 7B blocks 28 and 29 are
*near-orthogonal* to that frame (cos = -0.0090 and -0.0058) and that alignment
only appears at block 30.  Which means: for the whole pre-handoff stack, the
quantity being measured is alignment to a frame the stream has not entered yet.
Whatever structure appears there is structure in a near-zero cosine, and the
largest covariate of it turns out to be the model's own next-base uncertainty.

So use the reference each regime is actually approaching:

* **blocks 0..onset-1** -- reference ``h_{onset-1}`` (block 27 in 7B), the state
  the pre-handoff stack is building.  This asks *when the representation
  settles*, which is the biological question.
* **blocks onset..end** -- reference ``h_norm``, the state just before the ACGT
  logits.  This asks *when the output commits*, which is the decoding question.

One curve, two targets, switched at the handoff.  Neither half is new
machinery: each is the same cosine the earlier work used, pointed at the
endpoint of its own regime.

Three things this construction must not pretend
-----------------------------------------------
1. **The endpoint is in its own reference.**  ``cos(h27, h27) = 1`` by
   construction, so the pre-curve ends at 1 whatever the model does.  The shape
   of the approach is the finding; the endpoint is not.  A held-out control
   reference (``h26`` by default) is extracted alongside, and
   :func:`self_reference_control` compares the two curves: if the approach were
   an artefact of including the endpoint, the two would not agree away from
   their own endpoints.

2. **Thresholded depths censor.**  This is the follow-up manuscript's central
   result -- the published operating point never crosses for 77% of intronic
   positions, and the censoring rate itself varies with annotation class, which
   is enough to invert the ordering.  So every ``c_*`` returned here carries its
   censored fraction, and the continuous maxima are the primary readouts.

3. **Post-handoff is short in 7B.**  With onset 28 and rotation 30, the
   post-handoff regime is blocks 28-31, and 30 and 31 are a passthrough pair.
   That is three distinct points.  The construction is worth more at 40B, where
   26 blocks follow the rotation; in 7B the post-curve should be read as a jump
   with a shape, not as a curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats as S
from .config import MIN_N_PER_CLASS


@dataclass
class DualRef:
    """Which blocks belong to which regime, and which reference each uses."""

    onset: int = 28
    rotation: int = 30
    n_blocks: int = 32
    control_offset: int = 1        # held-out control reference, onset-1-this

    @property
    def pre_ref(self) -> int:
        return self.onset - 1

    @property
    def control_ref(self) -> int:
        return self.onset - 1 - self.control_offset

    @property
    def pre_blocks(self) -> List[str]:
        """Blocks measured against h_{onset-1}, excluding the reference itself."""
        return [f"{i:02d}" for i in range(0, self.pre_ref)]

    @property
    def post_blocks(self) -> List[str]:
        """Blocks measured against h_norm, from the onset to the last block.

        ``norm`` itself is excluded, exactly as ``pre_blocks`` excludes its own
        endpoint: ``cos(h_norm, h_norm) = 1`` for every position, so including it
        would make the threshold 1, the censored fraction 0 and the depth
        constant.  The earlier work made the same choice when it fixed the
        deepest tap below the reference.
        """
        return [f"{i:02d}" for i in range(self.onset, self.n_blocks)]

    def pre_col(self, tag: str, ref: Optional[int] = None, perp: bool = False) -> str:
        r = self.pre_ref if ref is None else ref
        return f"{'apreperp' if perp else 'apre'}{r:02d}_{tag}"

    def post_col(self, tag: str) -> str:
        return f"a_{tag}"


# --------------------------------------------------------------------------
# per-position readouts
# --------------------------------------------------------------------------

def continuous_readouts(df: pd.DataFrame, dr: DualRef,
                        perp: bool = False) -> pd.DataFrame:
    """The two continuous maxima -- primary, because they do not censor.

    ``a_pre``  max cosine to h_{onset-1} over the pre-handoff blocks
    ``a_post`` max cosine to h_norm from the onset onward
    """
    out = df.copy()
    pre_cols = [dr.pre_col(t, perp=perp) for t in dr.pre_blocks]
    pre_cols = [c for c in pre_cols if c in out.columns]
    post_cols = [dr.post_col(t) for t in dr.post_blocks]
    post_cols = [c for c in post_cols if c in out.columns]
    if not pre_cols or not post_cols:
        raise KeyError(
            "dual-reference columns missing: re-run step5-extract with this "
            "version so that apre*_<block> is stored alongside a_<block>"
        )
    suffix = "_perp" if perp else ""
    out[f"a_pre{suffix}"] = out[pre_cols].max(axis=1)
    out[f"a_post{suffix}"] = out[post_cols].max(axis=1)
    out[f"argmax_pre{suffix}"] = [
        int(dr.pre_blocks[i]) for i in np.asarray(out[pre_cols].to_numpy()).argmax(axis=1)]
    return out


def calibrate_gamma(df: pd.DataFrame, cols: Sequence[str],
                    quantile: float = 0.70) -> Dict[str, float]:
    """Pick a threshold from the panel, the way the earlier work picked gamma.

    Returned together with what it will cost: the fraction of positions that
    never cross it.  A threshold is only reportable beside that number.
    """
    cols = [c for c in cols if c in df.columns]
    m = df[cols].to_numpy(dtype=np.float64)
    best = np.nanmax(m, axis=1)
    gamma = float(np.nanquantile(best, 1.0 - quantile))
    censored = float(np.mean(best < gamma))
    return dict(gamma=gamma, censored_fraction=censored,
                quantile=quantile, n=int(np.isfinite(best).sum()))


def settle_depth(df: pd.DataFrame, cols: Sequence[str], blocks: Sequence[str],
                 gamma: float, persistence: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """First block whose cosine reaches ``gamma`` and stays there.

    Returns ``(depth, crossed)``.  Non-crossing positions get NaN rather than a
    ceiling value, because entering an average at the ceiling is exactly how the
    published ordering got inverted.
    """
    cols = [c for c in cols if c in df.columns]
    m = df[cols].to_numpy(dtype=np.float64)
    ok = m >= gamma
    n, L = ok.shape
    depth = np.full(n, np.nan)
    for j in range(L - persistence + 1):
        win = ok[:, j:j + persistence].all(axis=1)
        fresh = win & np.isnan(depth)
        # blocks are block indices; "norm" would have no index, and is excluded
        # from both pools above, but guard anyway rather than encode a fake one.
        if blocks[j] == "norm":
            continue
        depth[fresh] = float(blocks[j])
    return depth, np.isfinite(depth)


def dual_settle(df: pd.DataFrame, dr: DualRef, gamma_quantile: float = 0.70,
                perp: bool = False) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Both settling depths, each against its own reference, with censoring."""
    out = continuous_readouts(df, dr, perp=perp)
    suffix = "_perp" if perp else ""

    pre_cols = [dr.pre_col(t, perp=perp) for t in dr.pre_blocks]
    post_cols = [dr.post_col(t) for t in dr.post_blocks]

    g_pre = calibrate_gamma(out, pre_cols, gamma_quantile)
    g_post = calibrate_gamma(out, post_cols, gamma_quantile)

    d_pre, cr_pre = settle_depth(out, pre_cols, dr.pre_blocks, g_pre["gamma"])
    d_post, cr_post = settle_depth(out, post_cols, dr.post_blocks, g_post["gamma"])
    out[f"c_pre{suffix}"] = d_pre
    out[f"c_post{suffix}"] = d_post
    out[f"crossed_pre{suffix}"] = cr_pre.astype(np.float32)
    out[f"crossed_post{suffix}"] = cr_post.astype(np.float32)

    report = dict(
        pre=dict(reference=f"h{dr.pre_ref}", blocks=dr.pre_blocks, **g_pre),
        post=dict(reference="h_norm", blocks=dr.post_blocks, **g_post),
        note=("continuous a_pre / a_post are the primary readouts; c_pre / c_post "
              "are thresholded and are only interpretable beside their censored "
              "fractions, which is the failure this project already documented"),
    )
    return out, report


# --------------------------------------------------------------------------
# degeneracy: the failure mode this construction is most exposed to
# --------------------------------------------------------------------------

# No minimum range constant.  "Does this curve carry depth signal" is answered
# by a variance ratio whose reference is 1 -- does it move more across blocks
# than the same quantity scatters across positions at one block -- which needs
# no chosen number.  See exp1.nulls.variance_separation.


def curve_diagnostics(df: pd.DataFrame, dr: DualRef) -> Dict[str, object]:
    """Is each curve informative enough to threshold at all?

    The pre-handoff cosine is the exposed one.  The companion manuscript reports
    that the output reference is 94% a single global direction; a residual
    stream dominated by one persistent component gives ``cos(h_l, h_27)`` close
    to 1 at *every* block, and a settling depth cut on that measures the last
    digit of a constant.  The rehearsal reproduces exactly this -- range 0.009
    across twenty-seven blocks -- so the check runs before any depth is reported
    rather than after a figure has been drawn from it.
    """
    out: Dict[str, object] = {}
    for name, cols in (("to_pre", [dr.pre_col(t) for t in dr.pre_blocks]),
                       ("to_pre_perp", [dr.pre_col(t, perp=True) for t in dr.pre_blocks]),
                       ("to_output", [dr.post_col(t) for t in dr.post_blocks])):
        have = [c for c in cols if c in df.columns]
        if not have:
            out[name] = dict(available=False)
            continue
        from .nulls import variance_separation

        m = df[have].to_numpy(dtype=np.float64)
        means = np.nanmean(m, axis=0)
        vs = variance_separation(means, m)
        out[name] = dict(available=True, n_blocks=len(have),
                         first=round(float(means[0]), 5),
                         last=round(float(means[-1]), 5),
                         dynamic_range=round(float(np.nanmax(means) - np.nanmin(means)), 5),
                         variance_ratio=round(float(vs["variance_ratio"]), 4),
                         thresholdable=bool(vs["informative"]),
                         basis=vs["note"])
    usable = [k for k, v in out.items()
              if v.get("available") and v.get("thresholdable")]
    out["verdict"] = (
        "raw cosines are usable" if "to_pre" in usable else
        ("the raw pre-handoff cosine is flat; use the u-removed curve, whose "
         "range is above the margin" if "to_pre_perp" in usable else
         "neither pre-handoff curve has usable dynamic range; report the "
         "RELATIVE approach only and do not threshold either raw curve"))
    out["criterion"] = ("between-block variance of the mean curve against the "
                        "mean within-position variance at a block; informative "
                        "when the ratio exceeds 1. No threshold is chosen.")
    return out


def relative_approach(df: pd.DataFrame, dr: DualRef, perp: bool = False
                      ) -> pd.DataFrame:
    """Each position's approach rescaled to its OWN trajectory.

    ``rel_l = (cos_l - min_l cos) / (max_l cos - min_l cos)`` over the regime's
    blocks, so the readout is "how far along its own path is this position at
    block l", between 0 and 1 by construction.

    This is the readout that survives a residual stream with a large persistent
    component: a constant offset shared by every block cancels, and what is left
    is the *shape* of the approach, which is the thing the pre-handoff reference
    was introduced to measure.  It is also censoring-free in the sense that
    matters here -- every position reaches 1 somewhere within its own regime, so
    the depth is never undefined and never enters an average at a ceiling.

    What it gives up, and this must be said in the paper: it cannot compare
    *absolute* alignment between positions.  A position that never aligns with
    anything and one that aligns perfectly both reach 1 on their own scale.  So
    it is reported together with the absolute range per position, below.
    """
    out = df.copy()
    suffix = "_perp" if perp else ""
    cols = [dr.pre_col(t, perp=perp) for t in dr.pre_blocks]
    cols = [c for c in cols if c in out.columns]
    if not cols:
        raise KeyError("no pre-handoff columns; re-run step5-extract")
    m = out[cols].to_numpy(dtype=np.float64)
    lo = np.nanmin(m, axis=1, keepdims=True)
    hi = np.nanmax(m, axis=1, keepdims=True)
    span = hi - lo
    rel = np.where(span > 1e-12, (m - lo) / np.where(span > 1e-12, span, 1.0), np.nan)
    for j, t in enumerate(dr.pre_blocks):
        out[f"rel_pre{suffix}_{t}"] = rel[:, j].astype(np.float32)
    out[f"pre_span{suffix}"] = span.ravel().astype(np.float32)

    # depth on the relative scale: first block past the fraction, and the
    # area under the approach, which needs no threshold at all.
    for frac in (0.5, 0.9):
        depth = np.full(len(out), np.nan)
        reached = rel >= frac
        for j in range(rel.shape[1]):
            fresh = reached[:, j] & np.isnan(depth)
            depth[fresh] = float(dr.pre_blocks[j])
        out[f"c_pre_rel{suffix}_{int(frac * 100)}"] = depth
    out[f"auc_pre{suffix}"] = np.nanmean(rel, axis=1).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# profiles and contrasts
# --------------------------------------------------------------------------

def dual_profile(df: pd.DataFrame, dr: DualRef, group_col: Optional[str] = None,
                 perp: bool = False) -> pd.DataFrame:
    """Per-block mean of both cosines, optionally split by a grouping column.

    Both references are reported at *every* block, not only inside their own
    regime, so the crossover is visible rather than assumed.
    """
    rows: List[Dict[str, object]] = []
    tags = [f"{i:02d}" for i in range(dr.n_blocks)] + ["norm"]
    groups = [("all", df)] if group_col is None else list(df.groupby(group_col))
    for gname, g in groups:
        for tag in tags:
            rec: Dict[str, object] = dict(group=gname, tag=tag,
                                          block=(dr.n_blocks if tag == "norm" else int(tag)))
            pc = dr.pre_col(tag, perp=perp)
            if pc in g.columns:
                rec["to_pre"] = float(np.nanmean(g[pc]))
            cc = dr.pre_col(tag, ref=dr.control_ref, perp=perp)
            if cc in g.columns:
                rec["to_control"] = float(np.nanmean(g[cc]))
            oc = dr.post_col(tag)
            if oc in g.columns:
                rec["to_output"] = float(np.nanmean(g[oc]))
            rec["regime"] = ("R1" if rec["block"] < dr.onset - 1 else
                             "R2" if rec["block"] <= dr.rotation else "R3")
            rows.append(rec)
    return pd.DataFrame(rows)


def self_reference_control(profile: pd.DataFrame, dr: DualRef,
                           exclude_within: int = 3) -> Dict[str, object]:
    """Is the pre-handoff approach an artefact of its own endpoint?

    Compares the curve toward ``h_{onset-1}`` with the curve toward the held-out
    ``h_{onset-2}``, ignoring blocks within ``exclude_within`` of either
    reference, where each is trivially close to itself.  Agreement away from the
    endpoints is what licenses reading the approach as a real trajectory.
    """
    p = profile[profile["group"] == profile["group"].iloc[0]] if "group" in profile else profile
    p = p[p["block"] < dr.pre_ref - exclude_within]
    if "to_control" not in p.columns or p.empty:
        return dict(available=False,
                    reason="no control reference stored; set pre_ref_blocks to two blocks")
    a, b = p["to_pre"].to_numpy(), p["to_control"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return dict(available=False, reason="too few comparable blocks")
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return dict(available=True, n_blocks=int(ok.sum()),
                pearson_r=round(r, 4),
                max_abs_diff=round(float(np.nanmax(np.abs(a[ok] - b[ok]))), 4),
                verdict=("shapes agree away from the endpoints; the approach is not "
                         "an artefact of the reference choice" if r > 0.9 else
                         "shapes disagree; the pre-handoff curve depends on which "
                         "block is used as its endpoint and must be reported as such"))


def dual_context_table(df: pd.DataFrame, dr: DualRef, contexts: Sequence[str],
                       baseline: str = "intron", context_col: str = "context",
                       perp: bool = False) -> pd.DataFrame:
    """Within-window paired effect for both regimes, per class.

    The same estimand the follow-up manuscript fixed -- window as the resampling
    unit, cluster bootstrap -- applied to each reference separately, so
    "settles early in its own stack" and "commits early to an output" are two
    columns rather than one conflated number.
    """
    suffix = "_perp" if perp else ""
    rows: List[Dict[str, object]] = []
    for value in (f"a_pre{suffix}", f"a_post{suffix}"):
        if value not in df.columns:
            continue
        for ctx in contexts:
            if ctx == baseline:
                continue
            eff = S.window_effect(df, value=value, target=ctx, baseline=baseline,
                                  context_col=context_col)
            row = dict(regime="pre" if "pre" in value else "post",
                       reference=f"h{dr.pre_ref}" if "pre" in value else "h_norm",
                       context=ctx, **eff.as_row())
            # censoring is a property of the class, so report it beside the effect
            cc = f"crossed_{'pre' if 'pre' in value else 'post'}{suffix}"
            if cc in df.columns:
                sub = df[df[context_col] == ctx]
                row["censored_fraction"] = float(1.0 - np.nanmean(sub[cc])) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def dual_bridge(df: pd.DataFrame, dr: DualRef, controls: Sequence[str],
                perp: bool = False) -> Dict[str, object]:
    """Does settling early in the pre-handoff stack predict committing early?

    This is the question the whole two-reference construction exists to make
    askable: the predictor and the outcome are now measured against different
    targets, so the correlation is no longer two readings of one quantity.
    """
    suffix = "_perp" if perp else ""
    pre, post = f"a_pre{suffix}", f"a_post{suffix}"
    if pre not in df.columns or post not in df.columns:
        return dict(available=False, reason="run dual_settle first")
    have = [c for c in controls if c in df.columns]
    from dataclasses import asdict

    fit = S.bridge_partial_effect(df, predictor=pre, outcome=post, controls=have)
    res = dict(asdict(fit))
    res.update(available=True, predictor=pre, outcome=post,
               predictor_reference=f"h{dr.pre_ref}", outcome_reference="h_norm",
               controls=have)
    return res


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def plot_dual(profile: pd.DataFrame, dr: DualRef, out: Path,
              title: str = "settling against two references") -> Path:
    """One axis, both references, with the handoff marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(profile["group"]))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.axvspan(dr.onset - 1, dr.rotation, color="0.92", zorder=0)
    ax.axvline(dr.pre_ref, ls=":", lw=1, color="0.5")
    ax.axvline(dr.rotation, ls=":", lw=1, color="0.5")

    for gi, g in enumerate(groups):
        p = profile[profile["group"] == g].sort_values("block")
        if "to_pre" in p:
            ax.plot(p["block"], p["to_pre"], marker="o", ms=3,
                    label=f"{g} -> h{dr.pre_ref}" if len(groups) > 1 else f"-> h{dr.pre_ref}")
        if "to_control" in p:
            ax.plot(p["block"], p["to_control"], ls="--", lw=1, alpha=0.6,
                    label=f"-> h{dr.control_ref} (held-out control)" if gi == 0 else None)
        if "to_output" in p:
            ax.plot(p["block"], p["to_output"], marker="s", ms=3,
                    label=f"{g} -> h_norm" if len(groups) > 1 else "-> h_norm")

    ax.set_xlabel("block")
    ax.set_ylabel("cosine to reference")
    ax.set_title(title)
    ax.axhline(0.0, lw=0.8, color="0.7")
    ax.legend(fontsize=7, ncol=2)
    ax.text(dr.pre_ref - 0.3, ax.get_ylim()[0], f" h{dr.pre_ref}", fontsize=7,
            va="bottom", ha="right", color="0.4")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out
