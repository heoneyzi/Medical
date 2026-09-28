"""Steps 7-8: F1, F2, T1, stage-2 readouts and the bridge test.

Order matters and is enforced by the CLI: F1 decides how everything else is
read.  If the block-28 write turns out to be orthogonal rather than diluting,
the same channel numbers in F2 mean something different, so F1 is produced and
inspected first.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ADJUST_COVARIATES, CONTEXTS, result_path
from .io import load_table, save_json, save_table
from . import stats as S
from .metrics import early_ratio, settle_depth

BLOCK_TAGS = [f"{i:02d}" for i in range(64)]


def _tags_present(df: pd.DataFrame, prefix: str) -> List[str]:
    out = []
    for c in df.columns:
        if c.startswith(prefix + "_"):
            out.append(c[len(prefix) + 1 :])
    order = sorted([t for t in out if t != "norm"], key=lambda x: int(x))
    return order + (["norm"] if "norm" in out else [])


# --------------------------------------------------------------------------
# F1: numerator / denominator
# --------------------------------------------------------------------------

def f1_profile(df: pd.DataFrame, variant: str = "real") -> pd.DataFrame:
    """Per-block mean of log p, log n, log q, a -- window-weighted, not position-weighted."""
    d = df[df["variant"] == variant]
    tags = _tags_present(d, "n")
    rows = []
    for t in tags:
        p = d[f"p_{t}"].to_numpy(dtype=np.float64)
        n = d[f"n_{t}"].to_numpy(dtype=np.float64)
        q = d[f"q_{t}"].to_numpy(dtype=np.float64)
        a = d[f"a_{t}"].to_numpy(dtype=np.float64)
        win = d["window_id"].to_numpy()
        per_window = pd.DataFrame(dict(window_id=win, p=p, n=n, q=q, a=a)).groupby("window_id").mean()
        rows.append(dict(
            block=t,
            mean_p=float(per_window["p"].mean()),
            median_p=float(np.median(p)),
            mean_abs_p=float(np.mean(np.abs(p))),
            mean_n=float(per_window["n"].mean()),
            mean_q=float(per_window["q"].mean()),
            mean_a=float(per_window["a"].mean()),
            sd_a_windows=float(per_window["a"].std(ddof=1)) if len(per_window) > 1 else np.nan,
            frac_p_negative=float((p < 0).mean()),
        ))
    return pd.DataFrame(rows)


def f1_branch_decomposition(df: pd.DataFrame, variant: str = "real") -> pd.DataFrame:
    """Per-branch update projection (delta p) and magnitude, for the branch blocks."""
    d = df[df["variant"] == variant]
    tags = sorted({c.split("_")[1] for c in d.columns if c.startswith(("mxp_", "mlpp_"))})
    rows = []
    for t in tags:
        row = dict(block=t)
        for short, label in (("mx", "mixer"), ("mlp", "mlp")):
            pc, nc = f"{short}p_{t}", f"{short}n_{t}"
            if pc in d.columns:
                row[f"{label}_mean_dp"] = float(d[pc].mean())
                row[f"{label}_mean_abs_dp"] = float(d[pc].abs().mean())
                row[f"{label}_mean_norm"] = float(d[nc].mean())
                row[f"{label}_frac_dp_neg"] = float((d[pc] < 0).mean())
        if f"xinn_{t}" in d.columns:
            row["incoming_mean_norm"] = float(d[f"xinn_{t}"].mean())
            row["incoming_mean_p"] = float(d[f"xinp_{t}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def f1_decomposition(df: pd.DataFrame, onset: int, variant: str = "real",
                     window_col: str = "window_id") -> Dict[str, object]:
    """What happens to the aligned component across the onset, as an identity.

    Why this replaced a verdict
    ---------------------------
    The previous version classified the onset into dilution / orthogonal
    overwrite / active cancellation by comparing three ratios against three
    hand-set boundaries (0.02, 10, 0.5-2.0).  Those boundaries decided the
    paper's headline and nothing decided the boundaries, which makes the verdict
    a restatement of the constants.  Worse, the three labels are not three
    worlds: they are three regions of one continuum, and a position can be
    partly each.

    So nothing is classified here.  ``cos = p / n`` is a ratio, and across one
    block both parts change:

        log cos_28 - log cos_27  =  (log p_28 - log p_27)  -  (log n_28 - log n_27)
                                    \_____ numerator _____/    \____ denominator ___/

    which is exact, not a model.  And the numerator's own change is exactly the
    sum of the two branch projections, because the block is a residual sum:

        p_28 = p_27 + <mixer_28, u_hat> + <mlp_28, u_hat>

    also exact.  Both identities are *checked* here, not assumed, and every
    term is reported as a within-window mean with a window cluster-bootstrap
    interval.  The reader sees where the change came from, on the same scale,
    with uncertainty.  No cut-off appears anywhere in this function.
    """
    d = df[df["variant"] == variant] if "variant" in df.columns else df
    pre, post = f"{onset-1:02d}", f"{onset:02d}"
    need = [f"p_{pre}", f"p_{post}", f"n_{pre}", f"n_{post}"]
    if any(c not in d.columns for c in need):
        return dict(available=False, reason=f"missing {[c for c in need if c not in d.columns]}")

    p_pre, p_post = d[f"p_{pre}"].to_numpy(float), d[f"p_{post}"].to_numpy(float)
    n_pre, n_post = d[f"n_{pre}"].to_numpy(float), d[f"n_{post}"].to_numpy(float)

    eps = 1e-300
    terms = {
        "d_log_cos": np.log(np.abs(p_post) + eps) - np.log(n_post + eps)
                     - np.log(np.abs(p_pre) + eps) + np.log(n_pre + eps),
        "d_log_numerator": np.log(np.abs(p_post) + eps) - np.log(np.abs(p_pre) + eps),
        "d_log_denominator": np.log(n_post + eps) - np.log(n_pre + eps),
        "d_p": p_post - p_pre,
    }

    mx = d[f"mxp_{post}"].to_numpy(float) if f"mxp_{post}" in d.columns else None
    ml = d[f"mlpp_{post}"].to_numpy(float) if f"mlpp_{post}" in d.columns else None
    if mx is not None and ml is not None:
        terms["branch_mixer"] = mx
        terms["branch_mlp"] = ml
        terms["branch_residual"] = (p_post - p_pre) - mx - ml

    frame = pd.DataFrame(terms)
    frame[window_col] = d[window_col].to_numpy()

    out: Dict[str, object] = dict(available=True, onset=onset,
                                  pre_block=pre, post_block=post, n_positions=len(d))
    for name in terms:
        per_window = frame.groupby(window_col)[name].mean().to_numpy()
        bs = S.cluster_bootstrap_mean(per_window)
        out[name] = dict(mean=float(np.nanmean(per_window)),
                         ci_lo=float(bs["ci_lo"]), ci_hi=float(bs["ci_hi"]),
                         n_windows=int(np.isfinite(per_window).sum()))

    # identity checks: these must hold to machine precision, and if they do not
    # the decomposition is not describing this model's arithmetic.
    lhs = terms["d_log_cos"]
    rhs = terms["d_log_numerator"] - terms["d_log_denominator"]
    out["identity_log_error"] = float(np.nanmax(np.abs(lhs - rhs)))
    if "branch_residual" in terms:
        scale = np.nanmean(np.abs(p_post - p_pre)) + eps
        out["identity_branch_rel_error"] = float(
            np.nanmean(np.abs(terms["branch_residual"])) / scale)
        share_mlp = np.abs(ml) / (np.abs(mx) + np.abs(ml) + eps)
        pw = pd.DataFrame({"s": share_mlp, window_col: d[window_col].to_numpy()}) \
            .groupby(window_col)["s"].mean().to_numpy()
        bs = S.cluster_bootstrap_mean(pw)
        out["mlp_share_of_update"] = dict(mean=float(np.nanmean(pw)),
                                          ci_lo=float(bs["ci_lo"]),
                                          ci_hi=float(bs["ci_hi"]))
    out["reading"] = (
        "d_log_cos is exactly d_log_numerator minus d_log_denominator; whichever "
        "of the two has the larger magnitude is where the change in alignment "
        "came from. No threshold separates them -- compare the two intervals.")
    return out


def f1_hypothesis_check(decomp: Dict[str, object],
                        plan: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Test each pre-registered POINT prediction against its interval.

    A hypothesis that predicts a number is testable without a cut-off: either
    the interval covers the predicted value or it does not.  That is the only
    form of yes/no this analysis uses, and it is decided by the data's own
    uncertainty rather than by a constant.

    A note the formalisation forced, worth carrying into the paper
    --------------------------------------------------------------
    Writing the three pre-registered hypotheses as point predictions shows that
    **A and B are the same claim**.  "The aligned component is preserved"
    (A, a property of the state) and "the onset update has no aligned component"
    (B, a property of the update) are related by the residual identity
    ``p_post = p_pre + <update, u_hat>``: either both hold or neither does.
    They were never two hypotheses, and a three-way classifier between them was
    always going to be deciding on noise.

    It also shows a case the original set omitted: the update could have a
    *positive* aligned component, so that the onset ADDS alignment while the
    norm explodes.  Nothing ruled that out and nothing tested for it.

    So the honest object is one signed quantity, ``d_p``, and the reading is
    which side of zero its interval lies on -- a sign, decided by the data's own
    uncertainty, with no constant anywhere:

    ``covers 0``        the update is orthogonal; the cosine fell by dilution
                        (this is A and B at once)
    ``entirely < 0``    active cancellation (C)
    ``entirely > 0``    the onset adds alignment (unnamed in the original set)
    """
    if not decomp.get("available"):
        return pd.DataFrame()
    rows = []

    def covers(key: str, value: float) -> Optional[Dict[str, object]]:
        t = decomp.get(key)
        if not isinstance(t, dict):
            return None
        return dict(quantity=key, mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"],
                    predicted=value,
                    consistent=bool(t["ci_lo"] <= value <= t["ci_hi"]))

    a = covers("d_log_numerator", 0.0)
    if a:
        rows.append(dict(hypothesis="A_dilution",
                         prediction="the aligned component is unchanged (d log p = 0)", **a))
    b = covers("d_p", 0.0)
    if b:
        rows.append(dict(hypothesis="B_orthogonal_overwrite",
                         prediction="the onset update has no aligned component (dp = 0)", **b))
    t = decomp.get("d_p")
    if isinstance(t, dict):
        rows.append(dict(hypothesis="C_active_cancellation",
                         prediction="the update is negative (interval entirely below 0)",
                         quantity="d_p", mean=t["mean"], ci_lo=t["ci_lo"],
                         ci_hi=t["ci_hi"], predicted=0.0,
                         consistent=bool(t["ci_hi"] < 0.0)))
        rows.append(dict(hypothesis="D_adds_alignment",
                         prediction="the onset ADDS aligned component "
                                    "(interval entirely above 0) -- absent from the "
                                    "original pre-registered set",
                         quantity="d_p", mean=t["mean"], ci_lo=t["ci_lo"],
                         ci_hi=t["ci_hi"], predicted=0.0,
                         consistent=bool(t["ci_lo"] > 0.0)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["note"] = ("consistent = the interval covers the predicted value "
                       "(or, for C, lies wholly below it). More than one may be "
                       "consistent; that is information, not a failure to decide.")
    return out


def f1_rotation_share(df: pd.DataFrame, rotation: int, variant: str = "real",
                      window_col: str = "window_id") -> Dict[str, object]:
    """How much of the final aligned component the rotation block wrote.

    Reported as a share with a cluster-bootstrap interval, not as a two-sided
    classification against 0.5 and 0.9.  The two pre-registered readings become
    point predictions on the same axis: pure rotation predicts share 0 (the
    block turns what is already there), a pure new write predicts share 1.  The
    interval says which, if either, the data covers -- and an interval sitting
    at 0.6 is reported as 0.6 rather than forced into a label.
    """
    d = df[df["variant"] == variant] if "variant" in df.columns else df
    tag, pre = f"{rotation:02d}", f"{rotation-1:02d}"
    if f"p_{tag}" not in d.columns or f"p_{pre}" not in d.columns:
        return dict(available=False, reason="missing rotation blocks")
    p_pre = d[f"p_{pre}"].to_numpy(float)
    p_rot = d[f"p_{tag}"].to_numpy(float)
    eps = 1e-300
    share = np.abs(p_rot - p_pre) / (np.abs(p_rot - p_pre) + np.abs(p_pre) + eps)
    pw = pd.DataFrame({"s": share, window_col: d[window_col].to_numpy()}) \
        .groupby(window_col)["s"].mean().to_numpy()
    bs = S.cluster_bootstrap_mean(pw)
    return dict(available=True, rotation=rotation,
                written_share=float(np.nanmean(pw)),
                ci_lo=float(bs["ci_lo"]), ci_hi=float(bs["ci_hi"]),
                n_windows=int(np.isfinite(pw).sum()),
                covers_pure_rotation=bool(bs["ci_lo"] <= 0.0 <= bs["ci_hi"]),
                covers_pure_new_write=bool(bs["ci_lo"] <= 1.0 <= bs["ci_hi"]),
                note=("share of the post-rotation aligned component contributed by "
                      "the rotation block itself; 0 = it only turned what was there, "
                      "1 = it wrote all of it. No boundary is applied."))


def f2_channel_profile(df: pd.DataFrame, variant: str = "real") -> pd.DataFrame:
    d = df[df["variant"] == variant]
    tags = _tags_present(d, "au")
    rows = []
    for t in tags:
        rows.append(dict(
            block=t,
            mean_a=float(d[f"a_{t}"].mean()),
            mean_au=float(d[f"au_{t}"].mean()),
            mean_av=float(d[f"av_{t}"].mean()),
            r_entropy_a=_safe_corr(d[f"a_{t}"], d["entropy_final"]),
            r_entropy_au=_safe_corr(d[f"au_{t}"], d["entropy_final"]),
            r_entropy_av=_safe_corr(d[f"av_{t}"], d["entropy_final"]),
            r_repeat_av=_safe_corr(d[f"av_{t}"], d["repeat"]),
            r_phylop_av=_safe_corr(d[f"av_{t}"], d["phylop"]),
        ))
    return pd.DataFrame(rows)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def pool_max(df: pd.DataFrame, prefix: str, blocks: Sequence[str], out_col: str) -> pd.DataFrame:
    """Continuous readout: max over a pool of blocks (no threshold, no index)."""
    cols = [f"{prefix}_{b}" for b in blocks if f"{prefix}_{b}" in df.columns]
    df = df.copy()
    df[out_col] = df[cols].max(axis=1)
    df[out_col + "_argblock"] = df[cols].to_numpy().argmax(axis=1)
    return df


def t1_context_table(
    df: pd.DataFrame,
    value_cols: Dict[str, str],
    covariates: Sequence[str] = ADJUST_COVARIATES,
    baseline: str = "intron",
    contexts: Sequence[str] = tuple(c for c in CONTEXTS if c != "intron"),
    variant: str = "real",
) -> pd.DataFrame:
    """Table A3, extended with the channel axis.

    ``value_cols`` maps a display name to a column, e.g.
    ``{"a": "a_pool", "a_u": "au_pool", "a_v": "av_pool"}``.
    """
    d = df[df["variant"] == variant].copy()
    have = [c for c in covariates if c in d.columns]
    rows = []
    for label, col in value_cols.items():
        adj = S.residualise_within_window(d, col, have, out_col=f"{col}__adj")
        for ctx in contexts:
            raw = S.window_effect(d, value=col, target=ctx, baseline=baseline)
            adjusted = S.window_effect(adj, value=f"{col}__adj", target=ctx, baseline=baseline)
            ent_only = S.window_effect(
                S.residualise_within_window(d, col, [c for c in ("entropy_final",) if c in d.columns],
                                            out_col=f"{col}__ent"),
                value=f"{col}__ent", target=ctx, baseline=baseline)
            rows.append(dict(
                channel=label, context=ctx,
                d_unadj=raw.mean_d, lo_unadj=raw.ci_lo, hi_unadj=raw.ci_hi,
                d_entropy_only=ent_only.mean_d, lo_ent=ent_only.ci_lo, hi_ent=ent_only.ci_hi,
                d_adj=adjusted.mean_d, lo_adj=adjusted.ci_lo, hi_adj=adjusted.ci_hi,
                n_windows=raw.n_windows_used, n_dropped=raw.n_windows_dropped,
            ))
    return pd.DataFrame(rows)


def complexity_yardstick(df: pd.DataFrame, value: str, variant: str = "real") -> Dict[str, float]:
    """Repeat vs non-repeat **inside intron** -- the published size reference."""
    d = df[(df["variant"] == variant) & (df["context"] == "intron")].copy()
    d["context"] = np.where(d["repeat"] > 0.5, "intron_repeat", "intron_clean")
    eff = S.window_effect(d, value=value, target="intron_repeat", baseline="intron_clean")
    return eff.as_row()


# --------------------------------------------------------------------------
# stage 2 and the bridge
# --------------------------------------------------------------------------

def stage2_table(df: pd.DataFrame, onset: int, rotation: int, variant: str = "real") -> pd.DataFrame:
    d = df[df["variant"] == variant]
    tags = _tags_present(d, "kl")
    rows = []
    for t in tags:
        rows.append(dict(
            block=t,
            post_onset=(t == "norm") or (int(t) >= onset if t != "norm" else True),
            mean_kl=float(d[f"kl_{t}"].mean()),
            median_kl=float(d[f"kl_{t}"].median()),
            mean_entropy=float(d[f"ent_{t}"].mean()),
            top1_match_rate=float(d[f"top1_{t}"].mean()),
            mean_sctr=float(d[f"sctr_{t}"].mean()) if f"sctr_{t}" in d.columns else np.nan,
            mean_acgt_mass=float(d[f"acgtmass_{t}"].mean()),
        ))
    return pd.DataFrame(rows)


def stage2_position_features(df: pd.DataFrame, onset: int, rotation: int,
                             variant: str = "real") -> pd.DataFrame:
    """Per-position stage-2 summaries: early ratios (7B) and settle depth (long stacks)."""
    d = df[df["variant"] == variant].copy()
    tags = [t for t in _tags_present(d, "kl") if t != "norm"]
    post = [t for t in tags if int(t) >= onset]
    pre_tag = f"{onset-1:02d}"

    if f"sctr_{pre_tag}" in d.columns:
        d["r_A"] = early_ratio(d[f"sctr_{onset:02d}"], d[f"sctr_{pre_tag}"], d["sctr_norm"])
    if f"kl_{pre_tag}" in d.columns:
        d["r_D"] = 1.0 - d[f"kl_{onset:02d}"] / d[f"kl_{pre_tag}"].replace(0, np.nan)

    if len(post) >= 3:
        match = d[[f"top1_{t}" for t in post]].to_numpy()
        d["c_post"] = settle_depth(match, [int(t) for t in post], persistence=2)
        d["c_post_censored"] = ~np.isfinite(d["c_post"])
    return d


def bridge_test(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    controls: Sequence[str] = ("entropy_final", "gc", "repeat", "phylop", "boundary_dist"),
) -> Dict[str, object]:
    have = [c for c in controls if c in df.columns]
    fit = S.bridge_partial_effect(df, outcome=outcome, predictor=predictor, controls=have)
    quad = S.quadrant_assign(df.dropna(subset=[predictor, outcome]), x=predictor, y=outcome)
    counts = quad.value_counts().to_dict()
    ctx = (
        df.dropna(subset=[predictor, outcome])
        .assign(quadrant=quad)
        .groupby(["quadrant", "context"]).size().unstack(fill_value=0)
    )
    return dict(fit=asdict(fit), quadrant_counts=counts,
                quadrant_by_context=ctx.to_dict(), controls=have)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def plot_f1(profile: pd.DataFrame, out: Path, onset: Optional[int] = None,
            rotation: Optional[int] = None, title: str = "F1") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = profile[profile["block"] != "norm"].copy()
    d["b"] = d["block"].astype(int)
    d = d.sort_values("b")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].semilogy(d["b"], np.maximum(d["mean_abs_p"], 1e-300), label="|p| aligned component", lw=2)
    ax[0].semilogy(d["b"], np.maximum(d["mean_n"], 1e-300), label="n  total norm", lw=2)
    ax[0].semilogy(d["b"], np.maximum(d["mean_q"], 1e-300), label="q  orthogonal remainder", lw=2, ls="--")
    ax[0].set_ylabel("magnitude (log)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(title)

    ax[1].plot(d["b"], d["mean_a"], color="k", lw=2)
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_ylabel("a = p / n")
    ax[1].set_xlabel("block")

    for a in ax:
        if onset is not None:
            a.axvline(onset, color="C3", ls=":", lw=1.2)
        if rotation is not None:
            a.axvline(rotation, color="C0", ls=":", lw=1.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_f2(channels: pd.DataFrame, out: Path, title: str = "F2") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = channels[channels["block"] != "norm"].copy()
    d["b"] = d["block"].astype(int)
    d = d.sort_values("b")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(d["b"], d["mean_a"], label="a (total)", lw=2, color="k")
    ax[0].plot(d["b"], d["mean_au"], label="carrier  cos(h, u)", lw=2)
    ax[0].plot(d["b"], d["mean_av"], label="content  <h, v(t)>", lw=2)
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set_ylabel("mean alignment")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(title)

    ax[1].plot(d["b"], d["r_entropy_au"], label="corr(carrier, entropy)", lw=2)
    ax[1].plot(d["b"], d["r_entropy_av"], label="corr(content, entropy)", lw=2)
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_ylabel("Pearson r")
    ax[1].set_xlabel("block")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_bridge(df: pd.DataFrame, predictor: str, outcome: str, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[predictor, outcome])
    if len(d) > 40000:
        d = d.sample(40000, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    for ctx, g in d.groupby("context"):
        ax.scatter(g[predictor], g[outcome], s=2, alpha=0.25, label=ctx)
    ax.axvline(d[predictor].median(), color="0.5", lw=0.8)
    ax.axhline(d[outcome].median(), color="0.5", lw=0.8)
    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)
    ax.legend(frameon=False, fontsize=7, markerscale=4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
