"""Linear decodability of biology at every block.

What this measures, and what it does not
----------------------------------------
For each block, fit a linear probe from the full residual state to a biological
label and score it out of fold.  The result is a curve over the whole stack, one
per label.  It answers: *at which depth is this property linearly readable, and
how much of that readability survives the handoff?*

It does not measure information.  A property can be present and not linearly
readable, which is exactly what the handoff might do to it -- the pre-registered
hypothesis in this project's own plan distinguishes summarisation from
non-linear re-encoding, and a linear probe cannot tell them apart on its own.
So the curve is labelled *linear decodability*, and a drop is reported as a drop
in linear decodability, not as a loss of information.  Two controls make the
curve interpretable:

``shuffled labels``
    Labels permuted within window.  Any probe can fit noise given enough
    dimensions; this is what the floor actually looks like at this sample size.

``untrained model``
    The weight-shuffled extraction already used elsewhere in this package.  Run
    the same sweep on it to separate "depth does this" from "these weights do
    this".

Protocol
--------
Deliberately identical to the one the companion manuscript already uses, so the
numbers are comparable to its tables: standardise, 256 principal components,
logistic or ridge regression, folds grouped by window, a cap on positions per
class, and orientation-free AUROC for binary tasks.  Grouping by window is not
optional -- adjacent positions are 266x overdispersed, and ungrouped folds
inflate AUROC by whole points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_COMPONENTS = 256
N_FOLDS = 5
CAP_PER_CLASS = 4000
SEED = 42


@dataclass
class ProbeSpec:
    """One label to probe for."""

    name: str
    column: str
    kind: str                     # "binary" | "multiclass" | "continuous"
    positive: Optional[str] = None       # binary: the positive level
    negative: Optional[str] = None       # binary: the negative level, else "rest"
    min_per_class: int = 100

    def describe(self) -> str:
        if self.kind == "binary":
            return f"{self.column}=={self.positive} vs {self.negative or 'rest'}"
        return f"{self.column} ({self.kind})"


#: The default sweep.  Deliberately mixes annotation classes, motif classes,
#: measurements and model-internal quantities, because the interesting result is
#: how differently their curves behave across the handoff.
def default_specs(motif_names: Sequence[str] = ()) -> List[ProbeSpec]:
    specs = [
        ProbeSpec("coding_exon", "context", "binary", positive="coding_exon"),
        ProbeSpec("intron", "context", "binary", positive="intron"),
        ProbeSpec("splice_donor", "context", "binary", positive="splice_donor",
                  negative="intron", min_per_class=50),
        ProbeSpec("splice_acceptor", "context", "binary", positive="splice_acceptor",
                  negative="intron", min_per_class=50),
        ProbeSpec("context_all", "context", "multiclass"),
        ProbeSpec("canonical_core", "splice_canonical", "binary", positive="1.0",
                  min_per_class=50),
        ProbeSpec("agreement", "site_agreement", "multiclass", min_per_class=50),
        ProbeSpec("gc", "gc", "continuous"),
        ProbeSpec("phylop", "phylop", "continuous"),
        ProbeSpec("cpg_oe", "cpg_oe", "continuous"),
        ProbeSpec("ppt_fraction", "ppt_fraction", "continuous"),
        ProbeSpec("kmer3_entropy", "kmer3_entropy", "continuous"),
        ProbeSpec("next_base_entropy", "entropy_final", "continuous"),
        ProbeSpec("base_next", "base_next", "multiclass"),
    ]
    for m in motif_names:
        specs.append(ProbeSpec(f"motif_{m}", f"motif_{m}_score", "continuous"))
    return specs


def load_states(npz_path: str, meta_path_loader) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Load the stratified raw-state subsample and its metadata."""
    z = np.load(npz_path)
    states = {k: z[k] for k in z.files}
    meta = meta_path_loader()
    n = {k: v.shape[0] for k, v in states.items()}
    if len(set(n.values())) > 1:
        raise ValueError(f"raw-state blocks have different row counts: {n}")
    if states and len(meta) != next(iter(states.values())).shape[0]:
        raise ValueError(
            f"metadata has {len(meta)} rows but states have "
            f"{next(iter(states.values())).shape[0]}; they are not aligned")
    return states, meta


def _prepare(spec: ProbeSpec, meta: pd.DataFrame, rng: np.random.Generator
             ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Rows, target and groups for one spec, or None if it cannot be run."""
    if spec.column not in meta.columns:
        return None
    col = meta[spec.column]

    if spec.kind == "continuous":
        y = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isfinite(y)
        if keep.sum() < 200 or np.nanstd(y[keep]) == 0:
            return None
        return np.where(keep)[0], y[keep], meta.loc[keep, "window_id"].to_numpy(), "r2"

    lab = col.astype(str).to_numpy()
    if spec.kind == "binary":
        pos = lab == str(spec.positive)
        neg = (lab == str(spec.negative)) if spec.negative else ~pos
        keep = pos | neg
        if pos.sum() < spec.min_per_class or neg.sum() < spec.min_per_class:
            return None
        idx = np.where(keep)[0]
        # cap the majority class so one class does not set the metric
        y = pos[idx].astype(int)
        idx = _cap(idx, y, rng)
        y = pos[idx].astype(int)
        return idx, y, meta.loc[idx, "window_id"].to_numpy(), "auroc"

    # multiclass
    vc = pd.Series(lab).value_counts()
    ok = [k for k, v in vc.items() if v >= spec.min_per_class and k not in ("", "nan")]
    if len(ok) < 2:
        return None
    keep = np.isin(lab, ok)
    idx = np.where(keep)[0]
    codes = pd.Categorical(lab[idx], categories=ok).codes
    idx = _cap(idx, codes, rng)
    codes = pd.Categorical(lab[idx], categories=ok).codes
    return idx, codes, meta.loc[idx, "window_id"].to_numpy(), "macro_auroc"


def _cap(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator,
         cap: int = CAP_PER_CLASS) -> np.ndarray:
    out: List[int] = []
    for lev in np.unique(y):
        where = idx[y == lev]
        if len(where) > cap:
            where = rng.choice(where, size=cap, replace=False)
        out.extend(where.tolist())
    return np.asarray(sorted(out))


def probe_block(X: np.ndarray, y: np.ndarray, groups: np.ndarray, metric: str,
                seed: int = SEED, shuffle_labels: bool = False) -> Dict[str, float]:
    """Out-of-fold score of a linear probe, folds grouped by window."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    changed = np.nan
    if shuffle_labels:
        original = y.copy()
        y = y.copy()
        for g in np.unique(groups):          # permute inside window, not across
            m = groups == g
            y[m] = rng.permutation(y[m])
        changed = float(np.mean(y != original))
        # A within-window permutation can only do something if the label VARIES
        # inside windows.  For a label that is constant per window -- a window
        # -level property, or a class too rare to co-occur with its baseline --
        # this null is vacuous and would otherwise be reported as a floor equal
        # to the real score, which reads as "the probe learned nothing".

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return dict(score=float("nan"), sd=float("nan"), n=len(y), folds=0)
    folds = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=folds)

    scores: List[float] = []
    for tr, te in gkf.split(X, y, groups):
        # Grouped folds are uneven by construction -- a window-grouped split can
        # leave a training fold far smaller than n/folds -- so the component
        # count has to be recomputed per fold against the rows that fold
        # actually has, not against the whole sample.
        comps = int(min(N_COMPONENTS, X.shape[1], max(len(tr) - 1, 1)))
        if comps < 1 or len(tr) < 3:
            continue
        if metric == "r2":
            model = make_pipeline(StandardScaler(), PCA(n_components=comps,
                                                        random_state=seed), Ridge())
            model.fit(X[tr], y[tr])
            scores.append(float(r2_score(y[te], model.predict(X[te]))))
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(), PCA(n_components=comps, random_state=seed),
            LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        if metric == "auroc":
            a = float(roc_auc_score(y[te], proba[:, 1]))
            scores.append(max(a, 1.0 - a))          # orientation-free, as elsewhere
        else:
            try:
                scores.append(float(roc_auc_score(y[te], proba, multi_class="ovr",
                                                  average="macro")))
            except ValueError:
                continue
    if not scores:
        return dict(score=float("nan"), sd=float("nan"), n=len(y), folds=0,
                    shuffle_changed=changed)
    return dict(score=float(np.mean(scores)), sd=float(np.std(scores)),
                n=int(len(y)), folds=len(scores), shuffle_changed=changed)


def sweep(states: Dict[str, np.ndarray], meta: pd.DataFrame,
          specs: Sequence[ProbeSpec], blocks: Optional[Sequence[str]] = None,
          with_shuffled_control: bool = True, seed: int = SEED,
          n_null: int = 20, alpha: float = 0.05,
          progress: bool = True) -> pd.DataFrame:
    """One row per (block, label): the probe score and its fold spread."""
    rng = np.random.default_rng(seed)
    keys = list(blocks) if blocks else sorted(
        states, key=lambda k: int(k[1:]) if k[1:].isdigit() else 10**6)
    rows: List[Dict[str, object]] = []

    prepared = {}
    for spec in specs:
        got = _prepare(spec, meta, rng)
        if got is None:
            rows.append(dict(block=np.nan, label=spec.name, metric="",
                             score=np.nan, sd=np.nan, n=0, folds=0,
                             note="not runnable: column missing or too few per class"))
            continue
        prepared[spec.name] = (spec, got)

    for key in keys:
        X_all = states[key]
        block = int(key[1:]) if key[1:].isdigit() else -1
        for name, (spec, (idx, y, groups, metric)) in prepared.items():
            res = probe_block(X_all[idx], y, groups, metric, seed=seed)
            rows.append(dict(block=block, tap=key, label=name, metric=metric,
                             describe=spec.describe(), **res))
            if with_shuffled_control:
                # A DISTRIBUTION of permuted scores, not one draw. "Did this
                # probe decode anything" is then decided by whether the real
                # score exceeds the (1 - alpha) quantile of its own null --
                # a stated error rate -- instead of by comparing R^2 to 0.02.
                nulls = np.array([
                    probe_block(X_all[idx], y, groups, metric,
                                seed=seed + 1000 * k, shuffle_labels=True)["score"]
                    for k in range(max(n_null, 1))], dtype=float)
                from .nulls import null_exceedance
                exc = null_exceedance(res["score"], nulls, alpha=alpha)
                rows[-1].update(
                    null_mean=float(np.nanmean(nulls)),
                    null_cut=exc.get("cut"), exceeds_null=exc.get("exceeds"),
                    p_vs_null=exc.get("p_one_sided"), n_null=int(np.isfinite(nulls).sum()))
                ctl = probe_block(X_all[idx], y, groups, metric, seed=seed,
                                  shuffle_labels=True)
                note = ("" if ctl.get("shuffle_changed", 0) > 0 else
                        "VACUOUS: this label does not vary within windows, so the "
                        "within-window permutation changed nothing. Treat the floor "
                        "as undefined, not as equal to the real score.")
                rows.append(dict(block=block, tap=key, label=f"{name}__shuffled",
                                 metric=metric, describe="within-window label permutation",
                                 note=note, **ctl))
        if progress:
            print(f"[sweep] {key} done", flush=True)
    return pd.DataFrame(rows)


def retention_table(sw: pd.DataFrame, onset: int, rotation: int,
                    n_blocks: int = 32) -> pd.DataFrame:
    """How much of each label's peak decodability survives the handoff.

    ``peak_pre``   best score at any block before the onset
    ``at_onset``   score at the onset block
    ``final``      score at the last block
    ``retention``  final / peak_pre, above the shuffled floor
    """
    real = sw[~sw["label"].astype(str).str.endswith("__shuffled")]
    shuf = sw[sw["label"].astype(str).str.endswith("__shuffled")]
    if "shuffle_changed" in shuf.columns:        # drop vacuous floors entirely
        shuf = shuf[shuf["shuffle_changed"].fillna(0) > 0]
    floor = (shuf.assign(label=shuf["label"].astype(str).str.replace("__shuffled", "", regex=False))
                 .groupby("label")["score"].mean()) if not shuf.empty else pd.Series(dtype=float)
    rows: List[Dict[str, object]] = []
    for label, g in real.groupby("label"):
        g = g.dropna(subset=["score"])
        if g.empty:
            continue
        pre = g[g["block"] < onset]
        post = g[g["block"] >= onset]
        if pre.empty or post.empty:
            continue
        f = float(floor.get(label, np.nan))
        peak = float(pre["score"].max())
        peak_block = int(pre.loc[pre["score"].idxmax(), "block"])
        at_onset = float(post.loc[post["block"].idxmin(), "score"])
        final = float(post.loc[post["block"].idxmax(), "score"])
        denom = peak - f
        # Retention is only meaningful when the label was decodable ABOVE the
        # floor somewhere before the handoff.  A probe that never beat its own
        # null -- which is what a negative R^2 on a small panel means -- has no
        # peak to retain, and a ratio computed anyway would be noise with a
        # decimal point on it.
        # Usability is decided by the permutation null the sweep computed, not
        # by a constant: a label counts as decoded at a block if its score
        # exceeded the (1 - alpha) quantile of its own within-window
        # permutation distribution there.  No number is chosen here at all.
        if "exceeds_null" in g.columns:
            floor_ok = bool(g.loc[g["block"] < onset, "exceeds_null"].fillna(False).any())
        else:
            floor_ok = np.isfinite(peak)
        ok = np.isfinite(denom) and denom > 1e-6 and np.isfinite(f) and floor_ok
        rows.append(dict(
            label=label, floor=(round(f, 4) if np.isfinite(f) else np.nan),
            peak_pre=round(peak, 4), peak_block=peak_block,
            at_onset=round(at_onset, 4), final=round(final, 4),
            above_floor=bool(ok),
            retention=(round((final - f) / denom, 4) if ok else np.nan),
            drop_at_onset=round(peak - at_onset, 4),
            note=("" if ok else ("never exceeded its own permutation null before "
                                 "the handoff; retention undefined"))))
    out = pd.DataFrame(rows)
    return out.sort_values(["above_floor", "retention"], ascending=[False, True])


def plot_sweep(sw: pd.DataFrame, out: Path, onset: int, rotation: int,
               labels: Optional[Sequence[str]] = None,
               title: str = "linear decodability by block") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = sw[~sw["label"].astype(str).str.endswith("__shuffled")].dropna(subset=["block"])
    if labels:
        real = real[real["label"].isin(labels)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axvspan(onset - 1, rotation, color="0.92", zorder=0)
    for label, g in real.groupby("label"):
        g = g.sort_values("block")
        ax.plot(g["block"], g["score"], marker="o", ms=2.5, lw=1.2, label=str(label))
    shuf = sw[sw["label"].astype(str).str.endswith("__shuffled")]
    if not shuf.empty:
        m = shuf.groupby("block")["score"].mean()
        ax.plot(m.index, m.values, ls="--", lw=1, color="0.4",
                label="shuffled-label floor")
    ax.set_xlabel("block")
    ax.set_ylabel("out-of-fold score (AUROC or R^2)")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out
