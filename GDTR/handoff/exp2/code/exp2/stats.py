"""§1.6 — dependency clusters, bootstrap, IUT, equivalence.

The unit of inference is never a token and never a run. It is a genomic
dependency cluster: the connected component of a graph whose edges join
rows that share a locus, a donor, a recipient, or an overlapping input
window. Conditions / doses / layers inside a cluster are repeated measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Mapping, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# dependency clustering
# --------------------------------------------------------------------------


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class Row:
    """One measurement. `keys` are the sharing keys that create dependence."""
    value: float
    keys: tuple[str, ...]          # e.g. ("locus:chr8:1234", "donor:chr8:9999")
    window: Optional[tuple[str, int, int]] = None
    stratum: str = "all"           # annotation / task stratum, preserved in bootstrap
    meta: dict = field(default_factory=dict)
    # `keys` describe DEPENDENCE and are therefore deliberately allowed to
    # differ between paired conditions.  `unit_id` describes ALIGNMENT.  The
    # old code zipped row order, which silently paired different loci after a
    # filter/sort.  New confirmatory code always fills this field.
    unit_id: Optional[str] = None
    # Some contrasts combine rows whose donor and recipient windows differ.
    # A single `window` cannot represent that dependency union, so retain all
    # contributing half-open windows here.  `window` remains for backwards
    # compatibility with existing caches.
    windows: tuple[tuple[str, int, int], ...] = ()

    def all_windows(self) -> tuple[tuple[str, int, int], ...]:
        out = list(self.windows)
        if self.window is not None and self.window not in out:
            out.append(self.window)
        return tuple(out)


def build_clusters(rows: Sequence[Row]) -> np.ndarray:
    """Connected components over shared keys and overlapping windows.

    Returns an int array of cluster ids, one per row.
    """
    n = len(rows)
    dsu = _DSU(n)
    by_key: dict[str, int] = {}
    for i, row in enumerate(rows):
        for k in row.keys:
            j = by_key.setdefault(k, i)
            dsu.union(i, j)

    # overlapping windows on the same chromosome also create dependence
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for i, row in enumerate(rows):
        for c, a, b in row.all_windows():
            if not (a < b):
                raise ValueError(f"invalid half-open genomic window {c}:{a}-{b}")
            by_chrom.setdefault(c, []).append((a, b, i))
    for c, items in by_chrom.items():
        items.sort()
        active: list[tuple[int, int]] = []          # (end, idx)
        for a, b, i in items:
            active = [(e, j) for (e, j) in active if e > a]
            for _, j in active:
                dsu.union(i, j)
            active.append((b, i))

    roots = np.array([dsu.find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def cluster_means(rows: Sequence[Row], cluster_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§1.6: form the per-row contrast first, then average WITHIN cluster.

    Returns (per-cluster mean, per-cluster stratum label index).
    """
    vals = np.asarray([r.value for r in rows], dtype=float)
    strata = np.asarray([r.stratum for r in rows])
    k = cluster_ids.max() + 1 if len(cluster_ids) else 0
    means = np.zeros(k)
    labs = np.empty(k, dtype=object)
    for c in range(k):
        sel = cluster_ids == c
        means[c] = vals[sel].mean()
        # A dependency component can legitimately join rows from nominally
        # different strata through a shared donor/window.  It must never be
        # split for resampling.  Give that component a deterministic composite
        # stratum instead of assigning the majority label.
        labs[c] = "|".join(sorted(set(strata[sel].tolist())))
    return means, labs


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n_clusters: int
    n_rows: int
    method: str = "stratified dependency-cluster bootstrap"

    def as_row(self) -> dict:
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi,
                "n_clusters": self.n_clusters, "n_rows": self.n_rows}


def cluster_bootstrap(
    rows: Sequence[Row],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    min_clusters: int = 50,
    required: Optional[int] = None,
) -> Estimate:
    """Stratified cluster bootstrap. Strata (annotation / task) are preserved.

    Raises if usable clusters < max(min_clusters, required) -- §1.6 forbids a
    locked claim below that, and the correct response is to extend into the
    next unused split, not to lower the bar.
    """
    ids = build_clusters(rows)
    means, labs = cluster_means(rows, ids)
    k = len(means)
    need = max(min_clusters, required or 0)
    if k < need:
        raise RuntimeError(
            f"only {k} usable dependency clusters (< {need}); per §1.6 do not "
            f"make a locked claim -- extend into the next manifest split by "
            f"the pre-registered overflow rule."
        )
    rng = np.random.default_rng(seed)
    strata = np.unique(labs)
    idx_by_stratum = {s: np.flatnonzero(labs == s) for s in strata}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True) for idx in idx_by_stratum.values()
        ])
        boots[b] = statistic(means[pick])
    return Estimate(
        point=float(statistic(means)),
        lo=float(np.quantile(boots, alpha / 2)),
        hi=float(np.quantile(boots, 1 - alpha / 2)),
        n_clusters=k,
        n_rows=len(rows),
    )


def _default_join_key(row: Row) -> Hashable:
    """Return an explicit observational-unit key, never a list position.

    Old callers that pre-date `unit_id` still work when their dependency keys
    are identical (as in the original public tests).  Confirmatory callers
    should set `unit_id`; dependency keys are not generally alignment keys.
    """
    if row.unit_id is not None:
        return row.unit_id
    for name in ("unit_id", "pair_id", "locus_id", "locus", "id"):
        if name in row.meta:
            return (name, str(row.meta[name]))
    if row.keys:
        return ("keys", tuple(sorted(row.keys)))
    raise ValueError("paired row has neither unit_id, a recognised meta id, nor keys")


def align_rows(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    *,
    on: Optional[Callable[[Row], Hashable]] = None,
    require_complete: bool = True,
) -> list[tuple[Hashable, Row, Row]]:
    """Keyed inner join with duplicate/missing-unit checks.

    `zip` is prohibited for confirmatory contrasts because filtering one arm
    can shift every later pair.  Sorting is only for deterministic output and
    has no statistical meaning.
    """
    key = on or _default_join_key

    def index(rows: Sequence[Row], side: str) -> dict[Hashable, Row]:
        out: dict[Hashable, Row] = {}
        for r in rows:
            k = key(r)
            if k in out:
                raise ValueError(f"duplicate paired unit {k!r} on side {side}")
            out[k] = r
        return out

    ia, ib = index(rows_a, "a"), index(rows_b, "b")
    ka, kb = set(ia), set(ib)
    if require_complete and ka != kb:
        only_a = sorted(map(str, ka - kb))[:5]
        only_b = sorted(map(str, kb - ka))[:5]
        raise ValueError(
            "paired contrast has unmatched keyed units; "
            f"only_a={only_a}, only_b={only_b}"
        )
    common = sorted(ka & kb, key=str)
    return [(k, ia[k], ib[k]) for k in common]


def _union_windows(a: Row, b: Row) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted(set(a.all_windows()) | set(b.all_windows())))


def paired_difference_rows(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    *,
    on: Optional[Callable[[Row], Hashable]] = None,
    require_complete: bool = True,
) -> list[Row]:
    """Form a-b per keyed unit and union all dependence keys/windows first."""
    diffs: list[Row] = []
    for k, a, b in align_rows(rows_a, rows_b, on=on, require_complete=require_complete):
        if a.stratum != b.stratum:
            stratum = "|".join(sorted({a.stratum, b.stratum}))
        else:
            stratum = a.stratum
        diffs.append(Row(
            value=a.value - b.value,
            keys=tuple(sorted(set(a.keys) | set(b.keys))),
            window=None,
            windows=_union_windows(a, b),
            stratum=stratum,
            meta={"a": a.meta, "b": b.meta, "join_key": str(k)},
            unit_id=str(k),
        ))
    return diffs


def paired_contrast(rows_a: Sequence[Row], rows_b: Sequence[Row], **kw) -> Estimate:
    """Keyed paired contrast formed per unit FIRST, then cluster-averaged.

    Optional alignment controls are consumed here rather than passed to the
    bootstrap: `on=` and `require_complete=`.
    """
    on = kw.pop("on", None)
    complete = kw.pop("require_complete", True)
    diffs = paired_difference_rows(rows_a, rows_b, on=on, require_complete=complete)
    if not diffs:
        raise RuntimeError("paired contrast has no matched units")
    return cluster_bootstrap(diffs, **kw)


def required_cluster_count(manifest, analysis_key: str) -> int:
    """Read the simulation-power result for a locked confirmatory family.

    There is deliberately no fallback to 1 or 50.  A missing power result is
    an incomplete Step 2 calibration, not permission to weaken the gate.
    """
    try:
        n = int(manifest.margins.n_clusters_required[analysis_key])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"manifest is missing power-calibrated n_clusters_required[{analysis_key!r}]; "
            "run Step 2 and seal that value before confirmatory analysis"
        ) from exc
    if n <= 0:
        raise RuntimeError(f"invalid required cluster count for {analysis_key!r}: {n}")
    return n


def family_cluster_bootstrap(
    families: Mapping[str, Sequence[Row]],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    required: int,
    method: str = "bonferroni",
) -> dict[str, Estimate]:
    """Simultaneous familywise intervals for pre-registered contrasts.

    Bonferroni intervals are intentionally used here: they remain valid when
    contrasts have different missingness/dependency graphs.  A max-T routine
    would require a complete, shared cluster matrix and can silently become
    anti-conservative after eligibility filtering.
    """
    if not families:
        raise ValueError("empty contrast family")
    if method != "bonferroni":
        raise ValueError("only dependency-safe bonferroni intervals are implemented")
    fam_alpha = alpha / len(families)
    return {
        name: cluster_bootstrap(rows, n_boot=n_boot, alpha=fam_alpha,
                                seed=seed + i, required=required)
        for i, (name, rows) in enumerate(families.items())
    }


# --------------------------------------------------------------------------
# equivalence and IUT
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceResult:
    verdict: str                  # "equivalent" | "not-equivalent" | "ci-only"
    estimate: Estimate
    margin: Optional[float]
    note: str = ""


def equivalence(est: Estimate, margin: Optional[float]) -> EquivalenceResult:
    """TOST-style: the whole CI must sit inside +/- margin.

    §1.5: with no defensible margin, do NOT declare equivalence -- report the
    CI upper bound and the position inside R_nat instead.
    """
    if margin is None:
        return EquivalenceResult("ci-only", est, None,
                                 "no defensible margin; reporting CI upper bound only")
    inside = (est.lo > -margin) and (est.hi < margin)
    return EquivalenceResult("equivalent" if inside else "not-equivalent", est, margin)


def simultaneous_equivalence(
    estimates: Mapping[str, Estimate],
    margins: Mapping[str, Optional[float]] | Optional[float],
) -> dict[str, EquivalenceResult]:
    """Evaluate an already-simultaneous CI family against practical margins.

    The CIs should come from `family_cluster_bootstrap`; this function does
    not pretend that several ordinary 95% intervals are simultaneous.
    """
    out: dict[str, EquivalenceResult] = {}
    for name, est in estimates.items():
        margin = margins.get(name) if isinstance(margins, Mapping) else margins
        out[name] = equivalence(est, margin)
    return out


def positive(est: Estimate, margin: float = 0.0) -> bool:
    """Locked one-sided rule expressed on a two-sided familywise CI."""
    return bool(est.lo > margin)


@dataclass(frozen=True)
class IUTResult:
    passed: bool
    per_family: dict[str, tuple[Estimate, float, bool]]

    def failures(self) -> list[str]:
        return [k for k, (_, _, ok) in self.per_family.items() if not ok]


def intersection_union(
    contrasts: dict[str, tuple[Estimate, float]],
) -> IUTResult:
    """Step 9-5 specificity: EVERY control family must be beaten on its own
    margin. Heterogeneous controls are never pooled into one mean -- a weak
    random-subspace control must not be able to hide a failed wrong-pair one.
    """
    per = {}
    for name, (est, margin) in contrasts.items():
        per[name] = (est, margin, est.lo > margin)
    return IUTResult(all(ok for _, _, ok in per.values()), per)


# --------------------------------------------------------------------------
# Step 2 power by resampling the development clusters
# --------------------------------------------------------------------------


def simulation_power(
    dev_rows: Sequence[Row],
    *,
    effect: float,
    n_clusters: int,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """Resample DEVELOPMENT clusters (not a parametric sigma) to get power at
    a given effect and cluster count -- Step 2 "simulation-based power"."""
    ids = build_clusters(dev_rows)
    means, _ = cluster_means(dev_rows, ids)
    centered = means - means.mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        s = rng.choice(centered, size=n_clusters, replace=True) + effect
        boot = rng.choice(s, size=(400, n_clusters), replace=True).mean(1)
        if np.quantile(boot, alpha / 2) > 0:
            hits += 1
    return hits / n_sim
