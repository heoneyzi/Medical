"""Two ways to say "this is a splice site", and what they disagree about.

Why both
--------
The earlier gDTR work defined its classes from GENCODE: a position is a splice
donor because the annotation says a transcript boundary is there.  The motif
work in this package defines them from sequence: a position is a donor because a
matrix counted from real donors scores it highly.  These are different claims,
and a reviewer will ask whether the result depends on which one is used.

So run everything under both, and cross them:

======================  ======================================================
``both``                annotated *and* motif-strong -- the uncontroversial set
``annot_only``          annotated but the sequence is weak -- a site that needs
                        context, or an annotation artefact
``motif_only``          strong motif, no annotation -- a cryptic or unused site
``neither``             background
======================  ======================================================

The two off-diagonal cells are the interesting ones and they are *not*
interchangeable.  If a layer-wise readout separates ``both`` from ``neither``
but not ``annot_only`` from ``neither``, the model is tracking the sequence
motif.  If it separates ``annot_only`` too, it is tracking something the local
sequence does not carry.  That distinction is unavailable from either definition
on its own, which is the whole reason for computing both.

Calibrating the motif call
--------------------------
The threshold is not chosen by eye.  It is the ``q``-th percentile of the motif
score *at annotated sites of that class*, so "motif-strong" means "scores at
least as well as the weakest q% of real annotated sites".  That makes the cut a
property of the panel, states what it costs, and can be pre-registered.  The
calibration set is annotated sites, so ``annot_only`` is by construction about
``q`` of annotated sites -- the reported number to check is how much
``motif_only`` exceeds what that implies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats as S

#: Annotation class -> the motif whose score defines the sequence-based call.
DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("splice_donor", "splice_donor_U2"),
    ("splice_acceptor", "splice_acceptor_U2"),
)

AGREEMENT_CELLS = ("both", "annot_only", "motif_only", "neither")


@dataclass
class SiteDefinition:
    annot_context: str
    motif: str
    score_col: str
    threshold: float
    quantile: float
    n_annotated: int
    n_motif_called: int
    cells: Dict[str, int]

    def as_row(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k != "cells"}
        d.update({f"n_{k}": v for k, v in self.cells.items()})
        total = max(sum(self.cells.values()), 1)
        d["motif_only_rate"] = round(self.cells["motif_only"] / total, 6)
        return d


def calibrate_threshold(df: pd.DataFrame, score_col: str, annot_context: str,
                        null_scores: Optional[Sequence[float]] = None,
                        target_fdr: float = 0.05,
                        context_col: str = "context"):
    """The cut whose false-discovery rate against a composition-matched null
    is at most ``target_fdr``.

    The previous version took the 10th percentile of the score at annotated
    sites, and the rehearsal showed what that buys: 46% of ALL positions came
    out "motif-strong", because a nine-column PWM clears that bar on ordinary
    sequence. The quantile was a number with nothing behind it.

    The null here is not invented either. The extraction already runs a
    dinucleotide-preserving shuffle of every window as an input control, so the
    same PWM scored on ``variant == "shuffle_di"`` is a composition-matched null
    for free: same base composition, same dinucleotide structure, no motif
    arrangement. The cut is then the smallest score at which the expected
    contamination from that null falls under a *stated* rate.

    ``target_fdr`` is the one remaining free number, and it is the kind that can
    be argued with: it says how much of the called set the analysis is willing
    to have be noise. It is reported with every table.
    """
    from .nulls import fdr_threshold

    obs = df.loc[df[context_col] == annot_context, score_col]
    obs = obs[np.isfinite(obs)]
    if len(obs) < 30:
        return None
    if null_scores is None or len(np.asarray(null_scores)) < 30:
        return None
    return fdr_threshold(obs.to_numpy(), np.asarray(null_scores, dtype=float),
                         target_fdr=target_fdr,
                         null_name="dinucleotide-preserving shuffle (shuffle_di variant)")


def site_definitions(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]] = DEFAULT_PAIRS,
    target_fdr: float = 0.05,
    context_col: str = "context",
    null_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Add ``site_annot`` / ``site_motif`` / ``site_agreement`` columns.

    One agreement label per position, resolved across all pairs: a position is
    ``motif_only`` if any configured motif calls it and no annotation does.
    Positions with no motif score at all (no motif set was loaded) are returned
    unchanged with a reason, rather than silently labelled ``neither``.
    """
    out = df.copy()
    usable = [(ctx, m) for ctx, m in pairs if f"motif_{m}_score" in out.columns]
    if not usable:
        return out, [dict(available=False, reason=(
            "no motif score columns in this extraction; re-run step5-extract "
            "with --motifs so the annotation/motif comparison can be made"))]

    annot_any = np.zeros(len(out), dtype=bool)
    motif_any = np.zeros(len(out), dtype=bool)
    annot_lab = np.array([""] * len(out), dtype=object)
    motif_lab = np.array([""] * len(out), dtype=object)
    reports: List[Dict[str, object]] = []

    for ctx, motif in usable:
        score_col = f"motif_{motif}_score"
        null_scores = (null_df[score_col].to_numpy(dtype=float)
                       if null_df is not None and score_col in null_df.columns else None)
        cal = calibrate_threshold(out, score_col, ctx, null_scores,
                                  target_fdr=target_fdr, context_col=context_col)
        if cal is None:
            reports.append(dict(
                available=False, annot_context=ctx, motif=motif,
                reason=("no composition-matched null available: re-run "
                        "step5-extract with shuffle_di among the variants, so the "
                        "motif call can be FDR-calibrated instead of guessed"
                        if null_scores is None else
                        "fewer than 30 annotated sites to calibrate on")))
            continue
        if not cal.feasible:
            reports.append(dict(available=False, annot_context=ctx, motif=motif,
                                reason=cal.note, **cal.as_row()))
            continue
        thr = cal.threshold
        is_annot = (out[context_col] == ctx).to_numpy()
        is_motif = (out[score_col].to_numpy() >= thr) & np.isfinite(out[score_col].to_numpy())
        annot_lab[is_annot & (annot_lab == "")] = ctx
        motif_lab[is_motif & (motif_lab == "")] = motif
        annot_any |= is_annot
        motif_any |= is_motif

        cells = dict(
            both=int((is_annot & is_motif).sum()),
            annot_only=int((is_annot & ~is_motif).sum()),
            motif_only=int((~is_annot & is_motif).sum()),
            neither=int((~is_annot & ~is_motif).sum()))
        row = SiteDefinition(
            annot_context=ctx, motif=motif, score_col=score_col, threshold=thr,
            quantile=float("nan"), n_annotated=int(is_annot.sum()),
            n_motif_called=int(is_motif.sum()), cells=cells).as_row()
        row.pop("quantile", None)
        row.update({f"fdr_{k}": v for k, v in cal.as_row().items()})
        reports.append(row)

    out["site_annot"] = np.where(annot_any, annot_lab, "none")
    out["site_motif"] = np.where(motif_any, motif_lab, "none")
    out["site_agreement"] = np.select(
        [annot_any & motif_any, annot_any & ~motif_any, ~annot_any & motif_any],
        ["both", "annot_only", "motif_only"], default="neither")
    return out, reports


def concordance_table(df: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 as counts and as rates, for the paper."""
    if "site_agreement" not in df.columns:
        return pd.DataFrame()
    vc = df["site_agreement"].value_counts()
    total = int(vc.sum())
    rows = [dict(cell=c, n=int(vc.get(c, 0)),
                 fraction=round(float(vc.get(c, 0)) / max(total, 1), 6))
            for c in AGREEMENT_CELLS]
    return pd.DataFrame(rows)


def definition_contrast(
    df: pd.DataFrame,
    value: str,
    pairs: Sequence[Tuple[str, str]] = DEFAULT_PAIRS,
    baseline: str = "intron",
    context_col: str = "context",
) -> pd.DataFrame:
    """The same contrast under the annotation definition and the motif one.

    Reported side by side rather than differenced, because the two use different
    baselines by construction and a difference of two effect sizes estimated on
    overlapping windows is not a quantity with an honest interval here.  What is
    interpretable is whether the two intervals overlap.
    """
    rows: List[Dict[str, object]] = []
    for ctx, motif in pairs:
        if context_col in df.columns and (df[context_col] == ctx).any():
            eff = S.window_effect(df, value=value, target=ctx, baseline=baseline,
                                  context_col=context_col)
            rows.append(dict(definition="annotation", site=ctx, **eff.as_row()))
        if "site_motif" in df.columns and (df["site_motif"] == motif).any():
            eff = S.window_effect(df, value=value, target=motif, baseline="none",
                                  context_col="site_motif")
            rows.append(dict(definition="motif", site=motif, **eff.as_row()))
    return pd.DataFrame(rows)


def discordance_contrast(
    df: pd.DataFrame,
    value: str,
    baseline: str = "neither",
) -> pd.DataFrame:
    """Each agreement cell against background, on the same axis.

    ``annot_only`` versus ``motif_only`` is the comparison that says whether the
    readout is following the sequence motif or something beyond it.
    """
    rows: List[Dict[str, object]] = []
    for cell in AGREEMENT_CELLS:
        if cell == baseline:
            continue
        if (df.get("site_agreement") is None) or not (df["site_agreement"] == cell).any():
            continue
        eff = S.window_effect(df, value=value, target=cell, baseline=baseline,
                              context_col="site_agreement")
        rows.append(dict(cell=cell, **eff.as_row()))
    return pd.DataFrame(rows)


def discordance_profile(df: pd.DataFrame, prefix: str, blocks: Sequence[str]
                        ) -> pd.DataFrame:
    """Per-block mean of one family of columns, split by agreement cell."""
    rows: List[Dict[str, object]] = []
    for cell, g in df.groupby("site_agreement"):
        for tag in blocks:
            col = f"{prefix}_{tag}"
            if col not in g.columns:
                continue
            rows.append(dict(cell=cell, tag=tag,
                             block=(len(blocks) if tag == "norm" else int(tag)),
                             mean=float(np.nanmean(g[col])), n=int(len(g))))
    return pd.DataFrame(rows)
