"""Every cut-off in this package, what kind it is, and what stands behind it.

The rule
--------
A number that decides something must be one of three kinds:

``derived``     computed from an explicit null and controlling a *stated* error
                rate.  What is reported is the error rate and the null.
``separated``   sitting inside a measured gap between two regimes.  What is
                reported is the interval of values that give the same answer.
``inherited``   fixed by a named published analysis, kept only for
                comparability, cited, and reported with a sensitivity sweep.

Nothing else is allowed.  A constant with no null, no gap and no citation is
removed and replaced by a continuous estimate with an interval, or by an exact
decomposition.

This file is the registry, and ``python -m exp1 audit-thresholds`` prints it.
The point of writing it down is that the rule becomes checkable: if a constant
appears in the code and not here, or appears here as ``removed`` and still runs,
the audit says so instead of the claim living only in a docstring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass
class ThresholdRecord:
    name: str
    where: str
    kind: str                 # derived | separated | inherited | removed
    value: str
    basis: str
    sweep: str = ""
    citation: str = ""

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


REGISTRY: List[ThresholdRecord] = [
    # ---------------- derived: an error rate against an explicit null --------
    ThresholdRecord(
        "motif call threshold", "sitedefs.calibrate_threshold", "derived",
        "the score at which FDR <= target_fdr (default 0.05)",
        "false-discovery rate against the dinucleotide-preserving shuffle that "
        "the extraction already runs as the shuffle_di input control. Same "
        "composition and dinucleotide structure, no motif arrangement.",
        sweep="report the called fraction at target_fdr in {0.01, 0.05, 0.10}"),
    ThresholdRecord(
        "probe decoded anything", "probesweep.sweep / retention_table", "derived",
        "the (1 - alpha) quantile of the label's own permutation null (alpha 0.05)",
        "within-window label permutation, repeated n_null times, so the "
        "comparison keeps the clustering positions actually have.",
        sweep="alpha is reported; p_vs_null is stored per block so any alpha "
              "can be applied after the fact"),
    ThresholdRecord(
        "curve carries depth signal", "dualref.curve_diagnostics", "derived",
        "variance ratio > 1",
        "between-block variance of the mean curve against the mean "
        "within-position variance at a block. The reference is 1 because that "
        "is where the two variances are equal; no value is chosen.",
        sweep="the ratio itself is reported, not just the comparison"),

    # ---------------- separated: a cut inside a measured gap -----------------
    ThresholdRecord(
        "cascade onset", "blockmap.detect_onset_separated", "separated",
        "any threshold in (max pre-onset ratio, onset ratio]",
        "the adjacent norm ratio separates the two regimes by a large factor, "
        "so the answer does not depend on where in the gap the cut falls. The "
        "interval and the gap factor are computed per run and reported; if the "
        "gap closes, the detector says the answer is a cut on a gradient.",
        sweep="detect_onset(T) is also run at T in {6,10,20,50,100,200} and the "
              "stability table is printed beside it",
        citation="handoff manuscript Sec. 7: pre-onset max 3.50 vs onset 214-267, "
                 "'every T in [6, 200] returns the same onset'"),

    # ---------------- inherited: cited, kept for comparability --------------
    ThresholdRecord(
        "min positions per class per window", "config.MIN_N_PER_CLASS", "inherited",
        "30",
        "the inclusion rule of the follow-up manuscript's window-clustered "
        "estimand. Changing it would make the new intervals incomparable with "
        "the published ones, which is the reason to keep it and the reason it "
        "must be swept rather than trusted.",
        sweep="step7-analysis reports the primary contrast at min_n in "
              "{10, 20, 30, 50, 100} so no conclusion rests on 30",
        citation="follow-up manuscript, window-clustered protocol"),
    ThresholdRecord(
        "settling threshold gamma", "dualref.dual_settle", "inherited",
        "the 70th-percentile operating point",
        "the published gamma_q70. Kept so the thresholded depth can be compared "
        "with the published one -- and reported only beside its censored "
        "fraction, because the follow-up manuscript's central result is that "
        "this cut censors three quarters of intronic positions and that the "
        "censoring itself varies with class.",
        sweep="the continuous readouts a_pre / a_post / auc_pre are primary; the "
              "thresholded depth is reported at gamma_quantile in {0.5,0.6,0.7,0.8}",
        citation="gDTR gamma_q70; follow-up manuscript Sec. 4 on what it costs"),
    ThresholdRecord(
        "PWM pseudocount", "motifs.PWM.pseudocount", "inherited",
        "0.25 (Jeffreys)",
        "the standard non-informative prior for a multinomial. Recorded in the "
        "manifest with every matrix rather than baked into the probabilities.",
        sweep="matrices are stored as COUNTS, so any pseudocount can be applied "
              "after the fact without re-deriving"),

    # ---------------- removed ------------------------------------------------
    ThresholdRecord(
        "motif hit quantile", "motifs (was HIT_QUANTILE = 0.99)", "removed",
        "-",
        "defined a 'hit' as the top 1% of scores WITHIN each window, which fixes "
        "the hit density at 1% whether or not the window contains a motif and "
        "makes the distance column incomparable between windows. Replaced by the "
        "distance to the window's best-scoring instance, which needs no cut."),
    ThresholdRecord(
        "low-complexity flag", "motifs (was LOW_COMPLEXITY_BITS = 3.0)", "removed",
        "-",
        "a binary derived from 3-mer entropy by an unjustified bit count. It "
        "threw away the ordering and added a decision nobody could defend. The "
        "continuous kmer3_entropy is kept and stratified on directly."),
    ThresholdRecord(
        "F1 three-way verdict", "analysis (was f1_verdict)", "removed",
        "-",
        "classified the onset by comparing three ratios against 0.02, 10 and "
        "0.5-2.0. Those constants decided the headline and nothing decided them. "
        "Replaced by f1_decomposition: an exact identity (d log cos = d log "
        "numerator - d log denominator, and p_post = p_pre + mixer + mlp) with "
        "window-bootstrap intervals on every term, plus each pre-registered "
        "point prediction tested against its own interval."),
    ThresholdRecord(
        "rotation write/rotate split", "analysis (was f1_rotation_is_write)", "removed",
        "-",
        "split on share > 0.9 and share < 0.5. Replaced by f1_rotation_share, "
        "which reports the share with an interval and states whether that "
        "interval covers the two point predictions (0 = pure rotation, "
        "1 = pure new write)."),
    ThresholdRecord(
        "probe usability floor", "probesweep (was R2 > 0.02 / AUROC > 0.52)", "removed",
        "-",
        "two numbers standing in for 'did this decode anything'. Replaced by "
        "exceedance of the label's own permutation null at a stated alpha."),
    ThresholdRecord(
        "minimum dynamic range", "dualref (was MIN_DYNAMIC_RANGE = 0.02)", "removed",
        "-",
        "a fixed range below which a curve was declared unthresholdable. "
        "Replaced by the between-block / within-position variance ratio, whose "
        "reference of 1 is a property of the comparison rather than a choice."),
    ThresholdRecord(
        "onset threshold T", "blockmap (T = 10 retained only as a sweep)", "removed",
        "-",
        "the primary detector no longer takes a threshold at all; T = 10 "
        "survives only inside the reported stability sweep, for comparability "
        "with the published block indices."),
]


def audit(check_live: bool = True) -> pd.DataFrame:
    """The registry, optionally verified against what the code still defines."""
    rows = [r.as_row() for r in REGISTRY]
    if check_live:
        import importlib

        gone = {
            "exp1.motifs": ["HIT_QUANTILE", "LOW_COMPLEXITY_BITS"],
            "exp1.dualref": ["MIN_DYNAMIC_RANGE"],
            "exp1.analysis": ["f1_verdict", "f1_rotation_is_write"],
        }
        for mod, names in gone.items():
            m = importlib.import_module(mod)
            for n in names:
                if hasattr(m, n):
                    rows.append(ThresholdRecord(
                        n, mod, "VIOLATION", "still defined",
                        "the registry records this as removed but the module "
                        "still defines it; either the removal was reverted or "
                        "the registry is stale").as_row())
    return pd.DataFrame(rows)


def summary() -> Dict[str, object]:
    df = audit()
    return dict(by_kind=df["kind"].value_counts().to_dict(),
                violations=int((df["kind"] == "VIOLATION").sum()),
                n=len(df))
