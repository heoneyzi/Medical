"""Step 4: window panels.

Why 400 windows and not 100
---------------------------
The follow-up manuscript measured 266x binomial overdispersion, lag-1
autocorrelation 0.53 and window-level eta^2 = 0.106.  The effective sample size
is therefore the number of *windows*, and only 22-39 of the original 100 windows
are usable per context.  Every interval in both manuscripts is effectively an
n = 20-40 estimate.  Quadrupling the window count halves the interval width for
a few GPU-hours -- the cheapest power available in this project.

The original 100 windows are kept as an exact subset (``legacy=True``) so the
published numbers can be reproduced on the sub-panel and shown to be consistent
with the new estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PanelSpec
from .labels import CONTEXT_CODES, ChromTracks


def _is_usable(tracks: ChromTracks, start: int, window_bp: int, max_n_fraction: float) -> bool:
    if start < 0 or start + window_bp > tracks.length:
        return False
    if tracks.sequence is None:
        return True
    seg = tracks.sequence[start : start + window_bp]
    n_frac = float((seg == ord("N")).mean())
    return n_frac <= max_n_fraction


def build_panel(
    tracks: ChromTracks,
    spec: PanelSpec,
    legacy_starts: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Sample non-overlapping windows, legacy ones first."""
    rng = np.random.default_rng(spec.seed)
    window_bp = spec.window_bp
    chosen: List[Dict[str, object]] = []
    occupied: List[range] = []

    def overlaps(s: int) -> bool:
        return any(s < r.stop and (s + window_bp) > r.start for r in occupied)

    for s in legacy_starts or []:
        if _is_usable(tracks, s, window_bp, spec.max_n_fraction) and not overlaps(s):
            chosen.append(dict(chrom=tracks.chrom, start=int(s), legacy=True))
            occupied.append(range(s, s + window_bp))

    guard = 0
    while len(chosen) < spec.n_windows and guard < spec.n_windows * 500:
        guard += 1
        s = int(rng.integers(0, max(tracks.length - window_bp, 1)))
        if overlaps(s) or not _is_usable(tracks, s, window_bp, spec.max_n_fraction):
            continue
        chosen.append(dict(chrom=tracks.chrom, start=s, legacy=False))
        occupied.append(range(s, s + window_bp))

    df = pd.DataFrame(chosen).sort_values(["legacy", "start"], ascending=[False, True])
    df["window_id"] = [f"{tracks.chrom}:{int(s)}" for s in df["start"]]
    df["window_bp"] = window_bp
    df["scored_start"] = df["start"] + (window_bp - spec.scored_bp) // 2
    df["scored_end"] = df["scored_start"] + spec.scored_bp
    return df.reset_index(drop=True)


def window_labels(
    tracks: ChromTracks,
    scored_start: int,
    scored_end: int,
    window_start: Optional[int] = None,
    motif_set=None,
) -> pd.DataFrame:
    """Per-position label frame for one window's scored region.

    Two groups of extra columns are merged on when the window's sequence is
    available, and they are kept distinct on purpose:

    * **measurements** -- observed splice core and canonicality, polypyrimidine
      fraction, CpG o/e, 3-mer entropy, low-complexity.  These are counts of the
      sequence and are always produced.
    * **model scores** -- ``motif_*``, produced only when ``motif_set`` is a
      non-empty :class:`~exp1.motifs.MotifSet`.  There is no default set: if
      none is passed, the frame simply has no model columns, rather than
      carrying scores from matrices nobody chose."""
    sl = slice(scored_start, scored_end)
    codes = tracks.context[sl]
    inv = {v: k for k, v in CONTEXT_CODES.items()}
    context = np.asarray([inv[int(c)] for c in codes], dtype=object)
    frame = pd.DataFrame(
        dict(
            pos=np.arange(scored_start, scored_end, dtype=np.int64),
            context=context,
            repeat=tracks.repeat[sl].astype(np.float32),
            gc=tracks.gc[sl],
            phylop=tracks.phylop[sl],
            boundary_dist=tracks.boundary_dist[sl].astype(np.float32),
            feature_len=np.log1p(tracks.feature_len[sl]).astype(np.float32),
        )
    )
    if tracks.sequence is not None:
        from .motifs import annotate_window, measure_window

        ws = window_start if window_start is not None else scored_start
        we = ws + max(scored_end - scored_start, 0) + 2 * (scored_start - ws)
        we = min(max(we, scored_end), tracks.length)
        local = slice(scored_start - ws, scored_end - ws)
        seq = tracks.sequence[ws:we]
        cols = (annotate_window(seq, local, context, motif_set).to_dict()
                if motif_set else measure_window(seq, local, context))
        for k, v in cols.items():
            frame[k] = v
    return frame


def panel_summary(panel: pd.DataFrame, tracks: ChromTracks, spec: PanelSpec) -> Dict[str, object]:
    """Count usable windows per context -- the number that sets the power."""
    counts: Dict[str, int] = {}
    per_context_windows: Dict[str, int] = {}
    for _, row in panel.iterrows():
        lab = window_labels(tracks, int(row["scored_start"]), int(row["scored_end"]))
        vc = lab["context"].value_counts()
        for ctx, n in vc.items():
            counts[ctx] = counts.get(ctx, 0) + int(n)
            if n >= 30:
                per_context_windows[ctx] = per_context_windows.get(ctx, 0) + 1
    return dict(
        chrom=tracks.chrom,
        n_windows=len(panel),
        n_legacy=int(panel["legacy"].sum()),
        positions_per_context=counts,
        windows_with_30plus=per_context_windows,
    )
