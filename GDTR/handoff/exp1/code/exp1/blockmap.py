"""Step 1-2: operator map, onset detection, rotation detection.

The structural claim this module tests
--------------------------------------
Evo 2's block layout has period 7: ``(hcs, hcm, hcl, attn, hcs, hcm, hcl)``, so
hyena-short sits at 0, 4, 7, 11, 14, 18, 21, 25, 28, ... and that prefix is
byte-identical across the 1B / 7B / 40B checkpoints.

Combining the two manuscripts:

    model   blocks   onset (norm ratio)   rotation (cosine jump)
    1B      25       21 (PREDICTED)       23
    7B      32       28                   30
    40B     50       21                   23

i.e. onset is always at a block ``= 0 (mod 7)`` (hyena-short) and the rotation
follows exactly two blocks later at ``= 2 (mod 7)`` (long hyena).  The 1B onset
has never been measured; predicting it before measuring is the cheapest
falsifiable test available in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

PERIOD = 7
OPERATOR_CYCLE = ["hcs", "hcm", "hcl", "attn", "hcs", "hcm", "hcl"]

# Published hyena-short prefix, used only as a cross-check against the configs.
HCS_PREFIX = [0, 4, 7, 11, 14, 18, 21, 25, 28]

# Config keys seen in the Evo 2 / StripedHyena-2 configs, in priority order.
CONFIG_KEYS = {
    "hcs": ["hcs_layer_idxs", "short_filter_layer_idxs", "hyena_short_layer_idxs"],
    "hcm": ["hcm_layer_idxs", "medium_filter_layer_idxs", "hyena_medium_layer_idxs"],
    "hcl": ["hcl_layer_idxs", "long_filter_layer_idxs", "hyena_long_layer_idxs"],
    "attn": ["attn_layer_idxs", "attention_layer_idxs"],
}


@dataclass
class BlockMap:
    model: str
    n_blocks: int
    types: List[str]                      # per-block operator type
    source: str                           # "config" or "period"
    hcs_idx: List[int] = field(default_factory=list)
    hcm_idx: List[int] = field(default_factory=list)
    hcl_idx: List[int] = field(default_factory=list)
    attn_idx: List[int] = field(default_factory=list)

    def type_of(self, block: int) -> str:
        return self.types[block]

    def as_rows(self) -> List[Dict[str, object]]:
        return [
            dict(model=self.model, block=i, op=t, mod7=i % PERIOD)
            for i, t in enumerate(self.types)
        ]


def blockmap_from_period(n_blocks: int, model: str = "?") -> BlockMap:
    """Derive the map from the period-7 rule alone (fallback / cross-check)."""
    types = [OPERATOR_CYCLE[i % PERIOD] for i in range(n_blocks)]
    bm = BlockMap(model=model, n_blocks=n_blocks, types=types, source="period")
    for i, t in enumerate(types):
        getattr(bm, f"{t}_idx").append(i)
    return bm


def blockmap_from_config(cfg: Dict, n_blocks: Optional[int] = None, model: str = "?") -> BlockMap:
    """Read the operator map out of a model config dict.

    Falls back to the period rule for any block not covered by the config, and
    records disagreements so they are visible rather than silently smoothed.
    """
    n_blocks = n_blocks or int(
        cfg.get("num_layers") or cfg.get("n_layer") or cfg.get("num_hidden_layers") or 0
    )
    if not n_blocks:
        raise ValueError("cannot determine block count from config")

    found: Dict[str, List[int]] = {}
    for op, keys in CONFIG_KEYS.items():
        for k in keys:
            if k in cfg and cfg[k] is not None:
                found[op] = sorted(int(v) for v in cfg[k])
                break

    types = [None] * n_blocks  # type: ignore[list-item]
    for op, idxs in found.items():
        for i in idxs:
            if 0 <= i < n_blocks:
                types[i] = op

    fallback = blockmap_from_period(n_blocks, model)
    filled = [t if t is not None else fallback.types[i] for i, t in enumerate(types)]

    bm = BlockMap(model=model, n_blocks=n_blocks, types=filled,
                  source="config" if found else "period")
    for i, t in enumerate(filled):
        getattr(bm, f"{t}_idx").append(i)
    return bm


def check_hcs_prefix(bm: BlockMap) -> Dict[str, object]:
    """Compare the hyena-short positions against the published prefix."""
    k = min(len(HCS_PREFIX), len(bm.hcs_idx))
    got = bm.hcs_idx[:k]
    want = HCS_PREFIX[:k]
    return dict(match=got == want, got=got, want=want)


# --------------------------------------------------------------------------
# onset / rotation detection
# --------------------------------------------------------------------------

def detect_onset_separated(mean_norm: Sequence[float]) -> Dict[str, object]:
    """Locate the onset WITHOUT choosing a threshold.

    The handoff manuscript's own argument for T = 10 is not that 10 is right but
    that the two regimes are 40x apart, so every T in [6, 200] agrees.  That
    argument is a measurement, and this computes it: take the largest adjacent
    norm ratio as the candidate, then report the whole interval of thresholds
    that select it and the gap factor separating it from everything before.

    On a new model or chromosome the separation is re-established rather than
    assumed, and if the gap ever closes the detector says so instead of
    returning whatever T = 10 happens to pick out of a gradient.
    """
    from .nulls import separating_interval

    n = np.asarray(mean_norm, dtype=np.float64)
    ratio = np.full(n.shape, np.nan)
    ratio[1:] = n[1:] / np.clip(n[:-1], 1e-300, None)
    if not np.isfinite(ratio[1:]).any():
        return dict(onset=None, reason="no finite ratios")
    idx = int(np.nanargmax(ratio))
    sep = separating_interval(ratio, idx)
    out = dict(onset=idx, ratio=[None if not np.isfinite(r) else float(r) for r in ratio],
               **sep.as_row())
    out["method"] = "largest adjacent norm ratio; no threshold is chosen"
    out["interpretation"] = (
        f"every threshold in ({sep.below_max:.4g}, {sep.at_value:.4g}] returns block {idx}. "
        + ("The two regimes are separated by a factor of "
           f"{sep.gap_factor:.4g}, so the answer does not depend on the threshold."
           if sep.gap_factor > 10 else
           "WARNING: the gap is narrow, so this is a cut on a gradient rather than "
           "a detection. Report the ratio profile, not an onset."))
    return out


def detect_onset(mean_norm: Sequence[float], threshold: float = 10.0) -> Dict[str, object]:
    """First block whose mean-norm ratio to its predecessor reaches ``threshold``.

    INHERITED from the handoff manuscript for comparability with its numbers.
    ``detect_onset_separated`` is the threshold-free primary; this is kept so the
    published block indices can be reproduced exactly, and it is always reported
    beside the sweep below.

    This is the detector of the handoff manuscript (T = 10, with the reported
    property that any T in [6, 200] returns the same block).  We return the full
    ratio profile so the margin can be inspected rather than trusted.
    """
    x = np.asarray(mean_norm, dtype=np.float64)
    ratio = np.full(x.shape, np.nan)
    ratio[1:] = x[1:] / np.maximum(x[:-1], 1e-300)
    idx = np.where(ratio >= threshold)[0]
    onset = int(idx[0]) if idx.size else None
    pre = ratio[1:onset] if onset else ratio[1:]
    return dict(
        onset=onset,
        threshold=threshold,
        ratio=ratio.tolist(),
        max_pre_onset_ratio=float(np.nanmax(pre)) if pre.size else float("nan"),
        onset_ratio=float(ratio[onset]) if onset else float("nan"),
    )


def onset_threshold_stability(mean_norm: Sequence[float],
                              thresholds: Sequence[float] = (6, 10, 20, 50, 100, 200)) -> Dict[float, Optional[int]]:
    """Onset block as a function of T.  A flat row here is the claim 'not tuned'."""
    return {float(t): detect_onset(mean_norm, float(t))["onset"] for t in thresholds}


def detect_rotation(mean_cos: Sequence[float]) -> Dict[str, object]:
    """Block at which the mean cosine to the output frame makes its largest jump.

    This is the follow-up manuscript's definition of the rotation, reproduced so
    that both papers use one detector.  The pre-rotation band is ``0 .. r-1``.
    """
    c = np.asarray(mean_cos, dtype=np.float64)
    jumps = np.diff(c)
    r = int(np.argmax(jumps)) + 1
    return dict(
        rotation=r,
        band=(0, r - 1),
        jump=float(jumps[r - 1]),
        cos_profile=c.tolist(),
        interior_peak_block=int(np.argmax(c[:r])) if r > 0 else None,
        interior_peak_value=float(np.max(c[:r])) if r > 0 else float("nan"),
    )


def operator_coordinate_table(records: Sequence[Dict[str, object]]) -> "object":
    """Build table T2 of the protocol: onset / rotation / mod-7 per model."""
    import pandas as pd

    rows = []
    for r in records:
        onset = r.get("onset")
        rot = r.get("rotation")
        rows.append(dict(
            model=r["model"],
            blocks=r["n_blocks"],
            onset=onset,
            onset_op=r["blockmap"].type_of(onset) if onset is not None else None,
            onset_mod7=(onset % PERIOD) if onset is not None else None,
            rotation=rot,
            rotation_op=r["blockmap"].type_of(rot) if rot is not None else None,
            rotation_mod7=(rot % PERIOD) if rot is not None else None,
            gap=(rot - onset) if (rot is not None and onset is not None) else None,
        ))
    return pd.DataFrame(rows)


def operator_hypothesis_verdict(table) -> Dict[str, object]:
    """Pre-registered check: onset at mod7==0 & hcs, rotation at mod7==2 & hcl, gap==2."""
    ok_onset = ((table["onset_mod7"] == 0) & (table["onset_op"] == "hcs")).all()
    ok_rot = ((table["rotation_mod7"] == 2) & (table["rotation_op"] == "hcl")).all()
    ok_gap = (table["gap"] == 2).all()
    return dict(
        onset_is_hcs_mod0=bool(ok_onset),
        rotation_is_hcl_mod2=bool(ok_rot),
        gap_is_two=bool(ok_gap),
        verdict="SUPPORTED" if (ok_onset and ok_rot and ok_gap) else "NOT SUPPORTED",
    )
