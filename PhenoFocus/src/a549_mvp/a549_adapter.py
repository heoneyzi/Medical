"""Load LINCS A549 (cpg0004) consensus profiles and align them to the
PhenoCompass morphology input space.

PhenoCompass' morphology encoder is an MLP with a **fixed 3297-dimensional
input** (``phenocompass.configs.model.triple_gmc.yaml`` ->
``encoders_mod.morph.num_input_features: 3297``).  Those 3297 columns are the
JUMP (U2OS) CellProfiler features, in a specific training order.

The LINCS A549 consensus profiles are also CellProfiler features but come from a
*different* pipeline/cell line, so:

* they use largely the **same feature names** (both are CellProfiler:
  ``<Compartment>_<Category>_<Measurement>_<Channel>...``), but
* the **set differs** (A549 has ~1783 features; JUMP has 3297), and
* channel-pair ordering in ``Correlation``/``Colocalization`` features can be
  swapped (``DNA_RNA`` vs ``RNA_DNA``).

This module aligns A549 profiles onto the JUMP 3297-column layout by name
(with a light canonicalisation to recover swapped correlation pairs), leaving
JUMP columns that are absent in A549 as ``0`` (a sensible fill for
mean-centred / robustized profiles).  The fraction of JUMP columns actually
covered by A549 is reported so you can quantify the cross-line feature overlap
that underlies the generalisation claim.

Nothing here needs PyTorch — it is pure pandas/numpy so it can be unit-tested
without the model.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

METADATA_PREFIX = "Metadata_"

# LINCS batch folder names inside the lincs-cell-painting ``consensus`` tree.
LINCS_A549_BATCH = "2016_04_01_a549_48hr_batch1"
# Default consensus file: whole-plate normalized, MODZ aggregated, *no* feature
# selection (keeps the full ~1783 CellProfiler features so overlap with JUMP is
# maximised).  One row per compound x dose.
DEFAULT_CONSENSUS_FILENAME = f"{LINCS_A549_BATCH}_consensus_modz.csv.gz"


# ---------------------------------------------------------------------------
# Feature-name helpers
# ---------------------------------------------------------------------------

def is_feature_column(col: str) -> bool:
    """A CellProfiler *feature* column is any column not prefixed ``Metadata_``."""
    return not col.startswith(METADATA_PREFIX)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the CellProfiler feature columns of a profile DataFrame."""
    return [c for c in df.columns if is_feature_column(c)]


# Categories whose measurement is symmetric in a pair of channels; for these we
# sort the two channel tokens so ``DNA_RNA`` and ``RNA_DNA`` canonicalise equal.
_SYMMETRIC_CATEGORIES = ("Correlation", "Colocalization")
_CHANNELS = ("DNA", "RNA", "ER", "AGP", "Mito", "Brightfield", "Hoechst", "Mitochondria")


def canonicalize_feature_name(name: str) -> str:
    """Best-effort canonical form of a CellProfiler feature name.

    Only rewrites symmetric ``Correlation``/``Colocalization`` features by
    alphabetically ordering the two channel tokens.  All other names are
    returned unchanged.  This recovers name mismatches that are purely due to
    channel-pair ordering differences between the JUMP and LINCS pipelines.
    """
    parts = name.split("_")
    if len(parts) >= 5 and parts[1] in _SYMMETRIC_CATEGORIES:
        # e.g. Cells_Correlation_Correlation_DNA_RNA -> tokens at idx 3,4
        chans = [p for p in parts if p in _CHANNELS]
        if len(chans) == 2:
            i1, i2 = parts.index(chans[0]), parts.index(chans[1], parts.index(chans[0]) + 1)
            if parts[i1] > parts[i2]:
                parts[i1], parts[i2] = parts[i2], parts[i1]
            return "_".join(parts)
    return name


# ---------------------------------------------------------------------------
# Resolving the canonical JUMP 3297-feature list
# ---------------------------------------------------------------------------

EXPECTED_MORPH_DIM = 3297


def resolve_jump_feature_columns(
    final_data_dir: Optional[str] = None,
    feature_ref: Optional[str] = None,
    cache_path: Optional[str] = None,
    expected_dim: int = EXPECTED_MORPH_DIM,
    verbose: bool = True,
) -> List[str]:
    """Resolve the ordered list of JUMP CellProfiler feature columns (len 3297).

    Resolution order (real sources always beat the cache, so a stale cache can
    never silently override your actual data):

    1. ``feature_ref`` -- an explicit path to either a ``.json`` list of column
       names, or a ``.csv``/``.csv.gz``/``.parquet`` profile file whose
       non-``Metadata_`` columns are the feature list.
    2. Auto-search under ``final_data_dir`` (typically ``.../final_data``) for a
       JUMP morphology profile with ~``expected_dim`` numeric feature columns.
    3. ``cache_path`` -- a previously cached ``.json`` list (used only when
       neither an explicit ref nor ``final_data_dir`` is given).

    The resolved list is cached to ``cache_path`` (if given) for reuse.

    Raises
    ------
    FileNotFoundError
        If no source can be found. The message explains how to supply one.
    """
    # 1. explicit reference
    if feature_ref:
        cols = _load_feature_list_from_any(feature_ref)
        if verbose:
            print(f"[jump-features] loaded {len(cols)} columns from {feature_ref}")
        _maybe_cache(cols, cache_path)
        return cols

    # 2. auto-search final_data (real data beats any cache)
    if final_data_dir and os.path.isdir(final_data_dir):
        cand = _autosearch_jump_features(final_data_dir, expected_dim, verbose=verbose)
        if cand is not None:
            cols = cand
            _maybe_cache(cols, cache_path)
            return cols

    # 3. cache (only when no explicit ref / final_data was provided)
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cols = json.load(f)
        if verbose:
            print(f"[jump-features] loaded {len(cols)} columns from cache {cache_path}")
        return cols

    raise FileNotFoundError(
        "Could not resolve the JUMP 3297-feature column list. Provide one via "
        "`feature_ref=` pointing to either (a) a JSON list of the 3297 column "
        "names, or (b) any JUMP CellProfiler profile file (.parquet/.csv[.gz]) "
        "whose non-'Metadata_' columns are those features. Such a file ships in "
        "the PhenoCompass `final_data/` deposit (e.g. under jump_map/)."
    )


def _load_feature_list_from_any(path: str) -> List[str]:
    if path.endswith(".json"):
        with open(path) as f:
            cols = json.load(f)
        if not isinstance(cols, list):
            raise ValueError(f"{path} must contain a JSON list of column names")
        return [str(c) for c in cols]
    cols = read_column_names(path)
    if cols is None:
        raise ValueError(f"Could not read column names from {path}")
    return get_feature_columns(pd.DataFrame(columns=cols))


# File extensions we know how to read column names from.
PROFILE_EXTS = (".parquet", ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt",
                ".txt.gz", ".h5ad", ".h5", ".hdf5")


def read_column_names(path: str) -> Optional[List[str]]:
    """Return the ordered column / variable names of a profile file.

    Supports parquet (schema only), csv/tsv[.gz] (header only), and AnnData
    ``.h5ad`` / generic HDF5 ``.h5`` (via ``var_names`` or a discovered table).
    Returns ``None`` if the format can't be introspected.
    """
    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        if path.endswith((".csv", ".csv.gz")):
            return list(pd.read_csv(path, nrows=0).columns)
        if path.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
            return list(pd.read_csv(path, sep="\t", nrows=0).columns)
        if path.endswith(".h5ad"):
            import anndata as ad

            a = ad.read_h5ad(path, backed="r")
            return list(a.var_names)
        if path.endswith((".h5", ".hdf5")):
            return _read_hdf5_columns(path)
    except Exception:
        return None
    return None


def _read_hdf5_columns(path: str) -> Optional[List[str]]:
    """Best-effort column-name extraction from an HDF5 file.

    Tries a pandas HDFStore table first, then AnnData, then common attributes.
    """
    # 1) pandas HDFStore
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            if keys:
                return list(pd.read_hdf(store, keys[0], stop=1).columns)
    except Exception:
        pass
    # 2) AnnData stored as .h5
    try:
        import anndata as ad

        return list(ad.read_h5ad(path, backed="r").var_names)
    except Exception:
        pass
    # 3) raw h5py: look for a 'features'/'columns'/'var' dataset
    try:
        import h5py

        with h5py.File(path, "r") as f:
            for key in ("features", "columns", "feature_names", "var_names"):
                if key in f:
                    vals = f[key][:]
                    return [v.decode() if isinstance(v, bytes) else str(v) for v in vals]
    except Exception:
        pass
    return None


def _autosearch_jump_features(
    final_data_dir: str, expected_dim: int, verbose: bool = True
) -> Optional[List[str]]:
    for path, feats, _ in iter_feature_candidates(final_data_dir):
        if abs(len(feats) - expected_dim) <= max(5, expected_dim // 20):
            if verbose:
                print(f"[jump-features] auto-detected {len(feats)} feature cols in {path}")
            return feats
    return None


def iter_feature_candidates(root: str):
    """Yield ``(path, feature_columns, all_columns)`` for every profile-like
    file under ``root`` that we can introspect. Ordered jump_map-first."""
    patterns = [
        os.path.join(root, "jump_map", "**", "*"),
        os.path.join(root, "**", "*"),
    ]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            if path in seen or not os.path.isfile(path):
                continue
            if not path.endswith(PROFILE_EXTS):
                continue
            seen.add(path)
            cols = read_column_names(path)
            if not cols:
                continue
            feats = [c for c in cols if is_feature_column(c)]
            yield path, feats, cols


def _maybe_cache(cols: List[str], cache_path: Optional[str]) -> None:
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(list(cols), f)


# ---------------------------------------------------------------------------
# A549 profile loading + alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignmentReport:
    """Diagnostics from aligning A549 features onto the JUMP layout."""

    n_jump_features: int
    n_a549_features: int
    n_matched_exact: int
    n_matched_canonical: int
    n_filled_zero: int
    coverage: float
    matched_columns: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        d = self.__dict__.copy()
        d.pop("matched_columns", None)
        return d

    def __str__(self) -> str:
        return (
            f"JUMP features: {self.n_jump_features} | A549 features: "
            f"{self.n_a549_features} | matched exact: {self.n_matched_exact} | "
            f"+canonical: {self.n_matched_canonical} | zero-filled: "
            f"{self.n_filled_zero} | coverage: {self.coverage:.1%}"
        )


def find_consensus_file(a549_dir: str, filename: Optional[str] = None) -> str:
    """Locate the A549 consensus csv.gz inside ``a549_dir``.

    Accepts either a directory containing the file directly, the lincs-style
    nested layout (``consensus/<batch>/<file>``), or a direct file path.
    """
    if os.path.isfile(a549_dir):
        return a549_dir
    target = filename or DEFAULT_CONSENSUS_FILENAME
    candidates = [
        os.path.join(a549_dir, target),
        os.path.join(a549_dir, LINCS_A549_BATCH, target),
        os.path.join(a549_dir, "consensus", LINCS_A549_BATCH, target),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # last resort: glob for any a549 consensus modz file
    hits = glob.glob(os.path.join(a549_dir, "**", "*a549*consensus*modz*.csv.gz"), recursive=True)
    hits = [h for h in hits if "feature_select" not in os.path.basename(h)]
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(
        f"Could not find A549 consensus file (looked for {target}) under {a549_dir}. "
        f"Run a549_mvp/download_a549.py first."
    )


def load_a549_consensus(
    a549_dir: str,
    filename: Optional[str] = None,
    dose_recode: Optional[int] = 6,
    drop_dmso: bool = True,
) -> pd.DataFrame:
    """Load the A549 consensus profiles.

    Parameters
    ----------
    a549_dir : str
        Directory (or file) containing the consensus csv.gz.
    filename : str, optional
        Override the consensus filename.
    dose_recode : int or None, default 6
        Keep only this recoded dose level (6 ~= 10 uM, the highest common dose).
        Set to ``None`` to keep all doses.
    drop_dmso : bool, default True
        Drop DMSO / empty ``Metadata_broad_sample`` control rows.

    Returns
    -------
    pd.DataFrame
        The consensus profiles (metadata + feature columns).
    """
    path = find_consensus_file(a549_dir, filename)
    df = pd.read_csv(path)
    if "Metadata_dose_recode" in df.columns and dose_recode is not None:
        df = df[df["Metadata_dose_recode"] == dose_recode].copy()
    if drop_dmso and "Metadata_broad_sample" in df.columns:
        bs = df["Metadata_broad_sample"].astype(str)
        df = df[~bs.isin(["nan", "", "DMSO"])].copy()
    df = df.reset_index(drop=True)
    return df


def align_to_jump(
    a549_df: pd.DataFrame,
    jump_feature_columns: List[str],
    standardize: bool = False,
) -> Tuple[np.ndarray, AlignmentReport]:
    """Align A549 CellProfiler features onto the ordered JUMP 3297-feature layout.

    Parameters
    ----------
    a549_df : pd.DataFrame
        A549 consensus profiles (metadata + features).
    jump_feature_columns : list of str
        The ordered JUMP feature columns (length should be 3297).
    standardize : bool, default False
        If True, z-score each matched column across the A549 rows before
        returning (helps when the two pipelines differ in scale). Zero-filled
        columns stay zero.

    Returns
    -------
    (np.ndarray, AlignmentReport)
        The ``(n_samples, len(jump_feature_columns))`` aligned matrix and a
        diagnostics report.
    """
    a549_feats = get_feature_columns(a549_df)
    a549_set = set(a549_feats)

    # Canonical-name index for A549 columns (for recovering swapped pairs).
    canon_to_a549: Dict[str, str] = {}
    for c in a549_feats:
        canon_to_a549.setdefault(canonicalize_feature_name(c), c)

    n = len(a549_df)
    out = np.zeros((n, len(jump_feature_columns)), dtype=np.float32)

    matched_exact = 0
    matched_canon = 0
    matched_cols: List[str] = []

    a549_values = a549_df  # alias
    for j, col in enumerate(jump_feature_columns):
        src = None
        if col in a549_set:
            src = col
            matched_exact += 1
        else:
            canon = canonicalize_feature_name(col)
            if canon in canon_to_a549:
                src = canon_to_a549[canon]
                matched_canon += 1
        if src is not None:
            vals = pd.to_numeric(a549_values[src], errors="coerce").to_numpy(dtype=np.float32)
            out[:, j] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            matched_cols.append(col)

    n_matched = matched_exact + matched_canon
    if standardize and n_matched > 0:
        cols_idx = [j for j, c in enumerate(jump_feature_columns) if c in set(matched_cols)]
        sub = out[:, cols_idx]
        mu = sub.mean(axis=0, keepdims=True)
        sd = sub.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        out[:, cols_idx] = (sub - mu) / sd

    report = AlignmentReport(
        n_jump_features=len(jump_feature_columns),
        n_a549_features=len(a549_feats),
        n_matched_exact=matched_exact,
        n_matched_canonical=matched_canon,
        n_filled_zero=len(jump_feature_columns) - n_matched,
        coverage=n_matched / max(1, len(jump_feature_columns)),
        matched_columns=matched_cols,
    )
    return out, report
