"""Storage helpers.

Tables are written as parquet when pyarrow/fastparquet is available (the server
case) and transparently fall back to gzipped CSV otherwise, so the analysis and
self-test paths run anywhere.  Every artefact is written next to a small JSON
sidecar recording how it was produced -- the provenance lesson from the
follow-up manuscript, where two extractions of the same panel disagreed by up to
0.8 per position and a factor of two in block-29 norms.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return True
        except Exception:
            return False


def table_path(base: Path) -> Path:
    """Resolve ``base`` (extension-less) to the format available here."""
    return base.with_suffix(".parquet" if _parquet_available() else ".csv.gz")


def save_table(df: pd.DataFrame, base: Path, meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write atomically: a killed job must not leave a readable-looking file.

    An extraction that is interrupted -- a job time limit, an OOM, a preemption
    -- used to leave a truncated table in the cache.  Every later step then
    found the file, tried to read it, and died on an opaque decompression error
    several steps away from the cause.  Writing to a sibling temp file and
    renaming means the cache only ever contains complete tables: an interrupted
    run leaves nothing, and the step simply re-runs.
    """
    import os

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    p = table_path(base)
    tmp = p.with_name(p.name + f".partial-{os.getpid()}")
    try:
        if p.suffix == ".parquet":
            df.to_parquet(tmp, index=False)
        else:
            df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, p)                       # atomic on POSIX
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    write_sidecar(p, meta or {}, extra=dict(rows=len(df), cols=list(df.columns)))
    return p


class TruncatedTable(RuntimeError):
    """A cache file exists but is incomplete."""


def load_table(base: Path) -> pd.DataFrame:
    base = Path(base)
    for cand in (base, base.with_suffix(".parquet"), base.with_suffix(".csv.gz")):
        if not cand.exists():
            continue
        try:
            if cand.suffix == ".parquet":
                return pd.read_parquet(cand)
            if cand.name.endswith(".csv.gz"):
                return pd.read_csv(cand)
        except (EOFError, OSError, ValueError) as exc:
            # Say what is actually wrong and what to do, at the point of failure
            # rather than as a decompression traceback in an analysis step.
            raise TruncatedTable(
                f"{cand} is incomplete or corrupt ({type(exc).__name__}: {exc}). "
                "A run was almost certainly interrupted while writing it. Delete "
                "the file and re-run the step that produces it; nothing downstream "
                "can be trusted until you do."
            ) from exc
    raise FileNotFoundError(f"no table found for {base}")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_context() -> Dict[str, Any]:
    return dict(
        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=socket.gethostname(),
        user=getpass.getuser(),
        python=platform.python_version(),
        git=git_commit(),
    )


def write_sidecar(path: Path, meta: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    side = Path(str(path) + ".meta.json")
    payload = dict(artifact=str(path), **run_context(), **(extra or {}), **meta)
    side.write_text(json.dumps(payload, indent=2, default=str))
    return side


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
