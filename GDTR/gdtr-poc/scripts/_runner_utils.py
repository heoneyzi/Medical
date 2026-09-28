"""Common runner utilities: status files, done markers, structured logging."""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

GDTR_ROOT = Path("/path/to/gDTR")


def status_path(phase: str) -> Path:
    p = GDTR_ROOT / "results" / "status"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{phase}.status"


def done_marker(phase_dir: Path, step_name=None) -> Path:
    if step_name:
        return phase_dir / f"_{step_name}_done"
    return phase_dir / "_done"


def write_status(phase: str, state: str, extra: Optional[dict] = None) -> None:
    sp = status_path(phase)
    payload = {"phase": phase, "state": state}
    if extra:
        payload.update(extra)
    sp.write_text(json.dumps(payload, indent=2))


def setup_logging(phase: str, log_dir: Path = None) -> logging.Logger:
    if log_dir is None:
        log_dir = GDTR_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{phase}.log"
    logger = logging.getLogger(phase)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    # Also fan-out to root for our src.* modules
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        root.addHandler(sh)
    return logger


@contextmanager
def phase_context(phase: str, phase_dir: Path, step_name=None):
    """Idempotent + resilient phase wrapper.

    - If `_done` already exists in phase_dir, raise SystemExit(0) early.
    - Marks RUNNING -> PASS/FAIL in status file.
    - On exception, writes traceback and re-raises.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    if done_marker(phase_dir, step_name).exists():
        write_status(phase, "PASS", {"reason": "idempotent skip — _done marker exists"})
        print(f"[{phase}] _done marker exists at {phase_dir}; skipping.", flush=True)
        sys.exit(0)
    write_status(phase, "RUNNING")
    try:
        yield
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        write_status(phase, "FAIL", {"error": str(e), "traceback": tb})
        print(tb, flush=True)
        raise


def write_done(phase: str, phase_dir: Path, extra: Optional[dict] = None, step_name=None) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    done_marker(phase_dir, step_name).write_text(json.dumps(extra or {"ok": True}, indent=2))
    write_status(phase, "PASS", extra)


def add_repo_paths() -> None:
    """Make `src.*` (Phase 1) and `phase0_src.src.*` (Phase 0 reuse) importable."""
    sys.path.insert(0, str(GDTR_ROOT))
    # Install phase0_src.src.* alias so Phase 1 scripts can import Phase 0 modules
    # without colliding with /path/to/gDTR/src.
    import importlib.util
    import types
    from pathlib import Path

    if "phase0_src" not in sys.modules:
        p0 = Path("/path/to/gDTR-phase0")
        parent = types.ModuleType("phase0_src")
        parent.__path__ = [str(p0)]
        sys.modules["phase0_src"] = parent
        srcpkg = types.ModuleType("phase0_src.src")
        srcpkg.__path__ = [str(p0 / "src")]
        sys.modules["phase0_src.src"] = srcpkg
        for mod in ("constants", "gdtr", "stats"):
            spec = importlib.util.spec_from_file_location(
                f"phase0_src.src.{mod}", str(p0 / "src" / f"{mod}.py"),
            )
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"phase0_src.src.{mod}"] = m
            spec.loader.exec_module(m)


def patch_safe_globals() -> None:
    """Allow vortex checkpoints to load under torch.load(weights_only=True)."""
    import _codecs
    import torch.serialization as ts
    try:
        ts.add_safe_globals([_codecs.encode])
    except Exception:
        pass
