"""Importable alias to expose Phase 0 src/ modules under `phase0_src.src.*`.

Phase 0 lives at /path/to/gDTR-phase0; we re-import its src package as
`phase0_src.src` so Phase 1 scripts can do
    from phase0_src.src.gdtr import jsd_running_min_monotonic
without colliding with /path/to/gDTR/src.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PHASE0_ROOT = Path("/path/to/gDTR-phase0")


def install_alias() -> None:
    if "phase0_src" in sys.modules:
        return
    # Create phase0_src parent and phase0_src.src
    import types
    parent = types.ModuleType("phase0_src")
    parent.__path__ = [str(_PHASE0_ROOT)]
    sys.modules["phase0_src"] = parent

    src_pkg = types.ModuleType("phase0_src.src")
    src_pkg.__path__ = [str(_PHASE0_ROOT / "src")]
    sys.modules["phase0_src.src"] = src_pkg

    # Pre-import the modules we use
    for mod in ("gdtr", "stats", "constants"):
        spec = importlib.util.spec_from_file_location(
            f"phase0_src.src.{mod}",
            str(_PHASE0_ROOT / "src" / f"{mod}.py"),
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"phase0_src.src.{mod}"] = m
        spec.loader.exec_module(m)
