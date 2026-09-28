"""VCC 2026 frozen (training-free) baselines.

Two perspectives, one pipeline:
  A) Frozen foundation models (STATE / STACK)  -> adapters/
  B) Known Perturb-seq matching (Replogle GWPS) -> reference.py + methods.py

Everything funnels into a leaderboard-ready `cell-eval prep` submission and a
3-rung diagnostic ladder for meaningful metric analysis.
"""
__version__ = "0.2.0"

# Silence a benign, very chatty anndata warning emitted when we assign var/obs
# names (it auto-transforms the index to str, which is exactly what we want).
import warnings as _warnings  # noqa: E402
_warnings.filterwarnings("ignore", message="Transforming to str index")
