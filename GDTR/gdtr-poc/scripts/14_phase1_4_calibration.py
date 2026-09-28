"""Phase 1.4 — q70 calibration of gamma_cos per region (sanity_gc, sanity_shuf)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase1.4"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        from src.constants_evo2 import N_LAYERS
        from src.calibration import compute_q70_per_region

        traces = np.load(ru.GDTR_ROOT / "results" / "phase1.1" / "lens_traces.npz", allow_pickle=True)
        D_cos = traces["D_cos"]  # [N, L, T]
        regions = traces["regions"]  # [N]

        # Penultimate layer = N_LAYERS - 2 = 30
        penult = N_LAYERS - 2
        gamma_per_region = compute_q70_per_region(
            D_cos, regions, n_layers=N_LAYERS, quantile=0.70, layer_for_calibration=penult,
        )
        # Also compute q70 globally
        global_q70 = float(np.quantile(D_cos[:, penult, :].reshape(-1), 0.70))

        out = {
            "phase": PHASE,
            "layer_for_calibration": penult,
            "quantile": 0.70,
            "gamma_cos_per_region": gamma_per_region,
            "gamma_cos_global_q70": global_q70,
        }
        (PHASE_DIR / "calibration.json").write_text(json.dumps(out, indent=2))
        LOG.info("calibration: %s", out)
        ru.write_done(PHASE, PHASE_DIR, out)


if __name__ == "__main__":
    main()
