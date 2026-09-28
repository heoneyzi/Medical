#!/usr/bin/env bash
# Orchestrator — runs the full 3-phase TDiG pipeline on the H200 server.
#
# Usage:
#   ssh <gpu-host> 'cd ~/TDiG && bash scripts/run_pipeline.sh'
#
# Or detached in tmux (recommended):
#   ssh <gpu-host> 'tmux new-session -d -s tdig "cd ~/TDiG && bash scripts/run_pipeline.sh"'
#   ssh <gpu-host> 'tmux attach -t tdig'
#
# Each phase gates on sanity / verification before the next. On any failure,
# the orchestrator exits with non-zero status and the offending stage's
# _provenance.json is preserved for debugging.

set -euo pipefail

CACHE="/path/to/TDiG/data/cache"
LOGS="${CACHE}/_logs"
VENV="/path/to/gDTR/venv/bin/activate"

mkdir -p "${LOGS}"
START_TS=$(date +%s)

source "${VENV}"
cd /path/to/TDiG

# ─────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────
echo "[pre-flight] $(date -u)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader > "${LOGS}/preflight_gpu.txt"
df -h /root > "${LOGS}/preflight_disk.txt"
python -c "import evo2, torch; print(evo2.__version__, torch.__version__)" > "${LOGS}/preflight_versions.txt"

# ─────────────────────────────────────────────────────────────────────────
# Population stats + gamma calibration (~3-5 min)
# ─────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE}/population_stats/gamma_calibration.json" ]; then
    echo "[10_population_stats] $(date -u)"
    python scripts/10_population_stats.py 2>&1 | tee "${LOGS}/10_population_stats.log"
fi

# ─────────────────────────────────────────────────────────────────────────
# PHASE A — smoke test (~5 min). Gates PHASE B.
# ─────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE}/_phase_a_smoke.json" ] || ! python -c "import json; assert json.load(open('${CACHE}/_phase_a_smoke.json'))['phase_a_pass']"; then
    echo "[PHASE A — 11_smoke_batched] $(date -u)"
    python scripts/11_smoke_batched.py 2>&1 | tee "${LOGS}/11_smoke_batched.log"
    python -c "
import json
v = json.load(open('${CACHE}/_phase_a_smoke.json'))
assert v['phase_a_pass'], f'PHASE A FAILED: {v}'
print(f'PHASE A pass; batched path verified at batch_size={v[\"batch_size\"]}')
" || { echo "PHASE A failed — halt"; exit 1; }
fi

# ─────────────────────────────────────────────────────────────────────────
# PHASE B — chr22 + cross-arch concurrent (~15-20 min). Gates PHASE C.
# ─────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE}/chr22/_done" ]; then
    echo "[PHASE B — 15_chr22_forward --with-crossarch] $(date -u)"
    python scripts/15_chr22_forward.py --with-crossarch --resume 2>&1 | tee "${LOGS}/15_chr22_forward.log"
fi

# Sanity gate: re-derive M1 x Ref C splice signal, compare to gDTR baseline
python scripts/_verify_outputs.py --stage chr22 --against gdtr_baseline 2>&1 | tee "${LOGS}/_verify_chr22.log"
echo "PHASE B verified"

# ─────────────────────────────────────────────────────────────────────────
# PHASE C — chr17 + variants concurrent (~35-50 min)
# ─────────────────────────────────────────────────────────────────────────
if [ ! -f "${CACHE}/chr17/_done" ] || [ ! -f "${CACHE}/variants/_done" ]; then
    echo "[PHASE C — 16_chr17_forward & 18_variant_forward concurrent] $(date -u)"

    # Launch both in background, capture PIDs
    python scripts/16_chr17_forward.py --resume --memory-budget-gb 80 > "${LOGS}/16_chr17_forward.log" 2>&1 &
    PID_CHR17=$!
    sleep 30   # let chr17 warm up first to claim its memory share

    python scripts/18_variant_forward.py --resume --memory-budget-gb 30 > "${LOGS}/18_variant_forward.log" 2>&1 &
    PID_VARIANT=$!

    # Wait for both
    wait $PID_CHR17  || { echo "chr17 failed"; exit 1; }
    wait $PID_VARIANT || { echo "variants failed"; exit 1; }
fi

# ─────────────────────────────────────────────────────────────────────────
# Final verification
# ─────────────────────────────────────────────────────────────────────────
python scripts/_verify_outputs.py --stage all 2>&1 | tee "${LOGS}/_verify_all.log"

END_TS=$(date +%s)
WALL_MIN=$(( (END_TS - START_TS) / 60 ))

python - <<PYEOF
import json
import os
from datetime import datetime
summary = {
    "completed_utc": datetime.utcnow().isoformat() + "Z",
    "wall_minutes": ${WALL_MIN},
    "stages_done": ["population_stats", "phase_a_smoke", "phase_b_chr22_crossarch",
                    "phase_c_chr17", "phase_c_variants"],
}
with open("${CACHE}/_done_full_pipeline.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

echo "[pipeline complete] wall time: ${WALL_MIN} min"
