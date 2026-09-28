#!/bin/bash
# Phase 2 master orchestrator. Halts on first verify FAIL.
set -e
source /path/to/gDTR/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/path/to/gDTR:/path/to/gDTR/scripts:/path/to/gDTR-phase0:${PYTHONPATH:-}
cd /path/to/gDTR
mkdir -p results/phase2.0 results/phase2.1 results/phase2.2 results/phase2.3 \
         results/phase2.4 results/phase2.5 results/phase2.6 results/status logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }
verify() {
  python scripts/verify_phase.py "phase2_$1" || {
    echo "[$(ts)] [FAIL] phase2.$1 verify failed"; exit 1; }
}

echo "[$(ts)] phase2.0 -> chr17 prep (CPU)"
python scripts/20_phase2_0_prep_chr17.py 2>&1 | tee logs/phase2.0.log
verify 0

# --- WAIT for previously-running GPU agents (phase3-pilot + p1-fullup) ---
# If their _done markers never appear (e.g. agents not actually launched), do
# NOT block forever — fall through after MAX_WAIT_SEC of polling.
MAX_WAIT_SEC=${PHASE2_GPU_WAIT_SEC:-7200}
WAITED=0
while [ ! -f results/phase3_pilot/_done ] || [ ! -f results/phase1.followup_full/_done ]; do
  if [ "$WAITED" -ge "$MAX_WAIT_SEC" ]; then
    echo "[$(ts)] note: phase3-pilot+p1-fullup _done markers absent after ${MAX_WAIT_SEC}s; proceeding (GPU is idle)."
    break
  fi
  echo "[$(ts)] waiting for phase3-pilot + p1-fullup ..."
  sleep 30
  WAITED=$((WAITED+30))
done

echo "[$(ts)] phase2.1 -> chr17 forward (GPU)"
python scripts/21_phase2_1_chr17_forward.py 2>&1 | tee logs/phase2.1.log
verify 1

echo "[$(ts)] phase2.2 -> Gate B chr17"
python scripts/22_phase2_2_gate_b_chr17.py 2>&1 | tee logs/phase2.2.log
verify 2

echo "[$(ts)] phase2.3 -> cross-chromosome"
python scripts/23_phase2_3_cross_chr.py 2>&1 | tee logs/phase2.3.log
verify 3

echo "[$(ts)] phase2.4 -> gene-class stratification"
python scripts/24_phase2_4_gene_class.py 2>&1 | tee logs/phase2.4.log
verify 4

echo "[$(ts)] phase2.5 -> splice fine chr17"
python scripts/25_phase2_5_splice_chr17.py 2>&1 | tee logs/phase2.5.log
verify 5

echo "[$(ts)] phase2.6 -> writeup"
python scripts/26_phase2_6_writeup.py 2>&1 | tee logs/phase2.6.log
verify 6

echo "Phase 2 COMPLETE at $(ts)" > results/phase2_all.done
echo "[$(ts)] Phase 2 done."
