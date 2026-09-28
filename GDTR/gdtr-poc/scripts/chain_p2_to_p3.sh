#!/bin/bash
set -e
cd /path/to/gDTR
source /path/to/gDTR/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
echo "[$(date)] Chain P2->P3 started, waiting for Phase 2 done"
while [ ! -f /path/to/gDTR/results/phase2_all.done ]; do sleep 60; done
echo "[$(date)] Phase 2 done detected"
python scripts/31_phase3_main.py 2>&1 | tee logs/phase3-main-forward.log
python scripts/verify_phase.py phase3_main_forward || { echo FAIL_FORWARD; exit 1; }
python scripts/32_phase3_main_analysis.py 2>&1 | tee logs/phase3-main-analysis.log
python scripts/verify_phase.py phase3_main || { echo FAIL_ANALYSIS; exit 1; }
touch /path/to/gDTR/results/phase3_main/_done
echo "Phase 3 chain complete at $(date)" > /path/to/gDTR/results/phase3_main_chain.done
