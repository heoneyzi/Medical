#!/bin/bash
set -e
source /path/to/gDTR/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/path/to/gDTR:/path/to/gDTR/scripts:/path/to/gDTR-phase0:${PYTHONPATH:-}
cd /path/to/gDTR
mkdir -p results/phase1.6 logs results/status
python scripts/16_phase1_6_chr22_forward.py 2>&1 | tee logs/phase1.6_chr22.log
python scripts/verify_phase.py 1.6_chr22 || { echo "verify 1.6_chr22 FAILED"; exit 1; }
