#!/usr/bin/env bash
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
LOG=$GEOWORK/geomlogs/capacity_matched.log
echo "=== start $(date -u +%H:%M:%SZ)" >> "$LOG"
set +e; python3 scripts_run/capacity_matched.py >> "$LOG" 2>&1; rc=$?; set -e
echo "=== done $(date -u +%H:%M:%SZ) rc=$rc" >> "$LOG"
