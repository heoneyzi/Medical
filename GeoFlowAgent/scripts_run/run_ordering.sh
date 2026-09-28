#!/usr/bin/env bash
# 메모리 여유가 35GiB 이상일 때만 시작한다. OOM 사고(10:18~10:41Z)를 반복하지 않는다.
set -euo pipefail
cd $GEOACMG_WORK
LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
free_gib() { awk -v l="$LIMIT" '$1=="rss"{printf "%d",(l-$2)/1073741824}' /sys/fs/cgroup/memory/memory.stat; }
LOG=$GEOWORK/geomlogs/ordering.log
while [ "$(free_gib)" -lt 35 ]; do echo "$(date -u +%H:%M:%SZ) 대기: 여유 $(free_gib)GiB < 35" >> "$LOG"; sleep 60; done
export PYTHONPATH=src HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
echo "=== start $(date -u +%H:%M:%SZ) (여유 $(free_gib)GiB)" >> "$LOG"
set +e; python3 scripts_run/ordering_mechanism.py >> "$LOG" 2>&1; rc=$?; set -e
echo "=== done $(date -u +%H:%M:%SZ) rc=$rc" >> "$LOG"
