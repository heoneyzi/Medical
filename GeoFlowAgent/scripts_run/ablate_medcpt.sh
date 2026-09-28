#!/usr/bin/env bash
# 동결 MedCPT 뷰를 0 으로 채우고 같은 학습을 돌린다 — 기호 해시만으로 얼마나 가는가.
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
OUT=artifacts/acmg/checkpoints/geometry_ablate_medcpt
LOG=$GEOWORK/geomlogs/ablate_medcpt.log
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) zero_views=[medcpt] energies=cosine,euclidean seeds=17,29" >> "$LOG"
set +e
python3 -m geoflowagent.cli compare-value-geometries \
  --config configs/geoacmg_ablate_medcpt.yaml \
  --output-dir "$OUT" --energies cosine euclidean --seeds 17 29 >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done  $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" >> "$LOG"
