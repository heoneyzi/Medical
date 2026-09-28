#!/usr/bin/env bash
# 한 (거리, 시드) 조합을 자기 출력 폴더에 학습시킨다.
# 최종 집계는 나중에 compare-value-geometries 를 전체 목록으로 한 번 더 돌려서 만든다.
# 그 실행은 완료된 run 을 해시 6개로 검증하고 재사용하므로 재학습이 없다.
set -euo pipefail
GPU="$1"; ENERGY="$2"; SEED="$3"
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
OUT="artifacts/acmg/checkpoints/geometry_parts/${ENERGY}-${SEED}"
LOG="$GEOWORK/geomlogs/${ENERGY}-${SEED}.log"
echo "=== start $(date -u +%H:%M:%SZ) gpu=$GPU energy=$ENERGY seed=$SEED" >> "$LOG"
python3 -m geoflowagent.cli compare-value-geometries \
  --config configs/geoacmg.yaml \
  --output-dir "$OUT" \
  --energies "$ENERGY" \
  --seeds "$SEED" >> "$LOG" 2>&1
echo "=== done  $(date -u +%H:%M:%SZ) rc=$?" >> "$LOG"
