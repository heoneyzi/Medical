#!/usr/bin/env bash
# flow 계획기 학습. 파이프라인 train_flow 단계와 동일한 호출이다(pipeline.py:_stage_train_flow).
# 이것이 없으면 C3(계획 생성 > 단계별 선택)·P3·C5(계획기 타당성 = 표현 타당성)가 측정 불가다.
# 사다리도 hindsight(1.0)/random(0.0) 괄호만 있고 학습된 rung 이 비어 있다.
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
LOG=$GEOWORK/geomlogs/train_flow.log
echo "=== start $(date -u +%H:%M:%SZ)" >> "$LOG"
set +e
python3 -m geoflowagent.cli train-state-flow --config configs/geoacmg.yaml >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done $(date -u +%H:%M:%SZ) rc=$rc" >> "$LOG"
exit $rc
