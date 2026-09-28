#!/usr/bin/env bash
# flow 계획기 — root-only 평가로 재실행.
#
# 경위: 기본(전수) 평가가 8.1시간 중 6.5시간을 소모하고도 끝나지 않았다.
#   evaluate_search_state_flow 는 적격한 dev 상태 전부를 돌며 상태마다
#   계획 4샘플 x nfe 12 + 재계획 롤아웃 3종을 전개한다. 파이썬 루프라 CPU 1.2코어에 묶였다
#   (128코어 장비, GPU 5%). 진행률을 볼 수단이 없었다.
#
# 학습은 이미 끝났다(epoch 60/60, best 49). training_resume.pt 가 있으므로 재실행은
# 학습을 건너뛰고 평가부터 시작한다(state_flow.py:1227 재개 게이트).
#
# --root-only-evaluation 은 task 당 독립 루트 하나만 평가한다(CLI 도움말: "useful for
# multi-seed replication"). 전수 평가와 다른 프로토콜이며 그 사실을 결과에 명시한다.
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
LOG=$GEOWORK/geomlogs/train_flow_rootonly.log
echo "=== start $(date -u +%H:%M:%SZ) (root-only 평가)" >> "$LOG"
set +e
python3 -m geoflowagent.cli train-state-flow --config configs/geoacmg.yaml \
  --root-only-evaluation >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done $(date -u +%H:%M:%SZ) rc=$rc" >> "$LOG"
