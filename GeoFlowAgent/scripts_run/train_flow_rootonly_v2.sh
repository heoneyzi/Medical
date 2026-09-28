#!/usr/bin/env bash
# flow 계획기 — root-only 평가, train 롤아웃 생략.
#
# 경위 (RUN_LOG 2026-09-19 22:40Z, 2026-09-20 01:10Z):
#   1) 기본(전수) 평가는 8.1시간 중 6.5시간을 평가에 쓰고도 끝나지 않아 종료(rc=143).
#   2) --root-only-evaluation 재실행도 3시간을 넘겼다. root-only 는 상태 수를 줄이지만
#      상태당 비용은 더 크다 — 루트 롤아웃은 최대 12단계를 다 전개한다.
#   3) state_flow.py:1364 가 제공하는 축소 스위치를 뒤늦게 발견했다.
#
#   GEOFLOW_STATE_FLOW_REPORT_TRAIN=0 이면 train 롤아웃(1,753 루트)을 통째로 건너뛰고
#   dev(584 루트 x samples_per_state=4)만 평가한다. train 평가는 표본 내라 논문에 쓰지
#   않으므로 처음부터 이걸 썼어야 했다.
#
# 재개: 학습은 끝났다(epoch 60/60, best 49, dev 손실 1.4803). training_resume.pt 가
#   state_flow.py:1227 재개 게이트를 통과하면 학습을 건너뛰고 평가부터 시작한다.
#   2026-09-20 검증: 게이트가 비교하는 5개 해시가 전부 일치한다.
#     source_tree_sha256     f2fce4e1...  (src/geoflowagent)
#     cache_content_sha256   00b19e6e...
#     source_manifest_sha256 4044cf4c...
#     training_config_sha256 a921b7c0...
#     resume_version         geoflowagent.search-state-flow-resume.v2
#
# 프로토콜 명시: --root-only-evaluation 은 task 당 독립 루트 하나만 평가한다.
#   전수 평가와 다른 프로토콜이며, 그리고 이 실행은 train 롤아웃을 보고하지 않는다.
#   결과에 두 사실을 모두 붙인다. 전수 평가 수치가 필요하면 별도로 돌려야 한다.
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export GEOFLOW_STATE_FLOW_REPORT_TRAIN=0
LOG=$GEOWORK/geomlogs/train_flow_rootonly_v2.log
mkdir -p "$(dirname "$LOG")"
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) (root-only 평가, train 롤아웃 생략)" >> "$LOG"
set +e
python3 -m geoflowagent.cli train-state-flow --config configs/geoacmg.yaml \
  --root-only-evaluation >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" >> "$LOG"
