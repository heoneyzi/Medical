#!/usr/bin/env bash
# R1 — cosine↔euclidean 쌍 시드 2개 추가 학습.
#
# 감사 결과 (2026-09-20 08:55Z, findings/R1_pairing_audit.json):
#   기존 시드 17·29·43 은 두 family 의 trunk 초기 지문이 완전히 일치한다.
#   reusable=[17,29,43]  unusable=[]  new_pairs_needed=2  (target 5)
#
# 새 시드 59·71 을 고른 이유: 기존 17·29·43 과 같은 성격의 값(소수, 간격 12~16)을
# 이어간 것이다. 임계값이 아니라 복제 시드이고, 여기에 기록해 둔다.
#
# 한 프로세스에서 4런(2 family × 2 seed)을 돌려 store 를 한 번만 적재한다.
# 저널 §21 의 "조합마다 프로세스를 띄워 store 를 15번 적재" 를 반복하지 않는다.
#
# 출력은 별도 디렉터리로 뺀다 — geometry_comparison/comparison.json 을 덮어쓰면
# 기존 15런 집계가 깨진다 (comparison.CORRUPTED_BY_OVERWRITE.json 전례).
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
OUT=artifacts/acmg/checkpoints/geometry_r1_pairs
LOG=$GEOWORK/geomlogs/r1_paired_seeds.log
mkdir -p "$(dirname "$LOG")"
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) energies=cosine,euclidean seeds=59,71 out=$OUT" >> "$LOG"
set +e
python3 -m geoflowagent.cli compare-value-geometries \
  --config configs/geoacmg.yaml \
  --output-dir "$OUT" \
  --energies cosine euclidean \
  --seeds 59 71 >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done  $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" >> "$LOG"
