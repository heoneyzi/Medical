#!/usr/bin/env bash
# 한 프로세스가 여러 (거리 x 시드) 조합을 연속 실행한다.
#
# 왜 배치인가:
#   compare_value_geometries 는 SearchFeatureStore 를 딱 한 번 만들어(value.py:1508)
#   모든 조합에 _store_override 로 넘긴다(value.py:1525, 1047-1056).
#   그 store 가 프로세스 RSS 의 대부분(~29GiB: examples.jsonl 448,704행을 파이썬 dict 로)이다.
#   조합마다 프로세스를 띄우면 로딩(약 12분)과 29GiB 를 매번 새로 낸다.
#   한 프로세스에 묶으면 로딩 1회 · 메모리 1벌로 끝난다.
#
# 수치 동일성: train_value_geometry 는 매 run 시작에 seed_everything(seed) 로
#   random/numpy/torch/CUDA RNG 와 deterministic 설정을 전부 리셋한다(value.py:1040).
#   따라서 묶어 돌려도 개별 실행과 결과가 같다.
#
# 인자: GPU SEED ENERGY [ENERGY ...]     (--energies 는 nargs="+" 이므로 공백 구분)
set -euo pipefail
GPU="$1"; SEED="$2"; shift 2
ENERGIES=("$@")
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"

# 집계가 기대하는 레이아웃에 직접 쓴다: {out}/{energy}/seed-{seed}  (value.py:1516)
# 그래야 나중에 통합 이동이 필요 없다.
OUT=artifacts/acmg/checkpoints/geometry_comparison
LOG="$GEOWORK/geomlogs/batch-seed${SEED}.log"

echo "=== start $(date -u +%H:%M:%SZ) gpu=$GPU seed=$SEED energies=${ENERGIES[*]}" >> "$LOG"
set +e
python3 -m geoflowagent.cli compare-value-geometries \
  --config configs/geoacmg.yaml \
  --output-dir "$OUT" \
  --energies "${ENERGIES[@]}" \
  --seeds "$SEED" >> "$LOG" 2>&1
rc=$?
set -e
echo "=== done  $(date -u +%H:%M:%SZ) rc=$rc" >> "$LOG"
exit $rc
