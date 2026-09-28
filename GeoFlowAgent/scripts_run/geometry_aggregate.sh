#!/usr/bin/env bash
# geometry 스윕 집계. 15런이 전부 끝난 뒤에만 돌린다.
#
# 이 실행은 재학습을 하지 않는다. 완료된 run 을 8개 검사로 검증하고 그대로 재사용한다
# (value.py:1127 training_result_version + 1133-1140 해시 6개 + 1146-1151 checkpoint_sha256).
# 검증 실패 시 조용히 재학습하지 않고 RuntimeError 로 죽는다 (value.py:1153) — fail-loud.
#
# 단, 재학습이 0이어도 SearchFeatureStore 를 통째로 적재하므로 ~30GiB 를 잡는다.
# 그래서 학습 배치가 하나라도 살아 있으면 거부한다.
set -euo pipefail
cd $GEOACMG_WORK
export PYTHONPATH=src
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR/hub
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

OUT=artifacts/acmg/checkpoints/geometry_comparison
LOG=$GEOWORK/geomlogs/aggregate.log

# 안전장치 1: 학습이 돌고 있으면 거부
n=$(ps -eo args= | grep -c '^python3 -m geoflowagent\.cli compare-value-geometries' || true)
if [ "$n" -gt 0 ]; then
  echo "거부: 학습 프로세스 ${n}개가 아직 살아 있다. 집계는 ~30GiB 를 더 잡으므로 전부 끝난 뒤 돌린다."
  exit 1
fi

# 안전장치 2: 15런이 전부 완료됐는지 확인
missing=$(python3 - <<'PY'
import os
ENER=["cosine","euclidean","directed_quasimetric","poincare","pair_mlp"]
miss=[f"{e}/seed-{s}" for e in ENER for s in (17,29,43)
      if not os.path.exists(f"artifacts/acmg/checkpoints/geometry_comparison/{e}/seed-{s}/value_metrics.json")]
print(",".join(miss))
PY
)
if [ -n "$missing" ]; then
  echo "거부: 미완료 런이 있다 -> $missing"
  echo "  (일부만으로 집계하려면 --energies/--seeds 로 명시적으로 좁혀서 직접 실행할 것)"
  exit 1
fi

echo "=== 집계 시작 $(date -u +%H:%M:%SZ) — 15런 전부 완료 확인됨" | tee -a "$LOG"
python3 -m geoflowagent.cli compare-value-geometries \
  --config configs/geoacmg.yaml \
  --output-dir "$OUT" 2>&1 | tee -a "$LOG"
echo "=== 집계 종료 $(date -u +%H:%M:%SZ) rc=$?" | tee -a "$LOG"
