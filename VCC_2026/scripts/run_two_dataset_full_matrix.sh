#!/usr/bin/env bash
set -eEuo pipefail

cd "$(dirname "$0")/.."

ACK=0
PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --acknowledge-non-strict) ACK=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "usage: $0 --acknowledge-non-strict [--preflight-only]"; exit 2 ;;
  esac
done
if [[ "$ACK" != 1 ]]; then
  echo "REFUSED: this matrix uses same-dataset perturbation responses."
  echo "It is a non-strict shadow diagnostic, not complete zero-shot."
  echo "Re-run with: $0 --acknowledge-non-strict"
  exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python}"
JIANG_H5AD="${JIANG_H5AD:-data_public/perturbench/jiang24_IFNG_counts.h5ad}"
GSE_H5AD="${GSE_H5AD:-data_public/gse270828/GSE270828_counts.h5ad}"
JIANG_SPLIT="${JIANG_SPLIT:-data_shadow/jiang24_ifng_bxpc3_loco}"
GSE_SPLIT="${GSE_SPLIT:-data_shadow/gse270828_rep3_lobo_vcc2026}"
OUT_ROOT="${OUT_ROOT:-data_shadow/two_dataset_vcc2026_metrics}"
GSE_TARGET_MAP="${GSE_TARGET_MAP:-data_public/gse270828/GSE270828_feature_README.csv}"
METHODS="${METHODS:-no_effect,global_mean,gwps_direct,gwps_nearest,gwps_weighted}"
mkdir -p "$OUT_ROOT/logs"
LOG_FILE="${LOG_FILE:-$OUT_ROOT/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log}"
STATUS_FILE="$OUT_ROOT/progress.tsv"
touch "$STATUS_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$OUT_ROOT/logs/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

stage=0
total_stages=8
current_stage="preflight"
mark() {
  local status="$1" detail="$2"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$current_stage" "$status" "$detail" >> "$STATUS_FILE"
}
next_stage() {
  stage=$((stage + 1)); current_stage="$1"
  local pct=$((100 * stage / total_stages))
  printf '\n[pipeline %d/%d %3d%%] %s\n' "$stage" "$total_stages" "$pct" "$current_stage"
  mark START "$current_stage"
}
trap 'rc=$?; mark FAIL "line=$LINENO exit=$rc"; echo "[pipeline] FAILED stage=$current_stage line=$LINENO exit=$rc; resume with the same command"; exit $rc' ERR

next_stage "preflight"
echo "[protocol] NON-STRICT shadow response transfer (explicitly acknowledged)"
echo "[metrics] ArcInstitute/cell-eval2 0.16.0 preset=vcc2026: pds mse nmae fid reach jac"
echo "[scaling] each external dataset gets its own generic-response baseline and 5x split-half replicate anchors"
echo "[scope] exact metric implementation; EXTERNAL LOCAL scores, not the Challenge server/leaderboard"
echo "[plan] 2 datasets x 11 representation/method rows; each row checkpoints immediately"
echo "[progress] terminal + $STATUS_FILE"
echo "[log] $LOG_FILE"
for file in "$PYTHON" "$JIANG_H5AD" "$GSE_H5AD" \
            "$GSE_TARGET_MAP" models/SE-100M/model.safetensors models/Stack-Large/bc_large.ckpt; do
  [[ -s "$file" ]] || { echo "ERROR: missing $file"; exit 2; }
done
[[ -x .venv/bin/cell-eval2 ]] || { echo "ERROR: cell-eval2 missing"; exit 2; }
"$PYTHON" - <<'PY'
from vcc_baselines.vcc2026_eval import preflight
print("[scorer]", preflight(require_gpu=True))
PY
command -v state >/dev/null || { echo "ERROR: state missing"; exit 2; }
command -v stack-embedding >/dev/null || { echo "ERROR: stack-embedding missing"; exit 2; }
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
mark DONE "preflight passed"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[pipeline] PREFLIGHT COMPLETE; remove --preflight-only to execute"
  exit 0
fi

next_stage "prepare Jiang24 IFNG -> BxPC3 LOCO"
if [[ ! -s "$JIANG_SPLIT/shadow.yaml" ]]; then
  "$PYTHON" -m vcc_baselines prepare-shadow \
    --input "$JIANG_H5AD" --context-col cell_type --holdout-context bxpc3 \
    --out "$JIANG_SPLIT" --pert-col condition --control-label control \
    --matrix X --min-cells 30 --max-targets 100 --max-genes 4000 \
    --max-controls 500 --max-cells-per-pert 100 --cells-per-pert 50 --seed 2026
else
  echo "[pipeline] SKIP existing $JIANG_SPLIT/shadow.yaml"
fi
mark DONE "$JIANG_SPLIT/shadow.yaml"

next_stage "prepare GSE270828 rep1/rep2 -> rep3 LOBO"
if [[ ! -s "$GSE_SPLIT/shadow.yaml" ]]; then
  "$PYTHON" -m vcc_baselines prepare-shadow \
    --input "$GSE_H5AD" --context-col rep --holdout-context rep3 \
    --out "$GSE_SPLIT" --pert-col har --control-label Non-Targeting \
    --single-col "" --matrix X --min-genes 0 --min-counts 0 \
    --min-cells 30 --max-targets 0 --max-genes 4000 \
    --max-controls 500 --max-cells-per-pert 100 --cells-per-pert 50 --seed 2026 \
    --target-gene-map "$GSE_TARGET_MAP"
else
  echo "[pipeline] SKIP existing $GSE_SPLIT/shadow.yaml"
fi
mark DONE "$GSE_SPLIT/shadow.yaml"

next_stage "frozen STATE-SE + STACK embeddings for Jiang24"
RUN_LOCAL_BENCHMARK=0 bash scripts/run_frozen_contexts.sh "$JIANG_SPLIT"
mark DONE "$JIANG_SPLIT/frozen"

next_stage "frozen STATE-SE + STACK embeddings for GSE270828"
RUN_LOCAL_BENCHMARK=0 bash scripts/run_frozen_contexts.sh "$GSE_SPLIT"
mark DONE "$GSE_SPLIT/frozen"

next_stage "Jiang24 exact vcc2026 six-metric matrix"
"$PYTHON" -m vcc_baselines shadow-benchmark --config "$JIANG_SPLIT/shadow.yaml" \
  --embedding "state=$JIANG_SPLIT/frozen/state_contexts.npz" \
  --embedding "stack=$JIANG_SPLIT/frozen/stack_contexts.npz" \
  --methods "$METHODS" --engine cell-eval2 --resume \
  --out "$OUT_ROOT/jiang24_vcc2026.csv"
mark DONE "$OUT_ROOT/jiang24_vcc2026.csv"

next_stage "GSE270828 exact vcc2026 six-metric matrix"
"$PYTHON" -m vcc_baselines shadow-benchmark --config "$GSE_SPLIT/shadow.yaml" \
  --embedding "state=$GSE_SPLIT/frozen/state_contexts.npz" \
  --embedding "stack=$GSE_SPLIT/frozen/stack_contexts.npz" \
  --methods "$METHODS" --engine cell-eval2 --target-gene-map "$GSE_TARGET_MAP" --resume \
  --out "$OUT_ROOT/gse270828_vcc2026.csv"
mark DONE "$OUT_ROOT/gse270828_vcc2026.csv"

next_stage "merge metrics and changes from no-effect"
"$PYTHON" scripts/compare_six_metric_matrix.py \
  --result "Jiang24_IFNG_BxPC3=$OUT_ROOT/jiang24_vcc2026.csv" \
  --result "GSE270828_rep3=$OUT_ROOT/gse270828_vcc2026.csv" \
  --out-wide "$OUT_ROOT/comparison_wide.csv" \
  --out-long "$OUT_ROOT/metric_changes_long.csv"
mark DONE "$OUT_ROOT/comparison_wide.csv"

echo "[pipeline] COMPLETE"
echo "[pipeline] results=$OUT_ROOT/comparison_wide.csv"
echo "[pipeline] metric changes=$OUT_ROOT/metric_changes_long.csv"
echo "[pipeline] progress=$STATUS_FILE log=$LOG_FILE"
