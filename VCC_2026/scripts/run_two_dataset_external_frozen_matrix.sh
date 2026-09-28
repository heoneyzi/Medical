#!/usr/bin/env bash
set -eEuo pipefail

cd "$(dirname "$0")/.."

PROTOCOL="replogle-only"
MODELS="${MODELS:-uce,transcriptformer_cell,scgpt,scfoundation,scprint2}"
PREFLIGHT_ONLY=0
ALLOW_UNAVAILABLE=0
ACK=0
for arg in "$@"; do
  case "$arg" in
    --protocol=replogle-only) PROTOCOL="replogle-only" ;;
    --protocol=shadow-existing) PROTOCOL="shadow-existing" ;;
    --models=*) MODELS="${arg#--models=}" ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --allow-unavailable) ALLOW_UNAVAILABLE=1 ;;
    --acknowledge-non-strict) ACK=1 ;;
    *)
      echo "usage: $0 [--protocol=replogle-only|shadow-existing] [--models=a,b]"
      echo "          [--preflight-only] [--allow-unavailable] [--acknowledge-non-strict]"
      exit 2
      ;;
  esac
done

if [[ "$PROTOCOL" == "shadow-existing" && "$ACK" != 1 ]]; then
  echo "REFUSED: shadow-existing consumes same-dataset source-context responses."
  echo "Re-run with --acknowledge-non-strict or use --protocol=replogle-only."
  exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python}"
BASE_ROOT="${OUT_ROOT:-data_experiments/external_frozen_matrix/$PROTOCOL}"
mkdir -p "$BASE_ROOT/logs"
LOG_FILE="${LOG_FILE:-$BASE_ROOT/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log}"
STATUS_FILE="$BASE_ROOT/progress.tsv"
touch "$STATUS_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$BASE_ROOT/logs/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$PROTOCOL" == "replogle-only" ]]; then
  JIANG_CONFIG="data_experiments/replogle_only/jiang24/shadow.yaml"
  GSE_CONFIG="data_experiments/replogle_only/gse270828/shadow.yaml"
  METHODS="no_effect,global_mean,replogle_k562,gwps_direct,gwps_nearest,gwps_weighted"
  RESULT_PROTOCOL="strict-replogle-only"
else
  JIANG_CONFIG="data_shadow/jiang24_ifng_bxpc3_loco/shadow.yaml"
  GSE_CONFIG="data_shadow/gse270828_rep3_lobo_vcc2026/shadow.yaml"
  METHODS="no_effect,global_mean,gwps_direct,gwps_nearest,gwps_weighted"
  RESULT_PROTOCOL="non-strict-shadow"
fi

stage=0
if [[ "$PROTOCOL" == "replogle-only" ]]; then total_stages=10; else total_stages=7; fi
current_stage="startup"
mark() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$current_stage" "$1" "$2" >> "$STATUS_FILE"
}
next_stage() {
  stage=$((stage + 1)); current_stage="$1"
  printf '\n[external-pipeline %d/%d %3d%%] %s\n' "$stage" "$total_stages" "$((100*stage/total_stages))" "$current_stage"
  mark START "$current_stage"
}
trap 'rc=$?; mark FAIL "line=$LINENO exit=$rc"; echo "[external-pipeline] FAILED stage=$current_stage line=$LINENO exit=$rc"; exit $rc' ERR

next_stage "prepare protocol data"
if [[ "$PROTOCOL" == "replogle-only" ]]; then
  "$PYTHON" scripts/build_replogle_only_experiment.py
fi
for required in "$PYTHON" "$JIANG_CONFIG" "$GSE_CONFIG" manifests/gene_map.csv; do
  [[ -s "$required" ]] || { echo "ERROR: missing $required"; exit 2; }
done
if [[ "$PROTOCOL" == "replogle-only" ]]; then
  "$PYTHON" scripts/audit_external_experiment.py \
    --config "$JIANG_CONFIG" --config "$GSE_CONFIG" \
    --out "$BASE_ROOT/data_policy_audit.json"
fi
mark DONE "$PROTOCOL"

AVAIL_FLAGS=()
if [[ "$ALLOW_UNAVAILABLE" == 1 ]]; then AVAIL_FLAGS+=(--allow-unavailable); fi

next_stage "external model preflight"
"$PYTHON" scripts/run_external_frozen_embeddings.py \
  --config "$JIANG_CONFIG" --models "$MODELS" \
  --artifact-root "$BASE_ROOT/jiang24_embeddings" --preflight-only "${AVAIL_FLAGS[@]}"
mark DONE "$MODELS"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[external-pipeline] PREFLIGHT COMPLETE; remove --preflight-only to execute"
  exit 0
fi

next_stage "external embeddings Jiang24"
"$PYTHON" scripts/run_external_frozen_embeddings.py \
  --config "$JIANG_CONFIG" --models "$MODELS" \
  --artifact-root "$BASE_ROOT/jiang24_embeddings" "${AVAIL_FLAGS[@]}"
mark DONE "$BASE_ROOT/jiang24_embeddings/enrolled_models.json"

next_stage "external embeddings GSE270828"
"$PYTHON" scripts/run_external_frozen_embeddings.py \
  --config "$GSE_CONFIG" --models "$MODELS" \
  --artifact-root "$BASE_ROOT/gse270828_embeddings" "${AVAIL_FLAGS[@]}"
mark DONE "$BASE_ROOT/gse270828_embeddings/enrolled_models.json"

embedding_args() {
  local manifest="$1"
  "$PYTHON" - "$manifest" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
for name, path in d["embeddings"].items():
    print(f"{name}={path}")
PY
}

mapfile -t JIANG_EMBEDDINGS < <(embedding_args "$BASE_ROOT/jiang24_embeddings/enrolled_models.json")
mapfile -t GSE_EMBEDDINGS < <(embedding_args "$BASE_ROOT/gse270828_embeddings/enrolled_models.json")
JIANG_ARGS=(); for item in "${JIANG_EMBEDDINGS[@]}"; do JIANG_ARGS+=(--embedding "$item"); done
GSE_ARGS=(); for item in "${GSE_EMBEDDINGS[@]}"; do GSE_ARGS+=(--embedding "$item"); done

next_stage "Jiang24 exact VCC2026 six-metric matrix"
"$PYTHON" -m vcc_baselines shadow-benchmark --config "$JIANG_CONFIG" \
  "${JIANG_ARGS[@]}" --methods "$METHODS" --engine cell-eval2 --resume \
  --artifact-root "$BASE_ROOT/jiang24_details" --out "$BASE_ROOT/jiang24_vcc2026.csv"
mark DONE "$BASE_ROOT/jiang24_vcc2026.csv"

next_stage "GSE270828 exact VCC2026 six-metric matrix"
"$PYTHON" -m vcc_baselines shadow-benchmark --config "$GSE_CONFIG" \
  "${GSE_ARGS[@]}" --methods "$METHODS" --engine cell-eval2 --resume \
  --target-gene-map data_public/gse270828/GSE270828_feature_README.csv \
  --artifact-root "$BASE_ROOT/gse270828_details" --out "$BASE_ROOT/gse270828_vcc2026.csv"
mark DONE "$BASE_ROOT/gse270828_vcc2026.csv"

if [[ "$PROTOCOL" == "replogle-only" ]]; then
  next_stage "Jiang24 exact six metrics by Replogle coverage"
  "$PYTHON" scripts/score_external_coverage_strata.py \
    --config "$JIANG_CONFIG" \
    --coverage data_experiments/replogle_only/jiang24/target_coverage.csv \
    --overall "$BASE_ROOT/jiang24_vcc2026.csv" \
    --details "$BASE_ROOT/jiang24_details" \
    --out "$BASE_ROOT/jiang24_coverage_vcc2026.csv"
  mark DONE "$BASE_ROOT/jiang24_coverage_vcc2026.csv"

  next_stage "GSE270828 exact six metrics by Replogle coverage"
  "$PYTHON" scripts/score_external_coverage_strata.py \
    --config "$GSE_CONFIG" \
    --coverage data_experiments/replogle_only/gse270828/target_coverage.csv \
    --overall "$BASE_ROOT/gse270828_vcc2026.csv" \
    --details "$BASE_ROOT/gse270828_details" \
    --target-gene-map data_public/gse270828/GSE270828_feature_README.csv \
    --out "$BASE_ROOT/gse270828_coverage_vcc2026.csv"
  mark DONE "$BASE_ROOT/gse270828_coverage_vcc2026.csv"
fi

next_stage "merge whole-dataset six metrics and changes"
"$PYTHON" scripts/compare_external_frozen_matrix.py \
  --result "Jiang24_IFNG_BxPC3=$BASE_ROOT/jiang24_vcc2026.csv" \
  --result "GSE270828_rep3=$BASE_ROOT/gse270828_vcc2026.csv" \
  --protocol "$RESULT_PROTOCOL" --out-wide "$BASE_ROOT/comparison_wide.csv" \
  --out-long "$BASE_ROOT/metric_changes_long.csv"
mark DONE "$BASE_ROOT/comparison_wide.csv"

if [[ "$PROTOCOL" == "replogle-only" ]]; then
  next_stage "merge coverage-stratified six metrics and changes"
  "$PYTHON" scripts/compare_external_coverage_matrix.py \
    --result "Jiang24_IFNG_BxPC3=$BASE_ROOT/jiang24_coverage_vcc2026.csv" \
    --result "GSE270828_rep3=$BASE_ROOT/gse270828_coverage_vcc2026.csv" \
    --protocol "$RESULT_PROTOCOL" \
    --out-wide "$BASE_ROOT/coverage_comparison_wide.csv" \
    --out-long "$BASE_ROOT/coverage_metric_changes_long.csv"
  mark DONE "$BASE_ROOT/coverage_comparison_wide.csv"
fi

echo "[external-pipeline] COMPLETE root=$BASE_ROOT"
echo "[external-pipeline] progress=$STATUS_FILE log=$LOG_FILE"
