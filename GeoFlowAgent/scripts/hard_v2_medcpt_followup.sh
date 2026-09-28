#!/usr/bin/env bash
set -uo pipefail

# DEV-ONLY follow-up prompted by the completed input attribution: MedCPT-only
# was substantially stronger than the naive full fusion.  Test is never opened.
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${GEOFLOW_PYTHON:-$GEOFLOW_RUNTIME/venv-gpu/bin/python}"
dev="${GEOFLOW_HARD_RUN_ROOT:-$GEOFLOW_RUNS/search_hard_v2}/development"
persistent="${GEOFLOW_PERSISTENT_RESULT_ROOT:-$GEOFLOW_RESULTS/search_hard_v2/development}"
config="$project/configs/search_hard_v2_medcpt_only.yaml"
output="$dev/checkpoints/medcpt_only_controls"
backup="$persistent/medcpt_only_controls"
log="$persistent/autopilot/medcpt_followup.log"
mkdir -p "$output" "$backup" "$(dirname "$log")"

exec 7>"$dev/.hard_v2_medcpt_followup.lock"
flock -n 7 || { echo "MedCPT follow-up already active"; exit 0; }
export PYTHONPATH="$project/src"
export HF_HOME="${GEOFLOW_HF_CACHE:-$HF_CACHE_DIR}" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GEOFLOW_VALUE_PROGRESS_BACKUP_DIR="${GEOFLOW_VALUE_PROGRESS_BACKUP_DIR:-$persistent/value_training_progress}"

say() { echo "$(date -u +'%FT%TZ') $*" | tee -a "$log"; }
run() {
  local phase="$1"; shift
  say "START $phase $*"
  CUDA_VISIBLE_DEVICES=0 "$@" >>"$log" 2>&1
  local code=$?
  say "END $phase exit=$code"
  return "$code"
}
backup_stage() {
  local source="$1" name="$2"
  mkdir -p "$backup/$name"
  cp -a "$source/." "$backup/$name/"
  (
    cd "$backup/$name" || exit 1
    find . -type f ! -name checksums.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum >checksums.sha256
  )
}

cd "$project"
run geometry_front "$python" -m geoflowagent.cli compare-value-geometries \
  --config "$config" --output-dir "$output/geometry" \
  --energies cosine euclidean diagonal_mahalanobis lowrank_mahalanobis \
  --seeds 17 29 43 || exit $?

# A second GPU owns the remaining five non-overlapping energy directories.
# Wait on its lock, then run the full command once to validate all 27 artifacts
# under one config/source contract and write the authoritative comparison.
exec 6>"$dev/.hard_v2_medcpt_tail_worker.lock"
flock 6
run geometry_aggregate "$python" -m geoflowagent.cli compare-value-geometries \
  --config "$config" --output-dir "$output/geometry" || exit $?
backup_stage "$output/geometry" geometry

# The configured-capacity cosine runs are byte-identical in effective training
# configuration to geometry/cosine, so seed artifacts can be reused safely.
for seed in 17 29 43; do
  mkdir -p "$output/capacity_curve/configured/seed-$seed"
  cp -a "$output/geometry/cosine/seed-$seed/value_geometry.pt" \
    "$output/geometry/cosine/seed-$seed/value_metrics.json" \
    "$output/capacity_curve/configured/seed-$seed/"
done
run capacity "$python" -m geoflowagent.cli compare-value-capacities \
  --config "$config" --output-dir "$output/capacity_curve" || exit $?
backup_stage "$output/capacity_curve" capacity_curve

selected_geometry="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_energy"])' "$output/geometry/comparison.json")"
selected_capacity="$($python -c 'import json,sys; rows=json.load(open(sys.argv[1]))["aggregate"]; print(max(rows,key=lambda row: row["mean_dev_joint_accuracy"])["capacity"])' "$output/capacity_curve/comparison.json")"
say "DEV selections geometry=$selected_geometry cosine_capacity=$selected_capacity"

run closed_loop_geometry "$python" -m geoflowagent.cli evaluate-search-agent \
  --config "$project/configs/search_hard_v2_frozen.yaml" --split dev \
  --checkpoint "$output/geometry/$selected_geometry/seed-17/value_geometry.pt" \
  --output-dir "$dev/reports/search_agent_medcpt_only/dev_geometry_selected" || exit $?
run closed_loop_capacity "$python" -m geoflowagent.cli evaluate-search-agent \
  --config "$project/configs/search_hard_v2_frozen.yaml" --split dev \
  --checkpoint "$output/capacity_curve/$selected_capacity/seed-17/value_geometry.pt" \
  --output-dir "$dev/reports/search_agent_medcpt_only/dev_cosine_capacity_selected" || exit $?
backup_stage "$dev/reports/search_agent_medcpt_only" closed_loop
say "MEDCPT FOLLOW-UP COMPLETE; test remains sealed"
