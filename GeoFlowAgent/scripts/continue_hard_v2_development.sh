#!/usr/bin/env bash
set -euo pipefail

# Continue the two manually started hard-v2 GPU jobs without duplicating them,
# then run the remaining development-only controls. Test remains sealed.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GEOFLOW_PYTHON:-$GEOFLOW_RUNTIME/venv-gpu/bin/python}"
run_root="${GEOFLOW_HARD_RUN_ROOT:-$GEOFLOW_RUNS/search_hard_v2}"
development="$run_root/development"
frozen_config="$project_root/configs/search_hard_v2_frozen.yaml"
hash_config="$project_root/configs/search_hard_v2_hash.yaml"
hash_parameter_config="$project_root/configs/search_hard_v2_hash_parameter_matched.yaml"
value_progress_root="${GEOFLOW_VALUE_PROGRESS_BACKUP_DIR:-$GEOFLOW_RESULTS/search_hard_v2/development/value_training_progress}"
state_flow_pid="${GEOFLOW_EXISTING_STATE_FLOW_PID:-84588}"
geometry_pid="${GEOFLOW_EXISTING_GEOMETRY_PID:-80221}"

mkdir -p "$development" "$value_progress_root"
exec 9>"$development/.hard_v2_development.lock"
if ! flock -n 9; then
  echo "Another hard-v2 development controller already holds the lock" >&2
  exit 75
fi

export PYTHONPATH="$project_root/src"
export HF_HOME="${GEOFLOW_HF_CACHE:-$HF_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GEOFLOW_VALUE_PROGRESS_BACKUP_DIR="$value_progress_root"
export GEOFLOW_STATE_FLOW_REPORT_TRAIN=0

wait_for_existing_pid() {
  local pid="$1"
  local label="$2"
  if kill -0 "$pid" 2>/dev/null; then
    echo "$(date -u +'%FT%TZ') waiting for existing $label pid=$pid"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 20
    done
  fi
  echo "$(date -u +'%FT%TZ') existing $label is no longer running"
}

run_gpu0_queue() {
  export CUDA_VISIBLE_DEVICES=0
  wait_for_existing_pid "$state_flow_pid" "State Flow"

  local state_flow_dir="$development/checkpoints/state_flow_frozen_qwen15_small"
  if [[ ! -s "$state_flow_dir/state_flow_metrics.json" || ! -s "$state_flow_dir/search_state_flow.pt" ]]; then
    echo "$(date -u +'%FT%TZ') resuming State Flow finalization"
    "$python_bin" -m geoflowagent.cli train-state-flow --config "$frozen_config"
  fi

  echo "$(date -u +'%FT%TZ') starting parameter-matched hash control"
  "$python_bin" -m geoflowagent.cli compare-value-geometries \
    --config "$hash_parameter_config"

  echo "$(date -u +'%FT%TZ') starting frozen capacity comparison"
  "$python_bin" -m geoflowagent.cli compare-value-capacities \
    --config "$frozen_config"

  if [[ ! -s "$development/checkpoints/geometry_hash_small/comparison.json" ]]; then
    echo "$(date -u +'%FT%TZ') ensuring small hash baseline comparison"
    "$python_bin" -m geoflowagent.cli compare-value-geometries \
      --config "$hash_config" --energies euclidean --seeds 17 29 43
  fi
  echo "$(date -u +'%FT%TZ') GPU 0 development queue complete"
}

run_gpu1_queue() {
  export CUDA_VISIBLE_DEVICES=1
  wait_for_existing_pid "$geometry_pid" "frozen geometry sweep"

  local geometry_dir="$development/checkpoints/geometry_frozen_qwen15_small"
  if [[ ! -s "$geometry_dir/comparison.json" ]]; then
    echo "$(date -u +'%FT%TZ') resuming frozen geometry sweep"
    "$python_bin" -m geoflowagent.cli compare-value-geometries \
      --config "$frozen_config"
  fi

  local selected_energy
  selected_energy="$($python_bin -c \
    "import json; print(json.load(open('$geometry_dir/comparison.json'))['selected_energy'])")"
  case "$selected_energy" in
    cosine|euclidean|diagonal_mahalanobis|lowrank_mahalanobis|asymmetric_bilinear|order_violation|directed_quasimetric|pair_mlp|poincare) ;;
    *)
      echo "Unexpected selected energy: $selected_energy" >&2
      exit 64
      ;;
  esac

  echo "$(date -u +'%FT%TZ') starting Search-DAgger with $selected_energy"
  "$python_bin" -m geoflowagent.cli train-dagger \
    --config "$frozen_config" \
    --initial-checkpoint "$geometry_dir/$selected_energy/seed-17/value_geometry.pt"

  echo "$(date -u +'%FT%TZ') starting selected-policy dev evaluation"
  "$python_bin" -m geoflowagent.cli evaluate-search-agent \
    --config "$frozen_config" \
    --split dev \
    --checkpoint "$development/checkpoints/dagger_frozen_qwen15_small/value_geometry.pt" \
    --output-dir "$development/reports/search_agent_frozen_qwen15_small/dev_selected"

  echo "$(date -u +'%FT%TZ') starting frozen input ablations"
  "$python_bin" -m geoflowagent.cli compare-value-inputs \
    --config "$frozen_config"
  echo "$(date -u +'%FT%TZ') GPU 1 development queue complete"
}

cd "$project_root"
run_gpu0_queue > $GEOWORK/geoflow_hard_v2_gpu0_completion.log 2>&1 &
gpu0_queue_pid=$!
run_gpu1_queue > $GEOWORK/geoflow_hard_v2_gpu1_completion.log 2>&1 &
gpu1_queue_pid=$!

status=0
wait "$gpu0_queue_pid" || status=$?
wait "$gpu1_queue_pid" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "Hard-v2 development controller failed; inspect the two completion logs" >&2
  exit "$status"
fi

echo "Hard-v2 development controls complete; test remains physically sealed."
