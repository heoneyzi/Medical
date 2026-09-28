#!/usr/bin/env bash
set -euo pipefail

# Replicate the hard-v2 State Flow claim across training seeds after GPU 0's
# value controls finish.  Final evaluation uses one independent root per task;
# the seed-17 run already contains the more expensive all-state evaluation.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GEOFLOW_PYTHON:-$GEOFLOW_RUNTIME/venv-gpu/bin/python}"
wait_pid="${GEOFLOW_EXISTING_GPU0_QUEUE_PID:-}"
run_root="${GEOFLOW_HARD_RUN_ROOT:-$GEOFLOW_RUNS/search_hard_v2}"
output_root="$run_root/development/checkpoints/state_flow_seed_replicates"
robustness_root="$run_root/development/checkpoints/resume_v2_robustness"
report_root="$run_root/development/reports"
persistent_root="${GEOFLOW_PERSISTENT_RESULT_ROOT:-$GEOFLOW_RESULTS/search_hard_v2/development}"
progress_root="$persistent_root/state_flow_training_progress"
config="$project_root/configs/search_hard_v2_frozen.yaml"

if [[ -n "$wait_pid" ]] && kill -0 "$wait_pid" 2>/dev/null; then
  echo "$(date -u +'%FT%TZ') waiting for GPU 0 value-control queue pid=$wait_pid"
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 20
  done
fi

mkdir -p "$output_root" "$persistent_root/state_flow_seed_replicates" "$progress_root"
exec 9>"$run_root/development/.state_flow_seed_replicates.lock"
if ! flock -n 9; then
  echo "Another State Flow seed-replication controller holds the lock" >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$project_root/src"
export HF_HOME="${GEOFLOW_HF_CACHE:-$HF_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GEOFLOW_STATE_FLOW_REPORT_TRAIN=0
export GEOFLOW_STATE_FLOW_PROGRESS_BACKUP_DIR="$progress_root"

cd "$project_root"
echo "$(date -u +'%FT%TZ') completing half/configured/two-x value-capacity curve"
"$python_bin" -m geoflowagent.cli compare-value-capacities \
  --config "$config" \
  --output-dir "$robustness_root/capacity_curve"

echo "$(date -u +'%FT%TZ') rerunning selected frozen and parameter-matched hash controls with resume-v2 semantics"
"$python_bin" -m geoflowagent.cli compare-value-geometries \
  --config "$config" \
  --output-dir "$robustness_root/frozen_cosine" \
  --energies cosine \
  --seeds 17 29 43
"$python_bin" -m geoflowagent.cli compare-value-geometries \
  --config "$project_root/configs/search_hard_v2_hash_parameter_matched.yaml" \
  --output-dir "$robustness_root/hash_euclidean" \
  --energies euclidean \
  --seeds 17 29 43
"$python_bin" -m geoflowagent.cli compare-value-run-groups \
  --config "$config" \
  --left-dir "$robustness_root/frozen_cosine/cosine" \
  --right-dir "$robustness_root/hash_euclidean/euclidean" \
  --left-label frozen_cosine_resume_v2 \
  --right-label parameter_matched_hash_resume_v2 \
  --seeds 17 29 43 \
  --output "$report_root/frozen_vs_parameter_matched_hash_resume_v2.json"
mkdir -p "$persistent_root/resume_v2_robustness"
cp -a "$report_root/frozen_vs_parameter_matched_hash_resume_v2.json" \
  "$persistent_root/resume_v2_robustness/"

for seed in 29 43; do
  run_dir="$output_root/seed-$seed"
  echo "$(date -u +'%FT%TZ') training State Flow seed=$seed with root-only dev evaluation"
  "$python_bin" -m geoflowagent.cli train-state-flow \
    --config "$config" \
    --seed "$seed" \
    --output-dir "$run_dir" \
    --root-only-evaluation
  persistent_seed="$persistent_root/state_flow_seed_replicates/seed-$seed"
  mkdir -p "$persistent_seed"
  cp -a "$run_dir/state_flow_metrics.json" "$run_dir/search_state_flow.pt" "$persistent_seed/"
  sha256sum "$persistent_seed/state_flow_metrics.json" \
    "$persistent_seed/search_state_flow.pt" > "$persistent_seed/checksums.sha256"
done

echo "$(date -u +'%FT%TZ') State Flow seed replications complete; test remains sealed"
