#!/usr/bin/env bash
set -euo pipefail

# Idempotent, dev-only recovery path for the hard-v2 experiment. This script
# deliberately never passes --include-test or --split test.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GEOFLOW_PYTHON:-$GEOFLOW_RUNTIME/venv-gpu/bin/python}"
gpu_index="${GEOFLOW_GPU_INDEX:-0}"
hf_cache="${GEOFLOW_HF_CACHE:-$HF_CACHE_DIR}"
run_root="${GEOFLOW_HARD_RUN_ROOT:-$GEOFLOW_RUNS/search_hard_v2}"
value_progress_root="${GEOFLOW_VALUE_PROGRESS_BACKUP_DIR:-$GEOFLOW_RESULTS/search_hard_v2/development/value_training_progress}"
hash_config="$project_root/configs/search_hard_v2_hash.yaml"
hash_parameter_config="$project_root/configs/search_hard_v2_hash_parameter_matched.yaml"
frozen_config="$project_root/configs/search_hard_v2_frozen.yaml"
development="$run_root/development"
lock_path="$development/.hard_v2_development.lock"

mkdir -p "$development"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "Another hard-v2 development orchestrator holds $lock_path" >&2
  exit 75
fi

export PYTHONPATH="$project_root/src"
export HF_HOME="$hf_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export GEOFLOW_VALUE_PROGRESS_BACKUP_DIR="$value_progress_root"
export GEOFLOW_STATE_FLOW_REPORT_TRAIN="${GEOFLOW_STATE_FLOW_REPORT_TRAIN:-0}"

cd "$project_root"

"$python_bin" -m geoflowagent.cli prepare-search --config "$hash_config"
"$python_bin" -m geoflowagent.cli audit-search-data --config "$hash_config"
"$python_bin" -m geoflowagent.cli embed --config "$hash_config"
"$python_bin" -m geoflowagent.cli evaluate-embeddings --config "$hash_config"
"$python_bin" -m geoflowagent.cli embed --config "$frozen_config"
"$python_bin" -m geoflowagent.cli evaluate-embeddings --config "$frozen_config"

"$python_bin" -m geoflowagent.cli train-state-flow --config "$frozen_config"

"$python_bin" -m geoflowagent.cli compare-value-geometries --config "$frozen_config"
"$python_bin" -m geoflowagent.cli compare-value-geometries \
  --config "$hash_parameter_config"
selected_energy="$($python_bin -c \
  "import json; print(json.load(open('$development/checkpoints/geometry_frozen_qwen15_small/comparison.json'))['selected_energy'])")"
case "$selected_energy" in
  cosine|euclidean|diagonal_mahalanobis|lowrank_mahalanobis|asymmetric_bilinear|order_violation|directed_quasimetric|pair_mlp|poincare) ;;
  *)
    echo "Unexpected selected energy: $selected_energy" >&2
    exit 64
    ;;
esac
selected_checkpoint="$development/checkpoints/geometry_frozen_qwen15_small/$selected_energy/seed-17/value_geometry.pt"
"$python_bin" -m geoflowagent.cli train-dagger \
  --config "$frozen_config" \
  --initial-checkpoint "$selected_checkpoint"
"$python_bin" -m geoflowagent.cli evaluate-search-agent \
  --config "$frozen_config" \
  --split dev \
  --checkpoint "$development/checkpoints/dagger_frozen_qwen15_small/value_geometry.pt" \
  --output-dir "$development/reports/search_agent_frozen_qwen15_small/dev_selected"

"$python_bin" -m geoflowagent.cli compare-value-inputs --config "$frozen_config"
"$python_bin" -m geoflowagent.cli compare-value-capacities --config "$frozen_config"
"$python_bin" -m geoflowagent.cli compare-value-geometries \
  --config "$hash_config" \
  --energies euclidean \
  --seeds 17 29 43

echo "Hard-v2 development experiments complete; the test split remains sealed."
