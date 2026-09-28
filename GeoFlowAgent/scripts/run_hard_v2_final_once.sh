#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${GEOFLOW_PYTHON:-$GEOFLOW_RUNTIME/venv-gpu/bin/python}"
config="$project/configs/search_hard_v2_final_frozen.yaml"
final="${GEOFLOW_HARD_RUN_ROOT:-$GEOFLOW_RUNS/search_hard_v2}/final"
persistent="${GEOFLOW_PERSISTENT_RESULT_ROOT:-$GEOFLOW_RESULTS/search_hard_v2}/final"
log="$persistent/final_once.log"
mkdir -p "$final" "$persistent"
exec 9>"$final/.final_once.lock"
flock -n 9 || { echo "final protocol already active"; exit 0; }

export PYTHONPATH="$project/src"
export HF_HOME="${GEOFLOW_HF_CACHE:-$HF_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  echo "$(date -u +'%FT%TZ') START $1" >>"$log"
  local name="$1"; shift
  "$@" >>"$log" 2>&1
  echo "$(date -u +'%FT%TZ') END $name exit=0" >>"$log"
}

if pgrep -f '[p]ython.*evaluate-search-agent.*search_hard_v2.*--split dev' >/dev/null; then
  echo "Development closed-loop evaluation is still active; final remains sealed." >&2
  exit 3
fi

reason="one-time hard-v2 final evaluation frozen by preregistration v1 and sealed amendment v2"
if [[ ! -f "$final/processed/manifest.json" ]]; then
  run prepare "$python" -m geoflowagent.cli prepare-search --config "$config" \
    --include-test --test-access-reason "$reason"
fi
if [[ ! -f "$final/reports/data_quality.json" ]]; then
  run audit "$python" -m geoflowagent.cli audit-search-data --config "$config" \
    --include-test --test-access-reason "$reason"
fi
if [[ ! -f "$final/cache_frozen_qwen15/manifest.json" ]]; then
  run embed "$python" -m geoflowagent.cli embed --config "$config"
fi
if [[ ! -f "$final/reports/embeddings_frozen_qwen15/embedding_report.json" ]]; then
  run embedding_report "$python" -m geoflowagent.cli evaluate-embeddings \
    --config "$config" --include-test
fi

value_root="$GEOFLOW_RUNS/search_hard_v2/development/checkpoints"
if [[ ! -f "$final/reports/value_test/summary.json" ]]; then
  run value_metrics "$python" "$project/scripts/evaluate_final_value_checkpoints.py" \
    --config "$config" --output-dir "$final/reports/value_test" \
    --test-access-reason "$reason" \
    --checkpoint "$value_root/medcpt_only_controls/capacity_curve/two_x/seed-17/value_geometry.pt" \
    --checkpoint "$value_root/medcpt_only_controls/capacity_curve/two_x/seed-29/value_geometry.pt" \
    --checkpoint "$value_root/medcpt_only_controls/capacity_curve/two_x/seed-43/value_geometry.pt" \
    --checkpoint "$value_root/dagger_frozen_qwen15_small/round_02/value/value_geometry.pt"
fi

if [[ ! -f "$final/reports/search_agent/medcpt_two_x_seed17/test_summary.json" ]] \
  || [[ ! -f "$final/reports/search_agent/original_full_dagger_round2/test_summary.json" ]]; then
  echo "$(date -u +'%FT%TZ') START closed_loop_primary_and_original" >>"$log"
  pids=()
  if [[ ! -f "$final/reports/search_agent/medcpt_two_x_seed17/test_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$python" -m geoflowagent.cli evaluate-search-agent \
      --config "$config" --split test \
      --checkpoint "$value_root/medcpt_only_controls/capacity_curve/two_x/seed-17/value_geometry.pt" \
      --output-dir "$final/reports/search_agent/medcpt_two_x_seed17" \
      --test-access-reason "$reason" >>"$log" 2>&1 &
    pids+=("$!")
  fi
  if [[ ! -f "$final/reports/search_agent/original_full_dagger_round2/test_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES=1 "$python" -m geoflowagent.cli evaluate-search-agent \
      --config "$config" --split test \
      --checkpoint "$value_root/dagger_frozen_qwen15_small/round_02/value/value_geometry.pt" \
      --output-dir "$final/reports/search_agent/original_full_dagger_round2" \
      --test-access-reason "$reason" >>"$log" 2>&1 &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "$(date -u +'%FT%TZ') END closed_loop_primary_and_original exit=0" >>"$log"
fi

if [[ ! -f "$final/reports/state_flow_seed17_test.json" ]]; then
  run state_flow_17 "$python" -m geoflowagent.cli evaluate-state-flow \
    --config "$config" --split test --root-only \
    --checkpoint "$value_root/state_flow_frozen_qwen15_small/search_state_flow.pt" \
    --output "$final/reports/state_flow_seed17_test.json" --test-access-reason "$reason"
fi
if [[ ! -f "$final/reports/state_flow_seed29_test.json" ]]; then
  run state_flow_29 "$python" -m geoflowagent.cli evaluate-state-flow \
    --config "$config" --split test --root-only \
    --checkpoint "$value_root/state_flow_seed_replicates/seed-29/search_state_flow.pt" \
    --output "$final/reports/state_flow_seed29_test.json" --test-access-reason "$reason"
fi
if [[ ! -f "$final/reports/state_flow_seed43_test.json" ]]; then
  run state_flow_43 "$python" -m geoflowagent.cli evaluate-state-flow \
    --config "$config" --split test --root-only \
    --checkpoint "$value_root/state_flow_seed_replicates/seed-43/search_state_flow.pt" \
    --output "$final/reports/state_flow_seed43_test.json" --test-access-reason "$reason"
fi

cp -a "$final/reports/." "$persistent/reports/"
find "$persistent" -type f ! -name checksums.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"$persistent/checksums.sha256"
echo "$(date -u +'%FT%TZ') FINAL PROTOCOL COMPLETE" >>"$log"
touch "$persistent/FINAL_COMPLETE"
