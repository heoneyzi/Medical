#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SPLIT="${1:-data_shadow/jiang24_ifng_bxpc3_loco}"
STATE_MODEL="${STATE_MODEL:-models/SE-100M}"
STACK_MODEL="${STACK_MODEL:-models/Stack-Large}"
DEVICE="${DEVICE:-cuda}"
RUN_LOCAL_BENCHMARK="${RUN_LOCAL_BENCHMARK:-1}"
STATE_BIN="${STATE_BIN:-$(command -v state || true)}"
[[ -n "$STATE_BIN" ]] || { echo "ERROR: state CLI not found"; exit 2; }
STATE_PYTHON="${STATE_PYTHON:-$(head -n 1 "$STATE_BIN" | sed 's/^#!//')}"
STACK_BIN="${STACK_BIN:-$(command -v stack-embedding || true)}"
[[ -n "$STACK_BIN" ]] || { echo "ERROR: stack-embedding CLI not found"; exit 2; }
[[ -s "$STATE_MODEL/model.safetensors" ]] || { echo "ERROR: missing STATE checkpoint"; exit 2; }
[[ -s "$STACK_MODEL/bc_large.ckpt" ]] || { echo "ERROR: missing STACK checkpoint"; exit 2; }
HOLDOUT_DIRS=("$SPLIT"/validation/*)
[[ ${#HOLDOUT_DIRS[@]} -eq 1 && -d "${HOLDOUT_DIRS[0]}" ]] || { echo "expected one holdout under $SPLIT/validation"; exit 2; }
HOLDOUT="$(basename "${HOLDOUT_DIRS[0]}")"
mkdir -p "$SPLIT/frozen/state" "$SPLIT/frozen/stack" "$SPLIT/logs"

inputs=("$SPLIT"/reference_controls/*.h5ad "${HOLDOUT_DIRS[0]}/controls.h5ad")
total=$((2 * ${#inputs[@]} + 2 + (RUN_LOCAL_BENCHMARK == 1 ? 1 : 0)))
step=0
progress() {
  step=$((step + 1))
  local label="$1" state="$2"
  local pct=$((100 * step / total))
  printf '[frozen %02d/%02d %3d%%] %-5s %s\n' "$step" "$total" "$pct" "$state" "$label"
}

echo "[frozen] split=$SPLIT holdout=$HOLDOUT device=$DEVICE contexts=${#inputs[@]}"
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

state_inputs=()
stack_inputs=()
for input in "${inputs[@]}"; do
  if [[ "$input" == */controls.h5ad ]]; then ctx="$HOLDOUT"; else ctx="$(basename "$input" .h5ad)"; fi
  state_out="$SPLIT/frozen/state/$ctx.h5ad"
  stack_out="$SPLIT/frozen/stack/$ctx.h5ad"
  if [[ ! -s "$state_out" ]]; then
    progress "STATE $ctx" START
    started=$SECONDS
    "$STATE_PYTHON" scripts/state_safetensors_embed.py \
      --model-folder "$STATE_MODEL" --input "$input" --output "$state_out" \
      --embed-key X_state --batch-size "${STATE_BATCH_SIZE:-4}" --device "$DEVICE" \
      2>&1 | tee "$SPLIT/logs/state_${ctx}.log"
    echo "[frozen] STATE $ctx elapsed=$((SECONDS - started))s output=$state_out"
  else
    progress "STATE $ctx" SKIP
  fi
  if [[ ! -s "$stack_out" ]]; then
    progress "STACK $ctx" START
    started=$SECONDS
    "$STACK_BIN" --checkpoint "$STACK_MODEL/bc_large.ckpt" --adata "$input" \
      --genelist "$STACK_MODEL/basecount_1000per_15000max.pkl" --output "$stack_out" \
      --batch-size "${STACK_BATCH_SIZE:-8}" --num-workers 0 --random-seed 2026 \
      --device "$DEVICE" --obs-source "$input" 2>&1 | tee "$SPLIT/logs/stack_${ctx}.log"
    echo "[frozen] STACK $ctx elapsed=$((SECONDS - started))s output=$stack_out"
  else
    progress "STACK $ctx" SKIP
  fi
  state_inputs+=(--input "$ctx=$state_out")
  stack_inputs+=(--input "$ctx=$stack_out")
done

progress "pool STATE embeddings" START
"$PYTHON" -m vcc_baselines pool-embeddings "${state_inputs[@]}" \
  --obsm-key X_state --out "$SPLIT/frozen/state_contexts.npz"
progress "pool STACK embeddings" START
"$PYTHON" -m vcc_baselines pool-embeddings "${stack_inputs[@]}" \
  --obsm-key X --out "$SPLIT/frozen/stack_contexts.npz"
if [[ "$RUN_LOCAL_BENCHMARK" == 1 ]]; then
  progress "local frozen benchmark" START
  "$PYTHON" -m vcc_baselines shadow-benchmark --config "$SPLIT/shadow.yaml" \
    --embedding "state=$SPLIT/frozen/state_contexts.npz" \
    --embedding "stack=$SPLIT/frozen/stack_contexts.npz" \
    --methods gwps_nearest,gwps_weighted --engine local --resume \
    --out "$SPLIT/benchmark_frozen_local.csv"
fi
echo "[frozen] COMPLETE split=$SPLIT"
