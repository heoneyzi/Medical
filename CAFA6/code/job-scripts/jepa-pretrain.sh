#!/bin/bash
################################## Slurm options ##################################
#SBATCH --job-name=cafa6-jepa-pretrain
#SBATCH --partition=TYAN_A6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/cafa6/logs/train/jepa-%x-%j.out
#SBATCH --error=/path/to/cafa6/logs/train/jepa-%x-%j.err
###################################################################################
set -euo pipefail

# Slurm runs scripts from a spool dir; use the real project path.
REPO_ROOT="/path/to/cafa6"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[jepa] ERROR: missing python at $PY" >&2
  exit 1
fi

# ProtT5 tokenizer requires sentencepiece; fail early with a clear message.
if [[ "${MODEL_FAMILY:-prott5}" == "prott5" ]]; then
  if ! "$PY" -c "import sentencepiece" >/dev/null 2>&1; then
    echo "[jepa] ERROR: missing python package: sentencepiece" >&2
    echo "[jepa] Install it with: $PY -m pip install sentencepiece" >&2
    exit 1
  fi
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Defaults target ProtT5; override for ESM with:
#   MODEL_FAMILY=esm CKPT_NAME=facebook/esm2_t33_650M_UR50D bash job-scripts/jepa-pretrain.sh
MODEL_FAMILY="${MODEL_FAMILY:-prott5}"
CKPT_NAME="${CKPT_NAME:-Rostlab/prot_t5_xl_uniref50}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/models/jepa-${MODEL_FAMILY}}"

echo "[jepa] python=$("$PY" -c 'import sys; print(sys.executable)')"
echo "[jepa] model_family=$MODEL_FAMILY ckpt=$CKPT_NAME out=$OUT_DIR"

"$PY" -m src.pretrain.jepa_protein_train \
  --model_family "$MODEL_FAMILY" \
  --ckpt_name "$CKPT_NAME" \
  --output_dir "$OUT_DIR" \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --span_min "${SPAN_MIN:-8}" \
  --span_max "${SPAN_MAX:-128}" \
  --num_train_steps "${NUM_TRAIN_STEPS:-2000}" \
  --learning_rate "${LR:-1e-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-32}" \
  --ema_decay "${EMA_DECAY:-0.999}" \
  --log_every "${LOG_EVERY:-50}" \
  --save_every "${SAVE_EVERY:-500}"
