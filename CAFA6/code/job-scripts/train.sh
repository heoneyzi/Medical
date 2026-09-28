#!/bin/bash
################################## Slurm options ##################################
#SBATCH --job-name=cafa6-prott5-train
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/cafa6/logs/train/train-%x-%j.out
#SBATCH --error=/path/to/cafa6/logs/train/train-%x-%j.err
###################################################################################
set -euo pipefail

# Slurm runs scripts from a spool dir; use the real project path.
REPO_ROOT="/path/to/cafa6"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[train] ERROR: missing python at $PY" >&2
  exit 1
fi

if ! "$PY" -c "import sentencepiece" >/dev/null 2>&1; then
  echo "[train] ERROR: missing python package: sentencepiece" >&2
  echo "[train] Install it with: $PY -m pip install sentencepiece" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

TRAIN_FASTA="${TRAIN_FASTA:-$REPO_ROOT/data/Train/train_sequences.fasta}"
TRAIN_TERMS="${TRAIN_TERMS:-$REPO_ROOT/data/Train/train_terms.tsv}"
MODEL_CKPT="${MODEL_CKPT:-Rostlab/prot_t5_xl_uniref50}"

MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
mkdir -p "$MODEL_ROOT"

# Set these env vars to override defaults for quick experiments.
MAX_TRAIN_PROTEINS="${MAX_TRAIN_PROTEINS:-}" # e.g. 2000

COMMON_ARGS=(
  --model-ckpt "$MODEL_CKPT"
  --train_fasta "$TRAIN_FASTA"
  --train_terms_tsv "$TRAIN_TERMS"
  --min_label_count "${MIN_LABEL_COUNT:-5}"
  --max_labels "${MAX_LABELS:-4096}"
  --max_length "${MAX_LENGTH:-1024}"
  --batch-size "${TRAIN_BATCH_SIZE:-1}"
  --epochs "${EPOCHS:-1}"
  --lr "${LR:-2e-4}"
  --grad-accum "${GRAD_ACCUM:-1}"
  --val_fraction "${VAL_FRACTION:-0.01}"
  --do_eval "${DO_EVAL:-true}"
  --freeze_encoder "${FREEZE_ENCODER:-false}"
  --unfreeze_last_n_layers "${UNFREEZE_LAST_N_LAYERS:-2}"
  --logging_steps "${LOGGING_STEPS:-50}"
)

if [[ -n "$MAX_TRAIN_PROTEINS" ]]; then
  COMMON_ARGS+=( --max_train_proteins "$MAX_TRAIN_PROTEINS" )
fi

echo "[train] python=$("$PY" -c 'import sys; print(sys.executable)')"
echo "[train] train_fasta=$TRAIN_FASTA"
echo "[train] train_terms=$TRAIN_TERMS"
echo "[train] model_root=$MODEL_ROOT"

# Train separate models per subontology (task spec: MF/BP/CC).
"$PY" -m src.train.prott5_go_train \
  "${COMMON_ARGS[@]}" \
  --namespace MF \
  --out "$MODEL_ROOT/prott5-go-mf" \
  --run_name "${RUN_NAME_MF:-prott5-go-mf}"

"$PY" -m src.train.prott5_go_train \
  "${COMMON_ARGS[@]}" \
  --namespace BP \
  --out "$MODEL_ROOT/prott5-go-bp" \
  --run_name "${RUN_NAME_BP:-prott5-go-bp}"

"$PY" -m src.train.prott5_go_train \
  "${COMMON_ARGS[@]}" \
  --namespace CC \
  --out "$MODEL_ROOT/prott5-go-cc" \
  --run_name "${RUN_NAME_CC:-prott5-go-cc}"
