#!/bin/bash
#SBATCH --job-name=jepa-go-train-pred
#SBATCH --output=/path/to/cafa6/logs/train/jepa-go-%j.out
#SBATCH --error=/path/to/cafa6/logs/train/jepa-go-%j.err
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8

# ==============================================================================
# JEPA-GO Training and Prediction Script
# 
# Uses pretrained JEPA encoder (jepa-prott5-1) for GO term prediction.
# Trains classifier head on top of frozen encoder, then runs prediction.
# ==============================================================================

set -e

# Activate environment
cd /path/to/cafa6
source .venv/bin/activate

# Huggingface cache
export HF_HOME=.cache/huggingface

# Create log directory
mkdir -p logs/train

# ==============================================================================
# Configuration
# ==============================================================================

# Model settings
# MODEL_FAMILY="${MODEL_FAMILY:-prott5}"
# BACKBONE_CKPT="${BACKBONE_CKPT:-Rostlab/prot_t5_xl_uniref50}"
# JEPA_MODEL_DIR="${JEPA_MODEL_DIR:-models/jepa-prott5-1}"
MODEL_FAMILY="${MODEL_FAMILY:-esm}"
BACKBONE_CKPT="${BACKBONE_CKPT:-facebook/esm2_t33_650M_UR50D}"
JEPA_MODEL_DIR="${JEPA_MODEL_DIR:-}"

# Training settings (optimized for A6000 48GB)
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
DTYPE="${DTYPE:-bfloat16}"

# Label settings
NAMESPACE="${NAMESPACE:-}"  # Empty for all, or MF/BP/CC
MIN_LABEL_COUNT="${MIN_LABEL_COUNT:-5}"
MAX_LABELS="${MAX_LABELS:-8192}"

# Output settings
OUTPUT_DIR="${OUTPUT_DIR:-models/jepa-go-$(date +%Y%m%d)}"
RUN_NAME="${RUN_NAME:-jepa-go}"

# WandB settings
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cafa6}"

# ==============================================================================
# Build arguments
# ==============================================================================

COMMON_ARGS=(
    --model_family "$MODEL_FAMILY"
    --backbone_ckpt "$BACKBONE_CKPT"
    --jepa_model_dir "$JEPA_MODEL_DIR"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
    --max_length "$MAX_LENGTH"
    --dtype "$DTYPE"
    --min_label_count "$MIN_LABEL_COUNT"
    --max_labels "$MAX_LABELS"
    --output_dir "$OUTPUT_DIR"
    --name "$RUN_NAME"
)

# Add namespace if specified
if [ -n "$NAMESPACE" ]; then
    COMMON_ARGS+=(--namespace "$NAMESPACE")
fi

# Add wandb flag
if [ "$USE_WANDB" = "true" ]; then
    COMMON_ARGS+=(--use_wandb --wandb_project "$WANDB_PROJECT")
fi

# ==============================================================================
# Training
# ==============================================================================

echo "=============================================="
echo "JEPA-GO Training"
echo "=============================================="
echo "Model family: $MODEL_FAMILY"
echo "Backbone: $BACKBONE_CKPT"
echo "JEPA model: $JEPA_MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

python -m src.jepa_go.train_jepa_encoder "${COMMON_ARGS[@]}"

echo "Training complete!"

# ==============================================================================
# Prediction (Kaggle Format)
# ==============================================================================

echo ""
echo "=============================================="
echo "JEPA-GO Prediction (Kaggle Format)"
echo "=============================================="

SUBMISSION_OUTPUT="${OUTPUT_DIR}/submission.tsv"

python -m src.jepa_go.predict_jepa_encoder \
    --model_dir "$OUTPUT_DIR" \
    --test_fasta data/Test/testsuperset.fasta \
    --output "$SUBMISSION_OUTPUT" \
    --dtype "$DTYPE" \
    --batch_size 16 \
    --min_score 0.001

echo "=============================================="
echo "All done!"
echo "Submission: $SUBMISSION_OUTPUT"
echo ""
echo "Submission file info:"
ls -lh "$SUBMISSION_OUTPUT"
echo "Line count: $(wc -l < "$SUBMISSION_OUTPUT")"
echo "Unique proteins: $(cut -f1 "$SUBMISSION_OUTPUT" | sort -u | wc -l)"
echo "=============================================="
