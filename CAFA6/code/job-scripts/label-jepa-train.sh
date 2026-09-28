#!/bin/bash
#SBATCH --job-name=label-jepa-train
#SBATCH --output=/path/to/cafa6/logs/train/label-jepa-train-%j.out
#SBATCH --error=/path/to/cafa6/logs/train/label-jepa-train-%j.err
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

# ==============================================================================
# Label-Space JEPA Training Script
# 
# Trains Label-Space JEPA model:
# - Learns GO term embeddings jointly with protein representations
# - Uses JEPA-style masked prediction in label space
# - EMA target embeddings for stable training
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
MODEL_FAMILY="${MODEL_FAMILY:-prott5}"
BACKBONE_CKPT="${BACKBONE_CKPT:-Rostlab/prot_t5_xl_uniref50}"
JEPA_PRETRAIN="${JEPA_PRETRAIN:-}"  # Optional: path to pretrained JEPA encoder

# Training settings (optimized for A6000 48GB)
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
DTYPE="${DTYPE:-bfloat16}"

# Label-Space JEPA settings
LABEL_EMBED_DIM="${LABEL_EMBED_DIM:-512}"
CONTEXT_RATIO="${CONTEXT_RATIO:-0.6}"
EMA_DECAY="${EMA_DECAY:-0.999}"
JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-1.0}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-1.0}"

# Label settings
NAMESPACE="${NAMESPACE:-}"  # Empty for all, or MF/BP/CC
MIN_LABEL_COUNT="${MIN_LABEL_COUNT:-5}"
MAX_LABELS="${MAX_LABELS:-8192}"

# Output settings
OUTPUT_DIR="${OUTPUT_DIR:-models/label-jepa-$(date +%Y%m%d)}"
RUN_NAME="${RUN_NAME:-label-jepa}"

# WandB settings
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cafa6}"

# ==============================================================================
# Build arguments
# ==============================================================================

COMMON_ARGS=(
    --model_family "$MODEL_FAMILY"
    --backbone_ckpt "$BACKBONE_CKPT"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
    --max_length "$MAX_LENGTH"
    --dtype "$DTYPE"
    --label_embed_dim "$LABEL_EMBED_DIM"
    --context_ratio "$CONTEXT_RATIO"
    --ema_decay "$EMA_DECAY"
    --jepa_loss_weight "$JEPA_LOSS_WEIGHT"
    --classification_loss_weight "$CLS_LOSS_WEIGHT"
    --min_label_count "$MIN_LABEL_COUNT"
    --max_labels "$MAX_LABELS"
    --output_dir "$OUTPUT_DIR"
    --name "$RUN_NAME"
)

# Add pretrained JEPA if specified
if [ -n "$JEPA_PRETRAIN" ]; then
    COMMON_ARGS+=(--jepa_model_dir "$JEPA_PRETRAIN")
fi

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
echo "Label-Space JEPA Training"
echo "=============================================="
echo "Model family: $MODEL_FAMILY"
echo "Backbone: $BACKBONE_CKPT"
echo "Label embed dim: $LABEL_EMBED_DIM"
echo "Context ratio: $CONTEXT_RATIO"
echo "EMA decay: $EMA_DECAY"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

python -m src.jepa_go.train_label_jepa "${COMMON_ARGS[@]}"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="
