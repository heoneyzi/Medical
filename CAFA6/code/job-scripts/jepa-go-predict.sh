#!/bin/bash
#SBATCH --job-name=jepa-go-pred
#SBATCH --output=/path/to/cafa6/logs/test/jepa-go-pred-%j.out
#SBATCH --error=/path/to/cafa6/logs/test/jepa-go-pred-%j.err
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8

# ==============================================================================
# JEPA-GO Prediction Script
# 
# Generates GO term predictions in exact Kaggle submission format:
# - No header
# - One line per (protein, GO_term, score) triplet  
# - Score with 3 decimal places
# - All predictions with score >= 0.001
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
MODEL_DIR="${MODEL_DIR:-models/jepa-go}"

# Input/Output settings
TEST_FASTA="${TEST_FASTA:-data/Test/testsuperset.fasta}"
OUTPUT="${OUTPUT:-${MODEL_DIR}/submission.tsv}"

# Prediction settings
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-bfloat16}"
MIN_SCORE="${MIN_SCORE:-0.001}"

# ==============================================================================
# Prediction
# ==============================================================================

echo "=============================================="
echo "JEPA-GO Prediction (Kaggle Format)"
echo "=============================================="
echo "Model: $MODEL_DIR"
echo "Test FASTA: $TEST_FASTA"
echo "Output: $OUTPUT"
echo "Min score: $MIN_SCORE"
echo "=============================================="

python -m src.jepa_go.predict_jepa_encoder \
    --model_dir "$MODEL_DIR" \
    --test_fasta "$TEST_FASTA" \
    --output "$OUTPUT" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --min_score "$MIN_SCORE"

echo ""
echo "=============================================="
echo "Prediction complete!"
echo "=============================================="

# Show output file info
echo ""
echo "Output file info:"
ls -lh "$OUTPUT"
echo ""
echo "Line count:"
wc -l "$OUTPUT"
echo ""
echo "First 5 lines:"
head -5 "$OUTPUT"
echo ""
echo "Unique proteins:"
cut -f1 "$OUTPUT" | sort -u | wc -l
echo "=============================================="
