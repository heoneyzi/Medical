#!/bin/bash
################################## Slurm options ##################################
#SBATCH --job-name=cafa6-prott5-predict
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/cafa6/logs/test/%x-%j.out
#SBATCH --error=/path/to/cafa6/logs/test/%x-%j.err
###################################################################################
set -euo pipefail

# Slurm runs scripts from a spool dir; use the real project path.
REPO_ROOT="/path/to/cafa6"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[predict] ERROR: missing python at $PY" >&2
  exit 1
fi

if ! "$PY" -c "import sentencepiece" >/dev/null 2>&1; then
  echo "[predict] ERROR: missing python package: sentencepiece" >&2
  echo "[predict] Install it with: $PY -m pip install sentencepiece" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
TEST_FASTA="${TEST_FASTA:-$DATA_ROOT/Test/testsuperset.fasta}"
OBO_PATH="${OBO_PATH:-$DATA_ROOT/Train/go-basic.obo}"

MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
MODEL_MF="${MODEL_MF:-$MODEL_ROOT/prott5-go-mf}"
MODEL_BP="${MODEL_BP:-$MODEL_ROOT/prott5-go-bp}"
MODEL_CC="${MODEL_CC:-$MODEL_ROOT/prott5-go-cc}"

OUT_TSV="${OUT_TSV:-$REPO_ROOT/submissions/prott5_pred.tsv}"

echo "[predict] python=$("$PY" -c 'import sys; print(sys.executable)')"
echo "[predict] test_fasta=$TEST_FASTA"
echo "[predict] obo=$OBO_PATH"
echo "[predict] model_mf=$MODEL_MF"
echo "[predict] model_bp=$MODEL_BP"
echo "[predict] model_cc=$MODEL_CC"
echo "[predict] out=$OUT_TSV"

"$PY" -m src.test.prott5_go_predict \
  --model-dir "$MODEL_MF" \
  --model-dir "$MODEL_BP" \
  --model-dir "$MODEL_CC" \
  --test-fasta "$TEST_FASTA" \
  --out "$OUT_TSV" \
  --batch-size "${BATCH_SIZE:-1}" \
  --top-k "${TOP_K:-1500}" \
  --min-score "${MIN_SCORE:-0.001}" \
  --max-terms-per-protein "${MAX_TERMS_PER_PROTEIN:-1500}" \
  --obo "$OBO_PATH" \
  --propagate
