#!/usr/bin/env bash
# Idempotent env setup for gDTR-PoC Phase 0
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "[env_setup] Python: $(python3 --version)"
python3 -m pip install --no-warn-script-location -r requirements.txt
python3 -c "import torch, transformers, Bio, pyfaidx, pyBigWig, seaborn, huggingface_hub, gffutils, statannotations, tqdm, matplotlib, numpy, pandas, scipy, sklearn; print('[env_setup] imports OK; torch', torch.__version__, 'transformers', transformers.__version__)"
mkdir -p data/reference data/annotation data/variants data/baselines
mkdir -p src tests scripts results/figures results/tables results/runs
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "[env_setup] HF_HOME=$HF_HOME"
echo "[env_setup] DONE"
