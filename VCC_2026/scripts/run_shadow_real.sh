#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_DIR="${1:-data_public/scperturb}"
OUT_DIR="${2:-data_shadow/tian2019_neuron_loco}"

"$PYTHON" -m vcc_baselines prepare-shadow \
  --source "iPSC=$DATA_DIR/TianKampmann2019_iPSC.h5ad" \
  --holdout "day7_neuron=$DATA_DIR/TianKampmann2019_day7neuron.h5ad" \
  --out "$OUT_DIR" \
  --pert-col perturbation --control-label control --single-col nperts \
  --matrix X --min-cells 30 --max-targets 100 --max-genes 4000 \
  --max-controls 500 --max-cells-per-pert 100 --cells-per-pert 50 --seed 0

"$PYTHON" -m vcc_baselines shadow-benchmark \
  --config "$OUT_DIR/shadow.yaml" \
  --engine local --out "$OUT_DIR/benchmark_local.csv"

if command -v cell-eval >/dev/null || [[ -x "$(dirname "$PYTHON")/cell-eval" ]]; then
  "$PYTHON" -m vcc_baselines shadow-benchmark \
    --config "$OUT_DIR/shadow.yaml" \
    --engine cell-eval --profile vcc --out "$OUT_DIR/benchmark_cell_eval.csv"
fi
