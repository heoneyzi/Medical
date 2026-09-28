#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SOURCE_H5AD="${1:-/tmp/jiang24_processed.h5ad}"
COMPACT_H5AD="${2:-data_public/perturbench/jiang24_IFNG_counts.h5ad}"
OUT_DIR="${3:-data_shadow/jiang24_ifng_bxpc3_loco}"

if [[ ! -s "$COMPACT_H5AD" ]]; then
  "$PYTHON" -m vcc_baselines compact-shadow \
    --input "$SOURCE_H5AD" --where treatment=IFNG --matrix counts --out "$COMPACT_H5AD"
fi

"$PYTHON" -m vcc_baselines prepare-shadow \
  --input "$COMPACT_H5AD" --context-col cell_type --holdout-context bxpc3 \
  --out "$OUT_DIR" --pert-col condition --control-label control \
  --matrix X --min-cells 30 --max-targets 100 --max-genes 4000 \
  --max-controls 500 --max-cells-per-pert 100 --cells-per-pert 50 --seed 2026

"$PYTHON" -m vcc_baselines shadow-benchmark \
  --config "$OUT_DIR/shadow.yaml" --engine local --out "$OUT_DIR/benchmark_local.csv"
