#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
RDS="${GSE_RDS:-data_public/gse270828/GSE270828_Merged_CountMatrix.rds.gz}"
FEATURES="${GSE_FEATURES:-data_public/gse270828/GSE270828_feature_README.csv}"
GSE_H5AD="${GSE_H5AD:-data_public/gse270828/GSE270828_counts.h5ad}"
JIANG_H5AD="${JIANG_H5AD:-data_public/perturbench/jiang24_IFNG_counts.h5ad}"
OUT_ROOT="${OUT_ROOT:-data_zero_shot}"

if [[ ! -s "$GSE_H5AD" ]]; then
  EXPORT_DIR="$(mktemp -d /tmp/gse270828_export.XXXXXX)"
  Rscript scripts/export_gse270828_rds.R "$RDS" "$EXPORT_DIR"
  "$PYTHON" scripts/build_gse270828_h5ad.py \
    --export-dir "$EXPORT_DIR" --source-rds "$RDS" --feature-readme "$FEATURES" \
    --out "$GSE_H5AD" --compression lzf
  echo "[zero-shot] temporary binary export can now be removed: $EXPORT_DIR"
fi

JIANG_OUT="$OUT_ROOT/jiang24_ifng_bxpc3_strict"
GSE_OUT="$OUT_ROOT/gse270828_nsc_strict"

if [[ ! -s "$JIANG_OUT/blind/predict.yaml" ]]; then
  "$PYTHON" -m vcc_baselines prepare-zero-shot \
    --input "$JIANG_H5AD" --context-name bxpc3 --where cell_type=bxpc3 --where treatment=IFNG \
    --out "$JIANG_OUT" --pert-col condition --control-label control --matrix X \
    --max-genes 4000 --max-controls 500 --max-cells-per-pert 100 \
    --cells-per-pert 50 --seed 2026
fi

if [[ ! -s "$GSE_OUT/blind/predict.yaml" ]]; then
  "$PYTHON" -m vcc_baselines prepare-zero-shot \
    --input "$GSE_H5AD" --context-name H23555_NSC --out "$GSE_OUT" \
    --pert-col har --control-label Non-Targeting --matrix X --min-genes 0 --min-counts 0 \
    --max-genes 4000 --max-controls 500 --max-cells-per-pert 100 \
    --cells-per-pert 50 --seed 2026
fi

for spec in \
  "$JIANG_OUT:bxpc3:" \
  "$GSE_OUT:H23555_NSC:$FEATURES"
do
  IFS=: read -r bundle context target_map <<< "$spec"
  "$PYTHON" -m vcc_baselines predict --config "$bundle/blind/predict.yaml" \
    --method no_effect --include-controls
  map_args=()
  [[ -n "$target_map" ]] && map_args=(--target-gene-map "$target_map")
  "$PYTHON" -m vcc_baselines score-zero-shot --config "$bundle/blind/predict.yaml" \
    --pred "$bundle/blind/outputs/no_effect/prediction.h5ad" \
    --truth "$bundle/sealed/$context/truth.h5ad" \
    --engine cell-eval2 "${map_args[@]}" --out "$bundle/results/no_effect_vcc2026"
done

"$PYTHON" scripts/audit_zero_shot_models.py --out "$OUT_ROOT/frozen_model_eligibility.csv"
"$PYTHON" scripts/compare_zero_shot.py \
  --result "Jiang24_IFNG_BxPC3=$JIANG_OUT/results/no_effect_vcc2026/zero_shot_results.csv" \
  --result "GSE270828_H23555_NSC=$GSE_OUT/results/no_effect_vcc2026/zero_shot_results.csv" \
  --out "$OUT_ROOT/two_dataset_strict_comparison.csv"
