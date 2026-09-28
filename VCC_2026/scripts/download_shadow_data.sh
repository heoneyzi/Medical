#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data_public}"
MODE="${2:-tian}"
mkdir -p "$OUT_DIR/scperturb" "$OUT_DIR/perturbench" "$OUT_DIR/gse270828"

download() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "[shadow-data] resume/verify existing $out"
  fi
  curl -L --fail --retry 8 --retry-delay 10 -C - -o "$out" "$url"
}

if [[ "$MODE" == "tian" || "$MODE" == "all" ]]; then
  download "https://zenodo.org/records/13350497/files/TianKampmann2019_iPSC.h5ad?download=1" \
    "$OUT_DIR/scperturb/TianKampmann2019_iPSC.h5ad"
  download "https://zenodo.org/records/13350497/files/TianKampmann2019_day7neuron.h5ad?download=1" \
    "$OUT_DIR/scperturb/TianKampmann2019_day7neuron.h5ad"
fi

if [[ "$MODE" == "jiang24" || "$MODE" == "all" ]]; then
  download "https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/jiang24_processed.h5ad.gz?download=true" \
    "$OUT_DIR/perturbench/jiang24_processed.h5ad.gz"
  echo "dd890a8019b8a615010963b32e6e28ce42fd0b94a749602bc93efb2fa8af56cc  $OUT_DIR/perturbench/jiang24_processed.h5ad.gz" | sha256sum -c -
  if [[ -n "${JIANG_H5AD_OUT:-}" ]]; then
    mkdir -p "$(dirname "$JIANG_H5AD_OUT")"
    gzip -dc "$OUT_DIR/perturbench/jiang24_processed.h5ad.gz" > "$JIANG_H5AD_OUT"
  else
    echo "[shadow-data] verified compressed Jiang24; set JIANG_H5AD_OUT=/large/tmp/jiang24_processed.h5ad to decompress"
  fi
fi

if [[ "$MODE" == "gse270828" || "$MODE" == "all" ]]; then
  download "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270828/suppl/GSE270828_Merged_CountMatrix.rds.gz" \
    "$OUT_DIR/gse270828/GSE270828_Merged_CountMatrix.rds.gz"
  download "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270828/suppl/GSE270828_feature_README.csv" \
    "$OUT_DIR/gse270828/GSE270828_feature_README.csv"
  echo "6b48616b478b66c550d99d4e2980a46886e2056ca6074f0ec6d52201c3ca89dd  $OUT_DIR/gse270828/GSE270828_Merged_CountMatrix.rds.gz" | sha256sum -c -
  echo "a56978bcfc4f9b5a80a5f1c62ffcd589525a973f695be52c48eac13704889a2b  $OUT_DIR/gse270828/GSE270828_feature_README.csv" | sha256sum -c -
  echo "[shadow-data] GSE270828 RDS is double-gzipped; use export_gse270828_rds.R (do not gzip -dc once and call readRDS directly)."
fi
