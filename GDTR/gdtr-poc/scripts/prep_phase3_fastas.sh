#!/bin/bash
# Stage C.1: download 9 chromosome FASTAs needed for Phase 3 main.
set -euo pipefail
REFDIR=/path/to/gDTR/data/reference
mkdir -p "$REFDIR"
cd "$REFDIR"
CHROMS=(chr2 chr3 chr5 chr7 chr10 chr11 chr12 chr13 chr16)
EXPECTED_LEN_chr2=242193529
EXPECTED_LEN_chr3=198295559
EXPECTED_LEN_chr5=181538259
EXPECTED_LEN_chr7=159345973
EXPECTED_LEN_chr10=133797422
EXPECTED_LEN_chr11=135086622
EXPECTED_LEN_chr12=133275309
EXPECTED_LEN_chr13=114364328
EXPECTED_LEN_chr16=90338345

for c in "${CHROMS[@]}"; do
  if [ -f "$REFDIR/$c.fa.fai" ]; then
    echo "[skip] $c.fa.fai already present"
    continue
  fi
  if [ ! -f "$REFDIR/$c.fa.gz" ]; then
    echo "[get] downloading $c"
    wget -q --show-progress "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz" -O "$REFDIR/$c.fa.gz"
  fi
  if [ ! -f "$REFDIR/$c.fa" ]; then
    echo "[gunzip] $c"
    gunzip -k -f "$REFDIR/$c.fa.gz"
  fi
  echo "[index] $c"
  python -c "import pysam; pysam.faidx('$REFDIR/$c.fa')"
done

# MD5s
echo "" >> /path/to/gDTR/data/DATA_VERSIONS.txt
echo "# Phase 3 main FASTAs (added $(date '+%Y-%m-%d %H:%M:%S'))" >> /path/to/gDTR/data/DATA_VERSIONS.txt
for c in "${CHROMS[@]}"; do
  md5=$(md5sum "$REFDIR/$c.fa" | awk '{print $1}')
  echo "$c.fa  $md5" >> /path/to/gDTR/data/DATA_VERSIONS.txt
done

echo "[verify] checking lengths and indexing"
for c in "${CHROMS[@]}"; do
  if [ ! -f "$REFDIR/$c.fa.fai" ]; then
    echo "[ERR] missing $c.fa.fai"
    exit 1
  fi
  len=$(awk -v c="$c" '$1==c{print $2}' "$REFDIR/$c.fa.fai")
  expvar="EXPECTED_LEN_$c"
  exp=${!expvar}
  if [ "$len" != "$exp" ]; then
    echo "[ERR] $c length $len != expected $exp"
    exit 1
  fi
  echo "  [OK] $c length $len"
done
echo 'Phase 3 FASTAs complete' > /path/to/gDTR/data/reference/_phase3_fastas_done
