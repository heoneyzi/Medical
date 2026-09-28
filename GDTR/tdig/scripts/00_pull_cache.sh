#!/usr/bin/env bash
# Pull Evo 2 hidden-state cache from the DigitalOcean snapshot.
#
# Prerequisites:
#   - DigitalOcean GPU droplet revived from the gDTR-PoC snapshot (~600GB)
#   - SSH alias '<gpu-host>' configured in ~/.ssh/config (see gDTR-PoC reference)
#   - Local data/cache/ exists (already in .gitignore)
#
# Pulls:
#   - chr22 12,978 windows hidden states (~23 GB compressed)
#   - chr17 27,586 windows hidden states (~30 GB compressed)
#
# Both are too large for GitHub; live in server cache only.

set -euo pipefail

REMOTE_HOST="<gpu-host>"
REMOTE_BASE="~/gDTR/results"
LOCAL_BASE="$(dirname "$0")/../data/cache"

mkdir -p "$LOCAL_BASE"

echo "Pulling chr22 cache..."
rsync -avz --progress \
  "${REMOTE_HOST}:${REMOTE_BASE}/phase1.6/chr22_cache.h5" \
  "${LOCAL_BASE}/chr22_cache.h5"

echo "Pulling chr17 cache..."
rsync -avz --progress \
  "${REMOTE_HOST}:${REMOTE_BASE}/phase2.1/chr17_cache.h5" \
  "${LOCAL_BASE}/chr17_cache.h5"

echo "Pulling Phase 1 calibration outputs..."
rsync -avz --progress \
  "${REMOTE_HOST}:${REMOTE_BASE}/phase1.4/" \
  "${LOCAL_BASE}/phase1.4/"

echo "Pulling ClinVar variant features (for T-B)..."
rsync -avz --progress \
  "${REMOTE_HOST}:${REMOTE_BASE}/phase3_main/variants_features.csv" \
  "${LOCAL_BASE}/variants_features.csv"

echo "Done. Caches at: ${LOCAL_BASE}"
