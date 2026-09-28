#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
uv run geoflow compare-distances \
  --config configs/smoke.yaml \
  --distances euclidean cosine diagonal_mahalanobis lowrank_mahalanobis bilinear poincare

