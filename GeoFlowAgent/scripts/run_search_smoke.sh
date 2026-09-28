#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

uv sync --extra dev
uv run geoflow run-search \
  --config configs/search_smoke.yaml \
  --evaluation-split dev

# Train and independently reload/evaluate the genuine search-distilled
# multi-path State Flow.  Its weak learned-STOP result remains visible in the
# report; the verifier-gated score is a separately labelled ablation.
uv run geoflow train-state-flow --config configs/search_smoke.yaml
uv run geoflow evaluate-state-flow \
  --config configs/search_smoke.yaml \
  --split dev

# The full nine-geometry/three-seed comparison is intentionally opt-in because
# it trains 27 models.  It still uses dev only, so the test split stays sealed.
if [[ "${RUN_GEOMETRY_SWEEP:-0}" == "1" ]]; then
  uv run geoflow compare-value-geometries --config configs/search_smoke.yaml
fi

uv run pytest
