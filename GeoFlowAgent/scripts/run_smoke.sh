#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
uv sync --extra dev
uv run geoflow run-all --config configs/smoke.yaml --evaluation-split test
uv run pytest
