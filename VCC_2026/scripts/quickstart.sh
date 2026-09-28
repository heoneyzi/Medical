#!/usr/bin/env bash
# Offline demo: synthetic raw-count data -> ladder -> coverage -> real .vcc (vcc prep).
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the repository-local environment so an older system Python cannot be
# selected accidentally from an IDE terminal.
if [[ -x .venv/bin/python ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        "ERROR: Python >= 3.11 is required. Create .venv and install the project:\n"
        "  uv venv --python 3.11 .venv\n"
        "  uv pip install --python .venv/bin/python -e '.[dev]'"
    )
PY

command -v zstd >/dev/null || { echo "ERROR: install zstd (apt-get install -y zstd)"; exit 1; }
command -v vcc  >/dev/null || { echo "ERROR: install the project in the active environment: pip install -e ."; exit 1; }

echo "==> make-synth (raw counts, contexts A/B/C, pert_counts.csv)"
python -m vcc_baselines make-synth --out data_synth

echo "==> 3-rung ladder (local metrics)"
python -m vcc_baselines ladder --config configs/synthetic.yaml \
  --methods no_effect,global_mean,gwps_direct,gwps_nearest,gwps_weighted,esm2_knn

echo "==> coverage stratification (GWPS reach ceiling)"
python -m vcc_baselines coverage --config configs/synthetic.yaml

echo "==> predict + official vcc prep DRY-RUN (validation only)"
python -m vcc_baselines predict --config configs/synthetic.yaml --method gwps_weighted --dry-run

echo "==> predict + real .vcc  (raw counts, no controls, targets verified)"
python -m vcc_baselines predict --config configs/synthetic.yaml --method gwps_weighted --prep

echo "==> A1 official random dummy (vcc sample)"
python -m vcc_baselines sample --config configs/synthetic.yaml

echo "DONE. Submission: outputs/gwps_weighted/submission.prep.vcc"
