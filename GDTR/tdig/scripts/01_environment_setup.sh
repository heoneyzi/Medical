#!/usr/bin/env bash
# Server one-time bootstrap. Run on <gpu-host> via:
#   ssh <gpu-host> 'bash -s' < scripts/01_environment_setup.sh
#
# Steps:
#   1. Clone TDiG to /path/to/TDiG (if not present)
#   2. Verify Evo 2 weights, venv, deps
#   3. Create /path/to/TDiG/data/cache/
#   4. Write _provenance_baseline.json (model SHA, packages, host)

set -euo pipefail

TDIG_REPO="https://github.com/YAICON-8th-Think-Deep-in-Genome/TDiG.git"
TDIG_DIR="/path/to/TDiG"
GDTR_VENV="/path/to/gDTR/venv"

# 1. Clone or update
if [ ! -d "${TDIG_DIR}" ]; then
    git clone "${TDIG_REPO}" "${TDIG_DIR}"
else
    cd "${TDIG_DIR}" && git pull
fi

# 2. Verify environment
source "${GDTR_VENV}/bin/activate"
python -c "import evo2; print(f'evo2 version: {evo2.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
test -d ~/.cache/huggingface/hub/models--arcinstitute--evo2_7b_base || { echo "evo2 weights missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 3. Cache directories
mkdir -p "${TDIG_DIR}/data/cache/"{population_stats,chr22,chr17,crossarch,variants,_logs}

# 4. Provenance baseline
cd "${TDIG_DIR}"
python - <<'PYEOF'
import json
import platform
import subprocess
import sys
from datetime import datetime

baseline = {
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "host": platform.node(),
    "tdig_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "model": {
        "name": "evo2_7b_base",
        "hf_revision": "bda0089f92582d5baabf0f22d9fc85f3588f6b58",
        "weights_md5": "359ef88ccac2a62644035578de8a7db4",
    },
    "python_version": sys.version,
}
with open("data/cache/_provenance_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)
print(json.dumps(baseline, indent=2))
PYEOF

echo "Setup complete. Cache at ${TDIG_DIR}/data/cache/"
