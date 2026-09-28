#!/bin/bash
# Pack small JSON/CSV/PNG artefacts and emit base64 to stdout for the local agent.
# Usage: bash scripts/sync_to_local.sh phase1.X
set -e
PHASE="$1"
if [ -z "$PHASE" ]; then
    echo "usage: $0 phase1.X" >&2
    exit 1
fi
DIR="/path/to/gDTR/results/${PHASE}"
if [ ! -d "$DIR" ]; then
    echo "no such dir: $DIR" >&2
    exit 1
fi
# Bundle JSON + CSV + PNG only (skip large npz/h5/pt)
cd /path/to/gDTR/results
tar czf - --exclude='*.npz' --exclude='*.h5' --exclude='*.pt' "$PHASE" | base64 -w0
echo
