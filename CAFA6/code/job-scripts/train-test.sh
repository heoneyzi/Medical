#!/bin/bash
################################## Slurm options ##################################
#SBATCH --job-name=cafa6-prott5-train-test
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/cafa6/logs/total/%x-%j.out
#SBATCH --error=/path/to/cafa6/logs/total/%x-%j.err
###################################################################################
set -euo pipefail

# Slurm runs scripts from a spool dir; use the real project path.
REPO_ROOT="/path/to/cafa6"
cd "$REPO_ROOT"

echo "[train-test] starting in $REPO_ROOT"

bash job-scripts/train.sh
bash job-scripts/test.sh
