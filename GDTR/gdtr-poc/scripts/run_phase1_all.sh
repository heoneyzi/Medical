#!/bin/bash
set -e
source /path/to/gDTR/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/path/to/gDTR:/path/to/gDTR/scripts:/path/to/gDTR-phase0:${PYTHONPATH:-}
cd /path/to/gDTR
mkdir -p results/phase1.1 results/phase1.2 results/phase1.3 results/phase1.4 results/phase1.5 results/phase1.6 results/phase1.7 results/status logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }
verify() { python scripts/verify_phase.py "$1" || { echo "[$(ts)] verify $1 FAILED"; exit 1; }; }

echo "[$(ts)] phase1.1 -> Gate A_evo untuned"
python scripts/10_phase1_1_gate_a_evo.py 2>&1 | tee logs/phase1.1.log
verify 1.1

echo "[$(ts)] phase1.4 -> calibration"
python scripts/14_phase1_4_calibration.py 2>&1 | tee logs/phase1.4.log
verify 1.4

echo "[$(ts)] phase1.2 -> tuned lens training"
python scripts/12_phase1_2_train_tuned_lens.py 2>&1 | tee logs/phase1.2.log
verify 1.2

echo "[$(ts)] phase1.3 -> Gate A_evo tuned"
python scripts/13_phase1_3_gate_a_tuned.py 2>&1 | tee logs/phase1.3.log
verify 1.3

echo "[$(ts)] phase1.5 -> HP sweep"
python scripts/15_phase1_5_hp_sweep.py 2>&1 | tee logs/phase1.5.log
verify 1.5

echo "phase1-fwd done at $(ts)" > results/phase1_fwd.done

echo "[$(ts)] waiting for chr22 forward to finish ..."
while [ ! -f results/phase1.6/_chr22_forward_done ]; do sleep 60; done
echo "[$(ts)] chr22 cache ready, verifying ..."
verify 1.6_chr22

echo "[$(ts)] running Gate B ..."
python scripts/16b_phase1_6_gate_b.py 2>&1 | tee logs/phase1.6_gate_b.log
verify 1.6_gate_b

python scripts/17_phase1_7_writeup.py 2>&1 | tee logs/phase1.7.log
verify 1.7

echo "Phase 1 complete at $(ts)" > results/phase1_all.done
