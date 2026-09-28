#!/bin/bash
set -u
cd ~/gDTR && source venv/bin/activate
echo "[chain] $(date) waiting for rollout _done marker"
while [ ! -f ~/gDTR/results/tier1_baselines/_rollout_done ]; do sleep 30; done
echo "[chain] $(date) rollout done; launching IG"
python scripts/47_t12_ig.py 2>&1 | tee logs/ig.run.log
echo "[chain] $(date) IG exit=$?; launching comparison pipeline"
python scripts/48_t12_compare_pipeline.py 2>&1 | tee logs/compare.run.log
echo "[chain] $(date) compare exit=$?; launching cost benchmark"
python scripts/49_t24_cost.py 2>&1 | tee logs/cost.run.log
echo "[chain] $(date) ALL DONE exit=$?"
touch ~/gDTR/results/_chain_t12_done
