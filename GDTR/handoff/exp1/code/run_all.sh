#!/usr/bin/env bash
# Experiment 1, 7B only.  Stops at the first failing gate on purpose.
set -euo pipefail

MODEL=${MODEL:-7b}
CHROM=${CHROM:-chr22}
DEV=${DEV:-cuda:0}

echo "== 0. self-test + threshold audit (no GPU) =="
python -m exp1 selftest
python -m exp1 audit-thresholds

echo "== 0b. motif models (derived from held-out chromosomes) =="
python -m exp1 step0-motifs acquire --chroms "${MOTIF_CHROMS:-chr1,chr2,chr3}"
python -m exp1 step0-motifs verify

echo "== 1. operator map + T2 =="
python -m exp1 step1-blockmap

echo "== 4. panel (400 windows) =="
python -m exp1 step4-panel --chrom "$CHROM" --n-windows 400

echo "== 2. onset / rotation =="
python -m exp1 step2-onset --model "$MODEL" --chrom "$CHROM" --device "$DEV" --n-windows 10

echo "== 3. reconstruction gate + dtype ledger =="
python -m exp1 step3-fidelity --model "$MODEL" --chrom "$CHROM" --device "$DEV"

echo "== 5a. reference direction u =="
python -m exp1 step5-uref --model "$MODEL" --chrom "$CHROM" --device "$DEV" --n-windows 40

echo "== 6. freeze the pre-registration BEFORE any contrast =="
python -m exp1 step6-prereg

echo "== 5b. extraction (real + input controls) =="
python -m exp1 step5-extract --model "$MODEL" --chrom "$CHROM" --device "$DEV" \
    --variants real,random,polyA,shuffle_di

echo "== 5c. untrained control =="
python -m exp1 step5-extract --model "$MODEL" --chrom "$CHROM" --device "$DEV" \
    --variants real --shuffle-seed 1234 --tag shuffled

echo "== 5d. motif perturbations (paired edits at annotated splice sites) =="
python -m exp1 step5-extract --model "$MODEL" --chrom "$CHROM" --device "$DEV" \
    --variants real --perturb --perturb-families core_mut,flank_shuffle,rescue \
    --perturb-every 4 --tag perturb

echo "== 7. F1 / F2 / T1 =="
python -m exp1 step7-analysis --model "$MODEL" --chrom "$CHROM"

echo "== 7b/7c. three-regime and motif-stratified contrasts =="
python -m exp1 step7b-regimes --model "$MODEL" --chrom "$CHROM"
python -m exp1 step7c-motifs --model "$MODEL" --chrom "$CHROM" --tag perturb

echo "== 7d. settling against two references (h27 before, h_norm after) =="
python -m exp1 step7d-dualref --model "$MODEL" --chrom "$CHROM"

echo "== 7e. GENCODE-annotated sites vs motif-called sites =="
python -m exp1 step7e-sitedefs --model "$MODEL" --chrom "$CHROM"

echo "== 8. stage 2 + bridge =="
python -m exp1 step8-stage2 --model "$MODEL" --chrom "$CHROM"

echo "== 9. linear decodability at every block =="
python -m exp1 step9-profile --model "$MODEL" --chrom "$CHROM" --motif-probes

echo "done. results/ holds every table, figure and sidecar."
