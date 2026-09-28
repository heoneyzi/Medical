#!/usr/bin/env bash
# Master DAG runner for Paper 1 revision_v8 (Plan §5.3).
# Halts on first FAIL. Writes ~/gDTR/results/revision_v8_status.jsonl.
#
# Environment overrides:
#   GDTR_ROOT  default ~/gDTR
#   SMOKE      "1" runs each step in --smoke mode
#   FORCE      "1" passes --force to each step
#   RUN_CHR1   "1" enables the optional chr1 forward (GPU priority 2)

set -uo pipefail

ROOT="${GDTR_ROOT:-$HOME/gDTR}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
EXTRA_FLAGS=""
[ "${SMOKE:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --smoke"
[ "${FORCE:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --force"

mkdir -p "$ROOT/results"
LOGFILE="$ROOT/results/revision_v8_status.jsonl"

# --- helper: run a step then call verify_step.py to confirm ---
run_step () {
    local step="$1"; shift
    echo "=== STEP $step START $(date -Is) ==="
    if "$@"; then
        if python3 "$SCRIPTS/verify_step.py" "$step"; then
            echo "=== STEP $step PASS ==="
            printf '{"step":"%s","status":"PASS","ts":"%s"}\n' \
                "$step" "$(date -Is)" >> "$LOGFILE"
            return 0
        fi
    fi
    echo "=== STEP $step FAIL ==="
    printf '{"step":"%s","status":"FAIL","ts":"%s"}\n' \
        "$step" "$(date -Is)" >> "$LOGFILE"
    exit 1
}

# Preflight — invariants must reproduce v7 values before we touch anything.
echo "=== PRE-FLIGHT verify_invariants --baseline ==="
python3 "$SCRIPTS/verify_invariants.py" --baseline ${SMOKE:+--smoke} || {
    echo "BASELINE INVARIANTS FAILED — refusing to proceed"
    exit 2
}

# ---------------------------------------------------------------
# Phase A — CPU work that has no dependency on other steps.
# (Runs in parallel; bash `wait` waits for the group.)
# ---------------------------------------------------------------
run_step p1a       python3 "$SCRIPTS/p1a_calib_val_split.py" $EXTRA_FLAGS &
run_step p1b_label python3 "$SCRIPTS/p1b_canonical_splice_label.py" $EXTRA_FLAGS &
run_step p3b2      python3 "$SCRIPTS/p3b2_repeat_breakdown.py" $EXTRA_FLAGS &
wait

# ---------------------------------------------------------------
# Phase A2 — second wave of CPU jobs that depend on Phase A.
# ---------------------------------------------------------------
run_step p1b_compare      python3 "$SCRIPTS/p1b_canonical_splice_compare.py" $EXTRA_FLAGS &
run_step p2_snv_join      python3 "$SCRIPTS/p2_snv_class_join.py" $EXTRA_FLAGS &
run_step p3b1             python3 "$SCRIPTS/p3b1_functional_positive_control.py" $EXTRA_FLAGS &
run_step p3b3_chr17       python3 "$SCRIPTS/p3b3_q2_chr17.py" $EXTRA_FLAGS &
wait

# ---------------------------------------------------------------
# Phase A3 — depends on p2_snv_join.
# ---------------------------------------------------------------
run_step p2_snv_stats     python3 "$SCRIPTS/p2_snv_per_class_stats.py" $EXTRA_FLAGS
run_step p2_indel_select  python3 "$SCRIPTS/p2_indel_select.py" $EXTRA_FLAGS

# ---------------------------------------------------------------
# Phase B — GPU priority 1 (frameshift forward).
# ---------------------------------------------------------------
run_step p2_indel_forward python3 "$SCRIPTS/p2_indel_forward.py" $EXTRA_FLAGS
run_step p2_indel_stats   python3 "$SCRIPTS/p2_indel_per_class_stats.py" $EXTRA_FLAGS
run_step p2_case          python3 "$SCRIPTS/p2_case_v8.py" $EXTRA_FLAGS

# ---------------------------------------------------------------
# Phase C — OPTIONAL: GPU priority 2 (chr1 sub-sample forward).
# ---------------------------------------------------------------
if [ "${RUN_CHR1:-0}" = "1" ]; then
    run_step prep_chr1            python3 "$SCRIPTS/prep_chr1_windows.py" $EXTRA_FLAGS
    run_step p3b3_chr1_forward    python3 "$SCRIPTS/p3b3_chr1_forward.py" $EXTRA_FLAGS
    run_step p3b3_chr1_enrichment python3 "$SCRIPTS/p3b3_chr1_enrichment.py" $EXTRA_FLAGS
fi

# ---------------------------------------------------------------
# Phase D — manuscript build + post-build invariants.
# ---------------------------------------------------------------
if [ -x "$SCRIPTS/build_paper1_v8.py" ] || [ -f "$SCRIPTS/build_paper1_v8.py" ]; then
    run_step build_v8 python3 "$SCRIPTS/build_paper1_v8.py" $EXTRA_FLAGS || true
fi

echo "=== POST-BUILD verify_invariants --post-build ==="
python3 "$SCRIPTS/verify_invariants.py" --post-build ${SMOKE:+--smoke}

echo "ALL STEPS COMPLETED"
