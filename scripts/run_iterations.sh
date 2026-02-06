#!/bin/bash
set -euo pipefail

# ============================================================
# run_iterations.sh — Run multiple evolution iterations
#
# Usage:
#   bash scripts/run_iterations.sh [START] [END] [MODEL] [EVOLVER_MODEL] [ANNOTATE_CLINICAL]
#
# Examples:
#   bash scripts/run_iterations.sh 1 5 Qwen3_30B_A3B
#   bash scripts/run_iterations.sh 3 5 Qwen3_30B_A3B   # resume from v3
#   bash scripts/run_iterations.sh 1 5 Qwen3_30B_A3B claude-opus-4-6 False  # disable clinical annotations
# ============================================================

START="${1:-1}"
END="${2:-5}"
MODEL="${3:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${4:-claude-opus-4-6}"
ANNOTATE_CLINICAL="${5:-True}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMP_DIR="$PROJECT_DIR/comparisons"
PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis")
TOTAL_START=$SECONDS

echo "============================================================"
echo "MULTI-ITERATION EXPERIMENT: v${START} through v${END}"
echo "  Model:             $MODEL"
echo "  Evolver:           $EVOLVER_MODEL"
echo "  Annotate Clinical: $ANNOTATE_CLINICAL"
echo "  Iterations:        $((END - START + 1))"
echo "============================================================"
echo ""

# ============================================================
# Helper: print performance table for a given version
# ============================================================
print_performance() {
    local V="$1"
    echo ""
    echo "  v${V} Performance (baseline -> evolved):"
    echo "  ---------------------------------------------------------------"
    printf "  %-16s %-22s %-22s %-22s %-22s\n" "Pathology" "Diagnosis" "PE First" "Labs" "Imaging"
    echo "  ---------------------------------------------------------------"
    for P in "${PATHOLOGIES[@]}"; do
        local COMP="$COMP_DIR/v${V}_vs_baseline_${P}.md"
        if [ -f "$COMP" ]; then
            local DX PE LABS IMG
            DX=$(grep "Diagnosis accuracy" "$COMP" 2>/dev/null | sed 's/- Diagnosis accuracy: //')
            PE=$(grep "PE first" "$COMP" 2>/dev/null | sed 's/- PE first: //')
            LABS=$(grep "Lab score" "$COMP" 2>/dev/null | sed 's/- Lab score total: //')
            IMG=$(grep "Imaging score" "$COMP" 2>/dev/null | sed 's/- Imaging score total: //')
            printf "  %-16s %-22s %-22s %-22s %-22s\n" "$P" "$DX" "$PE" "$LABS" "$IMG"
        fi
    done
    echo "  ---------------------------------------------------------------"
}

# ============================================================
# Run iterations
# ============================================================
for V in $(seq "$START" "$END"); do
    ITER_START=$SECONDS
    echo ""
    echo "************************************************************"
    echo "  ITERATION v${V} of v${END}  (started $(date '+%Y-%m-%d %H:%M:%S'))"
    echo "************************************************************"
    echo ""

    bash "$PROJECT_DIR/scripts/run_experiment_multi.sh" "v${V}" "$MODEL" "$EVOLVER_MODEL" "$ANNOTATE_CLINICAL"

    ITER_ELAPSED=$((SECONDS - ITER_START))
    ITER_MIN=$((ITER_ELAPSED / 60))
    ITER_SEC=$((ITER_ELAPSED % 60))

    echo ""
    echo "==== ITERATION v${V} COMPLETE (${ITER_MIN}m ${ITER_SEC}s) ===="
    print_performance "$V"
    echo ""
done

# ============================================================
# Final cross-version summary
# ============================================================
TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "ALL ITERATIONS COMPLETE (v${START} through v${END})"
echo "  Total time: ${TOTAL_HR}h ${TOTAL_MIN}m"
echo "============================================================"
echo ""

# Cross-version trend tables with percentage change
# Helper: extract baseline value from comparison file
extract_base_val() {
    local KEY="$1" FILE="$2"
    grep "$KEY" "$FILE" 2>/dev/null | head -1 | sed "s/.*: \(.*\) -> .*/\1/"
}

# Print cross-version trend for a metric
# Args: TITLE GREP_KEY TYPE(binary|score)
print_metric_trend() {
    local TITLE="$1" GREP_KEY="$2" TYPE="$3"

    echo "  ${TITLE}:"
    echo "  --------------------------------------------------------------------------"
    printf "  %-16s %-12s" "Pathology" "Baseline"
    for V in $(seq "$START" "$END"); do
        printf " %-16s" "v${V}"
    done
    echo ""
    echo "  --------------------------------------------------------------------------"

    for P in "${PATHOLOGIES[@]}"; do
        local BASE_STR="-"
        for V in $(seq "$START" "$END"); do
            local COMP="$COMP_DIR/v${V}_vs_baseline_${P}.md"
            if [ -f "$COMP" ]; then
                BASE_STR=$(extract_base_val "$GREP_KEY" "$COMP")
                [ -z "$BASE_STR" ] && BASE_STR="-"
                break
            fi
        done

        printf "  %-16s %-12s" "$P" "$BASE_STR"

        for V in $(seq "$START" "$END"); do
            local COMP="$COMP_DIR/v${V}_vs_baseline_${P}.md"
            local CELL="-"
            if [ -f "$COMP" ]; then
                local LINE
                LINE=$(grep "$GREP_KEY" "$COMP" 2>/dev/null | head -1)
                if [ -n "$LINE" ]; then
                    if [ "$TYPE" = "binary" ]; then
                        # Binary: extract counts, compute percentage change
                        local EVOL_CT BASE_CT N_PAT
                        EVOL_CT=$(echo "$LINE" | sed 's/.*-> \([0-9]*\)\/.*/\1/')
                        BASE_CT=$(echo "$LINE" | sed "s/.*: \([0-9]*\)\/.*/\1/")
                        N_PAT=$(echo "$LINE" | sed 's/.*\/\([0-9]*\) .*/\1/')
                        if [ "${N_PAT:-0}" -gt 0 ] 2>/dev/null; then
                            local DIFF=$(( EVOL_CT - BASE_CT ))
                            local DIFF_PP=$(( DIFF * 100 / N_PAT ))
                            if [ "$DIFF_PP" -ge 0 ]; then
                                CELL="${EVOL_CT}/${N_PAT} (+${DIFF_PP}%)"
                            else
                                CELL="${EVOL_CT}/${N_PAT} (${DIFF_PP}%)"
                            fi
                        fi
                    else
                        # Score: show evolved + delta as-is
                        CELL=$(echo "$LINE" | sed 's/.*-> //')
                    fi
                fi
            fi
            printf " %-16s" "$CELL"
        done
        echo ""
    done
    echo "  --------------------------------------------------------------------------"
    echo ""
}

print_metric_trend "Diagnosis Accuracy" "Diagnosis accuracy" "binary"
print_metric_trend "PE First" "PE first" "binary"
print_metric_trend "Lab Score (total)" "Lab score total" "score"
print_metric_trend "Imaging Score (total)" "Imaging score total" "score"

# Skills summary
echo "  Skills:"
for V in $(seq "$START" "$END"); do
    SKILL="$PROJECT_DIR/skills/v${V}/acute_abdominal_pain.md"
    if [ -f "$SKILL" ]; then
        BYTES=$(wc -c < "$SKILL")
        WORDS=$(wc -w < "$SKILL")
        echo "    v${V}: ${WORDS} words, ${BYTES} bytes"
    fi
done
echo ""
echo "  Logs:     ls $PROJECT_DIR/logs/experiment_multi_v*.log"
echo "  Reports:  ls $COMP_DIR/v*_vs_baseline_*.md"
echo "  Skills:   ls $PROJECT_DIR/skills/v*/acute_abdominal_pain.md"
echo "============================================================"
