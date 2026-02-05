#!/bin/bash
set -euo pipefail

# ============================================================
# run_iterations.sh — Run multiple evolution iterations
#
# Usage:
#   bash scripts/run_iterations.sh [START] [END] [MODEL] [EVOLVER_MODEL]
#
# Examples:
#   bash scripts/run_iterations.sh 1 5 Qwen3_30B_A3B
#   bash scripts/run_iterations.sh 3 5 Qwen3_30B_A3B   # resume from v3
# ============================================================

START="${1:-1}"
END="${2:-5}"
MODEL="${3:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${4:-claude-sonnet-4-20250514}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMP_DIR="$PROJECT_DIR/comparisons"
PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis")
TOTAL_START=$SECONDS

echo "============================================================"
echo "MULTI-ITERATION EXPERIMENT: v${START} through v${END}"
echo "  Model:         $MODEL"
echo "  Evolver:       $EVOLVER_MODEL"
echo "  Iterations:    $((END - START + 1))"
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
    printf "  %-16s %-14s %-14s %-10s %-10s\n" "Pathology" "Diagnosis" "PE First" "Labs" "Imaging"
    echo "  ---------------------------------------------------------------"
    for P in "${PATHOLOGIES[@]}"; do
        local COMP="$COMP_DIR/v${V}_vs_baseline_${P}.md"
        if [ -f "$COMP" ]; then
            local DX PE LABS IMG
            DX=$(grep "Diagnosis accuracy" "$COMP" 2>/dev/null | sed 's/- Diagnosis accuracy: //')
            PE=$(grep "PE first" "$COMP" 2>/dev/null | sed 's/- PE first: //')
            LABS=$(grep "Lab score" "$COMP" 2>/dev/null | sed 's/- Lab score total: //')
            IMG=$(grep "Imaging score" "$COMP" 2>/dev/null | sed 's/- Imaging score total: //')
            printf "  %-16s %-14s %-14s %-10s %-10s\n" "$P" "$DX" "$PE" "$LABS" "$IMG"
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

    bash "$PROJECT_DIR/scripts/run_experiment_multi.sh" "v${V}" "$MODEL" "$EVOLVER_MODEL"

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

# Diagnosis accuracy across versions
echo "  Diagnosis Accuracy Trend (evolved):"
echo "  ---------------------------------------------------------------"
printf "  %-16s" "Pathology"
for V in $(seq "$START" "$END"); do
    printf " %-8s" "v${V}"
done
echo ""
echo "  ---------------------------------------------------------------"
for P in "${PATHOLOGIES[@]}"; do
    printf "  %-16s" "$P"
    for V in $(seq "$START" "$END"); do
        COMP="$COMP_DIR/v${V}_vs_baseline_${P}.md"
        if [ -f "$COMP" ]; then
            # Extract just the evolved score (e.g., "3/10" from "3/10 -> 5/10 (+2)")
            EVOLVED_DX=$(grep "Diagnosis accuracy" "$COMP" 2>/dev/null | sed 's/.*-> \([^ ]*\).*/\1/')
            printf " %-8s" "$EVOLVED_DX"
        else
            printf " %-8s" "-"
        fi
    done
    echo ""
done
echo "  ---------------------------------------------------------------"
echo ""

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
