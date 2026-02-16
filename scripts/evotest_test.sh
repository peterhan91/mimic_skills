#!/bin/bash
set -euo pipefail

# ============================================================
# evotest_test.sh — Evaluate best skill on test set (100 patients)
#
# Uses data_file= to point run.py directly at test.pkl without
# modifying the shared _hadm_info_first_diag.pkl file (safe to
# run concurrently with training).
#
# Usage:
#   bash scripts/evotest_test.sh [FLAGS] <BEST_SKILL_PATH> [MODEL] [ANNOTATE_CLINICAL]
#
# Flags (before positional args):
#   --agent ToT|ZeroShot     Agent type (default: ZeroShot)
#   --patient-sim            Enable patient simulator
#   --tot-max-depth N        ToT max search depth (recommend 15 for patsim)
#   --tot-breadth N          ToT frontier size
#   --tot-n-generate N       ToT candidates per step
#   --tot-temperature F      ToT generation temperature
#
# Examples:
#   bash scripts/evotest_test.sh skills/evo/episode_5.md
#   bash scripts/evotest_test.sh skills/evo/episode_5.md vLLM_Qwen3 True
#   bash scripts/evotest_test.sh --patient-sim skills/evo/episode_5.md vLLM_Qwen3 True
#   bash scripts/evotest_test.sh --agent ToT --patient-sim --tot-max-depth 15 skills/evo_tot_patsim/episode_4.md
#
# The script will:
#   1. Run baseline (no skill) on all 7 pathologies (100 patients each)
#   2. Run with best skill on all 7 pathologies
#   3. Evaluate + extract trajectories + compare
#
# Requires vLLM to be running if using a vLLM_* model.
# ============================================================

AGENT="ZeroShot"
PATIENT_SIMULATOR="False"
TOT_ARGS=()
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --agent)
            AGENT="${2:?--agent requires a value (ZeroShot or ToT)}"; shift 2 ;;
        --patient-sim)
            PATIENT_SIMULATOR="True"; shift ;;
        --tot-max-depth|--tot-breadth|--tot-n-generate|--tot-temperature)
            # Convert --tot-max-depth → tot_max_depth= for Hydra
            KEY=$(echo "${1#--}" | tr '-' '_')
            TOT_ARGS+=("${KEY}=${2:?$1 requires a value}"); shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

BEST_SKILL="${1:?Usage: $0 [--agent ToT] [--patient-sim] <BEST_SKILL_PATH> [MODEL] [ANNOTATE_CLINICAL]}"
MODEL="${2:-vLLM_Qwen3}"
ANNOTATE_CLINICAL="${3:-True}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_DIR/codes_Hager/MIMIC-Clinical-Decision-Making-Framework"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"
TRAJ_DIR="$PROJECT_DIR/trajectories"
COMP_DIR="$PROJECT_DIR/comparisons"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_eval_${TIMESTAMP}.log"

BASE_MODELS="${HF_HOME:-${HOME}/.cache/huggingface/hub}"
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"
PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis" "cholangitis" "bowel_obstruction" "pyelonephritis")

# Experiment prefix for trajectories/comparisons (prevents cross-experiment overwrites)
if [ "$AGENT" = "ToT" ] && [ "$PATIENT_SIMULATOR" = "True" ]; then
    TEST_PREFIX="totps"
elif [ "$AGENT" = "ToT" ]; then
    TEST_PREFIX="tot"
elif [ "$PATIENT_SIMULATOR" = "True" ]; then
    TEST_PREFIX="evops"
else
    TEST_PREFIX="evo"
fi

# ============================================================
# Helpers
# ============================================================
phase_start() {
    echo ""
    echo "$(date +%H:%M:%S) === PHASE $1: $2 ==="
    echo ""
    PHASE_START_SECONDS=$SECONDS
}

phase_end() {
    local elapsed=$((SECONDS - PHASE_START_SECONDS))
    echo ""
    echo "$(date +%H:%M:%S) --- Phase completed in ${elapsed}s ---"
    echo ""
}

die() {
    echo ""
    echo "ERROR: $1" >&2
    exit 1
}

# ============================================================
# PHASE 0: Setup and Prerequisites
# ============================================================
phase_start 0 "Setup and prerequisites"

mkdir -p "$RESULTS_DIR" "$TRAJ_DIR" "$COMP_DIR" "$LOG_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

# Resolve skill path
if [[ "$BEST_SKILL" != /* ]]; then
    BEST_SKILL="$PROJECT_DIR/$BEST_SKILL"
fi

echo "Test evaluation configuration:"
echo "  BEST_SKILL:        $BEST_SKILL"
echo "  MODEL:             $MODEL"
echo "  ANNOTATE_CLINICAL: $ANNOTATE_CLINICAL"
echo "  PATIENT_SIMULATOR: $PATIENT_SIMULATOR"
echo "  PATHOLOGIES:       ${PATHOLOGIES[*]}"
echo "  LOG_FILE:          $LOG_FILE"
echo ""

# Validate skill file
[ -f "$BEST_SKILL" ] || die "Skill file not found: $BEST_SKILL"
echo "  Skill content preview:"
head -5 "$BEST_SKILL" | sed 's/^/    /'
echo "    ..."
SKILL_WORDS=$(wc -w < "$BEST_SKILL")
echo "    ($SKILL_WORDS words total)"
echo ""

# Validate data
[ -f "$LAB_TEST_MAPPING" ] || die "Lab test mapping not found: $LAB_TEST_MAPPING"

for P in "${PATHOLOGIES[@]}"; do
    [ -f "$DATA_DIR/$P/test.pkl" ] || die "Test data not found: $DATA_DIR/$P/test.pkl"
done

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

echo "  All prerequisites OK"
phase_end

# ============================================================
# PHASE 1: Baseline run (no skill, test set)
# ============================================================
phase_start 1 "Baseline run (no skill, test set, all pathologies)"

declare -A BASELINE_RUN_DIRS
BASELINE_DESCR="_${TEST_PREFIX}_baseline_test100"

for P in "${PATHOLOGIES[@]}"; do
    echo ""
    echo "--- Baseline: $P (100 patients) ---"

    cd "$FRAMEWORK_DIR"
    BASELINE_CMD=(
        python run.py
        pathology="$P"
        model="$MODEL"
        agent="$AGENT"
        data_file="$DATA_DIR/$P/test.pkl"
        base_mimic="$DATA_DIR/$P"
        base_models="$BASE_MODELS"
        lab_test_mapping_path="$LAB_TEST_MAPPING"
        local_logging_dir="$RESULTS_DIR"
        summarize=True
        annotate_clinical="$ANNOTATE_CLINICAL"
        run_descr="$BASELINE_DESCR"
        "${TOT_ARGS[@]+"${TOT_ARGS[@]}"}"
    )
    if [ "$PATIENT_SIMULATOR" = "True" ]; then
        BASELINE_CMD+=(patient_simulator=True)
    fi
    "${BASELINE_CMD[@]}" || die "Baseline run failed for $P"

    BASELINE_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$P"*"$BASELINE_DESCR"* 2>/dev/null | head -1)
    [ -n "$BASELINE_RUN_DIR" ] || die "Could not find baseline results for $P"
    BASELINE_RUN_DIRS[$P]="$BASELINE_RUN_DIR"
    echo "  Results: $BASELINE_RUN_DIR"
done

phase_end

# ============================================================
# PHASE 2: Evaluate baseline + extract trajectories
# ============================================================
phase_start 2 "Evaluate baseline runs"

for P in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$P/test.pkl"
    BASELINE_TRAJ="$TRAJ_DIR/${TEST_PREFIX}_baseline_${P}_test100.json"

    echo ""
    echo "--- Evaluate baseline: $P ---"
    python "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "${BASELINE_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        || die "Baseline evaluation failed for $P"

    echo "--- Extract baseline trajectories: $P ---"
    python "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "${BASELINE_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        --output "$BASELINE_TRAJ" \
        || die "Baseline trajectory extraction failed for $P"
done

phase_end

# ============================================================
# PHASE 3: Best skill run (test set)
# ============================================================
phase_start 3 "Best skill run (test set, all pathologies)"

declare -A SKILL_RUN_DIRS
SKILL_DESCR="_${TEST_PREFIX}_evotest_best_test100"

for P in "${PATHOLOGIES[@]}"; do
    echo ""
    echo "--- Best skill: $P (100 patients) ---"

    cd "$FRAMEWORK_DIR"
    SKILL_CMD=(
        python run.py
        pathology="$P"
        model="$MODEL"
        agent="$AGENT"
        data_file="$DATA_DIR/$P/test.pkl"
        base_mimic="$DATA_DIR/$P"
        base_models="$BASE_MODELS"
        lab_test_mapping_path="$LAB_TEST_MAPPING"
        local_logging_dir="$RESULTS_DIR"
        summarize=True
        annotate_clinical="$ANNOTATE_CLINICAL"
        skill_path="$BEST_SKILL"
        run_descr="$SKILL_DESCR"
        "${TOT_ARGS[@]+"${TOT_ARGS[@]}"}"
    )
    if [ "$PATIENT_SIMULATOR" = "True" ]; then
        SKILL_CMD+=(patient_simulator=True)
    fi
    "${SKILL_CMD[@]}" || die "Skill run failed for $P"

    SKILL_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$P"*"$SKILL_DESCR"* 2>/dev/null | head -1)
    [ -n "$SKILL_RUN_DIR" ] || die "Could not find skill results for $P"
    SKILL_RUN_DIRS[$P]="$SKILL_RUN_DIR"
    echo "  Results: $SKILL_RUN_DIR"
done

phase_end

# ============================================================
# PHASE 4: Evaluate skill runs + extract + compare
# ============================================================
phase_start 4 "Evaluate skill runs and compare"

for P in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$P/test.pkl"
    BASELINE_TRAJ="$TRAJ_DIR/${TEST_PREFIX}_baseline_${P}_test100.json"
    SKILL_TRAJ="$TRAJ_DIR/${TEST_PREFIX}_evotest_best_${P}_test100.json"
    COMPARISON="$COMP_DIR/${TEST_PREFIX}_evotest_best_vs_baseline_${P}_test100.md"

    echo ""
    echo "--- Evaluate skill: $P ---"
    python "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "${SKILL_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        || die "Skill evaluation failed for $P"

    echo "--- Extract skill trajectories: $P ---"
    python "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "${SKILL_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        --output "$SKILL_TRAJ" \
        || die "Skill trajectory extraction failed for $P"

    echo "--- Compare baseline vs skill: $P ---"
    python "$PROJECT_DIR/scripts/compare_runs.py" \
        --baseline "$BASELINE_TRAJ" \
        --evolved "$SKILL_TRAJ" \
        --output "$COMPARISON" \
        || die "Comparison failed for $P"
done

phase_end

# ============================================================
# PHASE 5: Final Summary
# ============================================================
TOTAL_ELAPSED=$SECONDS
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "============================================================"
echo "TEST EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "  Skill:        $BEST_SKILL"
echo "  Model:        $MODEL"
echo "  Test set:     100 patients per pathology"
echo "  Duration:     ${TOTAL_HR}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""
echo "  Results:"
echo "  -----------------------------------------------------------------------------------"
printf "  %-16s %-35s %-35s\n" "Pathology" "Baseline Dir" "Skill Dir"
echo "  -----------------------------------------------------------------------------------"
for P in "${PATHOLOGIES[@]}"; do
    printf "  %-16s %-35s %-35s\n" "$P" "$(basename "${BASELINE_RUN_DIRS[$P]}")" "$(basename "${SKILL_RUN_DIRS[$P]}")"
done
echo "  -----------------------------------------------------------------------------------"
echo ""
echo "  Comparisons:"
for P in "${PATHOLOGIES[@]}"; do
    COMPARISON="$COMP_DIR/${TEST_PREFIX}_evotest_best_vs_baseline_${P}_test100.md"
    if [ -f "$COMPARISON" ]; then
        echo ""
        echo "  [$P]"
        # Print key metrics from comparison
        grep -E "(Diagnosis accuracy|PE first|Lab score|Imaging score)" "$COMPARISON" 2>/dev/null | sed 's/^/    /' || true
    fi
done
echo ""
echo "  Full comparison reports:"
for P in "${PATHOLOGIES[@]}"; do
    echo "    cat $COMP_DIR/${TEST_PREFIX}_evotest_best_vs_baseline_${P}_test100.md"
done
echo ""
echo "  Log: $LOG_FILE"
echo "============================================================"
