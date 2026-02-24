#!/bin/bash
set -euo pipefail

# ============================================================
# baseline_test.sh — Pure baselines: no skill, no lab annotations
#
# Runs the paper-equivalent baselines on the test set (100 patients)
# with annotate_clinical=False (no Approach 3 lab augmentation).
#
# Produces two baselines:
#   1. ZeroShot (ReAct) — paper's original agent architecture
#   2. ToT (Tree of Thoughts) — stronger search, still no skill
#
# Usage:
#   bash scripts/baseline_test.sh [FLAGS] [MODEL]
#
# Flags (before positional args):
#   --no-patient-sim         Disable patient simulator (on by default)
#   --tot-max-depth N        ToT max search depth (recommend 15 for patsim)
#   --tot-breadth N          ToT frontier size
#   --tot-n-generate N       ToT candidates per step
#   --tot-temperature F      ToT generation temperature
#   --parallel-pathologies   Run pathologies in parallel (all 7 at once)
#   --small-test             Use train.pkl (10 patients) on 2 pathologies for quick testing
#
# Examples:
#   bash scripts/baseline_test.sh                           # defaults
#   bash scripts/baseline_test.sh Qwen3_30B_A3B             # specify model
#   bash scripts/baseline_test.sh --no-patient-sim Qwen3_30B_A3B
#   bash scripts/baseline_test.sh --tot-max-depth 15 Qwen3_30B_A3B
#   bash scripts/baseline_test.sh --small-test --parallel-pathologies  # quick concurrency test
#
# Output structure:
#   results/     — raw run output (per pathology)
#   trajectories/ — extracted JSON trajectories
#   comparisons/  — ZeroShot vs ToT comparison markdown
#
# Requires vLLM to be running if using a vLLM_* model.
# ============================================================

PATIENT_SIMULATOR="True"
PATSIM_SUFFIX="_patsim"
PARALLEL_PATHOLOGIES=false
SMALL_TEST=false
DATA_SPLIT="test"
NUM_PATIENTS=100
TOT_ARGS=()
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --no-patient-sim)
            PATIENT_SIMULATOR="False"
            PATSIM_SUFFIX=""
            shift ;;
        --parallel-pathologies)
            PARALLEL_PATHOLOGIES=true; shift ;;
        --small-test)
            SMALL_TEST=true
            DATA_SPLIT="train"
            NUM_PATIENTS=10
            shift ;;
        --tot-max-depth|--tot-breadth|--tot-n-generate|--tot-temperature)
            KEY=$(echo "${1#--}" | tr '-' '_')
            TOT_ARGS+=("${KEY}=${2:?$1 requires a value}"); shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

MODEL="${1:-Qwen3_30B_A3B}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_DIR/codes_Hager/MIMIC-Clinical-Decision-Making-Framework"

# Detect Python (micromamba env, MIMIC_PYTHON override, or bare python)
source "$(dirname "$0")/_detect_python.sh"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"
TRAJ_DIR="$PROJECT_DIR/trajectories"
COMP_DIR="$PROJECT_DIR/comparisons"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/baseline_test_${TIMESTAMP}.log"

BASE_MODELS="${HF_HOME:-${HOME}/.cache/huggingface/hub}"
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"

if [ "$SMALL_TEST" = true ]; then
    # Quick concurrency test: 2 pathologies, 10 patients each
    PATHOLOGIES=("appendicitis" "cholecystitis")
else
    PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis" "cholangitis" "bowel_obstruction" "pyelonephritis")
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

exec > >(tee -a "$LOG_FILE") 2>&1

echo "==========================================================="
echo " BASELINE TEST: ZeroShot + ToT, no skill, no lab annotations"
echo "==========================================================="
echo ""
echo "  Model:             $MODEL"
echo "  Data split:        ${DATA_SPLIT} ($NUM_PATIENTS patients per pathology)"
echo "  Parallel:          $PARALLEL_PATHOLOGIES"
echo "  annotate_clinical: False (disabled for true baseline)"
echo "  skill_path:        (none)"
echo "  Patient Simulator: $PATIENT_SIMULATOR"
echo "  Pathologies:       ${PATHOLOGIES[*]}"
echo "  LOG_FILE:          $LOG_FILE"
echo ""

[ -f "$LAB_TEST_MAPPING" ] || die "Lab test mapping not found: $LAB_TEST_MAPPING"
for P in "${PATHOLOGIES[@]}"; do
    [ -f "$DATA_DIR/$P/test.pkl" ] || die "Test data not found: $DATA_DIR/$P/test.pkl"
done

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

echo "  All prerequisites OK"
phase_end

# ============================================================
# PHASE 1: ZeroShot baseline (no skill, annotate_clinical=False)
# ============================================================
phase_start 1 "ZeroShot baseline (no skill, no lab annotations)"

declare -A ZS_RUN_DIRS
ZS_DESCR="_zs_baseline${PATSIM_SUFFIX}_${DATA_SPLIT}${NUM_PATIENTS}"

run_zs_pathology() {
    local P="$1"
    cd "$FRAMEWORK_DIR"
    local CMD=(
        "$PYTHON" run.py
        pathology="$P"
        model="$MODEL"
        agent=ZeroShot
        data_file="$DATA_DIR/$P/${DATA_SPLIT}.pkl"
        base_mimic="$DATA_DIR/$P"
        base_models="$BASE_MODELS"
        lab_test_mapping_path="$LAB_TEST_MAPPING"
        local_logging_dir="$RESULTS_DIR"
        summarize=True
        annotate_clinical=False
        run_descr="$ZS_DESCR"
    )
    if [ "$PATIENT_SIMULATOR" = "True" ]; then
        CMD+=(patient_simulator=True)
    fi
    "${CMD[@]}"
}

if [ "$PARALLEL_PATHOLOGIES" = true ]; then
    echo "  Running ${#PATHOLOGIES[@]} pathologies in parallel"
    declare -A ZS_PIDS
    for P in "${PATHOLOGIES[@]}"; do
        echo "  Starting ZeroShot baseline: $P"
        run_zs_pathology "$P" > "$LOG_DIR/zs_baseline_${P}_${TIMESTAMP}.log" 2>&1 &
        ZS_PIDS[$P]=$!
    done
    for P in "${PATHOLOGIES[@]}"; do
        if ! wait "${ZS_PIDS[$P]}"; then
            die "ZeroShot baseline failed for $P (see $LOG_DIR/zs_baseline_${P}_${TIMESTAMP}.log)"
        fi
        echo "  Completed ZeroShot baseline: $P"
    done
else
    for P in "${PATHOLOGIES[@]}"; do
        echo ""
        echo "--- ZeroShot baseline: $P ($NUM_PATIENTS patients) ---"
        run_zs_pathology "$P" || die "ZeroShot baseline failed for $P"
    done
fi

for P in "${PATHOLOGIES[@]}"; do
    ZS_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$P"*"$ZS_DESCR"* 2>/dev/null | head -1)
    [ -n "$ZS_RUN_DIR" ] || die "Could not find ZeroShot baseline results for $P"
    ZS_RUN_DIRS[$P]="$ZS_RUN_DIR"
    echo "  Results: $ZS_RUN_DIR"
done

phase_end

# ============================================================
# PHASE 2: Evaluate ZeroShot baseline + extract trajectories
# ============================================================
phase_start 2 "Evaluate ZeroShot baseline"

for P in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$P/${DATA_SPLIT}.pkl"
    ZS_TRAJ="$TRAJ_DIR/zs_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"

    echo ""
    echo "--- Evaluate ZeroShot: $P ---"
    "$PYTHON" "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "${ZS_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        || die "ZeroShot evaluation failed for $P"

    echo "--- Extract ZeroShot trajectories: $P ---"
    "$PYTHON" "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "${ZS_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        --output "$ZS_TRAJ" \
        || die "ZeroShot trajectory extraction failed for $P"
done

phase_end

# ============================================================
# PHASE 3: ToT baseline (no skill, annotate_clinical=False)
# ============================================================
phase_start 3 "ToT baseline (no skill, no lab annotations)"

declare -A TOT_RUN_DIRS
TOT_DESCR="_tot_baseline${PATSIM_SUFFIX}_${DATA_SPLIT}${NUM_PATIENTS}"

run_tot_pathology() {
    local P="$1"
    cd "$FRAMEWORK_DIR"
    local CMD=(
        "$PYTHON" run.py
        pathology="$P"
        model="$MODEL"
        agent=ToT
        data_file="$DATA_DIR/$P/${DATA_SPLIT}.pkl"
        base_mimic="$DATA_DIR/$P"
        base_models="$BASE_MODELS"
        lab_test_mapping_path="$LAB_TEST_MAPPING"
        local_logging_dir="$RESULTS_DIR"
        summarize=True
        annotate_clinical=False
        run_descr="$TOT_DESCR"
        "${TOT_ARGS[@]+"${TOT_ARGS[@]}"}"
    )
    if [ "$PATIENT_SIMULATOR" = "True" ]; then
        CMD+=(patient_simulator=True)
    fi
    "${CMD[@]}"
}

if [ "$PARALLEL_PATHOLOGIES" = true ]; then
    echo "  Running ${#PATHOLOGIES[@]} pathologies in parallel"
    declare -A TOT_PIDS
    for P in "${PATHOLOGIES[@]}"; do
        echo "  Starting ToT baseline: $P"
        run_tot_pathology "$P" > "$LOG_DIR/tot_baseline_${P}_${TIMESTAMP}.log" 2>&1 &
        TOT_PIDS[$P]=$!
    done
    for P in "${PATHOLOGIES[@]}"; do
        if ! wait "${TOT_PIDS[$P]}"; then
            die "ToT baseline failed for $P (see $LOG_DIR/tot_baseline_${P}_${TIMESTAMP}.log)"
        fi
        echo "  Completed ToT baseline: $P"
    done
else
    for P in "${PATHOLOGIES[@]}"; do
        echo ""
        echo "--- ToT baseline: $P ($NUM_PATIENTS patients) ---"
        run_tot_pathology "$P" || die "ToT baseline failed for $P"
    done
fi

for P in "${PATHOLOGIES[@]}"; do
    TOT_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$P"*"$TOT_DESCR"* 2>/dev/null | head -1)
    [ -n "$TOT_RUN_DIR" ] || die "Could not find ToT baseline results for $P"
    TOT_RUN_DIRS[$P]="$TOT_RUN_DIR"
    echo "  Results: $TOT_RUN_DIR"
done

phase_end

# ============================================================
# PHASE 4: Evaluate ToT baseline + extract trajectories
# ============================================================
phase_start 4 "Evaluate ToT baseline"

for P in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$P/${DATA_SPLIT}.pkl"
    TOT_TRAJ="$TRAJ_DIR/tot_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"

    echo ""
    echo "--- Evaluate ToT: $P ---"
    "$PYTHON" "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "${TOT_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        || die "ToT evaluation failed for $P"

    echo "--- Extract ToT trajectories: $P ---"
    "$PYTHON" "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "${TOT_RUN_DIRS[$P]}" \
        --pathology "$P" \
        --patient_data "$PATIENT_DATA" \
        --output "$TOT_TRAJ" \
        || die "ToT trajectory extraction failed for $P"
done

phase_end

# ============================================================
# PHASE 5: Compare ZeroShot vs ToT baselines
# ============================================================
phase_start 5 "Compare ZeroShot vs ToT baselines"

for P in "${PATHOLOGIES[@]}"; do
    ZS_TRAJ="$TRAJ_DIR/zs_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"
    TOT_TRAJ="$TRAJ_DIR/tot_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"
    COMPARISON="$COMP_DIR/baseline_zs_vs_tot${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.md"

    echo ""
    echo "--- Compare ZeroShot vs ToT: $P ---"
    "$PYTHON" "$PROJECT_DIR/scripts/compare_runs.py" \
        --baseline "$ZS_TRAJ" \
        --evolved "$TOT_TRAJ" \
        --output "$COMPARISON" \
        || die "Comparison failed for $P"
done

phase_end

# ============================================================
# PHASE 6: Final Summary
# ============================================================
TOTAL_ELAPSED=$SECONDS
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "============================================================"
echo "BASELINE TEST COMPLETE"
echo "============================================================"
echo ""
echo "  Model:        $MODEL"
echo "  Data split:   ${DATA_SPLIT} ($NUM_PATIENTS patients per pathology)"
echo "  Duration:     ${TOTAL_HR}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""
echo "  Results:"
echo "  -----------------------------------------------------------------------------------"
printf "  %-20s %-35s %-35s\n" "Pathology" "ZeroShot Dir" "ToT Dir"
echo "  -----------------------------------------------------------------------------------"
for P in "${PATHOLOGIES[@]}"; do
    printf "  %-20s %-35s %-35s\n" "$P" "$(basename "${ZS_RUN_DIRS[$P]}")" "$(basename "${TOT_RUN_DIRS[$P]}")"
done
echo "  -----------------------------------------------------------------------------------"
echo ""
echo "  Comparisons (ZeroShot vs ToT):"
for P in "${PATHOLOGIES[@]}"; do
    COMPARISON="$COMP_DIR/baseline_zs_vs_tot${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.md"
    if [ -f "$COMPARISON" ]; then
        echo ""
        echo "  [$P]"
        grep -E "(Diagnosis accuracy|PE first|Lab score|Imaging score)" "$COMPARISON" 2>/dev/null | sed 's/^/    /' || true
    fi
done
echo ""
echo "  Trajectory files (for use as --baseline in compare_runs.py):"
for P in "${PATHOLOGIES[@]}"; do
    echo "    ZS:  $TRAJ_DIR/zs_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"
    echo "    ToT: $TRAJ_DIR/tot_baseline${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.json"
done
echo ""
echo "  Full comparison reports:"
for P in "${PATHOLOGIES[@]}"; do
    echo "    cat $COMP_DIR/baseline_zs_vs_tot${PATSIM_SUFFIX}_${P}_${DATA_SPLIT}${NUM_PATIENTS}.md"
done
echo ""
echo "  Next steps:"
echo "    - Run EvoTest with skill: bash scripts/evotest_full.sh 10 $MODEL"
echo "    - Compare skill results against these baselines using compare_runs.py"
echo ""
echo "  Log: $LOG_FILE"
echo "============================================================"
