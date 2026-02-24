#!/bin/bash
set -euo pipefail

# ============================================================
# cloud_baseline.sh — Run cloud API models on all test cases
#
# Supports: GPT5mini, ClaudeHaiku, ClaudeSonnet, ClaudeOpus, GPT5.2
# Runs ZeroShot agent with patient simulator on all 7 pathologies.
#
# Usage:
#   bash scripts/cloud_baseline.sh [FLAGS] [MODELS...]
#
# Flags:
#   --no-patient-sim         Disable patient simulator
#   --skill PATH             Inject skill from SKILL.md file
#   --skill-inject MODE      Skill injection mode: examples|system|both (default: examples)
#   --annotate-clinical      Enable lab result annotations (Approach 3)
#   --pathology P            Run only one pathology (can repeat)
#   --parallel               Run models in parallel (one model at a time per pathology)
#   --dry-run                Print commands without executing
#
# Examples:
#   bash scripts/cloud_baseline.sh                                  # all 3 default models
#   bash scripts/cloud_baseline.sh GPT5mini ClaudeHaiku             # specific models
#   bash scripts/cloud_baseline.sh --skill skills/v1/skill.md       # with skill injection
#   bash scripts/cloud_baseline.sh --pathology appendicitis GPT5mini  # one pathology
#   bash scripts/cloud_baseline.sh --annotate-clinical --skill skills/v1/skill.md
#
# Output:
#   results/   — raw run output (per model × pathology)
#   logs/      — per-model log files
#
# Environment:
#   OPENAI_API_KEY      — required for GPT models (via .env or env)
#   ANTHROPIC_API_KEY   — required for Claude models (via .env or env)
# ============================================================

# --- Defaults ---
PATIENT_SIMULATOR="True"
PATSIM_SUFFIX="_patsim"
SKILL_PATH=""
SKILL_INJECT="examples"
ANNOTATE_CLINICAL="False"
PARALLEL=false
DRY_RUN=false
SELECTED_PATHOLOGIES=()
MODELS=()

# --- Parse flags ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --no-patient-sim)
            PATIENT_SIMULATOR="False"
            PATSIM_SUFFIX=""
            shift ;;
        --skill)
            SKILL_PATH="${2:?--skill requires a path}"
            shift 2 ;;
        --skill-inject)
            SKILL_INJECT="${2:?--skill-inject requires examples|system|both}"
            shift 2 ;;
        --annotate-clinical)
            ANNOTATE_CLINICAL="True"
            shift ;;
        --pathology)
            SELECTED_PATHOLOGIES+=("${2:?--pathology requires a name}")
            shift 2 ;;
        --parallel)
            PARALLEL=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Remaining args are model names
while [[ $# -gt 0 ]]; do
    MODELS+=("$1"); shift
done

# Default models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("GPT5mini" "ClaudeHaiku" "ClaudeSonnet")
fi

# --- Paths ---
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_DIR/codes_Hager/MIMIC-Clinical-Decision-Making-Framework"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"
TRAJ_DIR="$PROJECT_DIR/trajectories"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"

ALL_PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis" "cholangitis" "bowel_obstruction" "pyelonephritis")

if [ ${#SELECTED_PATHOLOGIES[@]} -gt 0 ]; then
    PATHOLOGIES=("${SELECTED_PATHOLOGIES[@]}")
else
    PATHOLOGIES=("${ALL_PATHOLOGIES[@]}")
fi

# --- Helpers ---
die() { echo "ERROR: $1" >&2; exit 1; }

count_patients() {
    python3 -c "import pickle; d=pickle.load(open('$1','rb')); print(len(d))"
}

# --- Prerequisites ---
mkdir -p "$RESULTS_DIR" "$TRAJ_DIR" "$LOG_DIR"

# Source .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

[ -f "$LAB_TEST_MAPPING" ] || die "Lab test mapping not found: $LAB_TEST_MAPPING"
for P in "${PATHOLOGIES[@]}"; do
    [ -f "$DATA_DIR/$P/test.pkl" ] || die "Test data not found: $DATA_DIR/$P/test.pkl"
done

# Validate skill path if provided
if [ -n "$SKILL_PATH" ]; then
    # Resolve relative paths from project dir
    if [[ ! "$SKILL_PATH" = /* ]]; then
        SKILL_PATH="$PROJECT_DIR/$SKILL_PATH"
    fi
    [ -f "$SKILL_PATH" ] || die "Skill file not found: $SKILL_PATH"
fi

# --- Summary ---
SKILL_DESCR="(none)"
SKILL_SUFFIX=""
if [ -n "$SKILL_PATH" ]; then
    SKILL_NAME=$(basename "$SKILL_PATH" .md)
    SKILL_DESCR="$SKILL_PATH (inject=$SKILL_INJECT)"
    SKILL_SUFFIX="_SKILL_${SKILL_NAME}"
fi

ANNOT_SUFFIX=""
if [ "$ANNOTATE_CLINICAL" = "True" ]; then
    ANNOT_SUFFIX="_CLANNOT"
fi

echo "============================================================"
echo " CLOUD BASELINE: ZeroShot on all test cases"
echo "============================================================"
echo ""
echo "  Models:            ${MODELS[*]}"
echo "  Pathologies:       ${PATHOLOGIES[*]}"
echo "  Patient Simulator: $PATIENT_SIMULATOR"
echo "  Skill:             $SKILL_DESCR"
echo "  Annotate Clinical: $ANNOTATE_CLINICAL"
echo "  Parallel:          $PARALLEL"
echo ""

# Count total patients
TOTAL_PATIENTS=0
for P in "${PATHOLOGIES[@]}"; do
    N=$(count_patients "$DATA_DIR/$P/test.pkl")
    echo "  $P: $N patients"
    TOTAL_PATIENTS=$((TOTAL_PATIENTS + N))
done
echo "  ───────────────────────"
echo "  Total: $TOTAL_PATIENTS patients × ${#MODELS[@]} models = $((TOTAL_PATIENTS * ${#MODELS[@]})) runs"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN — commands printed below, not executed]"
    echo ""
fi

# --- Run function ---
run_model_pathology() {
    local MODEL="$1"
    local P="$2"
    local N=$(count_patients "$DATA_DIR/$P/test.pkl")
    local DESCR="_cloud_baseline${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}"

    local CMD=(
        python run.py
        pathology="$P"
        model="$MODEL"
        agent=ZeroShot
        data_file="$DATA_DIR/$P/test.pkl"
        base_mimic="$DATA_DIR/$P"
        lab_test_mapping_path="$LAB_TEST_MAPPING"
        local_logging_dir="$RESULTS_DIR"
        summarize=True
        annotate_clinical="$ANNOTATE_CLINICAL"
        patient_simulator="$PATIENT_SIMULATOR"
        run_descr="$DESCR"
    )

    # Skill injection
    if [ -n "$SKILL_PATH" ]; then
        CMD+=(skill_path="$SKILL_PATH" skill_inject="$SKILL_INJECT")
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] cd $FRAMEWORK_DIR && ${CMD[*]}"
        return 0
    fi

    echo ""
    echo "$(date +%H:%M:%S) --- $MODEL / $P ($N patients) ---"
    cd "$FRAMEWORK_DIR"
    "${CMD[@]}"
    echo "$(date +%H:%M:%S) --- $MODEL / $P complete ---"
}

# --- Main loop ---
START_SECONDS=$SECONDS

for MODEL in "${MODELS[@]}"; do
    MODEL_LOG="$LOG_DIR/cloud_${MODEL}_${TIMESTAMP}.log"
    echo ""
    echo "============================================================"
    echo "  Model: $MODEL"
    echo "  Log:   $MODEL_LOG"
    echo "============================================================"

    if [ "$PARALLEL" = true ] && [ "$DRY_RUN" = false ]; then
        # Run all pathologies for this model in parallel
        declare -A PIDS
        for P in "${PATHOLOGIES[@]}"; do
            P_LOG="$LOG_DIR/cloud_${MODEL}_${P}_${TIMESTAMP}.log"
            echo "  Starting: $MODEL / $P → $P_LOG"
            run_model_pathology "$MODEL" "$P" > "$P_LOG" 2>&1 &
            PIDS[$P]=$!
        done
        FAILED=false
        for P in "${PATHOLOGIES[@]}"; do
            if ! wait "${PIDS[$P]}"; then
                echo "  FAILED: $MODEL / $P (see $LOG_DIR/cloud_${MODEL}_${P}_${TIMESTAMP}.log)"
                FAILED=true
            else
                echo "  Done: $MODEL / $P"
            fi
        done
        unset PIDS
        if [ "$FAILED" = true ]; then
            echo "  WARNING: Some pathologies failed for $MODEL"
        fi
    else
        for P in "${PATHOLOGIES[@]}"; do
            run_model_pathology "$MODEL" "$P" 2>&1 | tee -a "$MODEL_LOG"
        done
    fi
done

# --- Evaluate all runs ---
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Evaluating all runs"
    echo "============================================================"

    DESCR="_cloud_baseline${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}"
    for MODEL in "${MODELS[@]}"; do
        for P in "${PATHOLOGIES[@]}"; do
            RUN_DIR=$(ls -td "$RESULTS_DIR"/*"${P}"*"${MODEL}"*"${DESCR}"* 2>/dev/null | head -1)
            if [ -z "$RUN_DIR" ]; then
                # Try matching with model_name instead of config name
                RUN_DIR=$(ls -td "$RESULTS_DIR"/*"${P}"*"${DESCR}"* 2>/dev/null | grep -i "${MODEL}" | head -1 || true)
            fi
            if [ -n "$RUN_DIR" ]; then
                echo ""
                echo "--- Evaluate: $MODEL / $P ---"
                python "$PROJECT_DIR/scripts/evaluate_run.py" \
                    --results_dir "$RUN_DIR" \
                    --pathology "$P" \
                    --patient_data "$DATA_DIR/$P/test.pkl" \
                    || echo "  WARNING: Evaluation failed for $MODEL / $P"

                TRAJ_FILE="$TRAJ_DIR/cloud_${MODEL}${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}_${P}.json"
                python "$PROJECT_DIR/scripts/extract_trajectories.py" \
                    --results_dir "$RUN_DIR" \
                    --pathology "$P" \
                    --patient_data "$DATA_DIR/$P/test.pkl" \
                    --output "$TRAJ_FILE" \
                    || echo "  WARNING: Trajectory extraction failed for $MODEL / $P"
            else
                echo "  WARNING: No results found for $MODEL / $P"
            fi
        done
    done
fi

# --- Summary ---
TOTAL_ELAPSED=$((SECONDS - START_SECONDS))
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "CLOUD BASELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Models:        ${MODELS[*]}"
echo "  Pathologies:   ${PATHOLOGIES[*]}"
echo "  Total patients: $TOTAL_PATIENTS × ${#MODELS[@]} models"
echo "  Duration:      ${TOTAL_HR}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Skill:         $SKILL_DESCR"
echo "  Annotate:      $ANNOTATE_CLINICAL"
echo ""
echo "  Results directory: $RESULTS_DIR"
echo "  Trajectory files:  $TRAJ_DIR/cloud_*"
echo "  Logs:              $LOG_DIR/cloud_*_${TIMESTAMP}.log"
echo ""
echo "  Cost estimates (approximate):"
echo "    GPT5mini:     ~\$6   for 706 patients"
echo "    ClaudeHaiku:  ~\$4   for 706 patients"
echo "    ClaudeSonnet: ~\$20  for 706 patients"
echo "    GPT5.2:       ~\$44  for 706 patients"
echo "    ClaudeOpus:   ~\$98  for 706 patients"
echo "============================================================"
