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
#   --include-remaining      Also run on remaining.pkl (all non-train patients)
#   --skill PATH             Inject skill from SKILL.md file
#   --skill-inject MODE      Skill injection mode: examples|system|both (default: examples)
#   --annotate-clinical      Enable lab result annotations (Approach 3)
#   --pathology P            Run only one pathology (can repeat)
#   --parallel               Run models in parallel (one model at a time per pathology)
#   --dry-run                Print commands without executing
#
# Examples:
#   bash scripts/cloud_baseline.sh                                  # test only, 3 default models
#   bash scripts/cloud_baseline.sh --include-remaining              # test + remaining (all non-train)
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
INCLUDE_REMAINING=false
SKILL_PATH=""
SKILL_INJECT="examples"
ANNOTATE_CLINICAL="False"
CONDITION="abdominal"
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
        --include-remaining)
            INCLUDE_REMAINING=true; shift ;;
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
        --condition)
            CONDITION="${2:?--condition requires a value (abdominal or chest)}"; shift 2 ;;
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

# Pathologies and lab test mapping by condition
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"
if [ "$CONDITION" = "chest" ]; then
    ALL_PATHOLOGIES=("myocardial_infarction" "pulmonary_embolism" "congestive_heart_failure" "aortic_stenosis" "mitral_regurgitation")
    CARDIAC_TOOLS_FLAG="cardiac_tools=True"
else
    ALL_PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis" "cholangitis" "bowel_obstruction" "pyelonephritis")
    CARDIAC_TOOLS_FLAG=""
fi

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

# --- Merge test + remaining if requested ---
# Creates a combined pkl per pathology (excludes only train.pkl)
if [ "$INCLUDE_REMAINING" = true ]; then
    MERGED_DIR="$DATA_DIR/_merged"
    mkdir -p "$MERGED_DIR"
    for P in "${PATHOLOGIES[@]}"; do
        MERGED_PKL="$MERGED_DIR/${P}_test_remaining.pkl"
        if [ ! -f "$MERGED_PKL" ] || [ "$DATA_DIR/$P/test.pkl" -nt "$MERGED_PKL" ]; then
            python3 -c "
import pickle, sys
merged = {}
for split in ['test', 'remaining']:
    path = '$DATA_DIR/$P/' + split + '.pkl'
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        merged.update(data)
    except FileNotFoundError:
        pass
with open('$MERGED_PKL', 'wb') as f:
    pickle.dump(merged, f)
print(f'  Merged $P: {len(merged)} patients (test + remaining)')
" || die "Failed to merge pkls for $P"
        fi
    done
    DATA_SUFFIX="_all"
else
    DATA_SUFFIX="_test"
fi

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

# --- Helper to get data file for a pathology ---
get_data_file() {
    local P="$1"
    if [ "$INCLUDE_REMAINING" = true ]; then
        echo "$MERGED_DIR/${P}_test_remaining.pkl"
    else
        echo "$DATA_DIR/$P/test.pkl"
    fi
}

DATA_LABEL="test only"
if [ "$INCLUDE_REMAINING" = true ]; then
    DATA_LABEL="test + remaining (all non-train)"
fi

echo "============================================================"
echo " CLOUD BASELINE: ZeroShot on ${DATA_LABEL}"
echo "============================================================"
echo ""
echo "  Models:            ${MODELS[*]}"
echo "  Pathologies:       ${PATHOLOGIES[*]}"
echo "  Data:              $DATA_LABEL"
echo "  Patient Simulator: $PATIENT_SIMULATOR"
echo "  Skill:             $SKILL_DESCR"
echo "  Annotate Clinical: $ANNOTATE_CLINICAL"
echo "  Parallel:          $PARALLEL"
echo ""

# Count total patients
TOTAL_PATIENTS=0
for P in "${PATHOLOGIES[@]}"; do
    N=$(count_patients "$(get_data_file "$P")")
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
    local DATA_FILE
    DATA_FILE=$(get_data_file "$P")
    local N=$(count_patients "$DATA_FILE")
    local DESCR="_cloud_baseline${DATA_SUFFIX}${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}"

    local CMD=(
        python run.py
        pathology="$P"
        model="$MODEL"
        agent=ZeroShot
        data_file="$DATA_FILE"
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
    [ -n "$CARDIAC_TOOLS_FLAG" ] && CMD+=("$CARDIAC_TOOLS_FLAG")

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

    DESCR="_cloud_baseline${DATA_SUFFIX}${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}"
    for MODEL in "${MODELS[@]}"; do
        for P in "${PATHOLOGIES[@]}"; do
            EVAL_DATA=$(get_data_file "$P")
            RUN_DIR=$(ls -td "$RESULTS_DIR"/*"${P}"*"${DESCR}"* 2>/dev/null | grep -i "${MODEL}" | head -1 || true)
            if [ -z "$RUN_DIR" ]; then
                RUN_DIR=$(ls -td "$RESULTS_DIR"/*"${P}"*"${MODEL}"*"${DESCR}"* 2>/dev/null | head -1)
            fi
            if [ -n "$RUN_DIR" ]; then
                echo ""
                echo "--- Evaluate: $MODEL / $P ---"
                python "$PROJECT_DIR/scripts/evaluate_run.py" \
                    --results_dir "$RUN_DIR" \
                    --pathology "$P" \
                    --patient_data "$EVAL_DATA" \
                    || echo "  WARNING: Evaluation failed for $MODEL / $P"

                TRAJ_FILE="$TRAJ_DIR/cloud_${MODEL}${DATA_SUFFIX}${PATSIM_SUFFIX}${SKILL_SUFFIX}${ANNOT_SUFFIX}_${P}.json"
                python "$PROJECT_DIR/scripts/extract_trajectories.py" \
                    --results_dir "$RUN_DIR" \
                    --pathology "$P" \
                    --patient_data "$EVAL_DATA" \
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
echo "  Data:          $DATA_LABEL"
echo "  Total patients: $TOTAL_PATIENTS × ${#MODELS[@]} models"
echo "  Duration:      ${TOTAL_HR}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Skill:         $SKILL_DESCR"
echo "  Annotate:      $ANNOTATE_CLINICAL"
echo ""
echo "  Results directory: $RESULTS_DIR"
echo "  Trajectory files:  $TRAJ_DIR/cloud_*"
echo "  Logs:              $LOG_DIR/cloud_*_${TIMESTAMP}.log"
echo ""
echo "  Cost estimates (approximate, per-patient × $TOTAL_PATIENTS):"
echo "    GPT5mini:     ~\$$(( TOTAL_PATIENTS * 9 / 1000 ))   (\$0.009/patient)"
echo "    ClaudeHaiku:  ~\$$(( TOTAL_PATIENTS * 6 / 1000 ))   (\$0.006/patient)"
echo "    ClaudeSonnet: ~\$$(( TOTAL_PATIENTS * 28 / 1000 ))  (\$0.028/patient)"
echo "    GPT5.2:       ~\$$(( TOTAL_PATIENTS * 63 / 1000 ))  (\$0.063/patient)"
echo "    ClaudeOpus:   ~\$$(( TOTAL_PATIENTS * 139 / 1000 )) (\$0.139/patient)"
echo "============================================================"
