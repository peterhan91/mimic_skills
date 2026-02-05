#!/bin/bash
set -euo pipefail

# ============================================================
# run_experiment_multi.sh — Multi-pathology evolution cycle
#
# Runs baseline + evolved experiments across ALL 4 pathologies,
# generates one general skill from combined trajectories,
# then compares each pathology individually.
#
# Pipeline per pathology:
#   Baseline run -> Evaluate -> Extract trajectories
# Then across all pathologies:
#   Evolver generates skill (from all trajectories) -> Sanitize
# Then per pathology:
#   Run with skill -> Evaluate -> Extract -> Compare
#
# Usage:
#   bash scripts/run_experiment_multi.sh [VERSION] [MODEL] [EVOLVER_MODEL]
#
# Examples:
#   bash scripts/run_experiment_multi.sh v2 Qwen3_30B_A3B
#   bash scripts/run_experiment_multi.sh v3 Qwen3_30B_A3B claude-opus-4-6
#   bash scripts/run_experiment_multi.sh v2 MedGemma4B
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================
VERSION="${1:-v1}"
MODEL="${2:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${3:-claude-opus-4-6}"
SPLIT="train"
PATHOLOGIES=("appendicitis" "cholecystitis" "diverticulitis" "pancreatitis")

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_DIR/codes_Hager/MIMIC-Clinical-Decision-Making-Framework"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"
TRAJ_DIR="$PROJECT_DIR/trajectories"
SKILLS_DIR="$PROJECT_DIR/skills"
COMP_DIR="$PROJECT_DIR/comparisons"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/experiment_multi_${VERSION}_${TIMESTAMP}.log"

# Base models cache (HuggingFace downloads here)
BASE_MODELS="${HF_HOME:-${HOME}/.cache/huggingface/hub}"
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"

# Previous skill (for iterative refinement)
V_NUM=0
PREV_VERSION=""
if [[ "$VERSION" =~ ^v([0-9]+)$ ]]; then
    V_NUM="${BASH_REMATCH[1]}"
    if [ "$V_NUM" -gt 1 ]; then
        PREV_V=$((V_NUM - 1))
        PREV_VERSION="v${PREV_V}"
    fi
fi

# ============================================================
# HELPER FUNCTIONS
# ============================================================
phase_start() {
    local phase_num="$1"
    local phase_desc="$2"
    echo ""
    echo "$(date +%H:%M:%S) === PHASE $phase_num: $phase_desc ==="
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
    echo "Last 20 lines of log:" >&2
    tail -20 "$LOG_FILE" 2>/dev/null || true
    exit 1
}

# ============================================================
# PHASE 0: Setup and Prerequisites
# ============================================================
phase_start 0 "Setup and prerequisites"

# Create directories
mkdir -p "$RESULTS_DIR" "$TRAJ_DIR" "$SKILLS_DIR/$VERSION" "$COMP_DIR" "$LOG_DIR"

# Start logging (tee to both stdout and log file)
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Multi-pathology experiment configuration:"
echo "  VERSION:        $VERSION"
echo "  MODEL:          $MODEL"
echo "  EVOLVER_MODEL:  $EVOLVER_MODEL"
echo "  SPLIT:          $SPLIT"
echo "  PATHOLOGIES:    ${PATHOLOGIES[*]}"
echo "  PROJECT_DIR:    $PROJECT_DIR"
echo "  BASE_MODELS:    $BASE_MODELS"
echo "  LOG_FILE:       $LOG_FILE"
if [ -n "$PREV_VERSION" ]; then
    echo "  PREV_VERSION:   $PREV_VERSION (will feed previous skill to Evolver)"
fi
echo ""

# Check prerequisites
[ -f "$LAB_TEST_MAPPING" ] || die "Lab test mapping not found: $LAB_TEST_MAPPING"

for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$PATHOLOGY/${SPLIT}.pkl"
    HAGER_DATA="$DATA_DIR/$PATHOLOGY/${PATHOLOGY}_hadm_info_first_diag.pkl"
    [ -f "$PATIENT_DATA" ] || die "Patient data not found: $PATIENT_DATA"
    [ -f "$HAGER_DATA" ] || die "Hager-format data not found: $HAGER_DATA"
done

# Load .env if present (for ANTHROPIC_API_KEY, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    die "ANTHROPIC_API_KEY not set. Add it to .env or export it before running."
fi

phase_end

# ============================================================
# PHASE 1: Baseline Runs (all pathologies)
# ============================================================
phase_start 1 "Baseline runs (${PATHOLOGIES[*]})"

declare -A BASELINE_RUN_DIRS

# Skip if baseline trajectories already exist (v2+ saves ~2h per iteration)
SKIP_BASELINE=false
if [ -n "$PREV_VERSION" ]; then
    SKIP_BASELINE=true
    for PATHOLOGY in "${PATHOLOGIES[@]}"; do
        if [ ! -f "$TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json" ]; then
            SKIP_BASELINE=false
            break
        fi
    done
fi

if [ "$SKIP_BASELINE" = true ]; then
    echo "Baseline trajectories already exist — skipping baseline runs"
    echo "(Delete $TRAJ_DIR/baseline_* to force fresh baselines)"
    for PATHOLOGY in "${PATHOLOGIES[@]}"; do
        BASELINE_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"_baseline_${SPLIT}10"* 2>/dev/null | head -1)
        BASELINE_RUN_DIRS[$PATHOLOGY]="${BASELINE_RUN_DIR:-unknown}"
    done
else

for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    BASELINE_DESCR="_baseline_${SPLIT}10"

    echo ""
    echo "--- Running baseline: $PATHOLOGY ---"
    echo "Running: cd $FRAMEWORK_DIR && python run.py \\"
    echo "  pathology=$PATHOLOGY model=$MODEL \\"
    echo "  base_mimic=$DATA_DIR/$PATHOLOGY \\"
    echo "  base_models=$BASE_MODELS \\"
    echo "  lab_test_mapping_path=$LAB_TEST_MAPPING \\"
    echo "  local_logging_dir=$RESULTS_DIR \\"
    echo "  summarize=True \\"
    echo "  run_descr=$BASELINE_DESCR"

    cd "$FRAMEWORK_DIR"
    python run.py \
        pathology="$PATHOLOGY" \
        model="$MODEL" \
        base_mimic="$DATA_DIR/$PATHOLOGY" \
        base_models="$BASE_MODELS" \
        lab_test_mapping_path="$LAB_TEST_MAPPING" \
        local_logging_dir="$RESULTS_DIR" \
        summarize=True \
        run_descr="$BASELINE_DESCR" \
        || die "Baseline run failed for $PATHOLOGY"

    BASELINE_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"$BASELINE_DESCR"* 2>/dev/null | head -1)
    [ -n "$BASELINE_RUN_DIR" ] || die "Could not find baseline results directory for $PATHOLOGY"
    BASELINE_RUN_DIRS[$PATHOLOGY]="$BASELINE_RUN_DIR"
    echo "Baseline results for $PATHOLOGY: $BASELINE_RUN_DIR"
done

fi  # SKIP_BASELINE

phase_end

# ============================================================
# PHASE 2: Evaluate + Extract Baseline Trajectories (all pathologies)
# ============================================================
phase_start 2 "Evaluate and extract baseline trajectories"

BASELINE_TRAJ_FILES=()

if [ "$SKIP_BASELINE" = true ]; then
    echo "Reusing existing baseline trajectories"
    for PATHOLOGY in "${PATHOLOGIES[@]}"; do
        BASELINE_TRAJ_FILES+=("$TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json")
        echo "  $TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json"
    done
else

for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$PATHOLOGY/${SPLIT}.pkl"
    BASELINE_RUN_DIR="${BASELINE_RUN_DIRS[$PATHOLOGY]}"
    BASELINE_TRAJ="$TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json"

    echo ""
    echo "--- Evaluating baseline: $PATHOLOGY ---"
    python "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "$BASELINE_RUN_DIR" \
        --pathology "$PATHOLOGY" \
        --patient_data "$PATIENT_DATA" \
        || die "Baseline evaluation failed for $PATHOLOGY"

    echo ""
    echo "--- Extracting baseline trajectories: $PATHOLOGY ---"
    python "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "$BASELINE_RUN_DIR" \
        --pathology "$PATHOLOGY" \
        --patient_data "$PATIENT_DATA" \
        --output "$BASELINE_TRAJ" \
        || die "Baseline trajectory extraction failed for $PATHOLOGY"

    BASELINE_TRAJ_FILES+=("$BASELINE_TRAJ")
done

fi  # SKIP_BASELINE

phase_end

# ============================================================
# PHASE 3: Evolver Generates Skill (from ALL pathologies)
# ============================================================
phase_start 3 "Evolver generates skill from all pathologies ($EVOLVER_MODEL)"

SKILL_PATH="$SKILLS_DIR/$VERSION/acute_abdominal_pain.md"

# Build --trajectories argument
# For v2+: use previous EVOLVED trajectories (what the skill actually did)
# For v1:  use baseline trajectories (no previous skill to evaluate)
TRAJ_ARGS=""
if [ -n "$PREV_VERSION" ]; then
    USE_PREV_EVOLVED=true
    for PATHOLOGY in "${PATHOLOGIES[@]}"; do
        PREV_EVOLVED="$TRAJ_DIR/${PREV_VERSION}_${PATHOLOGY}_${SPLIT}10.json"
        if [ ! -f "$PREV_EVOLVED" ]; then
            USE_PREV_EVOLVED=false
            echo "WARNING: Previous evolved trajectory not found: $PREV_EVOLVED"
            echo "         Falling back to baseline trajectories for Evolver"
            break
        fi
    done
    if [ "$USE_PREV_EVOLVED" = true ]; then
        echo "Using ${PREV_VERSION} evolved trajectories for Evolver"
        for PATHOLOGY in "${PATHOLOGIES[@]}"; do
            TRAJ_ARGS="$TRAJ_ARGS $TRAJ_DIR/${PREV_VERSION}_${PATHOLOGY}_${SPLIT}10.json"
        done
    else
        for TRAJ in "${BASELINE_TRAJ_FILES[@]}"; do
            TRAJ_ARGS="$TRAJ_ARGS $TRAJ"
        done
    fi
else
    # v1: use baseline trajectories
    for TRAJ in "${BASELINE_TRAJ_FILES[@]}"; do
        TRAJ_ARGS="$TRAJ_ARGS $TRAJ"
    done
fi

# Build optional --prev-skill argument
PREV_SKILL_ARG=""
if [ -n "$PREV_VERSION" ]; then
    PREV_SKILL="$SKILLS_DIR/$PREV_VERSION/acute_abdominal_pain.md"
    if [ -f "$PREV_SKILL" ]; then
        PREV_SKILL_ARG="--prev-skill $PREV_SKILL"
        echo "Feeding previous skill: $PREV_SKILL"
    else
        echo "WARNING: Previous skill not found at $PREV_SKILL, generating from scratch"
    fi
fi

echo "Running: python $PROJECT_DIR/scripts/evolve_skill.py \\"
echo "  --trajectories $TRAJ_ARGS \\"
echo "  --model $EVOLVER_MODEL \\"
echo "  $PREV_SKILL_ARG \\"
echo "  --output $SKILL_PATH"

python "$PROJECT_DIR/scripts/evolve_skill.py" \
    --trajectories $TRAJ_ARGS \
    --model "$EVOLVER_MODEL" \
    $PREV_SKILL_ARG \
    --output "$SKILL_PATH" \
    || die "Skill evolution failed"

phase_end

# ============================================================
# PHASE 4: Sanitize Skill
# ============================================================
phase_start 4 "Sanitize skill (remove disease name leakage)"

echo "Running: python $PROJECT_DIR/scripts/sanitize_skill.py \\"
echo "  $SKILL_PATH --inplace --report"

python "$PROJECT_DIR/scripts/sanitize_skill.py" \
    "$SKILL_PATH" --inplace --report \
    || die "Skill sanitization failed"

echo ""
echo "Sanitized skill content:"
echo "========================"
cat "$SKILL_PATH"
echo "========================"

phase_end

# ============================================================
# PHASE 5: Evolved Runs (all pathologies, with skill)
# ============================================================
phase_start 5 "Evolved runs with skill (${PATHOLOGIES[*]})"

declare -A EVOLVED_RUN_DIRS

for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    EVOLVED_DESCR="_${VERSION}_${SPLIT}10"

    echo ""
    echo "--- Running evolved: $PATHOLOGY ---"
    echo "Running: cd $FRAMEWORK_DIR && python run.py \\"
    echo "  pathology=$PATHOLOGY model=$MODEL \\"
    echo "  base_mimic=$DATA_DIR/$PATHOLOGY \\"
    echo "  base_models=$BASE_MODELS \\"
    echo "  lab_test_mapping_path=$LAB_TEST_MAPPING \\"
    echo "  local_logging_dir=$RESULTS_DIR \\"
    echo "  summarize=True \\"
    echo "  skill_path=$SKILL_PATH \\"
    echo "  run_descr=$EVOLVED_DESCR"

    cd "$FRAMEWORK_DIR"
    python run.py \
        pathology="$PATHOLOGY" \
        model="$MODEL" \
        base_mimic="$DATA_DIR/$PATHOLOGY" \
        base_models="$BASE_MODELS" \
        lab_test_mapping_path="$LAB_TEST_MAPPING" \
        local_logging_dir="$RESULTS_DIR" \
        summarize=True \
        skill_path="$SKILL_PATH" \
        run_descr="$EVOLVED_DESCR" \
        || die "Evolved run failed for $PATHOLOGY"

    EVOLVED_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"$EVOLVED_DESCR"* 2>/dev/null | head -1)
    [ -n "$EVOLVED_RUN_DIR" ] || die "Could not find evolved results directory for $PATHOLOGY"
    EVOLVED_RUN_DIRS[$PATHOLOGY]="$EVOLVED_RUN_DIR"
    echo "Evolved results for $PATHOLOGY: $EVOLVED_RUN_DIR"
done

phase_end

# ============================================================
# PHASE 6: Evaluate + Extract + Compare (all pathologies)
# ============================================================
phase_start 6 "Evaluate, extract, and compare evolved runs"

for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    PATIENT_DATA="$DATA_DIR/$PATHOLOGY/${SPLIT}.pkl"
    EVOLVED_RUN_DIR="${EVOLVED_RUN_DIRS[$PATHOLOGY]}"
    BASELINE_TRAJ="$TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json"
    EVOLVED_TRAJ="$TRAJ_DIR/${VERSION}_${PATHOLOGY}_${SPLIT}10.json"
    COMPARISON="$COMP_DIR/${VERSION}_vs_baseline_${PATHOLOGY}.md"

    echo ""
    echo "--- Evaluating evolved: $PATHOLOGY ---"
    python "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "$EVOLVED_RUN_DIR" \
        --pathology "$PATHOLOGY" \
        --patient_data "$PATIENT_DATA" \
        || die "Evolved evaluation failed for $PATHOLOGY"

    echo ""
    echo "--- Extracting evolved trajectories: $PATHOLOGY ---"
    python "$PROJECT_DIR/scripts/extract_trajectories.py" \
        --results_dir "$EVOLVED_RUN_DIR" \
        --pathology "$PATHOLOGY" \
        --patient_data "$PATIENT_DATA" \
        --output "$EVOLVED_TRAJ" \
        || die "Evolved trajectory extraction failed for $PATHOLOGY"

    echo ""
    echo "--- Comparing baseline vs evolved: $PATHOLOGY ---"
    python "$PROJECT_DIR/scripts/compare_runs.py" \
        --baseline "$BASELINE_TRAJ" \
        --evolved "$EVOLVED_TRAJ" \
        --output "$COMPARISON" \
        || die "Comparison failed for $PATHOLOGY"
done

phase_end

# ============================================================
# PHASE 7: Final Summary
# ============================================================
phase_start 7 "Final summary"

TOTAL_ELAPSED=$SECONDS
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "============================================================"
echo "MULTI-PATHOLOGY EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Model:        $MODEL"
echo "  Skill:        $VERSION"
echo "  Duration:     ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""
echo "  Per-pathology results:"
for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    echo ""
    echo "  [$PATHOLOGY]"
    echo "    Baseline results:     ${BASELINE_RUN_DIRS[$PATHOLOGY]}"
    echo "    Evolved results:      ${EVOLVED_RUN_DIRS[$PATHOLOGY]}"
    echo "    Baseline trajectories: $TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json"
    echo "    Evolved trajectories:  $TRAJ_DIR/${VERSION}_${PATHOLOGY}_${SPLIT}10.json"
    echo "    Comparison report:     $COMP_DIR/${VERSION}_vs_baseline_${PATHOLOGY}.md"
done
echo ""
echo "  Shared outputs:"
echo "    Skill file:            $SKILL_PATH"
echo "    Full log:              $LOG_FILE"
echo ""
echo "  Performance (baseline -> evolved):"
echo "  ---------------------------------------------------------------------------------------------------------"
printf "  %-16s %-22s %-22s %-22s %-22s\n" "Pathology" "Diagnosis" "PE First" "Labs" "Imaging"
echo "  ---------------------------------------------------------------------------------------------------------"
for PATHOLOGY in "${PATHOLOGIES[@]}"; do
    COMPARISON="$COMP_DIR/${VERSION}_vs_baseline_${PATHOLOGY}.md"
    if [ -f "$COMPARISON" ]; then
        DX=$(grep "Diagnosis accuracy" "$COMPARISON" 2>/dev/null | sed 's/- Diagnosis accuracy: //')
        PE=$(grep "PE first" "$COMPARISON" 2>/dev/null | sed 's/- PE first: //')
        LABS=$(grep "Lab score" "$COMPARISON" 2>/dev/null | sed 's/- Lab score total: //')
        IMG=$(grep "Imaging score" "$COMPARISON" 2>/dev/null | sed 's/- Imaging score total: //')
        printf "  %-16s %-22s %-22s %-22s %-22s\n" "$PATHOLOGY" "$DX" "$PE" "$LABS" "$IMG"
    fi
done
echo "  ---------------------------------------------------------------------------------------------------------"
echo ""
echo "  Next steps:"
echo "    - Review comparisons: ls $COMP_DIR/${VERSION}_vs_baseline_*.md"
echo "    - If improved, run on test split (100 patients per pathology)"
echo "    - If not, iterate: bash scripts/run_experiment_multi.sh v$((V_NUM + 1)) $MODEL"
echo "============================================================"

phase_end
