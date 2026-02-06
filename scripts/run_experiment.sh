#!/bin/bash
set -euo pipefail

# ============================================================
# run_experiment.sh — One full evolution cycle
#
# Pipeline: Baseline run -> Evaluate -> Extract trajectories
#           -> Evolver generates skill -> Sanitize -> Run with
#           skill -> Evaluate -> Extract -> Compare
#
# Usage:
#   bash scripts/run_experiment.sh [PATHOLOGY] [VERSION] [EVOLVER_MODEL] [ANNOTATE_CLINICAL]
#
# Examples:
#   bash scripts/run_experiment.sh cholecystitis v1
#   bash scripts/run_experiment.sh appendicitis v2 claude-opus-4-6
#   bash scripts/run_experiment.sh appendicitis v1 claude-opus-4-6 False  # disable clinical annotations
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================
PATHOLOGY="${1:-cholecystitis}"
VERSION="${2:-v1}"
EVOLVER_MODEL="${3:-claude-opus-4-6}"
ANNOTATE_CLINICAL="${4:-True}"
SPLIT="train"
MODEL="MedGemma4B"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_DIR="$PROJECT_DIR/codes_Hager/MIMIC-Clinical-Decision-Making-Framework"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"
TRAJ_DIR="$PROJECT_DIR/trajectories"
SKILLS_DIR="$PROJECT_DIR/skills"
COMP_DIR="$PROJECT_DIR/comparisons"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/experiment_${PATHOLOGY}_${VERSION}_${TIMESTAMP}.log"

# Base models cache (HuggingFace downloads here)
BASE_MODELS="${HF_HOME:-${HOME}/.cache/huggingface/hub}"
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"

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

echo "Experiment configuration:"
echo "  PATHOLOGY:         $PATHOLOGY"
echo "  VERSION:           $VERSION"
echo "  EVOLVER_MODEL:     $EVOLVER_MODEL"
echo "  ANNOTATE_CLINICAL: $ANNOTATE_CLINICAL"
echo "  SPLIT:             $SPLIT"
echo "  MODEL:             $MODEL"
echo "  PROJECT_DIR:       $PROJECT_DIR"
echo "  BASE_MODELS:       $BASE_MODELS"
echo "  LOG_FILE:          $LOG_FILE"
echo ""

# Check prerequisites
PATIENT_DATA="$DATA_DIR/$PATHOLOGY/${SPLIT}.pkl"
HAGER_DATA="$DATA_DIR/$PATHOLOGY/${PATHOLOGY}_hadm_info_first_diag.pkl"

[ -f "$PATIENT_DATA" ] || die "Patient data not found: $PATIENT_DATA"
[ -f "$HAGER_DATA" ] || die "Hager-format data not found: $HAGER_DATA (run scripts/prepare_split_for_hager.py first)"
[ -f "$LAB_TEST_MAPPING" ] || die "Lab test mapping not found: $LAB_TEST_MAPPING"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set — Phase 4 (Evolver) will fail"
fi

phase_end

# ============================================================
# PHASE 1: Baseline Run
# ============================================================
phase_start 1 "Baseline run ($PATHOLOGY, $MODEL, $SPLIT)"

BASELINE_DESCR="_baseline_${SPLIT}10"

echo "Running: cd $FRAMEWORK_DIR && python run.py \\"
echo "  pathology=$PATHOLOGY model=$MODEL \\"
echo "  base_mimic=$DATA_DIR/$PATHOLOGY \\"
echo "  base_models=$BASE_MODELS \\"
echo "  lab_test_mapping_path=$LAB_TEST_MAPPING \\"
echo "  local_logging_dir=$RESULTS_DIR \\"
echo "  summarize=True \\"
echo "  annotate_clinical=$ANNOTATE_CLINICAL \\"
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
    annotate_clinical="$ANNOTATE_CLINICAL" \
    run_descr="$BASELINE_DESCR" \
    || die "Baseline run failed"

# Find the results directory (most recent matching pattern)
BASELINE_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"$BASELINE_DESCR"* 2>/dev/null | head -1)
[ -n "$BASELINE_RUN_DIR" ] || die "Could not find baseline results directory"
echo "Baseline results: $BASELINE_RUN_DIR"

phase_end

# ============================================================
# PHASE 2: Evaluate Baseline
# ============================================================
phase_start 2 "Evaluate baseline"

echo "Running: python $PROJECT_DIR/scripts/evaluate_run.py \\"
echo "  --results_dir $BASELINE_RUN_DIR \\"
echo "  --pathology $PATHOLOGY \\"
echo "  --patient_data $PATIENT_DATA"

python "$PROJECT_DIR/scripts/evaluate_run.py" \
    --results_dir "$BASELINE_RUN_DIR" \
    --pathology "$PATHOLOGY" \
    --patient_data "$PATIENT_DATA" \
    || die "Baseline evaluation failed"

phase_end

# ============================================================
# PHASE 3: Extract Baseline Trajectories
# ============================================================
phase_start 3 "Extract baseline trajectories"

BASELINE_TRAJ="$TRAJ_DIR/baseline_${PATHOLOGY}_${SPLIT}10.json"

echo "Running: python $PROJECT_DIR/scripts/extract_trajectories.py \\"
echo "  --results_dir $BASELINE_RUN_DIR \\"
echo "  --pathology $PATHOLOGY \\"
echo "  --patient_data $PATIENT_DATA \\"
echo "  --output $BASELINE_TRAJ"

python "$PROJECT_DIR/scripts/extract_trajectories.py" \
    --results_dir "$BASELINE_RUN_DIR" \
    --pathology "$PATHOLOGY" \
    --patient_data "$PATIENT_DATA" \
    --output "$BASELINE_TRAJ" \
    || die "Baseline trajectory extraction failed"

phase_end

# ============================================================
# PHASE 4: Evolver Generates Skill
# ============================================================
phase_start 4 "Evolver generates skill ($EVOLVER_MODEL)"

SKILL_PATH="$SKILLS_DIR/$VERSION/acute_abdominal_pain.md"

echo "Running: python $PROJECT_DIR/scripts/evolve_skill.py \\"
echo "  --trajectories $BASELINE_TRAJ \\"
echo "  --model $EVOLVER_MODEL \\"
echo "  --output $SKILL_PATH"

python "$PROJECT_DIR/scripts/evolve_skill.py" \
    --trajectories "$BASELINE_TRAJ" \
    --model "$EVOLVER_MODEL" \
    --output "$SKILL_PATH" \
    || die "Skill evolution failed"

phase_end

# ============================================================
# PHASE 5: Sanitize Skill
# ============================================================
phase_start 5 "Sanitize skill (remove disease name leakage)"

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
# PHASE 6: Run with Evolved Skill
# ============================================================
phase_start 6 "Run with evolved skill ($VERSION)"

EVOLVED_DESCR="_${VERSION}_${SPLIT}10"

echo "Running: cd $FRAMEWORK_DIR && python run.py \\"
echo "  pathology=$PATHOLOGY model=$MODEL \\"
echo "  base_mimic=$DATA_DIR/$PATHOLOGY \\"
echo "  base_models=$BASE_MODELS \\"
echo "  lab_test_mapping_path=$LAB_TEST_MAPPING \\"
echo "  local_logging_dir=$RESULTS_DIR \\"
echo "  summarize=True \\"
echo "  annotate_clinical=$ANNOTATE_CLINICAL \\"
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
    annotate_clinical="$ANNOTATE_CLINICAL" \
    skill_path="$SKILL_PATH" \
    run_descr="$EVOLVED_DESCR" \
    || die "Evolved run failed"

# Find the evolved results directory
EVOLVED_RUN_DIR=$(ls -td "$RESULTS_DIR"/*"$PATHOLOGY"*"$EVOLVED_DESCR"* 2>/dev/null | head -1)
[ -n "$EVOLVED_RUN_DIR" ] || die "Could not find evolved results directory"
echo "Evolved results: $EVOLVED_RUN_DIR"

phase_end

# ============================================================
# PHASE 7: Evaluate Evolved Run
# ============================================================
phase_start 7 "Evaluate evolved run"

echo "Running: python $PROJECT_DIR/scripts/evaluate_run.py \\"
echo "  --results_dir $EVOLVED_RUN_DIR \\"
echo "  --pathology $PATHOLOGY \\"
echo "  --patient_data $PATIENT_DATA"

python "$PROJECT_DIR/scripts/evaluate_run.py" \
    --results_dir "$EVOLVED_RUN_DIR" \
    --pathology "$PATHOLOGY" \
    --patient_data "$PATIENT_DATA" \
    || die "Evolved evaluation failed"

phase_end

# ============================================================
# PHASE 8: Extract Evolved Trajectories + Compare
# ============================================================
phase_start 8 "Extract evolved trajectories and compare"

EVOLVED_TRAJ="$TRAJ_DIR/${VERSION}_${PATHOLOGY}_${SPLIT}10.json"
COMPARISON="$COMP_DIR/${VERSION}_vs_baseline_${PATHOLOGY}.md"

echo "Extracting evolved trajectories..."
python "$PROJECT_DIR/scripts/extract_trajectories.py" \
    --results_dir "$EVOLVED_RUN_DIR" \
    --pathology "$PATHOLOGY" \
    --patient_data "$PATIENT_DATA" \
    --output "$EVOLVED_TRAJ" \
    || die "Evolved trajectory extraction failed"

echo ""
echo "Comparing baseline vs evolved..."
python "$PROJECT_DIR/scripts/compare_runs.py" \
    --baseline "$BASELINE_TRAJ" \
    --evolved "$EVOLVED_TRAJ" \
    --output "$COMPARISON" \
    || die "Comparison failed"

phase_end

# ============================================================
# PHASE 9: Final Summary
# ============================================================
phase_start 9 "Final summary"

TOTAL_ELAPSED=$SECONDS
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Pathology:    $PATHOLOGY"
echo "  Model:        $MODEL"
echo "  Skill:        $VERSION"
echo "  Duration:     ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""
echo "  Outputs:"
echo "    Baseline results:     $BASELINE_RUN_DIR"
echo "    Evolved results:      $EVOLVED_RUN_DIR"
echo "    Baseline trajectories: $BASELINE_TRAJ"
echo "    Evolved trajectories:  $EVOLVED_TRAJ"
echo "    Skill file:            $SKILL_PATH"
echo "    Comparison report:     $COMPARISON"
echo "    Full log:              $LOG_FILE"
echo ""
echo "  Next steps:"
echo "    - Review the comparison: cat $COMPARISON"
echo "    - If improved, run on test split (100 patients)"
echo "    - If not, iterate: bash scripts/run_experiment.sh $PATHOLOGY v2"
echo "============================================================"

phase_end
