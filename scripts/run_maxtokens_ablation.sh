#!/bin/bash
set -euo pipefail

# ============================================================
# Max tokens ablation: run evotest_full with VLLM_MAX_TOKENS=1024 and 2048
# for both abdominal and chest conditions.
#
# Uses --resume so completed training is skipped (only missing test
# phases run).  Safe to re-run after a partial failure.
#
# Results saved to isolated directories via --experiment flag:
#   skills/evo_patsim_mt1024/        skills/evo_patsim_mt2048/
#   skills/evo_patsim_chest_mt1024/  skills/evo_patsim_chest_mt2048/
#   evotest_state_patsim_mt1024/     evotest_state_patsim_mt2048/
#   evotest_state_patsim_chest_mt1024/ ...
#
# Usage:
#   nohup bash scripts/run_maxtokens_ablation.sh > logs/maxtokens_ablation.log 2>&1 &
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

EPISODES=10
MODEL=Qwen3_5_35B_A3B

# Detect Python (for fixing missing evals)
source "$(dirname "$0")/_detect_python.sh"

echo "============================================================"
echo " MAX TOKENS ABLATION STUDY"
echo "============================================================"
echo ""
echo "  Will run 4 experiments sequentially (--resume skips done steps):"
echo "    1. Abdominal, VLLM_MAX_TOKENS=1024  (--experiment mt1024)"
echo "    2. Abdominal, VLLM_MAX_TOKENS=2048  (--experiment mt2048)"
echo "    3. Chest,     VLLM_MAX_TOKENS=1024  (--experiment mt1024)"
echo "    4. Chest,     VLLM_MAX_TOKENS=2048  (--experiment mt2048)"
echo ""
echo "  Each: ${EPISODES} episodes train → best skill → test (100/pathology)"
echo "  Model: $MODEL"
echo ""
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ============================================================
# Pre-flight: fix any missing eval pkls from previous partial runs
# ============================================================
echo "--- Pre-flight: checking for missing eval pkls ---"
DATA_DIR="$PROJECT_DIR/data_splits"
RESULTS_DIR="$PROJECT_DIR/results"

for RESULTS_SUBDIR in "$RESULTS_DIR"/*test100*/; do
    [ -d "$RESULTS_SUBDIR" ] || continue
    # Skip if eval pkl already exists
    if ls "$RESULTS_SUBDIR"/*_eval.pkl &>/dev/null; then
        continue
    fi
    # Skip if no results pkl
    if ! ls "$RESULTS_SUBDIR"/*_results.pkl &>/dev/null; then
        continue
    fi
    # Extract pathology name
    DIRNAME=$(basename "$RESULTS_SUBDIR")
    PATHOLOGY="${DIRNAME%%_ZeroShot_*}"
    PATIENT_DATA="$DATA_DIR/$PATHOLOGY/test.pkl"
    if [ -f "$PATIENT_DATA" ]; then
        echo "  Running missing evaluation: $PATHOLOGY ($(basename "$RESULTS_SUBDIR"))"
        "$PYTHON" "$PROJECT_DIR/scripts/evaluate_run.py" \
            --results_dir "$RESULTS_SUBDIR" \
            --pathology "$PATHOLOGY" \
            --patient_data "$PATIENT_DATA" || true
    fi
done
echo "--- Pre-flight complete ---"
echo ""

# --- 1. Abdominal, max_tokens=1024 ---
echo ""
echo "############ [1/4] Abdominal, max_tokens=1024 ############"
echo ""
VLLM_MAX_TOKENS=1024 bash scripts/evotest_full.sh \
    --resume --experiment mt1024 \
    "$EPISODES" "$MODEL" 2>&1 | tee logs/evotest_full_abdominal_mt1024.log

# --- 2. Abdominal, max_tokens=2048 ---
echo ""
echo "############ [2/4] Abdominal, max_tokens=2048 ############"
echo ""
VLLM_MAX_TOKENS=2048 bash scripts/evotest_full.sh \
    --resume --experiment mt2048 \
    "$EPISODES" "$MODEL" 2>&1 | tee logs/evotest_full_abdominal_mt2048.log

# --- 3. Chest, max_tokens=1024 ---
echo ""
echo "############ [3/4] Chest, max_tokens=1024 ############"
echo ""
VLLM_MAX_TOKENS=1024 bash scripts/evotest_full.sh \
    --resume --condition chest --experiment mt1024 \
    "$EPISODES" "$MODEL" 2>&1 | tee logs/evotest_full_chest_mt1024.log

# --- 4. Chest, max_tokens=2048 ---
echo ""
echo "############ [4/4] Chest, max_tokens=2048 ############"
echo ""
VLLM_MAX_TOKENS=2048 bash scripts/evotest_full.sh \
    --resume --condition chest --experiment mt2048 \
    "$EPISODES" "$MODEL" 2>&1 | tee logs/evotest_full_chest_mt2048.log

echo ""
echo "============================================================"
echo " MAX TOKENS ABLATION COMPLETE"
echo "============================================================"
echo "  Finished: $(date)"
echo ""
echo "  Results (3 configs × 12 pathologies = 36 baseline + 36 skill):"
echo "    Original (4096): evotest_state_patsim/ + evotest_state_patsim_chest/"
echo "    mt1024:          evotest_state_patsim_mt1024/ + evotest_state_patsim_chest_mt1024/"
echo "    mt2048:          evotest_state_patsim_mt2048/ + evotest_state_patsim_chest_mt2048/"
echo ""
echo "  Skills:"
echo "    Original (4096): skills/evo_patsim/ + skills/evo_patsim_chest/"
echo "    mt1024:          skills/evo_patsim_mt1024/ + skills/evo_patsim_chest_mt1024/"
echo "    mt2048:          skills/evo_patsim_mt2048/ + skills/evo_patsim_chest_mt2048/"
echo "============================================================"
