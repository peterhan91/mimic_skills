#!/bin/bash
set -euo pipefail

# ============================================================
# evotest_full.sh — Train on 4×10, test best skill on 7×100
#
# Usage:
#   bash scripts/evotest_full.sh [FLAGS] [EPISODES] [MODEL] [EVOLVER_MODEL] [ANNOTATE_CLINICAL]
#
# Flags (before positional args):
#   --agent ToT|ZeroShot     Agent type (default: ZeroShot)
#   --patient-sim            Enable patient simulator
#   --resume                 Resume from saved state
#   --tot-max-depth N        ToT max search depth (recommend 15 for patsim)
#   --tot-breadth N          ToT frontier size
#   --tot-n-generate N       ToT candidates per step
#   --tot-temperature F      ToT generation temperature
#
# Examples:
#   bash scripts/evotest_full.sh --agent ToT 10 Qwen3_30B_A3B
#   bash scripts/evotest_full.sh --agent ToT --patient-sim --tot-max-depth 15 10
#   bash scripts/evotest_full.sh --resume 10 Qwen3_30B_A3B
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse flags (--resume, --agent, --patient-sim, --tot-*)
RESUME_FLAG=()
AGENT_FLAG=()
PATSIM_FLAG=()
TOT_FLAGS=()
AGENT="ZeroShot"
PATIENT_SIMULATOR="False"
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --resume)
            RESUME_FLAG=(--resume); shift ;;
        --agent)
            AGENT="${2:?--agent requires a value (ZeroShot or ToT)}"
            AGENT_FLAG=(--agent "$AGENT"); shift 2 ;;
        --patient-sim)
            PATIENT_SIMULATOR="True"
            PATSIM_FLAG=(--patient-sim); shift ;;
        --tot-max-depth|--tot-breadth|--tot-n-generate|--tot-temperature)
            TOT_FLAGS+=("$1" "${2:?$1 requires a value}"); shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Agent × patient-sim → 2×2 matrix of parallel experiment dirs
if [ "$AGENT" = "ToT" ] && [ "$PATIENT_SIMULATOR" = "True" ]; then
    STATE_FILE="$PROJECT_DIR/evotest_state_tot_patsim/state.json"
    SKILLS_SUBDIR="skills/evo_tot_patsim"
elif [ "$AGENT" = "ToT" ]; then
    STATE_FILE="$PROJECT_DIR/evotest_state_tot/state.json"
    SKILLS_SUBDIR="skills/evo_tot"
elif [ "$PATIENT_SIMULATOR" = "True" ]; then
    STATE_FILE="$PROJECT_DIR/evotest_state_patsim/state.json"
    SKILLS_SUBDIR="skills/evo_patsim"
else
    STATE_FILE="$PROJECT_DIR/evotest_state/state.json"
    SKILLS_SUBDIR="skills/evo"
fi

EPISODES="${1:-10}"
MODEL="${2:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${3:-claude-opus-4-6}"
ANNOTATE_CLINICAL="${4:-True}"

echo ""
echo "==========================================================="
echo " FULL EXPERIMENT: Train 4×10 → Best Skill → Test 7×100"
echo "==========================================================="
echo "  Agent:     $AGENT"
echo "  Episodes:  $EPISODES"
echo "  Model:     $MODEL"
echo "  Evolver:   $EVOLVER_MODEL"
echo "  Pat. Sim:  $PATIENT_SIMULATOR"
echo ""

# ============================================================
# STEP 1: EvoTest training (4 pathologies × 10 patients)
# ============================================================
echo "########### STEP 1: EvoTest Training (4×10) ###########"
echo ""

bash "$PROJECT_DIR/scripts/evotest_train.sh" \
    "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}" \
    "${AGENT_FLAG[@]+"${AGENT_FLAG[@]}"}" \
    "${PATSIM_FLAG[@]+"${PATSIM_FLAG[@]}"}" \
    "${TOT_FLAGS[@]+"${TOT_FLAGS[@]}"}" \
    "$EPISODES" "$MODEL" "$EVOLVER_MODEL" "$ANNOTATE_CLINICAL"

# ============================================================
# STEP 2: Extract best skill from state.json
# ============================================================
echo ""
echo "########### STEP 2: Identify Best Skill ###########"
echo ""

[ -f "$STATE_FILE" ] || { echo "ERROR: $STATE_FILE not found"; exit 1; }

BEST_SKILL=$(python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    state = json.load(f)
# Pick the node with highest diagnosis accuracy (not composite score)
best_idx, best_dx = None, -1.0
for i, node in enumerate(state['nodes']):
    dx = node.get('per_metric', {}).get('Diagnosis', 0)
    if dx > best_dx:
        best_dx = dx
        best_idx = i
if best_idx is None:
    print('ERROR: No nodes in state', file=sys.stderr); sys.exit(1)
ep = state['nodes'][best_idx]['episode_num']
composite = state['nodes'][best_idx]['score']
print(f'$SKILLS_SUBDIR/episode_{ep}.md')
print(f'  Episode {ep}, Dx accuracy {best_dx:.3f}, composite {composite:.3f}', file=sys.stderr)
")

BEST_SKILL="$PROJECT_DIR/$BEST_SKILL"
[ -f "$BEST_SKILL" ] || { echo "ERROR: $BEST_SKILL not found"; exit 1; }
echo "  Best skill: $BEST_SKILL"
echo ""

# ============================================================
# STEP 3: Test on all 7 pathologies × 100 patients
# ============================================================
echo "########### STEP 3: Test Evaluation (7×100) ###########"
echo ""

bash "$PROJECT_DIR/scripts/evotest_test.sh" \
    "${AGENT_FLAG[@]+"${AGENT_FLAG[@]}"}" \
    "${PATSIM_FLAG[@]+"${PATSIM_FLAG[@]}"}" \
    "${TOT_FLAGS[@]+"${TOT_FLAGS[@]}"}" \
    "$BEST_SKILL" "$MODEL" "$ANNOTATE_CLINICAL"

echo ""
echo "==========================================================="
echo " FULL EXPERIMENT COMPLETE"
echo "==========================================================="
