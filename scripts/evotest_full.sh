#!/bin/bash
set -euo pipefail

# ============================================================
# evotest_full.sh — Train on 4×10, test best skill on 7×100
#
# Usage:
#   bash scripts/evotest_full.sh [EPISODES] [MODEL] [EVOLVER_MODEL] [ANNOTATE_CLINICAL]
#
# Resume after interruption (skips to test if training is done):
#   bash scripts/evotest_full.sh --resume [EPISODES] [MODEL] ...
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_FILE="$PROJECT_DIR/evotest_state/state.json"

# Parse --resume flag
RESUME_FLAG=()
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG=(--resume)
    shift
fi

EPISODES="${1:-10}"
MODEL="${2:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${3:-claude-opus-4-6}"
ANNOTATE_CLINICAL="${4:-True}"

echo ""
echo "==========================================================="
echo " FULL EXPERIMENT: Train 4×10 → Best Skill → Test 7×100"
echo "==========================================================="
echo "  Episodes:  $EPISODES"
echo "  Model:     $MODEL"
echo "  Evolver:   $EVOLVER_MODEL"
echo ""

# ============================================================
# STEP 1: EvoTest training (4 pathologies × 10 patients)
# ============================================================
echo "########### STEP 1: EvoTest Training (4×10) ###########"
echo ""

bash "$PROJECT_DIR/scripts/evotest_train.sh" \
    "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}" \
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
print(f'skills/evo/episode_{ep}.md')
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
    "$BEST_SKILL" "$MODEL" "$ANNOTATE_CLINICAL"

echo ""
echo "==========================================================="
echo " FULL EXPERIMENT COMPLETE"
echo "==========================================================="
