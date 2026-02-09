#!/bin/bash
set -euo pipefail

# ============================================================
# evotest_train.sh — Automated EvoTest evolutionary optimization
#
# Uses evotest_clinical.py (UCB tree exploration + regression
# protection) instead of a linear v1→v2→v3 chain.
#
# Features:
#   - UCB tree: selects best parent node, can branch and backtrack
#   - Regression protection: force-best-after-drop
#   - Negative memory: shows failed skills to the Evolver
#   - Resumable: --resume continues from evotest_state/state.json
#
# Usage:
#   bash scripts/evotest_train.sh [EPISODES] [MODEL] [EVOLVER_MODEL] [ANNOTATE_CLINICAL] [INITIAL_SKILL]
#
# Examples:
#   bash scripts/evotest_train.sh                                    # 10 episodes, defaults
#   bash scripts/evotest_train.sh 15 Qwen3_30B_A3B                  # 15 episodes
#   bash scripts/evotest_train.sh 10 Qwen3_30B_A3B claude-opus-4-6 True
#   bash scripts/evotest_train.sh 10 Qwen3_30B_A3B claude-opus-4-6 True skills/v2/acute_abdominal_pain.md
#
# Tree of Thoughts (parallel with ZeroShot, separate state):
#   bash scripts/evotest_train.sh --agent ToT 10 Qwen3_30B_A3B
#
# Resume after interruption:
#   bash scripts/evotest_train.sh --resume [EPISODES]
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Parse arguments (flags first, then positional)
# ============================================================
RESUME=false
AGENT="ZeroShot"
while [ "${1:-}" = "--resume" ] || [ "${1:-}" = "--agent" ]; do
    if [ "$1" = "--resume" ]; then
        RESUME=true
        shift
    elif [ "$1" = "--agent" ]; then
        AGENT="${2:?--agent requires a value (ZeroShot or ToT)}"
        shift 2
    fi
done

EPISODES="${1:-10}"
MODEL="${2:-Qwen3_30B_A3B}"
EVOLVER_MODEL="${3:-claude-opus-4-6}"
ANNOTATE_CLINICAL="${4:-True}"
INITIAL_SKILL="${5:-}"

# Agent-derived paths (must be after flag parsing)
if [ "$AGENT" = "ToT" ]; then
    SKILLS_DIR="$PROJECT_DIR/skills/evo_tot"
    STATE_FILE="$PROJECT_DIR/evotest_state_tot/state.json"
    RUN_PREFIX="tot"
else
    SKILLS_DIR="$PROJECT_DIR/skills/evo"
    STATE_FILE="$PROJECT_DIR/evotest_state/state.json"
    RUN_PREFIX="evo"
fi

# Train on 4 original pathologies; test on all 7 (via evotest_test.sh)
TRAIN_PATHOLOGIES=(appendicitis cholecystitis diverticulitis pancreatitis)

LOG_FILE="$LOG_DIR/evotest_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# ============================================================
# Prerequisites
# ============================================================
echo "============================================================"
echo "EVOTEST EVOLUTIONARY OPTIMIZATION"
echo "============================================================"
echo ""
echo "  Agent:             $AGENT"
echo "  Episodes:          $EPISODES"
echo "  Model:             $MODEL"
echo "  Evolver:           $EVOLVER_MODEL"
echo "  Annotate Clinical: $ANNOTATE_CLINICAL"
echo "  Resume:            $RESUME"
if [ -n "$INITIAL_SKILL" ]; then
echo "  Initial Skill:     $INITIAL_SKILL"
fi
echo "  Log:               $LOG_FILE"
echo ""

# Load .env if present
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Check ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it."
    exit 1
fi

# Check data prerequisites
LAB_TEST_MAPPING="$PROJECT_DIR/MIMIC-CDM-IV/lab_test_mapping.pkl"
[ -f "$LAB_TEST_MAPPING" ] || { echo "ERROR: Lab test mapping not found: $LAB_TEST_MAPPING"; exit 1; }

for PATHOLOGY in "${TRAIN_PATHOLOGIES[@]}"; do
    DATA="$PROJECT_DIR/data_splits/$PATHOLOGY/train.pkl"
    HAGER="$PROJECT_DIR/data_splits/$PATHOLOGY/${PATHOLOGY}_hadm_info_first_diag.pkl"
    [ -f "$DATA" ] || { echo "ERROR: Patient data not found: $DATA"; exit 1; }
    # Always copy train.pkl as _hadm_info_first_diag.pkl to prevent stale test data
    cp "$DATA" "$HAGER"
done

echo "  All prerequisites OK"
echo ""

# ============================================================
# Build evotest command
# ============================================================
EVOTEST_CMD=(
    python "$PROJECT_DIR/scripts/evotest_clinical.py"
    --episodes "$EPISODES"
    --model "$MODEL"
    --evolver-model "$EVOLVER_MODEL"
    --annotate-clinical "$ANNOTATE_CLINICAL"
    --agent "$AGENT"
    --pathologies "${TRAIN_PATHOLOGIES[@]}"
)

if [ "$RESUME" = true ]; then
    EVOTEST_CMD+=(--resume)
    if [ -f "$STATE_FILE" ]; then
        echo "  Resuming from: $STATE_FILE"
    else
        echo "  WARNING: No state file found at $STATE_FILE — starting fresh"
    fi
fi

if [ -n "$INITIAL_SKILL" ]; then
    if [ -f "$INITIAL_SKILL" ]; then
        EVOTEST_CMD+=(--initial-skill "$INITIAL_SKILL")
    else
        echo "WARNING: Initial skill not found: $INITIAL_SKILL — starting without seed"
    fi
fi

# Auto-detect baseline trajectories for caching
# Skip if resuming (state already has ep0) or if initial skill is set (cached
# baseline may have been run with a different/no skill)
TRAJ_DIR="$PROJECT_DIR/trajectories"
if [ "$RESUME" = false ] && [ -z "$INITIAL_SKILL" ] && [ -d "$TRAJ_DIR" ]; then
    BASELINE_OK=true
    for PATHOLOGY in "${TRAIN_PATHOLOGIES[@]}"; do
        if [ ! -f "$TRAJ_DIR/${RUN_PREFIX}_ep0_${PATHOLOGY}.json" ]; then
            BASELINE_OK=false
            break
        fi
    done
    if [ "$BASELINE_OK" = true ]; then
        echo "  Found existing baseline trajectories in $TRAJ_DIR — reusing"
        EVOTEST_CMD+=(--reuse-baseline "$TRAJ_DIR")
    fi
fi

# ============================================================
# Run EvoTest
# ============================================================
TOTAL_START=$SECONDS

echo "============================================================"
echo "  Starting EvoTest ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "  Command: ${EVOTEST_CMD[*]}"
echo "============================================================"
echo ""

# Tee output to log file
"${EVOTEST_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EVOTEST_EXIT=${PIPESTATUS[0]}

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "EVOTEST COMPLETE"
echo "============================================================"
echo ""
echo "  Exit code:   $EVOTEST_EXIT"
echo "  Duration:    ${TOTAL_HR}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Log:         $LOG_FILE"
echo ""

# ============================================================
# Summary: list generated skills with sizes
# ============================================================
if [ -d "$SKILLS_DIR" ]; then
    echo "  Generated skills:"
    for SKILL in "$SKILLS_DIR"/episode_*.md; do
        [ -f "$SKILL" ] || continue
        # Skip raw files
        [[ "$SKILL" == *_raw.md ]] && continue
        BASENAME=$(basename "$SKILL")
        WORDS=$(wc -w < "$SKILL")
        BYTES=$(wc -c < "$SKILL")
        echo "    $BASENAME: ${WORDS} words, ${BYTES} bytes"
    done
    echo ""
fi

# ============================================================
# Summary: print state.json best node info
# ============================================================
if [ -f "$STATE_FILE" ]; then
    BEST_IDX=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
best_idx = state.get('best_node_idx')
if best_idx is not None:
    node = state['nodes'][best_idx]
    print(f\"  Best node:     {best_idx} (episode {node['episode_num']})\")
    print(f\"  Best score:    {node['score']:.3f}\")
    pp = node.get('per_pathology', {})
    if pp:
        print(f\"  Per-pathology:\")
        for p, s in pp.items():
            print(f\"    {p:20s}: {s:.3f}\")
    best_skill = f'$SKILLS_DIR/episode_{node[\"episode_num\"]}.md'
    print(f\"  Best skill:    {best_skill}\")
    print()
    # Tree structure
    print(f\"  Tree ({len(state['nodes'])} nodes):\")
    for n in state['nodes']:
        indent = '  ' * n['depth']
        marker = ' ***BEST***' if n['idx'] == best_idx else ''
        print(f\"    {indent}ep{n['episode_num']} score={n['score']:.3f} depth={n['depth']}{marker}\")
else:
    print('  No best node found')
" 2>/dev/null || echo "  (Could not parse state file)")
    echo ""
fi

# ============================================================
# Next steps
# ============================================================
echo "  Next steps:"
echo "    - Review best skill: cat $SKILLS_DIR/episode_<N>.md"
echo "    - Continue evolving: bash scripts/evotest_train.sh --resume $((EPISODES + 5))"
echo "    - Run final eval on test set (100 patients per pathology):"
echo "      BEST_SKILL=$SKILLS_DIR/episode_<N>.md"
echo "      for P in appendicitis cholecystitis diverticulitis pancreatitis; do"
echo "        cp data_splits/\$P/test.pkl data_splits/\$P/\${P}_hadm_info_first_diag.pkl"
echo "        cd codes_Hager/MIMIC-Clinical-Decision-Making-Framework"
echo "        python run.py pathology=\$P model=$MODEL summarize=True \\"
echo "          annotate_clinical=$ANNOTATE_CLINICAL skill_path=../../\$BEST_SKILL \\"
echo "          run_descr=_evotest_best_test100"
echo "        cd ../.."
echo "      done"
echo "============================================================"

exit $EVOTEST_EXIT
