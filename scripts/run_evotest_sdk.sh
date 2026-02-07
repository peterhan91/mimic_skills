#!/usr/bin/env bash
# ===========================================================================
# EvoTest SDK — Evolutionary skill optimization with Claude Sonnet 4.5
#
# All three agents (Orchestrator, Lab Interpreter, Challenger) use Sonnet.
# The Evolver (skill generator) uses Opus for strongest reasoning.
#
# Usage:
#   bash scripts/run_evotest_sdk.sh              # Full run (10 episodes, all pathologies)
#   bash scripts/run_evotest_sdk.sh --resume     # Resume from checkpoint
#   bash scripts/run_evotest_sdk.sh --dry-run    # Dry run (no API calls)
#
# Any extra arguments are passed through to evotest_loop.py.
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
EPISODES=10
AGENT_MODEL="anthropic/claude-sonnet-4-5-20250929"
EVOLVER_MODEL="claude-opus-4-6"
SPLIT="train"                      # train (10 pts) | test (100 pts) | full
MAX_TURNS=20
ANNOTATE_CLINICAL=true             # true | false

# Pathologies to include (space-separated, or "all" for all 4)
PATHOLOGIES="all"
# PATHOLOGIES="appendicitis"       # Uncomment for fast single-pathology iteration

# Optional: warm-start from an existing skill
INITIAL_SKILL=""
# INITIAL_SKILL="skills/v2/acute_abdominal_pain.md"

# UCB hyperparameters (defaults are usually fine)
EXPLORATION_CONSTANT=1.0
DEPTH_CONSTANT=0.8
DROP_THRESHOLD=1.0

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Check API keys
# ---------------------------------------------------------------------------
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Export it or add to $PROJECT_DIR/.env"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
    python "$PROJECT_DIR/codes_openai_agent/evotest_loop.py"
    --episodes "$EPISODES"
    --litellm-model "$AGENT_MODEL"
    --evolver-model "$EVOLVER_MODEL"
    --split "$SPLIT"
    --max-turns "$MAX_TURNS"
    --exploration-constant "$EXPLORATION_CONSTANT"
    --depth-constant "$DEPTH_CONSTANT"
    --drop-threshold "$DROP_THRESHOLD"
)

if [ "$ANNOTATE_CLINICAL" = "true" ]; then
    CMD+=(--annotate-clinical)
else
    CMD+=(--no-annotate-clinical)
fi

if [ "$PATHOLOGIES" != "all" ]; then
    CMD+=(--pathologies $PATHOLOGIES)
fi

if [ -n "$INITIAL_SKILL" ]; then
    CMD+=(--initial-skill "$PROJECT_DIR/$INITIAL_SKILL")
fi

# Pass through any extra arguments (e.g. --resume, --dry-run)
CMD+=("$@")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "==========================================================="
echo "EvoTest SDK"
echo "==========================================================="
echo "  Agent model:   $AGENT_MODEL"
echo "  Evolver model: $EVOLVER_MODEL"
echo "  Episodes:      $EPISODES"
echo "  Split:         $SPLIT"
echo "  Pathologies:   $PATHOLOGIES"
echo "  Max turns:     $MAX_TURNS"
echo "  Annotate:      $ANNOTATE_CLINICAL"
if [ -n "$INITIAL_SKILL" ]; then
    echo "  Initial skill: $INITIAL_SKILL"
fi
echo "  Extra args:    $*"
echo "==========================================================="
echo ""

exec "${CMD[@]}"
