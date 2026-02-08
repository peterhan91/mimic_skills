#!/bin/bash
set -euo pipefail

# ============================================================
# run_in_container.sh — Run experiments inside Apptainer
#
# Requires vLLM to be running already (start with start_vllm.sh).
#
# Usage:
#   bash run_in_container.sh evotest 10                   # EvoTest 10 episodes
#   bash run_in_container.sh evotest --resume 15          # resume EvoTest
#   bash run_in_container.sh multi v1 vLLM_Qwen3         # multi-pathology
#   bash run_in_container.sh single cholecystitis v1      # single pathology
#   bash run_in_container.sh evotest-sdk 10               # EvoTest SDK
#   bash run_in_container.sh shell                        # interactive shell
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================
EXPERIMENT="${1:-multi}"
shift || true

SIF="/home/than/images/vllm_26.01-py3.sif"
OVERLAY="/home/than/images/overlay.img"
HF_CACHE="/home/than/.cache/huggingface"
PROJECT="/home/than/DeepLearning/mimic_skills"

VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# ============================================================
# Build experiment command based on mode
# ============================================================
case "$EXPERIMENT" in
    multi)
        VERSION="${1:-v1}"
        MODEL="${2:-vLLM_Qwen3}"
        EVOLVER="${3:-claude-opus-4-6}"
        ANNOTATE="${4:-True}"
        EXP_CMD="bash /workspace/scripts/run_experiment_multi.sh $VERSION $MODEL $EVOLVER $ANNOTATE"
        DESC="Multi-pathology $VERSION ($MODEL)"
        ;;
    single)
        PATHOLOGY="${1:-cholecystitis}"
        VERSION="${2:-v1}"
        EVOLVER="${3:-claude-opus-4-6}"
        ANNOTATE="${4:-True}"
        EXP_CMD="bash /workspace/scripts/run_experiment.sh $PATHOLOGY $VERSION $EVOLVER $ANNOTATE"
        DESC="Single-pathology $PATHOLOGY $VERSION"
        ;;
    evotest)
        EXP_CMD="bash /workspace/scripts/run_iterations.sh $*"
        DESC="EvoTest Hager ($*)"
        ;;
    evotest-sdk)
        EPISODES="${1:-10}"
        shift || true
        EXP_CMD="python /workspace/codes_openai_agent/evotest_loop.py --episodes $EPISODES --litellm-model openai/$VLLM_MODEL --litellm-base-url http://localhost:$VLLM_PORT/v1 --evolver-model claude-opus-4-6 $*"
        DESC="EvoTest SDK ($EPISODES episodes)"
        ;;
    shell)
        EXP_CMD="bash"
        DESC="Interactive shell"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 {multi|single|evotest|evotest-sdk|shell} [args...]"
        exit 1
        ;;
esac

echo "============================================================"
echo "  Experiment: $DESC"
echo "============================================================"

# ============================================================
# Preflight checks
# ============================================================
[ -f "$SIF" ]     || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[ -f "$OVERLAY" ] || { echo "ERROR: Overlay not found: $OVERLAY"; exit 1; }
[ -d "$HF_CACHE" ] || { echo "ERROR: HF cache not found: $HF_CACHE"; exit 1; }
[ -d "$PROJECT" ]  || { echo "ERROR: Project not found: $PROJECT"; exit 1; }

# Check vLLM is running (except for shell mode)
if [ "$EXPERIMENT" != "shell" ]; then
    if ! curl -s localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "ERROR: vLLM not running on port $VLLM_PORT"
        echo "Start it first:  bash scripts/start_vllm.sh"
        exit 1
    fi
    echo "  vLLM: running on port $VLLM_PORT"
fi
echo ""

# ============================================================
# Launch container — experiment only
# ============================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
apptainer exec --nv --fakeroot \
    --overlay "$OVERLAY":ro \
    --bind "$HF_CACHE":/root/.cache/huggingface \
    --bind "$PROJECT":/workspace \
    "$SIF" \
    bash -c "
set -euo pipefail
cd /workspace
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
$EXP_CMD
"
