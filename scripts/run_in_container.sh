#!/bin/bash
set -euo pipefail

# ============================================================
# run_in_container.sh — Run experiments inside Apptainer + vLLM
#
# Launches vLLM server inside the container, waits for it,
# then runs the chosen experiment script.
#
# Usage:
#   bash run_in_container.sh                              # default: multi v1
#   bash run_in_container.sh multi v1 vLLM_Qwen3         # multi-pathology
#   bash run_in_container.sh single cholecystitis v1      # single pathology
#   bash run_in_container.sh evotest 10                   # EvoTest Hager 10 episodes
#   bash run_in_container.sh evotest --resume 15          # resume Hager EvoTest
#   bash run_in_container.sh evotest-sdk 10               # EvoTest SDK 10 episodes
#   bash run_in_container.sh evotest-sdk 10 --resume      # resume SDK EvoTest
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

VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_TP=3
VLLM_GPU_UTIL=0.9
VLLM_MAX_LEN=32768
VLLM_PORT=8000
VLLM_TIMEOUT=300  # seconds to wait for vLLM startup

# ============================================================
# Build experiment command based on mode
# ============================================================
VLLM_EXTRA_ARGS=""  # Extra vLLM server flags (e.g. tool-choice for SDK)

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
        VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
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
echo "  Apptainer + vLLM Experiment Runner"
echo "============================================================"
echo "  Experiment:  $DESC"
echo "  SIF:         $SIF"
echo "  Overlay:     $OVERLAY"
echo "  vLLM model:  $VLLM_MODEL"
echo "  TP size:     $VLLM_TP"
echo "  Port:        $VLLM_PORT"
echo "============================================================"
echo ""

# ============================================================
# Preflight checks
# ============================================================
[ -f "$SIF" ]     || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[ -f "$OVERLAY" ] || { echo "ERROR: Overlay not found: $OVERLAY"; exit 1; }
[ -d "$HF_CACHE" ] || { echo "ERROR: HF cache not found: $HF_CACHE"; exit 1; }
[ -d "$PROJECT" ]  || { echo "ERROR: Project not found: $PROJECT"; exit 1; }

# ============================================================
# Launch container with everything inside
# ============================================================
apptainer exec --nv --fakeroot \
    --overlay "$OVERLAY" \
    --bind "$HF_CACHE":/root/.cache/huggingface \
    --bind "$PROJECT":/workspace \
    "$SIF" \
    bash -c "
set -euo pipefail

# Load environment
cd /workspace
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start vLLM server in background
echo ''
echo '>>> Starting vLLM server...'
python -m vllm.entrypoints.openai.api_server \\
    --model $VLLM_MODEL \\
    --tensor-parallel-size $VLLM_TP \\
    --gpu-memory-utilization $VLLM_GPU_UTIL \\
    --max-model-len $VLLM_MAX_LEN \\
    --port $VLLM_PORT \\
    $VLLM_EXTRA_ARGS \\
    > /tmp/vllm.log 2>&1 &
VLLM_PID=\$!

# Cleanup on exit
trap 'echo \">>> Shutting down vLLM (PID \$VLLM_PID)...\"; kill \$VLLM_PID 2>/dev/null; wait \$VLLM_PID 2>/dev/null' EXIT

# Wait for vLLM to be ready
echo \">>> Waiting for vLLM (PID \$VLLM_PID) on port $VLLM_PORT...\"
ELAPSED=0
while ! curl -s localhost:$VLLM_PORT/health > /dev/null 2>&1; do
    if ! kill -0 \$VLLM_PID 2>/dev/null; then
        echo 'ERROR: vLLM process died. Last 30 lines of log:'
        tail -30 /tmp/vllm.log
        exit 1
    fi
    if [ \$ELAPSED -ge $VLLM_TIMEOUT ]; then
        echo 'ERROR: vLLM startup timed out after ${VLLM_TIMEOUT}s. Last 30 lines:'
        tail -30 /tmp/vllm.log
        kill \$VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
    ELAPSED=\$((ELAPSED + 5))
    echo \"  ... waiting (\${ELAPSED}s)\"
done
echo \">>> vLLM ready after \${ELAPSED}s\"
echo ''

# Run experiment
echo '>>> Running: $EXP_CMD'
echo ''
$EXP_CMD
"
