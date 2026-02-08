#!/bin/bash
set -euo pipefail

# ============================================================
# start_vllm.sh — Start vLLM server inside Apptainer (keep running)
#
# Run this once, leave it running. Then run experiments separately:
#   bash scripts/run_in_container.sh evotest 10
#
# Usage:
#   bash scripts/start_vllm.sh              # default config
#   bash scripts/start_vllm.sh --tool-call  # enable tool calling (for SDK)
#
# Stop: Ctrl+C
# ============================================================

SIF="/home/than/images/vllm_26.01-py3.sif"
OVERLAY="/home/than/images/overlay.img"
HF_CACHE="/home/than/.cache/huggingface"
PROJECT="/home/than/DeepLearning/mimic_skills"

VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_TP=2
VLLM_GPU_UTIL=0.9
VLLM_MAX_LEN=32768
VLLM_PORT=8000

VLLM_EXTRA_ARGS=""
if [ "${1:-}" = "--tool-call" ]; then
    VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
fi

# Preflight
[ -f "$SIF" ]     || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[ -f "$OVERLAY" ] || { echo "ERROR: Overlay not found: $OVERLAY"; exit 1; }
[ -d "$HF_CACHE" ] || { echo "ERROR: HF cache not found: $HF_CACHE"; exit 1; }

# Check if already running
if curl -s localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "vLLM already running on port $VLLM_PORT"
    exit 0
fi

echo "============================================================"
echo "  Starting vLLM Server"
echo "============================================================"
echo "  Model:  $VLLM_MODEL"
echo "  TP:     $VLLM_TP"
echo "  Port:   $VLLM_PORT"
echo "  Extra:  ${VLLM_EXTRA_ARGS:-none}"
echo ""
echo "  Stop with Ctrl+C"
echo "  Run experiments in another terminal:"
echo "    bash scripts/run_in_container.sh evotest 10"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
apptainer exec --nv --fakeroot \
    --overlay "$OVERLAY":ro \
    --bind "$HF_CACHE":/root/.cache/huggingface \
    --bind "$PROJECT":/workspace \
    "$SIF" \
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --tensor-parallel-size "$VLLM_TP" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len "$VLLM_MAX_LEN" \
        --port "$VLLM_PORT" \
        $VLLM_EXTRA_ARGS
