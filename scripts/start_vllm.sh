#!/bin/bash
set -euo pipefail

# ============================================================
# start_vllm.sh — Start vLLM server inside Apptainer (keep running)
#
# Run this once, leave it running. Then run experiments separately:
#   bash scripts/container.sh evotest 10
#
# Usage:
#   bash scripts/start_vllm.sh              # default: Qwen3-30B-A3B
#   bash scripts/start_vllm.sh --qwen35     # Qwen3.5-35B-A3B (text-only)
#   bash scripts/start_vllm.sh --qwen35-27b # Qwen3.5-27B dense (text-only, with MTP)
#   bash scripts/start_vllm.sh --qwen3next  # Qwen3-Next-80B-A3B-FP8 (with MTP)
#   bash scripts/start_vllm.sh --tool-call  # enable tool calling (for SDK)
#   bash scripts/start_vllm.sh --qwen35 --tool-call  # combine flags
#
# Stop: Ctrl+C
# ============================================================

# Source local .env if it exists (machine-specific paths & GPU config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../.env" ] && source "$SCRIPT_DIR/../.env"

HF_CACHE="${MIMIC_HF_CACHE:-$HOME/.cache/huggingface}"
PROJECT="${MIMIC_PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_UTIL=0.95
VLLM_PORT=8000

# Parse flags
USE_QWEN35=false
USE_QWEN35_27B=false
USE_QWEN3NEXT=false
TOOL_CALL=false
for arg in "$@"; do
    case "$arg" in
        --qwen35)      USE_QWEN35=true ;;
        --qwen35-27b)  USE_QWEN35_27B=true ;;
        --qwen3next)   USE_QWEN3NEXT=true ;;
        --tool-call)   TOOL_CALL=true ;;
    esac
done

# Model-specific configuration
if $USE_QWEN3NEXT; then
    VLLM_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    VLLM_MAX_LEN=32768
    SIF="${MIMIC_SIF_QWEN35:-/cbica/home/hanti/containers/vllm-openai_nightly.sif}"
    OVERLAY="${MIMIC_OVERLAY_QWEN35:-/cbica/home/hanti/containers/vllm_overlay_qwen35.img}"
elif $USE_QWEN35_27B; then
    VLLM_MODEL="Qwen/Qwen3.5-27B"
    VLLM_MAX_LEN=32768
    SIF="${MIMIC_SIF_QWEN35:-/cbica/home/hanti/containers/vllm-openai_nightly.sif}"
    OVERLAY="${MIMIC_OVERLAY_QWEN35:-/cbica/home/hanti/containers/vllm_overlay_qwen35.img}"
elif $USE_QWEN35; then
    VLLM_MODEL="Qwen/Qwen3.5-35B-A3B"
    VLLM_MAX_LEN=32768
    SIF="${MIMIC_SIF_QWEN35:-/cbica/home/hanti/containers/vllm-openai_nightly.sif}"
    OVERLAY="${MIMIC_OVERLAY_QWEN35:-/cbica/home/hanti/containers/vllm_overlay_qwen35.img}"
else
    VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    VLLM_MAX_LEN=32768
    SIF="${MIMIC_SIF:?ERROR: Set MIMIC_SIF in .env (path to vllm SIF image)}"
    OVERLAY="${MIMIC_OVERLAY:?ERROR: Set MIMIC_OVERLAY in .env (path to overlay image)}"
fi

VLLM_EXTRA_ARGS=""

# Tool calling
if $TOOL_CALL; then
    if $USE_QWEN35 || $USE_QWEN35_27B; then
        VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder"
    else
        VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
    fi
fi

# Qwen3.5-specific flags (applies to both 35B-A3B and 27B)
if $USE_QWEN35 || $USE_QWEN35_27B; then
    # Reasoning parser for thinking mode, text-only to skip vision encoder
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS \
        --reasoning-parser qwen3 \
        --language-model-only"
fi

# MTP speculative decoding (Qwen3-Next only; Qwen3.5-27B OOMs with MTP on single 95GB GPU)
if $USE_QWEN3NEXT; then
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS \
        --speculative-config {\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}"
fi

# GH200 performance tuning
VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3"

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
echo "    bash scripts/container.sh evotest 10"
echo "============================================================"
echo ""

# Bind-mount GH200-tuned MoE kernel configs (copied from H200)
MOE_CONFIGS_BIND=""
if [ -d "$PROJECT/moe_configs" ]; then
    MOE_CONFIGS_BIND="--bind $PROJECT/moe_configs:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs"
    echo "  MoE configs: GH200-tuned (from H200)"
fi

# Python binary: nightly SIF uses python3 (no python symlink)
if $USE_QWEN35 || $USE_QWEN35_27B || $USE_QWEN3NEXT; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
apptainer exec --nv --fakeroot \
    --overlay "$OVERLAY"${OVERLAY_MOUNT:+:$OVERLAY_MOUNT} \
    --bind "$HF_CACHE":/root/.cache/huggingface \
    --bind "$PROJECT":/workspace \
    $MOE_CONFIGS_BIND \
    "$SIF" \
    $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --tensor-parallel-size "$VLLM_TP" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len "$VLLM_MAX_LEN" \
        --port "$VLLM_PORT" \
        $VLLM_EXTRA_ARGS
