#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${VLLM_ENV_FILE:-${PROJECT_ROOT}/model-serving/models.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

# Command-line overrides are applied after models.env for automated sequential runs.
DETACHED=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) VLLM_MODEL="${2:?--model requires a model ID}"; shift 2 ;;
    --port) VLLM_PORT="${2:?--port requires a port}"; shift 2 ;;
    --detach) DETACHED=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${VLLM_MODEL:?Set VLLM_MODEL to a local Hugging Face model ID}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_IMAGE_LIMIT="${VLLM_IMAGE_LIMIT:-1}"
VLLM_MODEL_CACHE="${VLLM_MODEL_CACHE:-${PROJECT_ROOT}/data/huggingface}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-qwen-vl-evaluation}"

mkdir -p "${VLLM_MODEL_CACHE}"
docker info >/dev/null
docker rm -f "${VLLM_CONTAINER_NAME}" >/dev/null 2>&1 || true
RUN_OPTIONS=()
if [[ "${DETACHED}" == true ]]; then
  RUN_OPTIONS+=(--detach)
fi
exec docker run --name "${VLLM_CONTAINER_NAME}" \
  "${RUN_OPTIONS[@]}" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p "${VLLM_PORT}:8000" \
  -v "${VLLM_MODEL_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  "${VLLM_IMAGE}" \
  "${VLLM_MODEL}" \
  --dtype half \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt "{\"image\":${VLLM_IMAGE_LIMIT},\"video\":0}"
