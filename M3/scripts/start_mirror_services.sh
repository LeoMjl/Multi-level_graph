#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.runtime/model-cache}"
WHEEL_ROOT="${WHEEL_ROOT:-${PROJECT_ROOT}/wheels}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MIRROR_MODEL="${MIRROR_MODEL:-MirrorAPI-Cache}"
MODEL_READY_ATTEMPTS="${MODEL_READY_ATTEMPTS:-450}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"

MIRROR_CONTAINER="m3-scale-mirrorapi-llm"
SERVER_CONTAINER="m3-scale-mirrorapi-server"

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }
}

wait_url() {
  local url="$1"
  local label="$2"
  local attempts="$3"
  local api_key="${4:-}"
  for ((i = 1; i <= attempts; i++)); do
    if [[ -n "$api_key" ]]; then
      if curl -fsS -H "Authorization: Bearer ${api_key}" "$url" >/dev/null 2>&1; then
        echo "$label is ready: $url"
        return 0
      fi
    elif curl -fsS "$url" >/dev/null 2>&1; then
      echo "$label is ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

require_dir "${MODEL_ROOT}/${MIRROR_MODEL}"
require_dir "${PROJECT_ROOT}/data/raw/repos/stabletoolbench/repository/server"
require_dir "${PROJECT_ROOT}/data/raw/repos/stabletoolbench/official_query_tools"
require_dir "$WHEEL_ROOT"
mkdir -p "$CACHE_ROOT"

if curl -fsS -H "Authorization: Bearer EMPTY" \
  "http://127.0.0.1:12345/v1/models" >/dev/null 2>&1; then
  echo "Reusing MirrorAPI-Cache model"
else
  docker rm -f "$MIRROR_CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$MIRROR_CONTAINER" \
    --gpus all \
    --network host \
    --ipc host \
    -v "${MODEL_ROOT}:/models:ro" \
    -v "${CACHE_ROOT}:/root/.cache" \
    "$IMAGE" \
    --model "/models/${MIRROR_MODEL}" \
    --served-model-name stabletoolbench-mirrorapi-cache \
    --api-key EMPTY \
    --host 0.0.0.0 \
    --port 12345 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.82 \
    --max-num-seqs 8 \
    --enforce-eager >/dev/null
  wait_url "http://127.0.0.1:12345/v1/models" \
    "MirrorAPI-Cache model" "$MODEL_READY_ATTEMPTS" EMPTY
fi

if curl -fsS "http://127.0.0.1:8080/docs" >/dev/null 2>&1; then
  echo "Reusing MirrorAPI virtual server"
else
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$SERVER_CONTAINER" \
    --network host \
    --entrypoint bash \
    -v "${PROJECT_ROOT}:/workspace/project:ro" \
    -v "${WHEEL_ROOT}:/wheels:ro" \
    "$IMAGE" \
    -lc 'pip install --no-index --find-links /wheels slowapi==0.1.9 >/tmp/pip.log && mkdir -p /tmp/stb-server && cp /workspace/project/data/raw/repos/stabletoolbench/repository/server/*.py /tmp/stb-server/ && cp /workspace/project/server.example.yml /tmp/stb-server/config_mirrorapi_cache.yml && cd /tmp/stb-server && python3 main_mirrorapi_cache.py' >/dev/null
  wait_url "http://127.0.0.1:8080/docs" \
    "MirrorAPI virtual server" "$SERVER_READY_ATTEMPTS"
fi

echo "M3 external-agent tool environment is ready"
