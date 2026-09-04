#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.runtime/model-cache}"
WHEEL_ROOT="${WHEEL_ROOT:-${PROJECT_ROOT}/wheels}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
AGENT_MODEL="${AGENT_MODEL:-Qwen3-14B-AWQ}"
MIRROR_MODEL="${MIRROR_MODEL:-MirrorAPI-Cache}"
AGENT_MAX_MODEL_LEN="${AGENT_MAX_MODEL_LEN:-32768}"
AGENT_MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS:-4}"
AGENT_TOOL_CALL_PARSER="${AGENT_TOOL_CALL_PARSER:-qwen3_xml}"
MODEL_READY_ATTEMPTS="${MODEL_READY_ATTEMPTS:-450}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"

AGENT_CONTAINER="m3-agent-qwen"
MIRROR_CONTAINER="m3-mirrorapi-llm"
SERVER_CONTAINER="m3-mirrorapi-server"
OLD_AGENT_CONTAINER="ocean-qwen"
BACKUP_AGENT_CONTAINER="ocean-qwen-before-m3"

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }
}

wait_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-120}"
  local api_key="${4:-}"
  for ((i = 1; i <= attempts; i++)); do
    if [[ -n "$api_key" ]]; then
      if curl -fsS -H "Authorization: Bearer ${api_key}" "$url" >/dev/null 2>&1; then
        echo "$label is ready: $url"
        return 0
      fi
    else
      if curl -fsS "$url" >/dev/null 2>&1; then
        echo "$label is ready: $url"
        return 0
      fi
    fi
    sleep 2
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

require_dir "${MODEL_ROOT}/${AGENT_MODEL}"
require_dir "${MODEL_ROOT}/${MIRROR_MODEL}"
require_dir "${PROJECT_ROOT}/data/raw/repos/stabletoolbench/repository/server"
require_dir "${PROJECT_ROOT}/data/raw/repos/stabletoolbench/official_query_tools"
require_dir "${WHEEL_ROOT}"

if docker container inspect "$OLD_AGENT_CONTAINER" >/dev/null 2>&1; then
  if docker container inspect "$BACKUP_AGENT_CONTAINER" >/dev/null 2>&1; then
    echo "Both $OLD_AGENT_CONTAINER and $BACKUP_AGENT_CONTAINER exist" >&2
    exit 2
  fi
  docker stop "$OLD_AGENT_CONTAINER" >/dev/null
  docker rename "$OLD_AGENT_CONTAINER" "$BACKUP_AGENT_CONTAINER"
fi

if curl -fsS "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
  echo "Reusing ready TaskGraph agent: http://127.0.0.1:8000/v1/models"
else
  if docker container inspect "$AGENT_CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$AGENT_CONTAINER" >/dev/null
  fi
  docker run -d \
    --name "$AGENT_CONTAINER" \
    --gpus all \
    --network host \
    --ipc host \
    -v "${MODEL_ROOT}:/models:ro" \
    -v "${CACHE_ROOT}:/root/.cache" \
    "$IMAGE" \
    --model "/models/${AGENT_MODEL}" \
    --served-model-name ocean-qwen gpt-qwen3-14b \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype half \
    --max-model-len "${AGENT_MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.47 \
    --max-num-seqs "${AGENT_MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "${AGENT_TOOL_CALL_PARSER}" >/dev/null
  wait_url "http://127.0.0.1:8000/v1/models" "TaskGraph agent" "$MODEL_READY_ATTEMPTS"
fi

if curl -fsS -H "Authorization: Bearer EMPTY" \
  "http://127.0.0.1:12345/v1/models" >/dev/null 2>&1; then
  echo "Reusing ready MirrorAPI-Cache model: http://127.0.0.1:12345/v1/models"
else
  if docker container inspect "$MIRROR_CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$MIRROR_CONTAINER" >/dev/null
  fi
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
    --gpu-memory-utilization 0.42 \
    --max-num-seqs 8 \
    --enforce-eager >/dev/null
  wait_url "http://127.0.0.1:12345/v1/models" \
    "MirrorAPI-Cache model" "$MODEL_READY_ATTEMPTS" EMPTY
fi

if curl -fsS "http://127.0.0.1:8080/docs" >/dev/null 2>&1; then
  echo "Reusing ready MirrorAPI-Cache virtual server: http://127.0.0.1:8080/docs"
else
  if docker container inspect "$SERVER_CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$SERVER_CONTAINER" >/dev/null
  fi
  docker run -d \
    --name "$SERVER_CONTAINER" \
    --network host \
    --entrypoint bash \
    -v "${PROJECT_ROOT}:/workspace/project:ro" \
    -v "${WHEEL_ROOT}:/wheels:ro" \
    "$IMAGE" \
    -lc 'pip install --no-index --find-links /wheels slowapi==0.1.9 >/tmp/pip.log && mkdir -p /tmp/stb-server && cp /workspace/project/data/raw/repos/stabletoolbench/repository/server/*.py /tmp/stb-server/ && cp /workspace/project/server.example.yml /tmp/stb-server/config_mirrorapi_cache.yml && cd /tmp/stb-server && python3 main_mirrorapi_cache.py' >/dev/null
  wait_url "http://127.0.0.1:8080/docs" \
    "MirrorAPI-Cache virtual server" "$SERVER_READY_ATTEMPTS"
fi
timeout 15s nvidia-smi \
  --query-compute-apps=pid,used_memory,process_name --format=csv,noheader \
  || echo "GPU process telemetry unavailable; service health checks passed" >&2
echo "Official M3 services are ready"
