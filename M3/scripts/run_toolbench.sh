#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${PROJECT_ROOT}/data/raw/repos/stabletoolbench/repository}"
TOOL_ROOT="${TOOL_ROOT:-${PROJECT_ROOT}/data/raw/repos/stabletoolbench/official_query_tools}"
QUERY_ROOT="${QUERY_ROOT:-${OFFICIAL_ROOT}/solvable_queries/test_instruction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/taskgraph}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-gpt-qwen3-14b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8080/virtual}"
TOOLBENCH_KEY="${TOOLBENCH_KEY:-}"
METHOD="${METHOD:-ours}"
SUBSET="${SUBSET:-all}"
NUM_THREADS="${NUM_THREADS:-1}"
MAX_TASKS="${MAX_TASKS:-0}"
TASK_OFFSET="${TASK_OFFSET:-0}"
ABLATION_MODE="${ABLATION_MODE:-full}"
SAMPLING_SEED="${SAMPLING_SEED:-20260831}"
ACTION_TEMPERATURE="${ACTION_TEMPERATURE:-0}"

PINNED_COMMIT="aa4ed9f4737ad98bd706663f01d63623c3427812"
case "${METHOD}" in
  ours) LABEL="ours_progressive"; UPSTREAM_METHOD="MLG" ;;
  cot) LABEL="official_stabletoolbench_cot"; UPSTREAM_METHOD="CoT@1" ;;
  dfs) LABEL="official_stabletoolbench_dfs"; UPSTREAM_METHOD="DFS_woFilter_w2" ;;
  *) echo "METHOD must be ours, cot, or dfs" >&2; exit 2 ;;
esac

if command -v git >/dev/null 2>&1; then
  ACTUAL_COMMIT="$(cd "${OFFICIAL_ROOT}" && git rev-parse HEAD)"
elif [[ -f "${OFFICIAL_ROOT}/.git/HEAD" ]]; then
  ACTUAL_COMMIT="$(tr -d '\r\n' < "${OFFICIAL_ROOT}/.git/HEAD")"
else
  echo "Cannot verify the StableToolBench checkout commit" >&2
  exit 2
fi
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "StableToolBench checkout is not pinned to ${PINNED_COMMIT}" >&2
  exit 2
fi
[[ -d "${TOOL_ROOT}" ]] || { echo "Missing tool environment: ${TOOL_ROOT}" >&2; exit 2; }
if [[ -n "${MODEL_API_KEY}" && "${MODEL_API_KEY}" != "EMPTY" ]]; then
  curl -fsS -H "Authorization: Bearer ${MODEL_API_KEY}" \
    "${BASE_URL}/models" >/dev/null
else
  curl -fsS "${BASE_URL}/models" >/dev/null
fi
curl -fsS "${SERVICE_URL%/virtual}/docs" >/dev/null

FORMAL_SUBSETS=(G2_instruction G2_category G3_instruction)
if [[ "${SUBSET}" == "all" ]]; then
  SUBSETS=("${FORMAL_SUBSETS[@]}")
else
  if [[ ! " ${FORMAL_SUBSETS[*]} " =~ " ${SUBSET} " ]]; then
    echo "SUBSET must be all, G2_instruction, G2_category, or G3_instruction" >&2
    exit 2
  fi
  SUBSETS=("${SUBSET}")
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${OFFICIAL_ROOT}:${OFFICIAL_ROOT}/toolbench/inference"
export SERVICE_URL
RAW_ROOT="${OUTPUT_ROOT}/raw/${LABEL}"
CONVERTED_ROOT="${OUTPUT_ROOT}/converted/${LABEL}"
mkdir -p "${RAW_ROOT}" "${CONVERTED_ROOT}" "${OUTPUT_ROOT}/smoke_inputs"

for group in "${SUBSETS[@]}"; do
  query_file="${QUERY_ROOT}/${group}.json"
  [[ -s "${query_file}" ]] || { echo "Missing query input: ${query_file}" >&2; exit 2; }
  if (( MAX_TASKS > 0 )); then
    selected_file="${OUTPUT_ROOT}/smoke_inputs/${group}_${TASK_OFFSET}_${MAX_TASKS}.json"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/select_queries.py" \
      --input "${query_file}" --output "${selected_file}" \
      --limit "${MAX_TASKS}" --offset "${TASK_OFFSET}"
    query_file="${selected_file}"
  fi
  raw_dir="${RAW_ROOT}/${group}"
  mkdir -p "${raw_dir}"
  if [[ "${METHOD}" == "ours" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_taskgraph.py" \
      --official-root "${OFFICIAL_ROOT}" \
      --input-query-file "${query_file}" \
      --output-answer-file "${raw_dir}" \
      --tool-root-dir "${TOOL_ROOT}" \
      --openai-key "${MODEL_API_KEY}" \
      --base-url "${BASE_URL}" \
      --chatgpt-model "${MODEL}" \
      --toolbench-key "${TOOLBENCH_KEY}" \
      --num-thread "${NUM_THREADS}" \
      --ablation-mode "${ABLATION_MODE}" \
      --sampling-seed "${SAMPLING_SEED}" \
      --action-temperature "${ACTION_TEMPERATURE}"
  else
    "${PYTHON_BIN}" "${OFFICIAL_ROOT}/toolbench/inference/qa_pipeline_multithread.py" \
      --tool_root_dir "${TOOL_ROOT}" \
      --backbone_model chatgpt_function \
      --chatgpt_model "${MODEL}" \
      --base_url "${BASE_URL}" \
      --openai_key "${MODEL_API_KEY}" \
      --method "${UPSTREAM_METHOD}" \
      --input_query_file "${query_file}" \
      --output_answer_file "${raw_dir}" \
      --toolbench_key "${TOOLBENCH_KEY}" \
      --num_thread "${NUM_THREADS}"
  fi
  (
    cd "${OFFICIAL_ROOT}/toolbench/tooleval"
    "${PYTHON_BIN}" convert_to_answer_format.py \
      --answer_dir "${raw_dir}" \
      --method "${UPSTREAM_METHOD}" \
      --output "${CONVERTED_ROOT}/${group}.json"
  )
done

echo "M3 StableToolBench artifacts: ${OUTPUT_ROOT}"
echo "method=${LABEL} subsets=${SUBSETS[*]} max_tasks=${MAX_TASKS} ablation=${ABLATION_MODE}"
