#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${PROJECT_ROOT}/data/raw/repos/stabletoolbench/repository}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/taskgraph}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
FAC_MODEL="${FAC_MODEL:-Evaluator}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
FAC_IMAGE="${FAC_IMAGE:-m3-fac-evaluator:official}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHOD="${METHOD:-ours}"
EVALUATE_TIMES="${EVALUATE_TIMES:-3}"
EVAL_MODEL="${EVAL_MODEL:-deepseek-v4-flash}"
EVAL_THINKING_MODE="${EVAL_THINKING_MODE:-disabled}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
API_POOL_FILE="${API_POOL_FILE:-}"
SKIP_SOWR="${SKIP_SOWR:-0}"

PINNED_COMMIT="aa4ed9f4737ad98bd706663f01d63623c3427812"
case "$METHOD" in
  ours) LABEL="ours_progressive" ;;
  cot) LABEL="official_stabletoolbench_cot" ;;
  dfs) LABEL="official_stabletoolbench_dfs" ;;
  *) echo "METHOD must be ours, cot, or dfs" >&2; exit 2 ;;
esac
REFERENCE="official_stabletoolbench_cot"
SUBSETS=(G2_instruction G2_category G3_instruction)

[[ "$SKIP_SOWR" == "0" || "$SKIP_SOWR" == "1" ]] || {
  echo "SKIP_SOWR must be 0 or 1" >&2
  exit 2
}

ACTUAL_COMMIT="$(cd "$OFFICIAL_ROOT" && git rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$PINNED_COMMIT" ]] || {
  echo "StableToolBench checkout is not pinned to $PINNED_COMMIT" >&2
  exit 2
}
[[ -d "${MODEL_ROOT}/${FAC_MODEL}" ]] || {
  echo "Missing official FAC model: ${MODEL_ROOT}/${FAC_MODEL}" >&2
  exit 2
}

CONVERTED_ROOT="${OUTPUT_ROOT}/converted"
PASS_ROOT="${OUTPUT_ROOT}/pass_rate"
PREFERENCE_ROOT="${OUTPUT_ROOT}/preference"
FAC_ROOT="${OUTPUT_ROOT}/fac"
mkdir -p "${PASS_ROOT}/${LABEL}" "$PREFERENCE_ROOT" "${FAC_ROOT}/${LABEL}"

for subset in "${SUBSETS[@]}"; do
  [[ -s "${CONVERTED_ROOT}/${LABEL}/${subset}.json" ]] || {
    echo "Missing converted artifact for ${LABEL}/${subset}" >&2
    exit 2
  }
done
if [[ "$METHOD" == "ours" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src" "$PYTHON_BIN" \
    "${PROJECT_ROOT}/scripts/validate_artifacts.py" \
    --official-root "$OFFICIAL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --method "$LABEL" \
    --output "${OUTPUT_ROOT}/${LABEL}_artifact_audit.json"
fi

if [[ -n "$API_POOL_FILE" ]]; then
  [[ -s "$API_POOL_FILE" ]] || { echo "Missing API pool: $API_POOL_FILE" >&2; exit 2; }
  [[ "$EVAL_THINKING_MODE" == "disabled" ]] || {
    echo "Formal DeepSeek evaluation requires EVAL_THINKING_MODE=disabled" >&2
    exit 2
  }
  export API_POOL_FILE EVAL_MODEL EVAL_THINKING_MODE EVAL_TEMPERATURE
  export PYTHONPATH="${PROJECT_ROOT}/src:${OFFICIAL_ROOT}:${OFFICIAL_ROOT}/toolbench/inference"
  for subset in "${SUBSETS[@]}"; do
    (
      cd "${OFFICIAL_ROOT}/toolbench/tooleval"
      "$PYTHON_BIN" eval_pass_rate.py \
        --converted_answer_path "$CONVERTED_ROOT" \
        --save_path "${PASS_ROOT}/${LABEL}" \
        --reference_model "$LABEL" \
        --test_ids "${OFFICIAL_ROOT}/solvable_queries/test_query_ids" \
        --evaluate_times "$EVALUATE_TIMES" \
        --max_eval_threads 30 \
        --overwrite \
        --test_set "$subset"
    )
  done
  if [[ "$METHOD" != "cot" && "$SKIP_SOWR" == "0" ]]; then
    [[ -d "${CONVERTED_ROOT}/${REFERENCE}" ]] || {
      echo "Run CoT inference before SoWR" >&2
      exit 2
    }
    for subset in "${SUBSETS[@]}"; do
      (
        cd "${OFFICIAL_ROOT}/toolbench/tooleval"
        "$PYTHON_BIN" eval_preference.py \
          --converted_answer_path "$CONVERTED_ROOT" \
          --reference_model "$REFERENCE" \
          --output_model "$LABEL" \
          --test_ids "${OFFICIAL_ROOT}/solvable_queries/test_query_ids" \
          --save_path "$PREFERENCE_ROOT" \
          --pass_rate_result_path "$PASS_ROOT" \
          --use_pass_rate true \
          --evaluate_times "$EVALUATE_TIMES" \
          --test_set "$subset"
      )
    done
  fi
else
  echo "API_POOL_FILE is empty: DeepSeek-judged SoPR/SoWR will be skipped"
fi

for container in m3-agent-qwen m3-mirrorapi-llm m3-mirrorapi-server; do
  docker stop "$container" >/dev/null 2>&1 || true
done

if ! docker image inspect "$FAC_IMAGE" >/dev/null 2>&1; then
  docker build \
    --build-arg "VLLM_IMAGE=${VLLM_IMAGE}" \
    -f "${PROJECT_ROOT}/docker/m3-fac-evaluator.Dockerfile" \
    -t "$FAC_IMAGE" \
    "$PROJECT_ROOT"
fi

for subset in "${SUBSETS[@]}"; do
  docker run --rm \
    --gpus all \
    --ipc host \
    --entrypoint python3 \
    -v "${MODEL_ROOT}:/models:ro" \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${OUTPUT_ROOT}:/workspace/results" \
    "$FAC_IMAGE" \
    /workspace/project/data/raw/repos/stabletoolbench/repository/toolbench/tooleval/fac_eval.py \
    --model_path "/models/${FAC_MODEL}" \
    --evaluation_path "/workspace/results/converted/${LABEL}/${subset}.json" \
    --output_path "/workspace/results/fac/${LABEL}/${subset}.csv" \
    --ids "/workspace/project/data/raw/repos/stabletoolbench/repository/solvable_queries/test_query_ids/${subset}.json"
done

IMPORT_ARGS=(
  --method "$LABEL"
  --subsets "${SUBSETS[@]}"
  --fac-root "$FAC_ROOT"
  --output "${OUTPUT_ROOT}/${LABEL}_summary.json"
)
if [[ -n "$API_POOL_FILE" ]]; then
  IMPORT_ARGS+=(--pass-rate-root "$PASS_ROOT")
  IMPORT_ARGS+=(
    --sopr-sowr-judge-model "$EVAL_MODEL"
    --sopr-sowr-thinking-mode "$EVAL_THINKING_MODE"
    --sopr-sowr-temperature "$EVAL_TEMPERATURE"
  )
  if [[ "$METHOD" != "cot" && "$SKIP_SOWR" == "0" ]]; then
    IMPORT_ARGS+=(--preference-root "$PREFERENCE_ROOT" --reference-model "$REFERENCE")
  fi
fi
PYTHONPATH="${PROJECT_ROOT}/src" "$PYTHON_BIN" \
  "${PROJECT_ROOT}/scripts/import_official_results.py" "${IMPORT_ARGS[@]}"

echo "Official StableToolBench evaluation complete: ${LABEL}"
if [[ "$METHOD" != "cot" && "$SKIP_SOWR" == "1" ]]; then
  echo "SoWR skipped explicitly: no per-query CoT reference artifact is available."
fi
