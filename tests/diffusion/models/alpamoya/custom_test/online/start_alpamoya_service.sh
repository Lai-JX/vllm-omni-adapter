#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${OMNI_ROOT}/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/share/models/Alpamayo-1.5-10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-alpamayo1.5}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-${OMNI_ROOT}/tests/diffusion/models/alpamoya/custom_test/online/alpamayo1_5.yaml}"
VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-vllm-omni}"
COLLECT_METRICS="${COLLECT_METRICS:-false}"
LOG_FILE="${LOG_FILE:-}"

export PYTHONPATH="${OMNI_ROOT}:${WORKSPACE_ROOT}/vllm:${PYTHONPATH:-}"
# export VLLM_TORCH_PROFILER_WITH_STACK="${VLLM_TORCH_PROFILER_WITH_STACK:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect-metrics)
      COLLECT_METRICS=true
      shift
      ;;
    --no-collect-metrics)
      COLLECT_METRICS=false
      shift
      ;;
    --log-file)
      if [[ $# -lt 2 ]]; then
        echo "error: --log-file requires a value" >&2
        exit 1
      fi
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${OMNI_ROOT}"
CMD=(
  "${VLLM_OMNI_BIN}" serve "${MODEL}"
  --omni
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --stage-configs-path "${STAGE_CONFIGS_PATH}"
)

if [[ "${COLLECT_METRICS}" == "true" ]]; then
  CMD+=(--log-stats --enable-diffusion-pipeline-profiler)
fi

CMD+=("${EXTRA_ARGS[@]}")

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  "${CMD[@]}"
fi
