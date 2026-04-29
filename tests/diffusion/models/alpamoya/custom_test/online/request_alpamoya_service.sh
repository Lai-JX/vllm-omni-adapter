#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${OMNI_ROOT}/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-alpamayo1.5}"
CLIP_ID="${1:-100ae358-f548-49b8-af4d-c0afdbcfe9ed}"
T0_US="${2:-5100000}"
USE_HELPER_MESSAGES="${USE_HELPER_MESSAGES:-false}"
REQUEST_ID="${REQUEST_ID:-alpamayo-shell-request}"
INCLUDE_RAW_RESPONSE="${INCLUDE_RAW_RESPONSE:-false}"

export PYTHONPATH="${OMNI_ROOT}:${WORKSPACE_ROOT}/vllm:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"
ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL}"
  --clip-id "${CLIP_ID}"
  --t0-us "${T0_US}"
  # --request-id "${REQUEST_ID}"
)

if [[ "${USE_HELPER_MESSAGES}" == "true" ]]; then
  ARGS+=(--use-helper-messages)
fi

if [[ "${INCLUDE_RAW_RESPONSE}" == "true" ]]; then
  ARGS+=(--include-raw-response)
fi

python alpamoya_openai_client.py "${ARGS[@]}"
