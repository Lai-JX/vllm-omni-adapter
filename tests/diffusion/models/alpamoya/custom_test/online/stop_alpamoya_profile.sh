#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# STAGES_JSON="${STAGES_JSON:-[1]}"
STAGES_JSON="${STAGES_JSON:-all}"
URL="http://${HOST}:${PORT}/stop_profile"

if [[ "${STAGES_JSON}" == "all" ]]; then
  BODY='{}'
else
  BODY="{\"stages\": ${STAGES_JSON}}"
fi

curl -sS -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  -d "${BODY}"
echo
