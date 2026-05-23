#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${FACE_SERVICE_URL_LOCAL:-http://localhost:8081}"
IMAGE="${1:-}"

if [ -z "${IMAGE}" ]; then
  echo "Usage: $0 /path/to/image.jpg"
  exit 2
fi

curl -fsS -F "file=@${IMAGE}" "${BASE_URL}/recognize"
echo

