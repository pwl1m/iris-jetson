#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${FACE_SERVICE_URL_LOCAL:-http://localhost:8081}"
SUBJECT="${1:-}"
IMAGE="${2:-}"

if [ -z "${SUBJECT}" ] || [ -z "${IMAGE}" ]; then
  echo "Usage: $0 subject /path/to/image.jpg"
  exit 2
fi

curl -fsS -F "subject=${SUBJECT}" -F "file=@${IMAGE}" "${BASE_URL}/enroll"
echo

