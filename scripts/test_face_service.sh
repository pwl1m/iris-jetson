#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${FACE_SERVICE_URL_LOCAL:-http://localhost:8081}"
IMAGE="${1:-}"
SUBJECT="${2:-test_subject}"

echo "== Face service health =="
curl -fsS "${BASE_URL}/health"
echo

if [ -n "${IMAGE}" ]; then
  echo "== Enroll =="
  curl -fsS -F "subject=${SUBJECT}" -F "file=@${IMAGE}" "${BASE_URL}/enroll"
  echo

  echo "== Recognize =="
  curl -fsS -F "file=@${IMAGE}" "${BASE_URL}/recognize"
  echo
else
  echo "No image provided. Usage: $0 vision-app/samples/snapshots/person.jpg subject_name"
fi
