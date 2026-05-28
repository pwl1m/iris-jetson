#!/usr/bin/env bash
set -euo pipefail

JETSON_TARGET="${1:-}"
REMOTE_DIR="${2:-~/iris-jetson}"

if [ -z "${JETSON_TARGET}" ]; then
  echo "Usage: $0 user@jetson-host [remote_dir]"
  exit 2
fi

rsync -az --delete \
  --exclude ".git" \
  --exclude "frigate/data" \
  --exclude "iris-mosquitto/data" \
  --exclude "iris-app/data-events/*.jsonl" \
  --exclude "__pycache__" \
  ./ "${JETSON_TARGET}:${REMOTE_DIR}/"

echo "Synced to ${JETSON_TARGET}:${REMOTE_DIR}"
