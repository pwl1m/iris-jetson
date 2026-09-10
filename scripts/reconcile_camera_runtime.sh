#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec 9>/tmp/iris-camera-reconcile.lock
flock -n 9 || exit 0
"$ROOT_DIR/scripts/resolve_camera_runtime.sh" --allow-empty
COMPOSE="$(docker compose version >/dev/null 2>&1 && printf 'docker compose' || printf 'docker-compose')"
compose_args=(--env-file .env --env-file .camera-runtime.env -f docker-compose.yml)
if [[ -f docker-compose.override.yml ]]; then
  compose_args+=(-f docker-compose.override.yml)
fi
compose_args+=(-f docker-compose.camera-runtime.yml)
${COMPOSE} "${compose_args[@]}" config --quiet
${COMPOSE} "${compose_args[@]}" up -d --no-build
