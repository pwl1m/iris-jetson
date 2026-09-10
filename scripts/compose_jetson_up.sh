#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/compose_lib.sh"

COMPOSE="$(compose_cmd)"
require_docker_access

"${SCRIPT_DIR}/resolve_camera_runtime.sh"
compose_args=(--env-file .env --env-file .camera-runtime.env -f docker-compose.yml)
if [[ -f docker-compose.override.yml ]]; then
  compose_args+=(-f docker-compose.override.yml)
fi
compose_args+=(-f docker-compose.camera-runtime.yml)
${COMPOSE} "${compose_args[@]}" config --quiet
${COMPOSE} "${compose_args[@]}" build iris-app
${COMPOSE} "${compose_args[@]}" up -d
