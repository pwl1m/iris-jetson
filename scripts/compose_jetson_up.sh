#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/compose_lib.sh"

COMPOSE="$(compose_cmd)"
require_docker_access

${COMPOSE} build iris-app
${COMPOSE} up -d --remove-orphans
