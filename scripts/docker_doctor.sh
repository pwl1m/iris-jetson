#!/usr/bin/env bash
set -euo pipefail

echo "== Docker binary =="
docker --version || true

echo
echo "== Compose binaries =="
docker compose version || true
docker-compose --version || true

echo
echo "== Session groups =="
id || true

echo
echo "== Docker group =="
getent group docker || true

echo
echo "== Docker socket =="
ls -l /var/run/docker.sock || true

echo
echo "== Docker daemon access =="
docker info >/dev/null 2>&1 && echo "ok" || echo "no access"

