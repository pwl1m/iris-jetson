#!/usr/bin/env bash
set -euo pipefail

echo "== Jetson preflight =="
echo "host: $(hostname)"
echo "kernel: $(uname -a)"

echo
echo "== L4T / JetPack =="
if [ -f /etc/nv_tegra_release ]; then
  cat /etc/nv_tegra_release
else
  echo "/etc/nv_tegra_release not found"
fi

echo
echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; this may be expected depending on JetPack"
fi

echo
echo "== Docker =="
docker --version
if docker compose version >/dev/null 2>&1; then
  docker compose version
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose --version
else
  echo "docker compose plugin missing"
fi

echo
echo "== NVIDIA container runtime =="
docker info 2>/dev/null | grep -i "runtimes" || true

echo
echo "== Camera =="
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || true
else
  echo "v4l2-ctl missing"
fi

echo
echo "== Resources =="
free -h
df -h .
