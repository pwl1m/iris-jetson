#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/runtime-libs"

mkdir -p "${OUT_DIR}"

copy_glob() {
  local pattern="$1"
  for path in ${pattern}; do
    if [[ -e "${path}" ]]; then
      cp -a "${path}" "${OUT_DIR}/"
    fi
  done
}

# cuDNN runtime libs needed by CUDA EP.
copy_glob "/lib/aarch64-linux-gnu/libcudnn*.so*"

# TensorRT runtime libs (optional today, required when enabling TRT EP).
copy_glob "/lib/aarch64-linux-gnu/libnvinfer*.so*"

echo "runtime-libs prepared in ${OUT_DIR}"
ls -la "${OUT_DIR}" | sed -n '1,120p'
