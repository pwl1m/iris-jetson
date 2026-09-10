#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${JETSON_DEPLOY_PATH:-/opt/jetson-orin-deploy}"
HEALTH_URL="${JETSON_HEALTH_URL:-http://localhost:8081/health}"
HEALTH_RETRIES="${JETSON_HEALTH_RETRIES:-20}"
HEALTH_DELAY="${JETSON_HEALTH_DELAY_SECONDS:-3}"

cd "${APP_DIR}"

if [ ! -f .env ]; then
  echo ".env not found in ${APP_DIR}. Create it before running deploy." >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "${APP_DIR} nao e um repositorio Git." >&2
  exit 1
fi

# O deploy nunca descarta alteracoes locais. Arquivos de runtime ignorados,
# como .env e o compose gerado para cameras, continuam preservados.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Worktree com alteracoes rastreadas; deploy cancelado para preservar dados locais:" >&2
  git status --short >&2
  echo "Versione, guarde ou reverta as alteracoes antes de tentar novamente." >&2
  exit 1
fi

echo "Fetching deploy/staging..."
git fetch --prune origin deploy/staging
git checkout deploy/staging
git merge --ff-only origin/deploy/staging

./scripts/jetson_preflight.sh
./scripts/compose_jetson_up.sh

attempt=1
while [ "${attempt}" -le "${HEALTH_RETRIES}" ]; do
  if curl -fsS "${HEALTH_URL}" >/dev/null; then
    echo "Health check passed at ${HEALTH_URL}"
    exit 0
  fi

  echo "Health check attempt ${attempt}/${HEALTH_RETRIES} failed; waiting ${HEALTH_DELAY}s"
  sleep "${HEALTH_DELAY}"
  attempt=$((attempt + 1))
done

echo "Health check failed after ${HEALTH_RETRIES} attempts" >&2
exit 1
