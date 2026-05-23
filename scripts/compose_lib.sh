#!/usr/bin/env bash

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    printf 'docker compose'
    return 0
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    printf 'docker-compose'
    return 0
  fi

  echo "Docker Compose nao encontrado. Instale docker-compose-plugin ou docker-compose." >&2
  return 1
}

require_docker_access() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  cat >&2 <<'EOF'
Sem acesso ao Docker daemon na sessao atual.

Causa comum neste host:
- `docker-compose` existe
- o usuario ja esta no grupo `docker`
- mas a sessao atual ainda nao herdou esse grupo

Correcoes:
1. faca logout/login e tente novamente
2. ou reinicie a maquina
3. ou execute temporariamente com sudo no seu terminal

Diagnostico rapido:
- `id`
- `getent group docker`
- `ls -l /var/run/docker.sock`
EOF
  return 1
}
