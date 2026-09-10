#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
allow_empty=false
[[ "${1:-}" == "--allow-empty" ]] && allow_empty=true
if [[ ! -f .env ]]; then
  echo ".env ausente; nao e possivel resolver cameras aprovadas." >&2
  exit 2
fi
dotenv_get() {
  local key="$1" value
  value="$(grep -m1 -E "^${key}=" .env || true)"
  value="${value#*=}"
  printf '%s' "$value"
}
runtime_env=".camera-runtime.env"
runtime_compose="docker-compose.camera-runtime.yml"
tmp_env="$(mktemp)"
tmp_compose="$(mktemp)"
trap 'rm -f "$tmp_env" "$tmp_compose"' EXIT
umask 077
printf '# Gerado por scripts/resolve_camera_runtime.sh. Nao editar.\n' > "$tmp_env"
printf 'services:\n  iris-app:\n    devices:\n' > "$tmp_compose"
mapfile -t discovered < <(find /dev/v4l/by-id -maxdepth 1 -type l -name '*-video-index0' -print 2>/dev/null | sort)
active_count=0
for slot in 1 2 3 4; do
  enabled_var="CAMERA_${slot}_ENABLED"
  host_var="CAMERA_${slot}_HOST_DEVICE"
  stable_var="CAMERA_${slot}_STABLE_PATH"
  serial_var="CAMERA_${slot}_SERIAL"
  device_var="CAMERA_${slot}_DEVICE"
  configured_enabled="$(dotenv_get "$enabled_var")"
  host_path="$(dotenv_get "$host_var")"
  if [[ -z "$host_path" ]]; then
    host_path="$(dotenv_get "$stable_var")"
  fi
  serial="$(dotenv_get "$serial_var")"
  container_path="$(dotenv_get "$device_var")"
  configured_enabled="${configured_enabled:-false}"
  selected=""
  if [[ "$configured_enabled" == "true" || "$configured_enabled" == "1" ]]; then
    if [[ -n "$host_path" && -c "$host_path" ]]; then
      selected="$host_path"
    elif [[ -n "$serial" ]]; then
      for candidate in "${discovered[@]}"; do
        if [[ "$candidate" == *"_${serial}-video-index0" ]]; then
          selected="$candidate"
          break
        fi
      done
    fi
  fi
  if [[ -n "$selected" && -n "$container_path" ]]; then
    printf 'CAMERA_%s_ENABLED=true\n' "$slot" >> "$tmp_env"
    printf '      - "%s:%s"\n' "$selected" "$container_path" >> "$tmp_compose"
    printf 'camera_slot=%s serial=%s status=active path=%s\n' "$slot" "${serial:-unknown}" "$selected"
    active_count=$((active_count + 1))
  else
    printf 'CAMERA_%s_ENABLED=false\n' "$slot" >> "$tmp_env"
    if [[ "$configured_enabled" == "true" || "$configured_enabled" == "1" ]]; then
      printf 'camera_slot=%s serial=%s status=absent_or_unapproved\n' "$slot" "${serial:-unknown}"
    fi
  fi
done
printf '    environment:\n' >> "$tmp_compose"
while IFS= read -r entry; do
  [[ -z "$entry" || "$entry" == \#* ]] && continue
  key="${entry%%=*}"
  value="${entry#*=}"
  printf '      %s: "%s"\n' "$key" "$value" >> "$tmp_compose"
done < "$tmp_env"
if (( active_count == 0 )) && [[ "$allow_empty" != "true" ]]; then
  echo "Nenhuma camera aprovada e presente; runtime nao sera iniciado." >&2
  exit 3
fi
if (( active_count == 0 )); then
  echo "Nenhuma camera aprovada presente; iniciando API sem workers."
fi
mv "$tmp_env" "$runtime_env"
mv "$tmp_compose" "$runtime_compose"
chmod 600 "$runtime_env" "$runtime_compose"
printf 'resolved_cameras=%s\n' "$active_count"
