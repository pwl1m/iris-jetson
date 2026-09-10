#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-}"
if [[ -z "$DEVICE" ]]; then
  mapfile -t candidates < <(find /dev/v4l/by-id -maxdepth 1 -type l -name "*-video-index0" -print 2>/dev/null | sort)
  if (( ${#candidates[@]} != 1 )); then
    echo "Usage: $0 /dev/v4l/by-id/<approved-camera>-video-index0" >&2
    printf "Found %s capture candidates\\n" "${#candidates[@]}" >&2
    exit 2
  fi
  DEVICE="${candidates[0]}"
fi
if [[ ! -c "$DEVICE" ]]; then
  echo "Camera device unavailable: $DEVICE" >&2
  exit 3
fi

echo "== Logitech C930e / V4L2 test =="
echo "device: ${DEVICE}"

v4l2-ctl --device="${DEVICE}" --all
v4l2-ctl --device="${DEVICE}" --list-formats-ext
ffmpeg -f v4l2 -list_formats all -i "${DEVICE}" >/tmp/ffmpeg-v4l2-info.txt 2>&1 || true
echo
echo "FFmpeg format probe saved to /tmp/ffmpeg-v4l2-info.txt"

echo
echo "Preview options:"
echo "  ffplay ${DEVICE}"
echo "  gst-launch-1.0 v4l2src device=${DEVICE} ! videoconvert ! autovideosink"
