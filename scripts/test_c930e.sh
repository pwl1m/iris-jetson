#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-/dev/video0}"

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
