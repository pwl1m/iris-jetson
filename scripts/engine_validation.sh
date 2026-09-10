#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8081}"

echo "[engine] ${BASE_URL}/debug/engine"
curl -fsS "${BASE_URL}/debug/engine"
echo

echo "[health] ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health"
echo

echo "[runtime] amostrando preview, captura e taxas do worker por 8s"
python3 - "$BASE_URL" <<'PY'
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

base_url = sys.argv[1].rstrip("/")
preview_lags = []
capture_lags = []
samples = []

def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

for _ in range(8):
    now = datetime.now(timezone.utc)
    started = time.monotonic()
    payload = json.loads(urllib.request.urlopen(f"{base_url}/health", timeout=3).read().decode())
    streams = payload.get("streams", [])
    stream = next(
        (item for item in streams if item.get("enabled") and item.get("thread_alive")),
        payload.get("stream", {}),
    )
    preview_at = parse_iso(stream.get("preview_at"))
    capture_at = parse_iso(stream.get("last_capture_at"))
    if preview_at:
        preview_lags.append((now - preview_at).total_seconds())
    if capture_at:
        capture_lags.append((now - capture_at).total_seconds())
    samples.append({
        "at": started,
        "frames_read": int(stream.get("frames_read") or 0),
        "captures_seen": int(stream.get("captures_seen") or 0),
        "events_written": int(stream.get("events_written") or 0),
    })
    time.sleep(1)

def summary(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_s": round(min(values), 4),
        "p50_s": round(statistics.median(values), 4),
        "max_s": round(max(values), 4),
    }

def rate(field):
    if len(samples) < 2:
        return None
    elapsed = samples[-1]["at"] - samples[0]["at"]
    if elapsed <= 0:
        return None
    return round((samples[-1][field] - samples[0][field]) / elapsed, 3)

print(json.dumps({
    "preview_lag": summary(preview_lags),
    "capture_lag": summary(capture_lags),
    "worker_rates_fps": {
        "frames_read": rate("frames_read"),
        "analyses_started": rate("captures_seen"),
        "events_written": rate("events_written"),
    },
}, ensure_ascii=False))
PY
