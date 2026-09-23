from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def shared_image_path(image_path: str, local_event_root: str, shared_event_root: str) -> str:
    """Translate an Iris event path to the orchestrator's read-only volume path."""
    source = Path(image_path).resolve()
    local_root = Path(local_event_root).resolve()
    relative = source.relative_to(local_root)
    return str(Path(shared_event_root) / relative)


def build_vlm_job(settings, kind: str, event: dict) -> dict:
    event_id = str(event["event_id"])
    prompt_id = settings.vlm_prompt_id.strip() or "iris-scene-v1"
    payload = {
        "source": "iris",
        "event_id": event_id,
        "camera_id": str(event.get("camera_id") or event.get("camera") or "entrada"),
        "prompt_id": prompt_id,
        "priority": int(settings.vlm_priority),
        "image_path": shared_image_path(
            str(event["image_path"]),
            str(Path(settings.event_log_path).parent),
            settings.vlm_shared_event_root,
        ),
        "metadata": {
            "kind": kind,
            "captured_at": event.get("captured_at"),
            "recognition_status": (event.get("recognition") or {}).get("status"),
            "subject": (event.get("recognition") or {}).get("subject"),
            "similarity": (event.get("recognition") or {}).get("similarity"),
        },
        "max_attempts": int(settings.vlm_job_max_attempts),
        "idempotency_key": f"iris:{kind}:{event_id}:{prompt_id}",
    }
    callback_url = settings.vlm_callback_url.strip()
    if callback_url:
        payload["callback_url"] = callback_url
    return payload


def _token(settings) -> str:
    path = settings.vlm_api_token_file.strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return settings.vlm_api_token.strip()


def submit_vlm_job(settings, payload: dict) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "iris-vlm-submit/1.0"}
    token = _token(settings)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        settings.vlm_orchestrator_url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(0.2, float(settings.vlm_submit_timeout_seconds))) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError(f"VLM orchestrator returned HTTP {response.status}")
        result = json.load(response)
    if not (result.get("job") or {}).get("id"):
        raise RuntimeError("VLM orchestrator response has no job id")
    return result
