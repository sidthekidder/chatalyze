"""
Upstash Redis-backed report storage for shareable links.
Uses the Upstash REST API — no SDK, just HTTP. Never pauses unlike Supabase free tier.
Stores only analysis result and computed features — no raw messages.
Reports expire after 90 days.
"""
import json
import os
import uuid
from typing import Optional

_REPORT_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days


def _headers() -> dict:
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _base_url() -> Optional[str]:
    return os.getenv("UPSTASH_REDIS_REST_URL")


def save_report(result: dict, features: dict) -> Optional[str]:
    import requests
    url = _base_url()
    headers = _headers()
    if not url or not headers:
        return None

    rid = str(uuid.uuid4())
    payload = json.dumps({"result": result, "features": features})

    # Upstash REST: POST /set/{key} with body [value, "EX", ttl]
    resp = requests.post(
        f"{url}/set/{rid}",
        headers={**headers, "Content-Type": "application/json"},
        json=[payload, "EX", _REPORT_TTL_SECONDS],
    )
    if resp.ok:
        return rid
    return None


def load_report(report_id: str) -> Optional[tuple[dict, dict]]:
    import requests
    url = _base_url()
    headers = _headers()
    if not url or not headers:
        return None

    resp = requests.get(f"{url}/get/{report_id}", headers=headers)
    if not resp.ok:
        return None

    raw = resp.json().get("result")
    if not raw:
        return None

    try:
        d = json.loads(raw)
        return d["result"], d["features"]
    except (json.JSONDecodeError, KeyError):
        return None
