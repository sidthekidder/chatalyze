"""
Supabase-backed report storage for shareable links.
Stores only the analysis result and computed features — no raw messages.
"""
import os
from typing import Optional


def _client():
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def save_report(result: dict, features: dict) -> Optional[str]:
    client = _client()
    if not client:
        return None
    resp = client.table("reports").insert({
        "data": {"result": result, "features": features}
    }).execute()
    if resp.data:
        return resp.data[0]["id"]
    return None


def load_report(report_id: str) -> Optional[tuple[dict, dict]]:
    client = _client()
    if not client:
        return None
    try:
        resp = client.table("reports").select("data").eq("id", report_id).single().execute()
        if resp.data:
            d = resp.data["data"]
            return d["result"], d["features"]
    except Exception:
        pass
    return None
