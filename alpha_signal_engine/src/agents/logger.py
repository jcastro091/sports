import os, requests
from typing import Any, Dict

AUDIT_URL = os.getenv("AUDIT_URL", "http://localhost:8099/api/audit_log")

def audit_log(user: str, model: str, input: Dict[str,Any], output: Dict[str,Any],
              decision: str, status: str, meta: Dict[str,Any] | None = None):
    try:
        requests.post(AUDIT_URL, json={
            "user": user, "model": model, "input": input, "output": output,
            "decision": decision, "status": status, "meta": meta or {}
        }, timeout=5)
    except Exception as e:
        print(f"[audit-log] warn: {e}")
