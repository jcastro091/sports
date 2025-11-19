# utils/heartbeat.py
from __future__ import annotations
import os, time, json, urllib.request
from pathlib import Path

# Touch a local heartbeat file (used by a simple watchdog)
def touch(path: str | os.PathLike = "/var/run/ss_autoposter_heartbeat") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

# Return a coarse time bucket index for "once per N minutes" logic
def every(minutes: int) -> int:
    if minutes <= 0:
        minutes = 1
    return int(time.time() // (minutes * 60))

# Ping a remote heartbeat URL (e.g., Healthchecks.io / Uptime Kuma push)
def ping(url_env: str, payload: dict | None = None, timeout: float = 5.0) -> None:
    url = os.getenv(url_env, "").strip()
    if not url:
        return
    try:
        if payload is not None:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=timeout)
        else:
            urllib.request.urlopen(url, timeout=timeout)
    except Exception as e:
        print(f"[heartbeat] warn: {e}")
