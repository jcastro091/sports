#!/usr/bin/env python3
"""
pagekite_hc.py
A wrapper that runs the vendor pagekite.py as a subprocess and adds
Healthchecks.io heartbeats.

Env:
  HEARTBEAT_URL_PAGEKITE   - Healthchecks ping URL (required to enable pings)
  HEARTBEAT_FILE_PAGEKITE  - Optional path to a local "touch" file for watchdogs
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
import subprocess
import signal
from pathlib import Path
from typing import Optional

# -------- Healthchecks (stdlib only: urllib) --------
try:
    from urllib.request import Request, urlopen
except Exception:  # extremely unlikely
    Request = None  # type: ignore
    urlopen = None  # type: ignore

HC_URL = os.getenv("HEARTBEAT_URL_PAGEKITE", "").strip()
HC_FILE = os.getenv("HEARTBEAT_FILE_PAGEKITE", "").strip()
_SERVICE = "pagekite"
_last_bucket = -1
_stop_evt = threading.Event()

def _hc_touch() -> None:
    if not HC_FILE:
        return
    try:
        p = Path(HC_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        pass  # never crash due to heartbeat I/O

def _hc_http(method: str, url: str, payload: Optional[dict] = None, timeout: float = 5.0) -> None:
    if Request is None or urlopen is None:
        return
    try:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url=url, data=data, headers=headers, method=method.upper())
        with urlopen(req, timeout=timeout):
            pass
    except Exception:
        pass

def _hc_ping(suffix: str = "", payload: Optional[dict] = None) -> None:
    if not HC_URL:
        return
    url = HC_URL.rstrip("/")
    if suffix:
        url = f"{url}/{suffix.lstrip('/')}"
    _hc_http("POST" if payload else "GET", url, payload)

def _hc_tick() -> None:
    global _last_bucket
    now_bucket = int(time.time() // 60)
    if now_bucket != _last_bucket:
        _last_bucket = now_bucket
        _hc_ping("", {"svc": _SERVICE, "ts": int(time.time())})
        _hc_touch()

def _hc_background() -> None:
    while not _stop_evt.is_set():
        try:
            _hc_tick()
        except Exception:
            pass
        _stop_evt.wait(60)

# -------- Runner --------
def main() -> int:
    # Locate vendor script next to this file
    here = Path(__file__).resolve().parent
    vendor = here / "pagekite.py"
    if not vendor.exists():
        print(f"[pagekite_hc] ERROR: Could not find vendor script at: {vendor}", file=sys.stderr)
        return 2

    # Send startup ping + start ticker
    _hc_ping("start", {"svc": _SERVICE, "status": "start", "ts": int(time.time())})
    _hc_touch()
    t = threading.Thread(target=_hc_background, name="hc-ticker", daemon=True)
    t.start()

    # Launch vendor script (pass through all args)
    cmd = [sys.executable, "-u", str(vendor), *sys.argv[1:]]
    proc = subprocess.Popen(cmd)  # inherit stdio

    # Forward Ctrl+C gracefully
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        try:
            # Best-effort terminate on both platforms
            proc.terminate()
        except Exception:
            pass
        rc = proc.wait()

    # Stop ticker and send final ping
    _stop_evt.set()
    try:
        t.join(timeout=2)
    except Exception:
        pass

    if rc == 0:
        _hc_ping("stop", {"svc": _SERVICE, "status": "stop", "ts": int(time.time())})
        _hc_touch()
    else:
        _hc_ping("fail", {"svc": _SERVICE, "status": "fail", "ts": int(time.time()), "exit_code": rc})
        _hc_touch()
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
