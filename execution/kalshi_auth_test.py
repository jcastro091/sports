#!/usr/bin/env python3
"""
Quick Kalshi auth sanity check.

Usage:
    py -3.11 kalshi_auth_test.py
"""

import sys, os, json, requests
from pathlib import Path

# ---- USER CONFIG (edit if needed) ----
BASE_URL = "https://api.elections.kalshi.com"
KEY_ID = os.getenv("KALSHI_KEY_ID") or "YOUR_REAL_KEY_ID"
PRIVATE_KEY_FILE = os.getenv("KALSHI_PRIVATE_KEY_FILE") or r"C:\Users\Jcast\OneDrive\Documents\sports-repo\keys\kalshi-demo-keyPROD.pem"
# -------------------------------------

def load_pem(p):
    t = Path(p).read_text().strip()
    if "PRIVATE KEY" not in t:
        raise ValueError(f"Doesn't look like a PEM private key: {p}")
    return t

def get_token(sess, base, key_id, pem):
    # Elections host uses /auth/token (no /trade-api/v2 prefix)
    paths = ["/auth/token", "/trade-api/v2/auth/token"]
    last = None
    for path in paths:
        r = sess.post(f"{base}{path}", json={"key_id": key_id, "private_key": pem}, timeout=10)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("application/json"):
            js = r.json()
            if js.get("token"): return js["token"], path
        last = (r.status_code, r.text[:300], path)
    raise RuntimeError(f"Auth failed on both endpoints. Last: {last}")

def main():
    print(f"🔍 Checking Kalshi auth at {BASE_URL} …")
    try:
        pem = load_pem(PRIVATE_KEY_FILE)
        sess = requests.Session()
        sess.headers["Content-Type"] = "application/json"

        token, used = get_token(sess, BASE_URL, KEY_ID, pem)
        print(f"✅ Auth OK via {used}")
        sess.headers["Authorization"] = f"Bearer {token}"

        r = sess.get(f"{BASE_URL}/trade-api/v2/portfolio/balance", timeout=10)
        print("Balance:", r.status_code, r.text[:400])
        if r.status_code == 200:
            print("✅ Connected!")
        else:
            print("❌ Balance call failed; token accepted but scope/env may be off.")
            sys.exit(2)

    except Exception as e:
        print("❌", e)
        sys.exit(1)

if __name__ == "__main__":
    main()