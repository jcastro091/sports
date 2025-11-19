#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffer_check.py
---------------
Quick Buffer connectivity tester.
• Reads BUFFER_ACCESS_TOKEN from .env or env
• Lists your available profiles with names + IDs
• Optional: --post "Your test message" to queue a test post (now=False)
Usage:
  py -3.11 buffer_check.py
  py -3.11 buffer_check.py --post "TEST from SharpsSignal"
"""

import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

BUFFER_ACCESS_TOKEN = os.getenv("BUFFER_ACCESS_TOKEN")

API_BASE = "https://api.bufferapp.com/1"

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def list_profiles():
    url = f"{API_BASE}/profiles.json"
    params = {"access_token": BUFFER_ACCESS_TOKEN}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        die(f"❌ Failed to fetch profiles: {r.status_code} {r.text}")
    profiles = r.json()
    if not isinstance(profiles, list):
        die(f"❌ Unexpected response: {profiles}")
    print("✅ Buffer profiles found:")
    for p in profiles:
        name = p.get("formatted_username") or p.get("service_username") or p.get("service", "profile")
        pid = p.get("id")
        service = p.get("service")
        print(f"  - {service:6s}  {name}  | profile_id={pid}")
    return [p.get("id") for p in profiles if p.get("id")]

def post_test(message: str, profile_ids):
    if not profile_ids:
        die("No profile IDs available to post.")
    url = f"{API_BASE}/updates/create.json"
    headers = {"Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}"}
    for pid in profile_ids:
        data = {"profile_ids[]": pid, "text": message, "now": False}
        r = requests.post(url, data=data, headers=headers, timeout=20)
        if r.status_code in (200,201):
            print(f"✅ Queued TEST post to {pid}")
        else:
            print(f"❌ Failed to queue to {pid}: {r.status_code} {r.text}")

def main():
    if not BUFFER_ACCESS_TOKEN:
        die("❌ BUFFER_ACCESS_TOKEN missing. Put it in .env next to this script or set the env var.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", dest="post", help="Queue a test post to all profiles (now=False)")
    args = parser.parse_args()
    pids = list_profiles()
    if args.post:
        post_test(args.post, pids)

if __name__ == "__main__":
    main()
