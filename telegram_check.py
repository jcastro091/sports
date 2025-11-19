#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_check.py
-----------------
Checks Telegram bot connectivity and verifies it can message your channel.
Usage:
  py -3.11 telegram_check.py --test "Hello from SharpsSignal"
Requires:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
in your .env or environment.
"""

import os
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

BOT = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

def main():
    if not BOT:
        print("❌ TELEGRAM_BOT_TOKEN missing in env/.env")
        return
    # getMe
    r = requests.get(f"https://api.telegram.org/bot{BOT}/getMe", timeout=15)
    if r.status_code == 200:
        me = r.json().get("result", {})
        print(f"✅ Bot OK: @{me.get('username')} (id {me.get('id')})")
    else:
        print(f"❌ getMe failed: {r.status_code} {r.text}")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", dest="test", help="Send a test message to TELEGRAM_CHAT_ID")
    args = parser.parse_args()

    if args.test:
        if not CHAT:
            print("❌ TELEGRAM_CHAT_ID missing in env/.env")
            return
        payload = {"chat_id": CHAT, "text": args.test, "disable_web_page_preview": True}
        r2 = requests.post(f"https://api.telegram.org/bot{BOT}/sendMessage", json=payload, timeout=20)
        if r2.status_code == 200:
            print("✅ Sent test message to chat.")
        else:
            print(f"❌ Failed to send: {r2.status_code} {r2.text}")

if __name__ == "__main__":
    main()
