# ops/notify_tg.py
import os, sys, urllib.parse, urllib.request

BOT  = os.getenv("TG_BOT_TOKEN", "")
CHAT = os.getenv("TG_CHAT_ID", "")

def send(msg: str) -> None:
    if not BOT or not CHAT:
        print("[notify_tg] missing TG_BOT_TOKEN or TG_CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage?chat_id={CHAT}&text={urllib.parse.quote(msg)}&parse_mode=Markdown"
    urllib.request.urlopen(url, timeout=5)

if __name__ == "__main__":
    send(sys.argv[1] if len(sys.argv) > 1 else "ping")
