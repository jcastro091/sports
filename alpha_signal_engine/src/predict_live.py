import os, yaml, requests
from datetime import datetime
import argparse

def maybe_load_settings(path="config/settings.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def send_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()

def build_demo_prediction():
    # TODO: swap in your real predictor when ready
    return {"sport":"NBA","market":"H2H","pick":"LAL","confidence":0.72,"kelly":0.018}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Print a demo alert and exit (no Telegram).")
    args = p.parse_args()

    s = maybe_load_settings()
    token = os.getenv("TELEGRAM_BOT_TOKEN", s.get("apis", {}).get("telegram_bot_token", ""))
    chat_id = os.getenv("TELEGRAM_CHAT_ID", s.get("apis", {}).get("telegram_chat_id", ""))

    pred = build_demo_prediction()
    msg = (
        f"🤖 <b>AI Signal</b>\n"
        f"Sport: {pred['sport']} | Market: {pred['market']}\n"
        f"Pick: <b>{pred['pick']}</b>\n"
        f"Confidence: {pred['confidence']:.2f} | Kelly: {pred['kelly']:.3f}\n"
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    )

    if args.smoke:
        print("[predict_live] SMOKE:", msg)
        return

    if token and chat_id and not token.startswith("REPLACE_ME"):
        send_telegram(token, chat_id, msg)
        print("[predict_live] sent telegram alert")
    else:
        print("[predict_live] Telegram disabled (missing token/chat).")
        print("[predict_live] (dry-run) would send:", msg)

if __name__ == "__main__":
    main()
