import requests
from typing import Optional
from .base import BetOrder, Executor
from config.settings import Settings

class TelegramNotifyExecutor(Executor):
    """
    Sends a Telegram message with the proposed order. 
    (For safety, we do NOT auto-place real bets. You confirm manually.)
    """
    def __init__(self):
        if not Settings.TELEGRAM_BOT_TOKEN or not Settings.TELEGRAM_CHAT_ID:
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        self.base = f"https://api.telegram.org/bot{Settings.TELEGRAM_BOT_TOKEN}"

    def execute(self, order: BetOrder) -> str:
        text = (
            f"🧾 *Bet Proposal*\n"
            f"Sport: `{order.sport_key}`\n"
            f"Event: `{order.event_id}`\n"
            f"Market: `{order.market}` Side: `{order.side}`\n"
            f"Book: *{order.bookmaker}* @ `{order.american_odds}`\n"
            f"Stake: `${order.stake}`\n"
            f"Reason: {order.reason}\n\n"
            "_Note: This is a paper/notification flow. Manually place if you agree._"
        )
        r = requests.post(
            f"{self.base}/sendMessage",
            json={"chat_id": Settings.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=20,
        )
        r.raise_for_status()
        msg = r.json()
        return str(msg.get("result", {}).get("message_id", ""))
