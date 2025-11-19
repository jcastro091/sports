import os

def _csv(env_name: str, default: str = ""):
    raw = os.getenv(env_name, default)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

class Settings:
    ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # The Odds API v4 key
    ODDS_API_REGION = os.getenv("ODDS_API_REGION", "us")  # us / us2 / eu / uk
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # numeric chat id or @channelusername

    DEFAULT_BANKROLL = float(os.getenv("DEFAULT_BANKROLL", "1000"))
    DEFAULT_KELLY_FRACTION = float(os.getenv("DEFAULT_KELLY_FRACTION", "0.5"))
    MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.02"))  # 2% cap default

    # Use Odds API bookmaker names; we’ll normalize case.
    PREFERRED_BOOKS = _csv("PREFERRED_BOOKS", "Pinnacle,DraftKings,FanDuel")
    BLOCKED_BOOKS = _csv("BLOCKED_BOOKS", "")
