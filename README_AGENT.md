# Odds Shopping + Personalization + Safe Execution (v1)

This module:
1) Pulls current quotes from The Odds API for a specific event/market.
2) Finds the best available price across your allowed books.
3) Sizes the stake using Kelly (scaled).
4) Executes safely via:
   - Paper trade (CSV log)
   - Telegram notification (you confirm manually)

## Env (.env.local or OS env)
- ODDS_API_KEY=your_the_odds_api_key
- ODDS_API_REGION=us
- TELEGRAM_BOT_TOKEN=123456:ABC...
- TELEGRAM_CHAT_ID=@YourChannelOrNumericId
- DEFAULT_BANKROLL=2000
- DEFAULT_KELLY_FRACTION=0.5
- MAX_STAKE_PCT=0.02
- PREFERRED_BOOKS=Pinnacle,DraftKings,FanDuel
- BLOCKED_BOOKS=

## Install
```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate
pip install -r requirements-agents.txt
