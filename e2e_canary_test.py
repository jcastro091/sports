#!/usr/bin/env python3
import os, time
from pipes import send_telegram, append_allbets, autopost_to_x, build_allbets_row_from_prediction

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

pred = {
    "event_id": f"CANARY-{int(time.time())}",
    "sport": "americanfootball_nfl",
    "league": "NFL",
    "market": "H2H Away",
    "side": "away",
    "team": "Canary Team",
    "opponent": "Opponent FC",
    "american_odds": -110,
    "decimal_odds": 1.91,
    "bin": "Fav",
    "proba": 0.95,
    "ev_percent": -1.0,
    "stake": 0.0,
    "reason": "External e2e_canary_test",
    "is_canary": 1
}

msg = (f"🧪 SELF-TEST PICK\n{pred['league']} {pred['market']}\n"
       f"{pred['team']} ({pred['side']}) @ {pred['opponent']}\n"
       f"p={pred['proba']:.2f} | odds={pred['american_odds']} | bin={pred['bin']}\n"
       f"reason={pred['reason']} | is_canary=1")

# 1) Telegram
if DRY_RUN:
    print("[DRY] TG:", msg)
else:
    send_telegram(msg)

# 2) AllBets
row = build_allbets_row_from_prediction(pred)
if DRY_RUN:
    print("[DRY] AllBets row:", row)
else:
    append_allbets(row)

# 3) X
tweet = f"🧪 Canary: {pred['team']} ({pred['side']}) {pred['market']} p={pred['proba']:.2f} odds={pred['american_odds']} #SharpsSignal"
if DRY_RUN:
    print("[DRY] X:", tweet)
else:
    autopost_to_x(tweet, is_canary=True)

print("e2e_canary_test complete.")
