#!/usr/bin/env python3
import os
import logging
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from gspread.utils import rowcol_to_a1
from telegram import Bot

# === CONFIG ===
SHEET_ID           = os.getenv("SHEET_ID",        "1Vl9ceOFI5LSqXuVoGGUE3h4JZ685oZfRbbPIPdnGsGo")
WORKSHEET_NAME     = os.getenv("WORKSHEET_NAME",  "ConfirmedBets")
CREDENTIALS_FILE   = os.getenv("GOOGLE_CREDS_JSON",
                               r"C:\Users\Jcast\OneDrive\Documents\sports-repo\telegrambetlogger-35856685bc29.json")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN",  "7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","-1002623647951")

# === SHEET HEADER NAMES ===
HEADER_POSTED      = "Posted?"
HEADER_SPORT       = "Sport"
HEADER_AWAY        = "Away Team"
HEADER_HOME        = "Home Team"
HEADER_MARKET      = "Market"
HEADER_DIRECTION   = "Direction"
HEADER_PREDICTED   = "Predicted"
HEADER_CONFIDENCE  = "Confidence"
HEADER_GAME_TIME   = "Game Time"
HEADER_ODDS        = "Odds taken"
HEADER_RISK        = "Risk"
HEADER_PROFIT      = "Profit"

# === LOGGING ===
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    # 1. Authenticate with Google Sheets
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)

    # 2. Open spreadsheet & worksheet
    sh  = client.open_by_key(SHEET_ID)
    ws  = sh.worksheet(WORKSHEET_NAME)
    headers = ws.row_values(1)

    # 3. Locate “Posted?” column
    try:
        posted_col_idx = headers.index(HEADER_POSTED) + 1
    except ValueError:
        logging.error("'%s' column not found. Available headers: %s",
                      HEADER_POSTED, headers)
        return

    # 4. Load all records
    records = ws.get_all_records()

    # 5. Initialize Telegram bot
    bot = Bot(token=TELEGRAM_TOKEN)

    # 6. Iterate rows and send any that aren't marked TRUE
    for row_idx, rec in enumerate(records, start=2):
        if str(rec.get(HEADER_POSTED)).upper() != "TRUE":
            # Build your custom alert message
            msg = (
                f"📣 Bet Alert: {rec.get(HEADER_SPORT)} | "
                f"{rec.get(HEADER_AWAY)} @ {rec.get(HEADER_HOME)}\n"
                f"Market: {rec.get(HEADER_MARKET)}, {rec.get(HEADER_DIRECTION)}➤{rec.get(HEADER_PREDICTED)}\n"
                f"Confidence: {rec.get(HEADER_CONFIDENCE)}, Odds: {rec.get(HEADER_ODDS)}, Risk: {rec.get(HEADER_RISK)}\n"
                f"Game Time: {rec.get(HEADER_GAME_TIME)}"
            )

            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logging.info("Posted row %d: %s", row_idx, msg)
            except Exception as e:
                logging.error("Failed posting row %d: %s", row_idx, e)
                continue

            # 7. Mark this row as posted
            cell = rowcol_to_a1(row_idx, posted_col_idx)
            ws.update_acell(cell, "TRUE")

    logging.info("✅ All new confirmed bets have been posted.")


if __name__ == "__main__":
    main()
