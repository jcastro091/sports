import csv
import requests
import os
from datetime import datetime
from dateutil import parser
import logging
from difflib import get_close_matches
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import pytz
import time

# === CONFIG ===
API_KEY = "32d6c615af0e04784499615af0e04784499615c4b78d013"
SPORTS = [
    'basketball_nba', 'baseball_mlb', 'tennis_wta_french_open', 'tennis_atp_french_open',
    'boxing_boxing', 'basketball_wnba', 'basketball_euroleague',
    'soccer_usa_mls', 'baseball_kbo', 'mma_mixed_martial_arts'
]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Google Sheets Setup ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("telegrambetlogger-35856685bc29.json", scope)
client = gspread.authorize(creds)

allbets_sheet = client.open("ConfirmedBets").worksheet("AllBets")
confirmedbets_sheet = client.open("ConfirmedBets").worksheet("ConfirmedBets")

def tag_row(row):
    tags = []
    try:
        predicted = row.get("Predicted", "")
        movement = row.get("Movement", "")
        time_str = row.get("Game Time", "")
        timestamp_str = row.get("Timestamp", "")

        logging.debug(f"Processing row: Predicted={predicted}, Movement={movement}, Game Time={time_str}, Timestamp={timestamp_str}")

        # Basic parsing
        eastern = pytz.timezone("US/Eastern")
        game_time = parser.parse(time_str).astimezone(eastern) if time_str else None
        bet_time = parser.parse(timestamp_str).astimezone(eastern) if timestamp_str else None

        # Tag by time
        if game_time and bet_time:
            diff_minutes = (game_time - bet_time).total_seconds() / 60
            logging.debug(f"Time difference (min): {diff_minutes}")
            if diff_minutes < 15:
                tags.append("Late Bet")
            elif diff_minutes > 90:
                tags.append("Early Bet")

        # Tag by market
        if "Spread" in predicted:
            tags.append("Spread Bet")
        elif predicted in ["Over", "Under"]:
            tags.append("Total Bet")
        elif predicted:
            if predicted == row.get("Away Team") or predicted == row.get("Home Team"):
                # Try to infer underdog by odds movement
                if movement.startswith("+"):
                    try:
                        odds_val = int(movement[1:4])
                        if odds_val >= 120:
                            tags.append("H2H Underdog")
                    except:
                        logging.warning(f"Failed to parse odds from movement: {movement}")

        # Tag based on sharp move
        try:
            start, peak, end = movement.replace("→", "").split()
            start_val = int(start)
            peak_val = int(peak)
            end_val = int(end.split()[0])
            if abs(peak_val - end_val) >= 20:
                tags.append("A+ Movement")
        except Exception as e:
            logging.warning(f"Could not parse movement shift: {movement} → {e}")

    except Exception as e:
        logging.error(f"Tagging error: {e}")

    return ", ".join(tags)

# === Update Tags in Both Sheets ===
for sheet, label in [(allbets_sheet, "AllBets"), (confirmedbets_sheet, "ConfirmedBets")]:
    logging.info(f"Processing sheet: {label}")
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # Ensure Tags column exists
    if "Tags" not in df.columns:
        sheet.add_cols(1)
        current_headers = sheet.row_values(1)
        sheet.update_cell(1, len(current_headers) + 1, "Tags")
        df["Tags"] = ""

    tags_col_idx = df.columns.get_loc("Tags") + 1  # 1-based index for gspread

    for i, row in df.iterrows():
        existing = str(row.get("Tags", "")).strip()
        if existing:
            logging.info(f"Skipping row {i+2} in {label} (already tagged: {existing})")
            continue

        tag = tag_row(row)
        sheet.update_cell(i + 2, tags_col_idx, tag)
        logging.info(f"Tagged row {i+2} in {label}: {tag}")
        time.sleep(1)  # adjust pacing as needed

print("✅ Tags updated for AllBets and ConfirmedBets (only blank rows)")

