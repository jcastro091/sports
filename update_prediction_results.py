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

# === CONFIG ===
API_KEY = "32d6c615af0e04784499615c4b78d013"
# CSV_FILE = "prediction_alert_log - Copy.csv"
# TEMP_FILE = "temp_predictions.csv"
SPORTS = [
    'basketball_nba', 'baseball_mlb', 'tennis_wta_french_open', 'tennis_atp_french_open',
    'boxing_boxing', 'basketball_wnba', 'basketball_euroleague',
    'soccer_usa_mls', 'baseball_kbo', 'mma_mixed_martial_arts'
]
REQUIRED_FIELDS = [
    "Timestamp", "Sport", "Away Team", "Home Team", "Market", "Direction",
    "Movement", "Predicted", "Game Time", "Spread Line Home", "Spread Line Away", "Total Line"
]

# === Google Sheets Setup ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("telegrambetlogger-35856685bc29.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ConfirmedBets").worksheet("AllBets")
rows = sheet.get_all_records()
df = pd.DataFrame(rows)

# === LOGGING ===
logging.basicConfig(
    filename="task_scheduler_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("✅ Script started successfully!")

# === HELPERS ===
def get_final_scores(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/?daysFrom=3&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"❌ Error fetching scores for {sport_key}: {e}")
        return []

def to_int(val):
    try:
        return int(val)
    except:
        return None

def is_match(team1, team2):
    return bool(get_close_matches(team1.lower(), [team2.lower()], cutoff=0.8))

def normalize_date(value):
    try:
        return parser.parse(value).date()
    except:
        return None

# Preprocess Game Date in the Google Sheet
df['Game Date'] = df['Game Time'].apply(normalize_date)

# === STEP 1: Load CSV ===
correct_headers = REQUIRED_FIELDS
with open(CSV_FILE, newline='', encoding='utf-8') as infile:
    next(infile)
    reader = csv.DictReader(infile, fieldnames=correct_headers)
    clean_rows = list(reader)
    logging.info(f"📄 Loaded {len(clean_rows)} rows from {CSV_FILE}")

# === STEP 2: Evaluate and Compare ===
wins, losses, pending = 0, 0, 0
seen_games = set()
full_fieldnames = correct_headers + ["Actual Winner", "Prediction Result"]

with open(TEMP_FILE, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=full_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for row in clean_rows:
        try:
            if None in row or any(k is None for k in row.keys()):
                logging.warning(f"⚠️ Skipping malformed row: {row}")
                continue

            original_row = row.copy()
            game_key = f"{row['Home Team'].strip().lower()}_{row['Away Team'].strip().lower()}_{row['Game Time'].split()[0]}"

            if game_key in seen_games:
                writer.writerow(row)
                continue

            if row.get("Prediction Result", "").strip() in ["Win", "Lose", "Skipped"]:
                seen_games.add(game_key)
                writer.writerow(row)
                continue

            sport_key = row['Sport']
            predicted = row['Predicted']
            away = row['Away Team']
            home = row['Home Team']

            game_time = parser.parse(row['Game Time'], tzinfos={"EDT": -14400}).astimezone(None).date()
            row_game_date = normalize_date(row['Game Time'])

            total_line = float(row.get('Total Line') or 0)
            spread_line_home = float(row.get('Spread Line Home') or 0)
            spread_line_away = float(row.get('Spread Line Away') or 0)

            if sport_key not in SPORTS or not predicted.strip():
                original_row['Actual Winner'] = "N/A"
                original_row['Prediction Result'] = "Skipped"
                writer.writerow(original_row)
                continue

            scores = get_final_scores(sport_key)
            match = next(
                (
                    g for g in scores
                    if is_match(g['home_team'], home) and is_match(g['away_team'], away)
                    and abs((parser.parse(g['commence_time']).replace(tzinfo=None).date() - game_time).days) <= 1
                    and (g.get('completed') or any(int(s.get('score', 0)) > 0 for s in g.get("scores", [])))
                ),
                None
            )

            if not match or len(match.get("scores", [])) < 2:
                original_row['Actual Winner'] = "Pending"
                original_row['Prediction Result'] = "Pending"
                pending += 1
                writer.writerow(original_row)
                continue

            score_data = match.get("scores")
            home_score = to_int(score_data[0]['score']) if match['home_team'] == score_data[0]['name'] else to_int(score_data[1]['score'])
            away_score = to_int(score_data[0]['score']) if match['away_team'] == score_data[0]['name'] else to_int(score_data[1]['score'])

            if home_score is None or away_score is None:
                original_row['Actual Winner'] = "Pending"
                original_row['Prediction Result'] = "Pending"
                pending += 1
                writer.writerow(original_row)
                continue

            total_score = home_score + away_score
            actual_winner = home if home_score > away_score else away
            original_row['Actual Winner'] = actual_winner

            result = "Unknown"
            if predicted == home or predicted == away:
                result = "Win" if predicted == actual_winner else "Lose"
            elif predicted == "Over":
                result = "Win" if total_score > total_line else "Lose"
            elif predicted == "Under":
                result = "Win" if total_score < total_line else "Lose"
            elif predicted == f"{home} (Spread)":
                result = "Win" if (home_score - away_score) > spread_line_home else "Lose"
            elif predicted == f"{away} (Spread)":
                result = "Win" if (away_score - home_score) > spread_line_away else "Lose"

            original_row['Prediction Result'] = result
            wins += result == "Win"
            losses += result == "Lose"
            pending += result == "Pending"
            writer.writerow(original_row)

            # === Google Sheet Update Logic ===
            df_idx = df[
                (df['Home Team'].str.strip().str.lower() == home.strip().lower()) &
                (df['Away Team'].str.strip().str.lower() == away.strip().lower()) &
                (df['Game Date'] == row_game_date)
            ].index

            if not df_idx.empty:
                row_number = df_idx[0] + 2  # Offset for header
                sheet.update_cell(row_number, df.columns.get_loc("Actual Winner") + 1, actual_winner)
                sheet.update_cell(row_number, df.columns.get_loc("Prediction Result") + 1, result)
            else:
                logging.warning(f"⚠️ No matching row found in Google Sheet for {home} vs {away} on {row['Game Time']}")

        except Exception as e:
            logging.error(f"❌ Error processing row: {e}")
            continue

    logging.info(f"✍️ Writing results to {TEMP_FILE}")

# === STEP 3: Replace Original File ===
try:
    os.replace(TEMP_FILE, CSV_FILE)
    logging.info(f"✅ Overwrote {CSV_FILE} with updated predictions.")
    print(f"✅ Predictions updated. {wins} Wins, {losses} Losses, {pending} Pending.")
except Exception as e:
    logging.error(f"❌ Could not overwrite {CSV_FILE}: {e}")

logging.info(f"📊 Summary — Wins: {wins}, Losses: {losses}, Pending: {pending}")
input("Press Enter to close...")
