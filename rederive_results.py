import csv
import requests
from dateutil import parser
from difflib import get_close_matches
from datetime import timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)

API_KEY    = "32d6c615af0e04784499615c4b78d013"
INPUT_CSV  = "ConfirmedBets - AllBets (9).csv"
OUTPUT_CSV = "ConfirmedBets_AllBets_redrived.csv"
SPORTS     = {
    'basketball_nba','baseball_mlb','tennis_wta_french_open','tennis_atp_french_open',
    'boxing_boxing','basketball_wnba','basketball_euroleague',
    'soccer_usa_mls','baseball_kbo'
}

def get_final_scores(sport_key):
    url   = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    # first try your full 30-day window
    resp  = requests.get(url, params={"daysFrom": 30, "apiKey": API_KEY})
    if resp.status_code == 422:
        # fallback to last 3 days
        logging.warning(f"daysFrom=30 not allowed for {sport_key}, retrying with daysFrom=3")
        resp = requests.get(url, params={"daysFrom": 3, "apiKey": API_KEY})
    resp.raise_for_status()
    return resp.json()

def to_int(x):
    try: return int(x)
    except: return 0

def is_match(a, b):
    return bool(get_close_matches(a.lower(), [b.lower()], cutoff=0.8))

with open(INPUT_CSV, newline='', encoding='utf8') as fin, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf8') as fout:

    reader  = csv.DictReader(fin)
    fieldnames = reader.fieldnames
    writer  = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        sport      = row['Sport']
        predicted  = row['Predicted'].strip()
        home       = row['Home Team'].strip()
        away       = row['Away Team'].strip()
        gm         = row['Game Time']

        # default to leaving original if we can’t resolve
        actual_winner = row['Actual Winner']
        result        = row['Prediction Result']

        if sport in SPORTS and predicted:
            # parse your kickoff (EDT → UTC−4)
            try:
                sheet_dt = parser.parse(gm, tzinfos={"EDT": -14400})
            except:
                pass
            else:
                # pull scores once per sport
                scores = get_final_scores(sport)

                # find completed game between these two within ±3h
                candidates = []
                for g in scores:
                    if not g.get('completed'): continue
                    if not (is_match(g['home_team'], home) and is_match(g['away_team'], away)):
                        continue
                    api_dt = parser.parse(g['commence_time'])  # this is UTC
                    if abs(api_dt - sheet_dt) <= timedelta(hours=3):
                        candidates.append((abs(api_dt - sheet_dt), g))

                if candidates:
                    # pick the closest
                    _, game = min(candidates, key=lambda x: x[0])
                    # map scores by name
                    score_map = {s['name'].lower(): to_int(s['score']) 
                                 for s in game['scores']}
                    hscore = score_map.get(home.lower(), 0)
                    ascore = score_map.get(away.lower(), 0)

                    # draw detection
                    if hscore > ascore:
                        actual_winner = home
                    elif ascore > hscore:
                        actual_winner = away
                    else:
                        actual_winner = "Draw"

                    # H2H: only a correct pick of Draw would Win
                    result = "Win" if predicted == actual_winner else "Lose"

        # overwrite and write row
        row['Actual Winner']     = actual_winner
        row['Prediction Result'] = result
        writer.writerow(row)

print(f"✅ Wrote re-derived results to {OUTPUT_CSV}")
