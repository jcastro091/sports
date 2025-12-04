# sheets.py — update ONLY AllObservations with robust matching + better logging + sport mapping
import logging
import re
from datetime import timedelta

import requests
import gspread
import pandas as pd
import pytz
import os
from pathlib import Path

from dateutil import parser
from difflib import get_close_matches
from gspread.utils import rowcol_to_a1
from oauth2client.service_account import ServiceAccountCredentials

# ── CONFIG ───────────────────────────────────────────────────────────
API_KEY = "7e79a3f0859a50153761270b1d0e867f"

# Valid Odds API keys we support directly
SPORTS = [
    "baseball_mlb",
    "basketball_nba",
    "basketball_wnba",
    "basketball_ncaab",
    "icehockey_nhl",
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "boxing_boxing",
    "mma_mixed_martial_arts",
    "soccer_epl",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_usa_mls",
]


# Map human sheet labels (Sport, League) → Odds API keys
SPORT_KEY_MAP = {
    ("Basketball", "WNBA"): "basketball_wnba",
    ("Basketball", "NBA"): "basketball_nba",
    ("Basketball", "EUROLEAGUE"): "basketball_euroleague",
    ("Baseball", "MLB"): "baseball_mlb",
    ("Baseball", "KBO"): "baseball_kbo",
    ("Soccer", "MLS"): "soccer_usa_mls",
    ("Soccer", "EPL"): "soccer_epl",
    ("Soccer", "LIGUE 1"): "soccer_france_ligue_one",
    ("Soccer", "LIGUE ONE"): "soccer_france_ligue_one",  # common typo
    ("Soccer", "BUNDESLIGA"): "soccer_germany_bundesliga",
    ("Soccer", "SERIE A"): "soccer_italy_serie_a",
    ("Ice Hockey", "NHL"): "icehockey_nhl",
    ("MMA", ""): "mma_mixed_martial_arts",
    ("American Football", "NFL"): "americanfootball_nfl",
    ("American Football", "NCAAF"): "americanfootball_ncaaf",
    ("Boxing", ""): "boxing_boxing",
}

# How far from the *row timestamp* we’re willing to look for a matching game.
NAME_MATCH_WINDOW = timedelta(days=5)   # handles schedule drift, day rollovers, etc.

TZ_EASTERN = pytz.timezone("US/Eastern")
TZ_UTC = pytz.UTC


# ── LOGGING (robust on Windows) ──────────────────────────────────────
import sys, errno

class SafeStreamHandler(logging.StreamHandler):
    """Swallow console flush errors (e.g., EINVAL/EPIPE on Windows/redirects)."""
    def handleError(self, record):
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.EINVAL, errno.EPIPE):
            return  # ignore
        try:
            super().handleError(record)
        except Exception:
            pass

logging.raiseExceptions = False  # never crash on logging errors

# File handler (UTF-8) + safe console handler
_log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
file_handler = logging.FileHandler("task_scheduler_log.txt", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(_log_fmt))

console_handler = SafeStreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

root = logging.getLogger()
root.setLevel(logging.DEBUG)
# Clear any previously added handlers (if this module is reloaded)
for h in list(root.handlers):
    root.removeHandler(h)
root.addHandler(file_handler)
root.addHandler(console_handler)

logging.debug("Logger initialized (file + safe console)")


# ── SHEETS ───────────────────────────────────────────────────────────
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Path to Google service account JSON (can be overridden via env var GOOGLE_CREDS_FILE)
BASE_DIR = Path(__file__).resolve().parent
CREDS_FILE = os.getenv(
    "GOOGLE_CREDS_FILE",
    str(BASE_DIR / "creds" / "telegrambetlogger-35856685bc29.json"),
)

CREDS = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)

CLIENT = gspread.authorize(CREDS)

# --- Add this helper block (right after CLIENT = gspread.authorize(CREDS)) ---
import time
from gspread.exceptions import APIError

def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    if isinstance(e, APIError):
        try:
            # gspread APIError often has response/status_code
            sc = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
        except Exception:
            sc = None
        if sc and 500 <= sc < 600:
            return True
    return any(x in s for x in ["apierror: [503]", "apierror: [500]", "service is currently unavailable", "backend error"])

def _retry(fn, *args, retries=5, base_delay=0.8, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_transient(e) and i < retries - 1:
                delay = base_delay * (2 ** i)
                logging.warning(f"Transient error calling {getattr(fn, '__name__', fn)}: {e} → retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise

def open_spreadsheet_with_retry(client, title: str):
    return _retry(client.open, title)

def worksheet_with_retry(ss, title: str):
    return _retry(ss.worksheet, title)

# --- Use the retried open + worksheet getters instead of direct calls ---
SS = open_spreadsheet_with_retry(CLIENT, "ConfirmedBets")
WS = worksheet_with_retry(SS, "AllObservations")
WS_ALLBETS = worksheet_with_retry(SS, "AllBets")



# ── HELPERS ──────────────────────────────────────────────────────────
_scores_cache: dict[str, list[dict]] = {}  # (sport_key) -> list of games (Odds API)

TEAM_ALIASES = {
    # minimal alias expansion to help fuzzy matching
    "la": "los angeles",
    "st.": "saint",
    "st ": "saint ",
    "ny": "new york",
    "sf": "san francisco",
    "sd": "san diego",
    "tb": "tampa bay",
}

def scores_from_game(game: dict) -> tuple[int | None, int | None]:
    if not game or not game.get("completed"):
        return None, None
    scores = {s["name"].lower(): int(float(s["score"])) for s in game.get("scores", []) if "name" in s}
    home = (game.get("home_team") or "").lower()
    away = (game.get("away_team") or "").lower()
    return scores.get(home), scores.get(away)

def totals_result(final_home: int, final_away: int, total_line) -> str | None:
    try:
        tl = float(str(total_line).strip())
    except Exception:
        return None
    total = final_home + final_away
    if total > tl:  return "Over"
    if total < tl:  return "Under"
    return "Push"

def spread_cover(final_home: int, final_away: int, spread_home, spread_away) -> str | None:
    sh = None
    try: sh = float(str(spread_home).strip())
    except Exception: pass
    sa = None
    try: sa = float(str(spread_away).strip())
    except Exception: pass
    if sh is None and sa is None:
        return None
    if sh is None and sa is not None:
        sh = -sa  # away line exists ⇒ home line is its negative
    home_adj = final_home + sh
    if home_adj > final_away:  return "Home"
    if home_adj < final_away:  return "Away"
    return "Push"

def normalize_market(s: str) -> str:
    s = (s or "").strip().lower()
    if "total" in s: return "totals"
    if "spread" in s: return "spread"
    if s in ("h2h", "moneyline", "ml"): return "h2h"
    return s

def norm_team(s: str) -> str:
    s2 = re.sub(r"[^a-z0-9& ]+", " ", (s or "").lower()).strip()
    for k, v in TEAM_ALIASES.items():
        s2 = re.sub(rf"\b{k}\b", v, s2)
    s2 = re.sub(r"\s+", " ", s2)
    return s2

def close_same(a: str, b: str, cutoff: float = 0.72) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    hits = get_close_matches(a, [b], cutoff=cutoff)
    return bool(hits)

def as_utc(dt):
    if dt.tzinfo is None:
        dt = TZ_EASTERN.localize(dt)
    return dt.astimezone(TZ_UTC)

def _days_from_for_sport(sport_key: str) -> int:
    return 5 if sport_key == "baseball_mlb" else 3

def get_final_scores(sport_key: str) -> list[dict]:
    if sport_key in _scores_cache:
        return _scores_cache[sport_key]

    def _fetch(days_from: int) -> list[dict]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        r = requests.get(url, params={"daysFrom": days_from, "apiKey": API_KEY})
        if r.status_code == 401:
            logging.error(f"[{sport_key}] Unauthorized — check API key")
            return []
        if r.status_code == 422:
            if days_from > 3:
                logging.warning(f"[{sport_key}] daysFrom={days_from} returned 422; retrying with 3")
                return _fetch(3)
        r.raise_for_status()
        return r.json()

    days_from = _days_from_for_sport(sport_key)
    data = _fetch(days_from)
    _scores_cache[sport_key] = data or []
    logging.debug(f"[scores] cached {sport_key} with daysFrom={min(days_from,3) if not data else days_from}, games={len(_scores_cache[sport_key])}")
    return _scores_cache[sport_key]

def actual_winner_from_game(game: dict) -> str | None:
    if not game or not game.get("completed"):
        return None
    scores = {s["name"].lower(): int(float(s["score"])) for s in game.get("scores", []) if "name" in s}
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    if not home or not away:
        return None
    hs = scores.get(home.lower(), 0)
    as_ = scores.get(away.lower(), 0)
    if hs > as_: return home
    if hs < as_: return away
    return "Draw"

def pick_team_from_pred(home: str, away: str, pred_val: str) -> str | None:
    s = str(pred_val or "").strip().lower()
    if not s:
        return None
    try:
        p = float(s)
        if 0.0 <= p <= 1.0:
            return home if p >= 0.5 else away
    except Exception:
        pass
    nh, na = norm_team(home), norm_team(away)
    sp = norm_team(s)
    if sp == nh: return home
    if sp == na: return away
    hits = get_close_matches(sp, [nh, na], cutoff=0.72)
    if hits: return home if hits[0] == nh else away
    return None

def label_from_pred_and_actual(home: str, away: str, actual_team: str | None, pred_val: str) -> int | None:
    if not actual_team:
        return None
    pred_team = pick_team_from_pred(home, away, pred_val)
    if not pred_team:
        return None
    return 1 if pred_team.strip().lower() == actual_team.strip().lower() else 0
    
    
import time
from typing import List

def _values_get_with_retry(spreadsheet, a1_range: str, retries: int = 5, delay: float = 0.5) -> List[List[str]]:
    """Call spreadsheets.values_get with retries on 5xx."""
    for i in range(retries):
        try:
            resp = spreadsheet.values_get(a1_range)
            return resp.get("values", [])
        except Exception as e:
            # Retry on 5xx or generic transient errors
            msg = str(e).lower()
            if "apierror: [500]" in msg or "internal error" in msg or "backend" in msg:
                sleep = delay * (2 ** i)
                logging.warning(f"values_get {a1_range} failed (attempt {i+1}/{retries}): {e} → retrying in {sleep:.1f}s")
                time.sleep(sleep)
                continue
            raise
    logging.error(f"values_get {a1_range} failed after {retries} retries; returning empty.")
    return []

def _safe_header_row(ws, approx_cols: int = 64) -> list[str]:
    """
    Fetch header row (row 1) without asking for the entire row width.
    We try a modest width first, then expand if needed.
    """
    title = ws.title
    # try a couple of widths to avoid super-wide A1:1 requests
    for width in (approx_cols, 256, 512):
        a1 = f"{title}!A1:{rowcol_to_a1(1, width)}"
        vals = _values_get_with_retry(ws.spreadsheet, a1)
        if vals:
            return [str(x) for x in (vals[0] if vals else [])]
    # Last resort: fall back to gspread's row_values (still wrap by retry)
    try:
        return ws.row_values(1)
    except Exception as e:
        logging.error(f"row_values(1) failed: {e}")
        return []


def ensure_headers(ws, required_cols: list[str]) -> dict[str, int]:
    # read current header with a narrow range and retries
    header = _safe_header_row(ws, approx_cols=max(64, len(required_cols) + 8))
    changed = False

    # Append any missing required columns
    for col in required_cols:
        if col not in header:
            header.append(col)
            changed = True

    if changed:
        # Write back only the slice we need (not the whole row)
        end_a1 = rowcol_to_a1(1, len(header))
        ws.update(range_name=f"A1:{end_a1}", values=[header])
        # refresh header
        header = _safe_header_row(ws, approx_cols=len(header) + 8)

    # map: column name -> 1-based index
    return {name: idx + 1 for idx, name in enumerate(header)}


def parse_minutes(val) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def find_best_completed_game(sport_key: str, home: str, away: str, ref_time_utc) -> dict | None:
    nh, na = norm_team(home), norm_team(away)
    games = get_final_scores(sport_key)
    best = None
    best_delta = None
    for g in games:
        if not g.get("completed"):
            continue
        gh, ga = norm_team(g.get("home_team", "")), norm_team(g.get("away_team", ""))
        if not (close_same(gh, nh) and close_same(ga, na)):
            continue
        commence = as_utc(parser.parse(g["commence_time"]))
        delta = abs(commence - ref_time_utc)
        if delta <= NAME_MATCH_WINDOW and (best_delta is None or delta < best_delta):
            best, best_delta = g, delta
    return best

# Flexible Home/Away headers
def pick_col(row, *candidates):
    for c in candidates:
        if c in row and str(row.get(c, "")).strip() != "":
            return str(row.get(c, "")).strip()
    return ""

def pick_sport_key(row) -> str:
    """Return an Odds API 'sport' key from either:
       - existing 'Sport' if it already contains a valid key, or
       - human labels 'Sport' + 'League' using SPORT_KEY_MAP.
    """
    raw = str(row.get("Sport", "")).strip()
    if raw in SPORTS:
        return raw
    sport = str(row.get("Sport", "")).strip().title()
    league = str(row.get("League", "")).strip().upper()
    key = SPORT_KEY_MAP.get((sport, league))
    if key:
        return key
    # sport-only fallback (e.g., Boxing with blank league)
    key = SPORT_KEY_MAP.get((sport, ""))
    return key or ""

# ── CORE ─────────────────────────────────────────────────────────────
def update_allobservations(ws):
    records = _get_all_records_with_retry(ws)
    df = pd.DataFrame(records).fillna("")
    if df.empty:
        logging.info("AllObservations: sheet is empty.")
        return

    # Header checks (accept either Home/Away or Home Team/Away Team)
    req_any = [("Home", "Home Team"), ("Away", "Away Team")]
    hard_reqs = ["Sport", "Timestamp", "MinutesToStart"]
    for c in hard_reqs:
        if c not in df.columns:
            logging.warning(f"AllObservations missing required column '{c}', aborting.")
            return
    if not any(x in df.columns for x in req_any[0]) or not any(x in df.columns for x in req_any[1]):
        logging.warning("AllObservations missing Home/Away column(s), aborting.")
        return

    header_map = ensure_headers(ws, list(df.columns) + ["Actual Winner", "Prediction Result"])
    col_aw = header_map["Actual Winner"]
    col_pr = header_map["Prediction Result"]

    batch = []
    upd_aw = upd_pr = 0

    for idx, row in df.iterrows():
        rownum = idx + 2
        sport = pick_sport_key(row)
        if not sport:
            logging.debug(f"Row {rownum}: no sport key (Sport='{row.get('Sport','')}', League='{row.get('League','')}'), skip")
            continue

        home = pick_col(row, "Home", "Home Team")
        away = pick_col(row, "Away", "Away Team")
        ts_raw = str(row.get("Timestamp", "")).strip()
        pred_val = str(row.get("Predicted", "")).strip()

        if not (home and away and ts_raw):
            logging.debug(f"Row {rownum}: missing team names or timestamp, skip")
            continue

        try:
            ts = parser.parse(ts_raw)
            ts_utc = as_utc(ts)
        except Exception as e:
            logging.error(f"Row {rownum}: bad Timestamp '{ts_raw}': {e}")
            continue

        mins = parse_minutes(row.get("MinutesToStart", ""))
        ref_time = ts_utc + timedelta(minutes=mins) if mins is not None else ts_utc

        game = find_best_completed_game(sport, home, away, ref_time)
        if not game:
            logging.debug(
                f"[NoMatch] Row {rownum}: {away} @ {home} | sport={sport} | "
                f"ref={ref_time.isoformat()} | window=±{NAME_MATCH_WINDOW}"
            )
            continue

        fh, fa = scores_from_game(game)
        if fh is None or fa is None:
            logging.debug(f"Row {rownum}: matched game not completed yet or missing scores, skip")
            continue

        market = normalize_market(str(row.get("Market", "")))
        actual_value = None
        pr_value = None

        if market == "totals":
            total_line = row.get("Total Line", "")
            tr = totals_result(fh, fa, total_line)
            if tr:
                actual_value = tr  # Over / Under / Push
                s = (pred_val or "").strip().lower()
                pick = "Over" if s in ("o", "over") else ("Under" if s in ("u", "under") else None)
                if pick:
                    pr_value = "P" if tr == "Push" else (1 if pick == tr else 0)

        elif market == "spread":
            sh = row.get("Spread Line Home", "")
            sa = row.get("Spread Line Away", "")
            cover = spread_cover(fh, fa, sh, sa)  # Home / Away / Push
            if cover:
                actual_value = home if cover == "Home" else (away if cover == "Away" else "Push")
                s = (pred_val or "").strip().lower()
                pick_side = None
                if s in ("home", "h"): pick_side = "Home"
                elif s in ("away", "a"): pick_side = "Away"
                else:
                    nh, na = norm_team(home), norm_team(away)
                    sp = norm_team(s)
                    if sp and (close_same(sp, nh) or close_same(sp, na)):
                        pick_side = "Home" if close_same(sp, nh) else "Away"
                if pick_side:
                    pr_value = "P" if cover == "Push" else (1 if pick_side == cover else 0)

        else:
            actual = actual_winner_from_game(game)
            if actual:
                actual_value = actual
                if pred_val:
                    label01 = label_from_pred_and_actual(home, away, actual, pred_val)
                    if label01 is not None:
                        pr_value = label01

        # Batch writes if changed
        if actual_value is not None:
            curr_aw = str(row.get("Actual Winner", "")).strip()
            if curr_aw != actual_value:
                a1 = rowcol_to_a1(rownum, col_aw)
                batch.append({"range": f"AllObservations!{a1}", "values": [[actual_value]]})
                upd_aw += 1

        if pr_value is not None:
            curr_pr = str(row.get("Prediction Result", "")).strip()
            try:
                curr_norm = "" if curr_pr == "" else ("P" if str(curr_pr).upper() == "P" else str(int(float(curr_pr))))
            except Exception:
                curr_norm = {"win": "1", "lose": "0", "w": "1", "l": "0", "p": "P"}.get(curr_pr.lower(), "")
            new_norm = pr_value if pr_value == "P" else str(pr_value)
            if curr_norm != new_norm:
                a1 = rowcol_to_a1(rownum, col_pr)
                batch.append({"range": f"AllObservations!{a1}", "values": [[new_norm]]})
                upd_pr += 1

    logging.info(f"Scanned {len(df)} rows; pending updates -> AW:{upd_aw}, PR:{upd_pr}")
    if batch:
        logging.info(f"AllObservations: applying {upd_aw} Actual Winner and {upd_pr} Prediction Result updates")
        body = {"valueInputOption": "USER_ENTERED", "data": batch}
        try:
            ws.spreadsheet.values_batch_update(body)
        except Exception as e:
            logging.warning(f"values_batch_update via worksheet failed ({e}); retrying on spreadsheet handle")
            SS.values_batch_update(body=body)
    else:
        logging.info("AllObservations: Nothing to update.")
        
        
def update_allbets(ws):
    """
    Update AllBets:
      - Column N:  Actual Winner
      - Column O:  Prediction Result (1/0 or 'P' for pushes)
    Required columns in AllBets: 
      Timestamp, Sport, Away Team, Home Team, Market, Predicted
    Optional (preferred for game matching): Game Time
    """
    records = _get_all_records_with_retry(ws)
    df = pd.DataFrame(records).fillna("")
    if df.empty:
        logging.info("AllBets: sheet is empty.")
        return

    # Ensure headers exist (append if missing)
    header_map = ensure_headers(ws, list(df.columns) + ["Actual Winner", "Prediction Result"])
    col_aw = header_map["Actual Winner"]
    col_pr = header_map["Prediction Result"]

    upd_aw = upd_pr = 0
    batch = []

    for idx, row in df.iterrows():
        rownum = idx + 2

        # sport key
        sport = pick_sport_key(row)
        if not sport:
            logging.debug(f"[AllBets r{rownum}] no sport key; skip")
            continue

        home = pick_col(row, "Home Team", "Home")  # AllBets headers use *Team*
        away = pick_col(row, "Away Team", "Away")
        if not (home and away):
            logging.debug(f"[AllBets r{rownum}] missing teams; skip")
            continue

        # Timestamp (prediction logged time)
        ts_raw = str(row.get("Timestamp", "")).strip()
        if not ts_raw:
            logging.debug(f"[AllBets r{rownum}] missing Timestamp; skip")
            continue

        # Preferred: Game Time (kickoff)
        gt_raw = str(row.get("Game Time", "")).strip()

        try:
            ts = parser.parse(ts_raw)
            ts_utc = as_utc(ts)
        except Exception as e:
            logging.error(f"[AllBets r{rownum}] bad Timestamp '{ts_raw}': {e}")
            continue

        ref_time = ts_utc
        if gt_raw:
            try:
                # Game Time is local (often EDT); localize and convert
                gt = parser.parse(gt_raw)
                ref_time = as_utc(gt)
            except Exception as e:
                logging.debug(f"[AllBets r{rownum}] Game Time parse failed '{gt_raw}': {e}")

        market = normalize_market(str(row.get("Market", "")))
        pred_val = str(row.get("Predicted", "")).strip()

        # Find the completed game closest to ref_time
        game = find_best_completed_game(sport, home, away, ref_time)
        if not game:
            logging.debug(
                f"[AllBets NoMatch r{rownum}] {away} @ {home} | sport={sport} | "
                f"ref={ref_time.isoformat()} | window=±{NAME_MATCH_WINDOW}"
            )
            continue

        # Scores
        fh, fa = scores_from_game(game)
        if fh is None or fa is None:
            logging.debug(f"[AllBets r{rownum}] matched game not completed yet or missing scores")
            continue

        actual_value = None
        pr_value = None

        if market == "totals":
            total_line = row.get("Total Line", "")
            tr = totals_result(fh, fa, total_line)  # Over/Under/Push
            if tr:
                actual_value = tr
                s = pred_val.lower()
                pick = "Over" if s in ("o", "over") else ("Under" if s in ("u", "under") else None)
                if pick:
                    pr_value = "P" if tr == "Push" else (1 if pick == tr else 0)

        elif market == "spread":
            sh = row.get("Spread Line Ho", row.get("Spread Line Home", ""))
            sa = row.get("Spread Line Aw", row.get("Spread Line Away", ""))
            cover = spread_cover(fh, fa, sh, sa)  # Home/Away/Push
            if cover:
                actual_value = home if cover == "Home" else (away if cover == "Away" else "Push")
                s = pred_val.lower()
                pick_side = None
                if s in ("home", "h"): pick_side = "Home"
                elif s in ("away", "a"): pick_side = "Away"
                else:
                    nh, na = norm_team(home), norm_team(away)
                    sp = norm_team(s)
                    if sp and (close_same(sp, nh) or close_same(sp, na)):
                        pick_side = "Home" if close_same(sp, nh) else "Away"
                if pick_side:
                    pr_value = "P" if cover == "Push" else (1 if pick_side == cover else 0)

        else:  # h2h / moneyline
            actual = actual_winner_from_game(game)
            if actual:
                actual_value = actual
                if pred_val:
                    label01 = label_from_pred_and_actual(home, away, actual, pred_val)
                    if label01 is not None:
                        pr_value = label01

        # Batch writes only if changed
        if actual_value is not None:
            curr_aw = str(row.get("Actual Winner", "")).strip()
            if curr_aw != actual_value:
                a1 = rowcol_to_a1(rownum, col_aw)
                batch.append({"range": f"AllBets!{a1}", "values": [[actual_value]]})
                upd_aw += 1

        if pr_value is not None:
            curr_pr = str(row.get("Prediction Result", "")).strip()
            try:
                curr_norm = "" if curr_pr == "" else ("P" if str(curr_pr).upper() == "P" else str(int(float(curr_pr))))
            except Exception:
                curr_norm = {"win": "1", "lose": "0", "w": "1", "l": "0", "p": "P"}.get(curr_pr.lower(), "")
            new_norm = pr_value if pr_value == "P" else str(pr_value)
            if curr_norm != new_norm:
                a1 = rowcol_to_a1(rownum, col_pr)
                batch.append({"range": f"AllBets!{a1}", "values": [[new_norm]]})
                upd_pr += 1

    logging.info(f"[AllBets] pending updates -> AW:{upd_aw}, PR:{upd_pr}")
    if batch:
        body = {"valueInputOption": "USER_ENTERED", "data": batch}
        try:
            ws.spreadsheet.values_batch_update(body)
        except Exception as e:
            logging.warning(f"[AllBets] values_batch_update via worksheet failed ({e}); retrying on spreadsheet handle")
            SS.values_batch_update(body=body)
    else:
        logging.info("[AllBets] Nothing to update.")
        
        
        
        
def _get_all_records_with_retry(ws, retries: int = 3, delay: float = 0.5):
    for i in range(retries):
        try:
            return ws.get_all_records()
        except Exception as e:
            msg = str(e).lower()
            if "apierror: [500]" in msg or "internal error" in msg:
                sleep = delay * (2 ** i)
                logging.warning(f"get_all_records failed (attempt {i+1}/{retries}): {e} → retrying in {sleep:.1f}s")
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("get_all_records failed after retries")



# ── RUN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # Update AllBets first (this is the source of truth for bets)
        update_allbets(WS_ALLBETS)

        # Optional: still update AllObservations if you want
        try:
            update_allobservations(WS)
        except Exception as e:
            logging.exception(f"update_allobservations failed: {e}")

    except Exception as e:
        logging.exception(f"sheets.py failed: {e}")

