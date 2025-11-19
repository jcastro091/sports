#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SharpsSignal — Daily Picks Publisher (Telegram-only + Social caption export)
+ Robust Google Sheets opening (by ID or Title)
"""

import os
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import pytz
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------------- Config ----------------------

load_dotenv()

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
# Prefer ID when provided (more reliable). Get it from sheet URL between /d/ and /edit
GSPREAD_SHEET_ID = os.getenv("GSPREAD_SHEET_ID")
GSPREAD_SHEET_NAME = os.getenv("GSPREAD_SHEET_NAME", "ConfirmedBets - AllObservations")
GSPREAD_WORKSHEET_NAME = os.getenv("GSPREAD_WORKSHEET_NAME", "AllObservations")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # PRO channel
TZ_NAME = os.getenv("TIMEZONE", "America/New_York")

# Window mins
WINDOW_MIN = int(os.getenv("WINDOW_MIN", "15"))
WINDOW_MAX = int(os.getenv("WINDOW_MAX", "60"))

# Files
BASE_DIR = os.path.dirname(__file__)
CACHE_FILE = os.path.join(BASE_DIR, ".published_cache.csv")
DAILY_LOCK_FILE = os.path.join(BASE_DIR, ".daily_posted.txt")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

EXPECTED_COLUMNS = {
    "sport": ["Sport", "sport"],
    "league": ["League", "league"],  # optional; we’ll derive from Sport if missing
    "home": ["Home", "home", "Team", "team_home"],
    "away": ["Away", "away", "Opponent", "team_away"],
    "market": ["Market", "market", "BetType"],
    "direction": ["Direction", "direction"],  # new: to derive selection
    "selection": ["Selection", "selection", "Side", "Pick"],  # optional; derived if missing
    "odds_american": ["Odds (Am)", "Odds", "odds", "AmericanOdds", "odds_american"],
    "game_time": ["Game Time", "GameTime", "game_time", "Start Time", "Start"],  # optional; derived if missing
    "timestamp": ["Timestamp", "timestamp"],  # new: to compute game_time
    "minutes_to_start": ["MinutesToStart", "MinutesTo", "MinutesToSt", "Minutes To Start", "Minutes To"],  # new
    "confidence": ["Confidence", "confidence", "AI_Confidence"],
}


def prettify_league_from_sport(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    # sport tokens look like "soccer_france_ligue_one" or "baseball_mlb"
    parts = s.split("_", 1)
    league_part = parts[1] if len(parts) > 1 else s
    # normalize
    txt = league_part.replace("_", " ").title()
    # Common acronyms back to uppercase
    txt = txt.replace("Mlb", "MLB").replace("Nba", "NBA").replace("Nfl", "NFL").replace("Nhl", "NHL")
    return txt

def derive_selection(direction: str, home: str, away: str) -> str:
    if not isinstance(direction, str):
        return ""
    d = direction.lower()
    if "over" in d:
        return "Over"
    if "under" in d:
        return "Under"
    if "home" in d:
        return home
    if "away" in d:
        return away
    return ""

def derive_game_time(row, tz_name: str, ts_col: str, m_col: str):
    """If explicit game_time missing, use Timestamp + MinutesToStart."""
    ts = row.get(ts_col)
    mins = row.get(m_col)
    if ts is None or mins is None:
        return None
    try:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")  # your timestamps have 'Z'
        if pd.isna(dt):
            return None
        # Add minutes and convert to target timezone
        dt2 = dt + pd.to_timedelta(float(mins), unit="m")
        return dt2.tz_convert(pytz.timezone(tz_name)).to_pydatetime()
    except Exception:
        return None


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def now_tz() -> datetime:
    return datetime.now(pytz.timezone(TZ_NAME))

def parse_time(val, tz_name: str) -> Optional[datetime]:
    tz = pytz.timezone(tz_name)
    # If it's already a datetime / Timestamp, normalize to tz
    if isinstance(val, (pd.Timestamp, datetime)):
        dt = val.to_pydatetime() if isinstance(val, pd.Timestamp) else val
        return tz.localize(dt) if dt.tzinfo is None else dt.astimezone(tz)

    if not isinstance(val, str):
        return None

    s = val.strip()
    # Normalize common tz abbreviations in strings like "Aug 30, 1:00 PM EDT"
    # by stripping the trailing abbrev and localizing ourselves.
    for abbrev in (" EDT", " EST", " CDT", " CST", " PDT", " PST", " UTC", " GMT"):
        if s.endswith(abbrev):
            s = s[: -len(abbrev)]
            break

    fmts = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M %p",
        "%Y-%m-%d %I:%M %p",
        "%b %d, %I:%M %p",          # e.g., "Aug 30, 1:00 PM"
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return tz.localize(dt)
        except Exception:
            continue

    # last resort: pandas parser (without abbrev) then localize
    try:
        dt = pd.to_datetime(s, utc=False, errors="coerce")
        if pd.isna(dt):
            return None
        if isinstance(dt, pd.Timestamp):
            return tz.localize(dt.to_pydatetime()) if dt.tzinfo is None else dt.tz_convert(tz).to_pydatetime()
    except Exception:
        return None
    return None


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p %Z").lstrip("0")

def post_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram creds; skipping.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        print(f"❌ Telegram failed: {r.status_code} {r.text}")
    else:
        print("✅ Posted to Telegram.")

def _authorize_client():
    scopes = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    if not GOOGLE_SERVICE_ACCOUNT_JSON or not os.path.exists(GOOGLE_SERVICE_ACCOUNT_JSON):
        raise RuntimeError("Service account JSON not found. Set GOOGLE_SERVICE_ACCOUNT_JSON.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SERVICE_ACCOUNT_JSON, scopes)
    return gspread.authorize(creds)

def open_sheet() -> pd.DataFrame:
    client = _authorize_client()
    try:
        if GSPREAD_SHEET_ID:
            sh = client.open_by_key(GSPREAD_SHEET_ID)
        else:
            sh = client.open(GSPREAD_SHEET_NAME)
    except gspread.SpreadsheetNotFound as e:
        # Give actionable help
        who = _whoami_service_account()
        raise RuntimeError(
            f"""Spreadsheet not found.

• If using title, check GSPREAD_SHEET_NAME exactly matches the doc title.
• Or set GSPREAD_SHEET_ID from the URL (between /d/ and /edit).
• Also share the sheet with this service account email: {who}
"""
        ) from e



    try:
        ws = sh.worksheet(GSPREAD_WORKSHEET_NAME)
    except gspread.WorksheetNotFound as e:
        titles = [w.title for w in sh.worksheets()]
        raise RuntimeError(f"Worksheet '{GSPREAD_WORKSHEET_NAME}' not found. Available tabs: {titles}") from e

    rows = ws.get_all_records()
    return pd.DataFrame(rows)

def _whoami_service_account() -> str:
    # read client_email from the JSON file for convenience
    try:
        import json
        with open(GOOGLE_SERVICE_ACCOUNT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("client_email","<unknown service account>")
    except Exception:
        return "<unknown service account>"

from hashlib import md5
def game_key(league, home, away, market, selection) -> str:
    base = f"{league}-{home}-{away}-{market}-{selection}"
    return md5(base.encode("utf-8")).hexdigest()

def load_cache() -> set:
    if not os.path.exists(CACHE_FILE): return set()
    try:
        return set(pd.read_csv(CACHE_FILE)["key"].astype(str).tolist())
    except Exception:
        return set()

def save_cache(keys: set):
    pd.DataFrame({"key": list(keys)}).to_csv(CACHE_FILE, index=False)

def read_daily_lock() -> Optional[str]:
    if not os.path.exists(DAILY_LOCK_FILE): return None
    try:
        return open(DAILY_LOCK_FILE, "r", encoding="utf-8").read().strip()
    except Exception:
        return None

def write_daily_lock(today_str: str):
    with open(DAILY_LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(today_str)

@dataclass
class Play:
    league: str
    home: str
    away: str
    market: str
    selection: str
    odds_american: Optional[float]
    game_time: datetime
    key: str

EXPECTED_COLUMNS = {
    "sport": ["Sport", "sport"],
    "league": ["League", "league"],  # optional; we’ll derive from Sport if missing
    "home": ["Home", "home", "Team", "team_home"],
    "away": ["Away", "away", "Opponent", "team_away"],
    "market": ["Market", "market", "BetType"],
    "direction": ["Direction", "direction"],  # new: to derive selection
    "selection": ["Selection", "selection", "Side", "Pick"],  # optional; derived if missing
    "odds_american": ["Odds (Am)", "Odds", "odds", "AmericanOdds", "odds_american"],
    "game_time": ["Game Time", "GameTime", "game_time", "Start Time", "Start"],  # optional; derived if missing
    "timestamp": ["Timestamp", "timestamp"],  # new: to compute game_time
    "minutes_to_start": ["MinutesToStart", "MinutesTo", "MinutesToSt", "Minutes To Start", "Minutes To"],  # new
    "confidence": ["Confidence", "confidence", "AI_Confidence"],
}


def extract_plays(df: pd.DataFrame) -> List[Play]:
    col = {k: find_col(df, v) for k, v in EXPECTED_COLUMNS.items()}

    # minimally required to build a message
    req_base = ["home", "away", "market", "odds_american"]
    for r in req_base:
        if not col.get(r):
            raise RuntimeError(f"Missing required column for '{r}'. Adjust EXPECTED_COLUMNS.")

    plays: List[Play] = []
    for _, row in df.iterrows():
        try:
            sport = str(row.get(col["sport"], "")).strip() if col.get("sport") else ""
            # league: use explicit league col if present; else derive from sport
            if col.get("league"):
                league = str(row[col["league"]]).strip()
            else:
                league = prettify_league_from_sport(sport)

            home = str(row[col["home"]]).strip()
            away = str(row[col["away"]]).strip()
            market = str(row[col["market"]]).strip()

            # selection: prefer explicit; else derive from Direction
            if col.get("selection") and pd.notna(row.get(col["selection"])):
                selection = str(row[col["selection"]]).strip()
            else:
                direction = str(row.get(col["direction"], "")).strip() if col.get("direction") else ""
                selection = derive_selection(direction, home, away)

            # odds
            odds_raw = row[col["odds_american"]]
            try:
                odds_american = float(str(odds_raw).replace("+", ""))
            except Exception:
                odds_american = None

            # game time: prefer explicit; else derive from timestamp + minutes_to_start
            gt = None
            if col.get("game_time") and pd.notna(row.get(col["game_time"])):
                gt = parse_time(row[col["game_time"]], TZ_NAME)
            if gt is None and col.get("timestamp") and col.get("minutes_to_start"):
                gt = derive_game_time(row, TZ_NAME, col["timestamp"], col["minutes_to_start"])
            if gt is None:
                # Skip if we still don't have a start time (we need it for the 15–60 min window)
                continue

            k = game_key(league, home, away, market, selection)
            plays.append(
                Play(
                    league=league,
                    home=home,
                    away=away,
                    market=market,
                    selection=selection or "N/A",
                    odds_american=odds_american,
                    game_time=gt,
                    key=k,
                )
            )
        except Exception:
            continue
    return plays


def within_window(gt: datetime, now_local: datetime) -> bool:
    delta_min = (gt - now_local).total_seconds()/60.0
    return WINDOW_MIN <= delta_min <= WINDOW_MAX

def fmt_telegram(p: Play) -> str:
    odds = f"{int(p.odds_american):+d}" if isinstance(p.odds_american, (float,int)) else "N/A"
    lines = [
        "🔥 <b>SharpsSignal PRO Pick</b> 🔥",
        f"<b>{p.league}</b> • {p.home} vs {p.away}",
        f"<b>{p.market}</b>: {p.selection} ({odds})",
        f"⏰ {fmt_time(p.game_time)} (posted 15–60m pre-game)",
        "",
        "👉 <a href=\"https://sharps-signal.com\">Join PRO for full card & tracking</a>",
    ]
    return "\n".join(lines)

def fmt_social(p: Play) -> str:
    odds = f"{int(p.odds_american):+d}" if isinstance(p.odds_american, (float,int)) else "N/A"
    return (
        f"🚨 Today’s Pick ({p.league}) 🚨\n"
        f"{p.home} vs {p.away}\n"
        f"{p.market}: {p.selection} ({odds})\n"
        f"⏰ {fmt_time(p.game_time)}\n"
        f"Full card 👉 sharps-signal.com\n"
        f"#sportsbetting #picks #AI #SharpsSignal"
    )

def export_caption(today_str: str, caption: str) -> str:
    path = os.path.join(EXPORT_DIR, f"{today_str}_caption.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(caption)
    return path

def main_once():
    now_local = now_tz()
    today_str = now_local.strftime("%Y-%m-%d")

    # one-per-day lock
    last = read_daily_lock()
    if last == today_str:
        print(f"[{now_local}] Already posted today ({today_str}); skipping.")
        return

    df = open_sheet()
    if df.empty:
        print("No data from sheet.")
        return

    plays = extract_plays(df)
    if not plays:
        print("No plays in sheet.")
        return

    # Only consider plays in the 15–60m window
    in_window = [p for p in plays if within_window(p.game_time, now_local)]
    if not in_window:
        print("No plays within 15–60m window.")
        return

    # Remove any that were already posted (per-game cache)
    posted_keys = load_cache()
    fresh = [p for p in in_window if p.key not in posted_keys]
    if not fresh:
        print("All in-window plays already posted.")
        return

    # Choose exactly ONE: earliest game time
    pick = sorted(fresh, key=lambda x: x.game_time)[0]

    # Publish to Telegram
    post_telegram(fmt_telegram(pick))

    # Export caption for manual IG/X posting
    caption_path = export_caption(today_str, fmt_social(pick))
    print(f"📝 Saved social caption → {caption_path}")

    # Save locks
    new_keys = set(posted_keys)
    new_keys.add(pick.key)
    save_cache(new_keys)
    write_daily_lock(today_str)
    print(f"✅ Posted one pick for {today_str}: {pick.league} {pick.home} vs {pick.away}")

if __name__ == "__main__":
    import time
    INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "300"))  # default: 5 minutes
    while True:
        try:
            main_once()
        except Exception as e:
            print(f"Top-level error: {e}")
        time.sleep(INTERVAL_SEC)

