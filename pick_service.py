from pathlib import Path
from dotenv import load_dotenv
import os, json, logging

# NEW imports that were missing
import pandas as pd
import gspread
from typing import Optional, Iterator
from dateutil import parser, tz
from google.oauth2.service_account import Credentials
from datetime import datetime   # ← add this


ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)
load_dotenv(dotenv_path=ROOT / "autoposter" / ".env", override=False)


def _mask(s, show=4):
    s = str(s or "")
    return s if len(s) <= show else s[:show] + "…"

SHEET_ID  = (os.getenv("SHEET_ID") or os.getenv("GOOGLE_SHEET_ID", "")).strip()
WORKSHEET = (os.getenv("WORKSHEET") or os.getenv("GOOGLE_WORKSHEET_NAME", "AllBets")).strip()

TZ_NAME   = os.getenv("TZ", "America/New_York")

# ← This is the single source of truth for credentials
CREDS_PATH = (
    os.getenv("GOOGLE_CREDENTIALS_JSON")
    or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    or str(ROOT / "google-credentials.json")   # final fallback
)

logger = logging.getLogger(__name__)

logger.info("[pick_service] SHEET_ID(raw)=%s (len=%d)  WORKSHEET=%s",
            _mask(SHEET_ID), len(SHEET_ID), WORKSHEET)
logger.info("[pick_service] Using credentials at: %s", CREDS_PATH)

# Log the service account email so you can share the sheet with it
try:
    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        client_email = json.load(f).get("client_email", "")
        if client_email:
            logger.info("[pick_service] Service account email: %s", client_email)
except Exception:
    pass


def _ws():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)  # ← use CREDS_PATH
    gc     = gspread.authorize(creds)

    
    
    try:
        sh = gc.open_by_key(SHEET_ID)
    except gspread.SpreadsheetNotFound as e:
        raise RuntimeError(
            "Spreadsheet 404. Most common fixes:\n"
            f"  • Ensure this exact ID is correct: {SHEET_ID}\n"
            "  • Share the sheet with the service account email shown above (Editor or Viewer is fine)\n"
            "  • Make sure we loaded the right .env (repo root or autoposter/.env)\n"
        ) from e
   
    
    
    try:
        return sh.worksheet(WORKSHEET)
    except gspread.WorksheetNotFound as e:
        raise RuntimeError(
            f"[pick_service] Worksheet '{WORKSHEET}' not found in sheet {_mask(SHEET_ID)}. "
            f"Available tabs: {[w.title for w in sh.worksheets()]}"
        ) from e


# ---- Column aliases ----
ALIASES = {
    "Timestamp":        ["Timestamp","Time","Created At","RowTimestamp"],
    "Sport":            ["Sport","League","sport","league"],
    "Market":           ["Market","Type","Bet Type"],
    "Predicted":        ["Predicted","Pick","Selection","Team"],
    "Game Time":        ["Game Time","Kickoff","Start Time"],
    "Odds Taken (AM)":  ["Odds Taken (AM)","Odds (Am)","Closing Odds (Am)","Opening Price","Odds"],
    "Reason":           ["Reason","Why","Explanation"],
    "Home":             ["Home","Home Team"],
    "Away":             ["Away","Away Team"],
    "Total Line":       ["Total Line","Total","O/U","Line"],
    "bet_id": ["bet_id", "Bet ID", "ID", "Bet Slip URL/ID", "Bet Placed?"],
    "Confidence":       ["Confidence","Conf","Kelly","Stake %"],
}

def _col(cols, key):
    cols = list(cols)
    for a in ALIASES.get(key, [key]):
        if a in cols:
            return a
    return key if key in cols else None
    
    
def _val_with_aliases(row: pd.Series, cols, key: str) -> str:
    """Return first non-empty value among alias columns for a given key."""
    for a in ALIASES.get(key, [key]):
        if a in cols:
            v = str(row.get(a, "")).strip()
            if v:
                return v
    # if exact key exists and has something, keep it
    if key in cols:
        v = str(row.get(key, "")).strip()
        if v:
            return v
    return ""


# ---- Public API ----
def fetch_recent_rows(limit: int = 200) -> pd.DataFrame:
    ws = _ws()
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header = [c.strip() for c in values[0]]
    data   = values[1:]
    df = pd.DataFrame(data, columns=header).dropna(how="all")
    return df.tail(limit) if len(df) > limit else df

def _parse_local(dt_str: str, tz_name: str) -> Optional[datetime]:
    try:
        local_tz = tz.gettz(tz_name)
        d = parser.parse(str(dt_str), tzinfos={"EDT": local_tz, "EST": local_tz})
        return d if d.tzinfo else d.replace(tzinfo=local_tz)
    except Exception:
        return None

def normalize_row(r: pd.Series, cols, tz_name: str = TZ_NAME) -> dict:
    sport_col  = _col(cols, "Sport")
    mkt_col    = _col(cols, "Market")
    pick_col   = _col(cols, "Predicted")
    gtime_col  = _col(cols, "Game Time")
    odds_col   = _col(cols, "Odds Taken (AM)")
    reason_col = _col(cols, "Reason")
    home_col   = _col(cols, "Home")
    away_col   = _col(cols, "Away")
    total_col  = _col(cols, "Total Line")
    conf_col   = _col(cols, "Confidence")
    bet_id_val = _val_with_aliases(r, cols, "bet_id")

    dt = _parse_local(r.get(gtime_col, ""), tz_name) if gtime_col else None

    return {
        "bet_id":       bet_id_val,
        "league":       r.get(sport_col, ""),
        "market":       r.get(mkt_col, ""),
        "pick":         r.get(pick_col, ""),
        "odds":         r.get(odds_col, ""),
        "home":         r.get(home_col, ""),
        "away":         r.get(away_col, ""),
        "total_line":   r.get(total_col, ""),
        "confidence":   r.get(conf_col, ""),
        "game_time_dt": dt,
        "reason":       r.get(reason_col, ""),
    }

def iter_picks_today(limit: int = 200) -> Iterator[dict]:
    df = fetch_recent_rows(limit=limit)
    if df.empty:
        return iter(())
    local_tz = tz.gettz(TZ_NAME)
    now = datetime.now(local_tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = now.replace(hour=23, minute=59, second=59, microsecond=0)
    for _, row in df.iterrows():
        n = normalize_row(row, df.columns, TZ_NAME)
        dt = n["game_time_dt"]
        if dt and start <= dt <= end:
            yield n

def find_pick_for_matchup(home: str, away: str, when: str = "today", limit: int = 300) -> Optional[dict]:
    home_l = (home or "").lower()
    away_l = (away or "").lower()
    df = fetch_recent_rows(limit=limit)
    if df.empty:
        return None
    for _, row in df.iterrows():
        n = normalize_row(row, df.columns, TZ_NAME)
        if not n.get("home") or not n.get("away"):
            continue
        if home_l in str(n["home"]).lower() and away_l in str(n["away"]).lower():
            if when == "today":
                dt = n.get("game_time_dt")
                if not dt:
                    continue
                local_tz = tz.gettz(TZ_NAME)
                now = datetime.now(local_tz)
                if dt.date() != now.date():
                    continue
            return n
    return None
