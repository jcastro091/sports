#!/usr/bin/env python3
# Grades NBA/NHL player props using balldontlie (NBA) + NHL Stats API
# Writes "Actual Value" and "Prediction Result" (1/0/"P") to ConfirmedBets/PlayerProps

import os, re, time, logging
from datetime import datetime, date, timedelta
from difflib import get_close_matches
from functools import lru_cache
from typing import Dict, Any, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import gspread
from dateutil import parser
from gspread.utils import rowcol_to_a1
from oauth2client.service_account import ServiceAccountCredentials



from dateutil import tz
TZINFOS = {
    "EST": tz.gettz("America/New_York"),
    "EDT": tz.gettz("America/New_York"),
}


# ========= CONFIG =========
BAL_KEY = os.getenv("BAL_KEY", "d1dcf8f7-769a-4933-bf79-7bdc02109f57")  # balldontlie
NBA_BASE = "https://api.balldontlie.io/v1"

NHL_SCHEDULE = "https://statsapi.web.nhl.com/api/v1/schedule"
NHL_BOX = "https://statsapi.web.nhl.com/api/v1/game/{gamePk}/boxscore"

# Auth: try Bearer first (current doc), fall back to X-API-Key if we see 401s.
BAL_BEARER = {"Authorization": f"Bearer {BAL_KEY}"} if BAL_KEY else {}
BAL_XAPI    = {"X-API-Key": BAL_KEY} if BAL_KEY else {}
HEADERS = BAL_BEARER or BAL_XAPI
API_SLEEP = float(os.getenv("API_SLEEP", "0.35"))  # throttle between calls
ENABLE_NHL = os.getenv("ENABLE_NHL", "1") == "1"   # set ENABLE_NHL=0 to skip NHL
ENABLE_NBA = os.getenv("ENABLE_NBA", "1") == "1"

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS = ServiceAccountCredentials.from_json_keyfile_name(
    "telegrambetlogger-35856685bc29.json", SCOPE
)
GC = gspread.authorize(CREDS)
SS = GC.open("ConfirmedBets")
WS = SS.worksheet("PlayerProps")

# ========= LOGGING =========
VERBOSE = ("--verbose" in os.sys.argv)
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO
logging.basicConfig(
    filename="playerprops_results_log.txt",
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)
console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logging.getLogger().addHandler(console)
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ========= HTTP SESSION (retry/backoff) =========
SESSION = requests.Session()
retry = Retry(
    total=4,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods={"GET"},
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

def req_json(name: str, url: str, params: dict, headers: Optional[dict] = None) -> dict:
    logging.debug(f"[HTTP] {name} → GET {url} params={params}")
    try:
        r = SESSION.get(url, headers=headers, params=params, timeout=25)
        # If explicit 429, respect Retry-After once more
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", "2"))
            logging.warning(f"[HTTP] {name} 429 rate-limited; sleeping {wait}s")
            time.sleep(wait)
            r = SESSION.get(url, headers=headers, params=params, timeout=25)
        logging.debug(f"[HTTP] {name} status={r.status_code}")
            
        if r.status_code == 401 and headers is BAL_BEARER and BAL_XAPI:
            # Retry once with X-API-Key header
            logging.warning("[HTTP] %s 401 with Bearer; retrying with X-API-Key", name)
            r = SESSION.get(url, headers=BAL_XAPI, params=params, timeout=25)
        if r.status_code != 200:
            snippet = r.text[:200].replace("\n", " ")
            logging.warning(f"[HTTP] {name} non-200={r.status_code} body≈{snippet}")
            return {}
            
            
        data = r.json()
        if API_SLEEP:
            time.sleep(API_SLEEP)
        return data
    except Exception as e:
        
        if VERBOSE:
            logging.exception(f"[HTTP] {name} failed: {e}")
        else:
            logging.warning(f"[HTTP] {name} failed: {e.__class__.__name__}: {e}")        
      
        return {}

# ========= HELPERS =========
TEAM_ALIASES = {"la": "los angeles", "st.": "saint", "ny": "new york", "tb": "tampa bay"}

_nba_stats_cache: Dict[int, List[dict]] = {}

def nba_stats_for_game(game_id: int) -> List[dict]:
    """Fallback to /v1/stats if /v1/box_scores is 401/empty."""
    if game_id in _nba_stats_cache:
        return _nba_stats_cache[game_id]
    all_stats: List[dict] = []
    page = 1
    while True:
                        
        data = req_json("nba_stats", f"{NBA_BASE}/stats",
                        {"game_ids[]": game_id, "page": page, "per_page": 100},
                        headers=HEADERS)                        
                        
        chunk = data.get("data", [])
        all_stats.extend(chunk)
        total_pages = data.get("meta", {}).get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
    _nba_stats_cache[game_id] = all_stats
    return all_stats


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    for k, v in TEAM_ALIASES.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return re.sub(r"\s+", " ", s).strip()

def fuzzy_eq(a: str, b: str, cutoff: float = 0.72) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return bool(get_close_matches(a, [b], cutoff=cutoff))

def to_float(x) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def ensure_cols(ws, cols: List[str]) -> Dict[str, int]:
    header = ws.row_values(1)
    changed = False
    for c in cols:
        if c not in header:
            header.append(c)
            changed = True
    if changed:
        ws.update(range_name=f"A1:{rowcol_to_a1(1, len(header))}", values=[header])
        header = ws.row_values(1)
    return {name: i + 1 for i, name in enumerate(header)}

def synth_game_raw(row) -> str:
    # Your sheet uses "Game (Away @ Home)"
    combo = (row.get("Game (Away @ Home)") or row.get("Game (Away@Home)") or "").strip()
    if combo:
        if "@" in combo:
            away, home = [x.strip() for x in combo.split("@", 1)]
            if away and home:
                return f"{away} @ {home}"
        return combo

    # Fallbacks
    game = (row.get("Game (raw)") or row.get("Game") or "").strip()
    if game:
        return game

    away_keys = ["Away Team", "Away", "Away Team Name", "Team Away", "Visitor", "Visitor Team"]
    home_keys = ["Home Team", "Home", "Home Team Name", "Team Home", "Home Team (Raw)"]
    away = next((str(row.get(k, "")).strip() for k in away_keys if row.get(k)), "")
    home = next((str(row.get(k, "")).strip() for k in home_keys if row.get(k)), "")
    return f"{away} @ {home}" if away and home else ""

# ========= MARKET MAPS =========
def nba_stat_from_market(m: str):
    m = (m or "").lower()
    if "points" in m:   return "pts"
    if "assists" in m:  return "ast"
    if "rebounds" in m: return "reb"
    if "threes" in m:   return "fg3m"
    return None

def nhl_stat_from_market(m: str):
    m = (m or "").lower()
    if "assists" in m: return "assists"
    if "shots" in m or "sog" in m: return "shots"
    if "points" in m: return "points"
    return None

# ========= NBA LOOKUPS (cached) =========
@lru_cache(maxsize=256)
def _nba_games_for_date_str(dstr: str) -> List[dict]:
    return req_json("nba_games_for_date", f"{NBA_BASE}/games",
                    {"dates[]": dstr, "per_page": 100}, headers=HEADERS).get("data", [])

def nba_games_for_date(d: date) -> List[dict]:
    return _nba_games_for_date_str(d.isoformat())

_nba_box_cache: Dict[int, List[dict]] = {}


def nba_box_scores_for_game(game_id: int) -> List[dict]:
    # unchanged cache header…
    if game_id in _nba_box_cache:
        return _nba_box_cache[game_id]

    all_stats: List[dict] = []
    page = 1
    got_401 = False

    while True:
        data = req_json("nba_box_scores", f"{NBA_BASE}/box_scores",
                        {"game_ids[]": game_id, "page": page, "per_page": 100},
                        headers=HEADERS)
        # Detect 401/empty response and bail to /stats
        if not data:
            got_401 = True
            break
        chunk = data.get("data", [])
        if chunk is None:
            got_401 = True
            break
        all_stats.extend(chunk)
        total_pages = data.get("meta", {}).get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1

    if got_401 or not all_stats:
        logging.warning(f"[NBA] /box_scores unavailable; falling back to /stats for game {game_id}")
        # Convert /stats schema to the shape we read later
        stats = nba_stats_for_game(game_id)
        # Normalize to mimic box_scores rows enough for our code path
        normalized = []
        for s in stats:
            p = s.get("player") or {}
            name = (p.get("first_name",""), p.get("last_name",""))
            normalized.append({
                "player": {"first_name": name[0], "last_name": name[1]},
                "stats": {
                    "pts":   s.get("pts"),
                    "ast":   s.get("ast"),
                    "reb":   s.get("reb"),
                    "fg3m":  s.get("fg3m"),
                }
            })
        _nba_box_cache[game_id] = normalized
        return normalized

    _nba_box_cache[game_id] = all_stats
    return all_stats


# ========= NHL LOOKUPS (cached) =========
@lru_cache(maxsize=256)
def _nhl_schedule_for_date_str(dstr: str) -> List[dict]:
    data = req_json("nhl_schedule", NHL_SCHEDULE, {"date": dstr})
    dates = data.get("dates", [])
    return (dates[0].get("games", []) if dates else [])

def nhl_schedule_for_date(d: date) -> List[dict]:
    return _nhl_schedule_for_date_str(d.isoformat())

_nhl_box_cache: Dict[int, dict] = {}
def nhl_boxscore_for_game(game_pk: int) -> dict:
    if game_pk not in _nhl_box_cache:
        _nhl_box_cache[game_pk] = req_json("nhl_boxscore", NHL_BOX.format(gamePk=game_pk), {})
    return _nhl_box_cache[game_pk]

# ========= MATCH + FETCH ACTUAL =========
def match_game(games: List[dict], game_raw: str, league: str) -> Optional[dict]:
    try:
        away_name, home_name = [x.strip() for x in game_raw.split("@", 1)]
    except Exception:
        logging.debug(f"[{league}] bad game_raw='{game_raw}'")
        return None

    for g in games:
        if league == "NBA":
            home = g["home_team"]["full_name"]; away = g["visitor_team"]["full_name"]
        else:
            home = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
        if fuzzy_eq(home_name, home) and fuzzy_eq(away_name, away):
            logging.debug(f"[{league}] matched game: {away} @ {home}")
            return g
    logging.debug(f"[{league}] no match for '{game_raw}' among {len(games)} games")
    return None

def nba_find_player_actual(game_raw: str, when: str, player_name: str, stat_key: str) -> Optional[float]:
    if not ENABLE_NBA:
        return None    
    
    # Normalize date around event date with tz handling
    base = parser.parse(when, tzinfos=TZINFOS).date() if when else datetime.utcnow().date()
    
    

    for d in (base, base - timedelta(days=1), base + timedelta(days=1)):
        games = nba_games_for_date(d)
        g = match_game(games, game_raw, "NBA")
        if not g:
            continue
        stats = nba_box_scores_for_game(g["id"])
        tgt = norm(player_name)
        for s in stats:
            p = s.get("player") or {}
            name = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if name and fuzzy_eq(tgt, name):
                val = s.get("stats", {}).get(stat_key, 0) or 0
                logging.debug(f"[NBA] {name} {stat_key}={val}")
                return float(val)
    return None

def nhl_find_player_actual(game_raw: str, when: str, player_name: str, stat_key: str) -> Optional[float]:
    if not ENABLE_NHL:
        return None

    # Normalize date around event date with tz handling
    base = parser.parse(when, tzinfos=TZINFOS).date() if when else datetime.utcnow().date()
    for d in (base, base - timedelta(days=1), base + timedelta(days=1)):
        games = nhl_schedule_for_date(d)
        g = match_game(games, game_raw, "NHL")
        if not g:
            continue
        box = nhl_boxscore_for_game(g["gamePk"])
        tgt = norm(player_name)
        for side in ("home", "away"):
            for pdata in (box.get("teams", {}).get(side, {}).get("players", {}) or {}).values():
                name = pdata.get("person", {}).get("fullName", "")
                if name and fuzzy_eq(tgt, name):
                    st = pdata.get("stats", {}).get("skaterStats", {}) or {}
                    if stat_key == "assists": return float(st.get("assists", 0) or 0)
                    if stat_key == "shots":   return float(st.get("shots", 0) or 0)
                    if stat_key == "points":  return float((st.get("goals", 0) or 0) + (st.get("assists", 0) or 0))
    return None

# ========= GRADING =========
def grade(side: str, line: Optional[float], actual: Optional[float]) -> Optional[str]:
    if line is None or actual is None:
        return None
    s = (side or "").lower().strip()
    if actual == line: return "P"
    if s == "over":    return "1" if actual > line else "0"
    if s == "under":   return "1" if actual < line else "0"
    return None

# ========= MAIN =========
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
if DRY_RUN:
    logging.info("🧪 DRY_RUN=1 → no writes will be made to Google Sheets")



def update_playerprops(ws):
    rows = ws.get_all_records()
    logging.info(f"PlayerProps total rows={len(rows)}")
    header = ensure_cols(ws, ["Actual Value", "Prediction Result"])
    c_actual, c_result = header["Actual Value"], header["Prediction Result"]

    now_local = datetime.now()
    batch: List[Dict[str, Any]] = []
    touched = 0
    eligible = 0

    for i, r in enumerate(rows):
        rnum = i + 2  # sheet row number
        sport = str(r.get("Sport", "")).strip()

        if sport not in ("basketball_nba", "icehockey_nhl"):
            continue
        if sport == "icehockey_nhl" and not ENABLE_NHL:
            logging.debug(f"r{rnum} skipping NHL due to ENABLE_NHL=0")
            continue
        if sport == "basketball_nba" and not ENABLE_NBA:
            logging.debug(f"r{rnum} skipping NBA due to ENABLE_NBA=0")
            continue

        game   = synth_game_raw(r)
        player = str(r.get("Player", "")).strip()
        market = str(r.get("Market", "")).strip()
        side   = str(r.get("Side") or r.get("Pick Side") or r.get("A/ Pub") or "").strip()
        line   = to_float(r.get("Line") or r.get("Pick Line") or r.get("Threshold"))
        when   = r.get("Event Start (ET)") or r.get("Game Time") or r.get("Timestamp (ET)") or r.get("Timestamp", "")

        # Skip future events to avoid hammering APIs before games start
        try:
            event_dt = parser.parse(when, tzinfos=TZINFOS)
            if event_dt > now_local:
                logging.debug(f"r{rnum} future event {event_dt}, skipping for now")
                continue
        except Exception:
            pass

        # Verify required fields
        missing = []
        if not game:        missing.append("game (Game (Away @ Home) or Game)")
        if not player:      missing.append("player")
        if not market:      missing.append("market")
        if not side:        missing.append("side")
        if line is None:    missing.append("line")
        if not when:        missing.append("event time")
        if missing:
            logging.debug(f"r{rnum} ineligible → missing: {', '.join(missing)}")
            continue

        # Resolve stat key
        if sport == "basketball_nba":
            key = nba_stat_from_market(market)
        else:
            key = nhl_stat_from_market(market)
        if not key:
            logging.debug(f"r{rnum} unsupported market '{market}'")
            continue

        eligible += 1

        # Fetch actual
        if sport == "basketball_nba":
            actual = nba_find_player_actual(game, when, player, key)
        else:
            actual = nhl_find_player_actual(game, when, player, key)

        if actual is None:
            logging.debug(f"r{rnum} no actual found for {player} {market} | game='{game}' | when='{when}'")
            continue

        result = grade(side, line, actual)
        if result is None:
            logging.debug(f"r{rnum} could not grade (side='{side}', line={line}, actual={actual})")
            continue

        logging.debug(f"r{rnum}: {player} {market} {side} {line} → actual={actual}, result={result}")

        if not DRY_RUN:
            batch.append({"range": f"PlayerProps!{rowcol_to_a1(rnum, c_actual)}", "values": [[actual]]})
            batch.append({"range": f"PlayerProps!{rowcol_to_a1(rnum, c_result)}", "values": [[result]]})
            touched += 1

        # Flush every 100 cells
        if len(batch) >= 100 and not DRY_RUN:
            SS.values_batch_update({"valueInputOption": "USER_ENTERED", "data": batch})
            logging.debug("batch write (100 cells)")
            batch.clear()
            time.sleep(0.2)

    if batch and not DRY_RUN:
        SS.values_batch_update({"valueInputOption": "USER_ENTERED", "data": batch})

    logging.info(f"Eligible rows={eligible}; graded={touched}")

if __name__ == "__main__":
    try:
        update_playerprops(WS)
    except Exception as e:
        logging.exception(f"playerprops_grade failed: {e}")
