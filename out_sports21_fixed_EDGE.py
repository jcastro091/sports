#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Imports / Setup =========
import os, sys, re, json, math, time, html, asyncio, logging, secrets, csv, signal, argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
import aiohttp
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Bot
from dateutil import parser
from gspread.exceptions import APIError
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# --- NEW: load .env so GOOGLE_CREDS_FILE and others are picked up ---
from dotenv import load_dotenv
load_dotenv()  # will read .env in the current working directory


def getenv_int(key: str, default: int) -> int:
    val = os.getenv(key, str(default))
    # Remove inline comments
    val = val.split("#")[0].strip()
    return int(val)


# ===== ML hook =====


try:
    # Preferred: when alpha_signal_engine is installed as a package
    from src.ml_predict import predict_win_prob  # type: ignore
except ModuleNotFoundError:
    # Fallback: load ml_predict.py from a *relative* path in this repo
    base_dir = Path(__file__).resolve().parent / "alpha_signal_engine" / "src"
    ml_path = base_dir / "ml_predict.py"

    spec = spec_from_file_location("ml_predict", ml_path)
    _mod = module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(_mod)
    predict_win_prob = _mod.predict_win_prob  # type: ignore[attr-defined]




# ========= Argparse / Logging =========
from logging.handlers import RotatingFileHandler

# 1) Parse CLI args first
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Verbose logging")
parser.add_argument("--trace-pinnacle", action="store_true",
                    help="Dump Pinnacle market/outcome details when nulls/missing")

args = parser.parse_args()

TRACE_PINNACLE = (
    args.trace_pinnacle
    or os.getenv("TRACE_PINNACLE", "0").lower() in ("1", "true", "yes", "y")
)
logging.debug("TRACE_PINNACLE=%s", TRACE_PINNACLE)


def _safe_log_path():
    p1 = (Path(__file__).parent / "logs")
    try:
        p1.mkdir(exist_ok=True)
        test = p1 / ".__writetest"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return p1
    except Exception:
        pass
    p2 = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "SharpSignal" / "logs"
    p2.mkdir(parents=True, exist_ok=True)
    return p2

log_dir = _safe_log_path()
log_file = log_dir / "sports.log"

handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
]
logging.basicConfig(
    level=(logging.DEBUG if args.verbose else logging.INFO),
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=handlers,
    force=True,
)

print(f"[boot] logging to: {log_file}")  # shows even if logging misconfigured
logging.info("✅ sports.py started (verbose=%s)", args.verbose)
logging.info("📂 Logging to %s", log_file)


logging.info("⚙️ ML hook loaded and ready")





# ---- Healthchecks.io (optional) ----
import requests, time
HC_URL = os.getenv("HEARTBEAT_URL_SPORTS", "").strip()
_last_hb_min = None

def hb_ping(suffix=""):
    if not HC_URL:
        return
    url = HC_URL.rstrip("/") + ("/" + suffix.lstrip("/") if suffix else "")
    try:
        r = requests.get(url, timeout=5)
        logging.debug("Healthcheck ping %s -> %s %s", url, r.status_code, (r.text or "")[:120])
    except Exception as e:
        logging.debug("Healthcheck ping error %s: %s", url, e)


def hb_tick():
    global _last_hb_min
    m = int(time.time() // 60)
    if m != _last_hb_min:
        _last_hb_min = m
        hb_ping("")  # minute heartbeat


# ========= Config =========
TIMEZONE = pytz.timezone("US/Eastern")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4")
PRO_CHAT_ID = int(os.getenv("PRO_CHAT_ID", "-1002756836961"))
ENTERPRISE_CHAT_ID = int(os.getenv("ENTERPRISE_CHAT_ID", "-1002635869925"))

API_KEY = os.getenv("ODDS_API_KEY", "7e79a3f0859a50153761270b1d0e867f")
PRIMARY_BOOKMAKER = "pinnacle"
BOOKMAKERS = ["pinnacle", "lowvig", "betonlineag"]
REGION = "us,eu"
MARKETS = ["h2h", "spreads", "totals"]
INCLUDE_BET_LIMITS = True
PINNACLE_ONLY_PEAKS = True  # only Pinnacle prices seed/update opening/peaks
RECORD_ON_SEND_ONLY = os.getenv("RECORD_ON_SEND_ONLY", "1").lower() in ("1","true","yes")

# ========= Tiering thresholds (env-overridable) =========
TIER_A_MIN_PROBA = float(os.getenv("TIER_A_MIN_PROBA", "0.60"))
TIER_A_MIN_EDGE  = float(os.getenv("TIER_A_MIN_EDGE",  "0.08"))   # 8%
TIER_A_MIN_REV_CENTS = int(os.getenv("TIER_A_MIN_REV_CENTS", "20"))

TIER_B_MIN_PROBA = float(os.getenv("TIER_B_MIN_PROBA", "0.55"))
TIER_B_MIN_EDGE  = float(os.getenv("TIER_B_MIN_EDGE",  "0.06"))   # 6%
TIER_B_MIN_REV_CENTS = int(os.getenv("TIER_B_MIN_REV_CENTS", "20"))

TIER_C_MIN_PROBA = float(os.getenv("TIER_C_MIN_PROBA", "0.50"))
TIER_C_MIN_EDGE  = float(os.getenv("TIER_C_MIN_EDGE",  "0.04"))   # 4%
TIER_C_MIN_REV_CENTS = int(os.getenv("TIER_C_MIN_REV_CENTS", "10"))


# Use a small list first; add more when you confirm it’s running chatty
SPORTS_LIST: List[str] = [
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




SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "ConfirmedBets")
TAB_BETS = os.getenv("TAB_BETS", "AllBets")
TAB_OBS  = os.getenv("TAB_OBS", "AllObservations")
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE", os.path.join(os.path.dirname(__file__), "telegrambetlogger-35856685bc29.json"))

# ========= Config =========
ALERT_WINDOW_MIN = getenv_int("ALERT_WINDOW_MIN", 60)
SEEN_RECENT_SEC  = getenv_int("SEEN_RECENT_SEC", 3600)
MIN_MINUTES_SINCE_PEAK = getenv_int("MIN_MINUTES_SINCE_PEAK", 5)
TRACK_BASE = getenv_int("TRACK_BASE", 300)

DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "10000"))

STRONG_CONFIG_JSON = os.getenv("STRONG_CONFIG_JSON", "strong_config.json")


PEAK_PRICES_FILE = "peak_prices.json"
TRACKED_GAMES_FILE = "tracked_games.json"
RECENT_PICKS_FILE = "recent_picks.json"


hb_ping("")

logging.info("ENV: ODDS_API_KEY=%s, TELEGRAM_BOT_TOKEN=%s, CREDS=%s",
             ("set" if API_KEY else "MISSING"),
             ("set" if TELEGRAM_BOT_TOKEN else "MISSING"),
             GOOGLE_CREDS_FILE)

# ========= Telegram =========
bot = Bot(token=TELEGRAM_BOT_TOKEN)
def send_pro(text: str):
    try: bot.send_message(chat_id=PRO_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e: logging.error("Telegram Pro send failed: %s", e)
def send_enterprise(text: str):
    try: bot.send_message(chat_id=ENTERPRISE_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e: logging.error("Telegram Ent send failed: %s", e)

# ========= Sheets =========
def _authorize_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
    gc = gspread.authorize(creds)
    return gc

try:
    gc = _authorize_sheets()
    sheet_bets = gc.open(SPREADSHEET_NAME).worksheet(TAB_BETS)
    obs_sheet = gc.open(SPREADSHEET_NAME).worksheet(TAB_OBS)
    logging.info("🧾 Google Sheets connected: %s / %s, %s", SPREADSHEET_NAME, TAB_BETS, TAB_OBS)
except Exception as e:
    logging.error("❌ Google Sheets auth/open failed: %s", e)
    # We keep running; rows just won't append if creds are wrong.
    gc = None
    sheet_bets = None
    obs_sheet = None

# ========= Small utils =========
import json
from pathlib import Path

TIER_CONFIG_PATH = Path(__file__).parent / "tier_config.json"

def load_tier_config(path: Path = TIER_CONFIG_PATH):
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tiers = cfg.get("tiers", [])
    default = cfg.get("default", {"code": "", "label": "Pass"})
    return tiers, default


TIERS, DEFAULT_TIER = load_tier_config()


def assign_tier(proba: float, sport: str, market: str, odds_bin: str):
    """
    Map model output -> Tier Code / Label using tier_config.json.
    """
    sport = (sport or "").strip()
    market = (market or "").strip()
    odds_bin = (odds_bin or "").strip()

    for rule in TIERS:
        if sport not in rule["sports"]:
            continue
        if market not in rule["markets"]:
            continue
        if odds_bin not in rule["odds_bin"]:
            continue

        if proba < rule.get("min_proba", 0.0):
            continue
        if proba > rule.get("max_proba", 1.0):
            continue

        return rule["code"], rule["label"]

    return DEFAULT_TIER["code"], DEFAULT_TIER["label"]


def append_by_header(ws, values: dict):
    """
    Append to ws by matching dict keys to the sheet's header names.
    Unknown headers are left blank; safe if columns are reordered.
    """
    headers = ws.row_values(1)
    row = [values.get(h, "") for h in headers]
    ws.append_row(row, value_input_option="USER_ENTERED")



def format_sport(key: str) -> str:
    # "basketball_nba" -> "NBA", "icehockey_nhl" -> "NHL", "americanfootball_ncaaf" -> "NCAAF", etc.
    tail = (key or "").split("_")[-1]
    return tail.upper() if tail else key.upper()



def _fit_to_headers(row, ws):
    """Pad or trim a row to exactly match the sheet's header count."""
    try:
        header_len = len(ws.row_values(1))
    except Exception:
        header_len = len(row)
    if len(row) < header_len:
        row = row + [""] * (header_len - len(row))
    elif len(row) > header_len:
        row = row[:header_len]
    return row


# ---- Per-game alert candidates (pick best per market group) ----
def _choose_best(cands):
    # cands: list[dict] with keys: group_key, edge, p_ml
    if not cands:
        return None
    # higher edge wins; tie -> higher p_ml
    return sorted(cands, key=lambda c: (c["edge"], c["p_ml"]), reverse=True)[0]


def market_group_from_label(label: str) -> str:
    if label.startswith("H2H"): return "H2H"
    if label.startswith("Spread"): return "Spread"
    if label.startswith("Total"): return "Totals"
    return "Other"



def parse_commence_utc(s) -> datetime:
    # Accept datetime or string; always return tz-aware UTC datetime
    if isinstance(s, datetime):
        dt = s
    else:
        s = str(s or "").replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            dt = parser.isoparse(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _dump_pinnacle_markets(game, why=""):
    if not TRACE_PINNACLE: return
    b = _get_book(game, "pinnacle")
    logging.debug("🔎 [PIN] %s %s@%s commence=%s", why, game.get("away_team"), game.get("home_team"), game.get("commence_time"))
    if not b:
        logging.debug("🔎 [PIN] no Pinnacle bookmaker on this event")
        return
    for m in b.get("markets") or []:
        key = (m.get("key") or "").lower()
        logging.debug("🔎 [PIN] market=%s outcomes=%d", key, len(m.get("outcomes") or []))
        for o in m.get("outcomes") or []:
            nm = (o.get("name") or o.get("participant") or o.get("description") or "").strip()
            logging.debug("    - %s | price=%s point=%s", nm, o.get("price"), o.get("point"))


def decimal_to_american(dec) -> Optional[int]:
    try: d = float(dec)
    except: return None
    if d <= 1.0: return None
    if d >= 2.0: return int(round((d - 1.0) * 100))
    return int(round(-100 / (d - 1.0)))
    

def live_odds_bin(dec) -> str:
    try: imp = 1.0 / float(dec)
    except: return "unknown"
    if imp >= 0.70: return "HeavyFav"
    if imp >= 0.55: return "Fav"
    if imp >= 0.40: return "Balanced"
    if imp >= 0.20: return "Dog"
    return "Longshot"
    
    
def dynamic_cents_threshold(american_at_peak: int) -> int:
    a = american_at_peak
    # Favorites (negative): shorter favs require bigger reversal
    if a <= -300: return 50
    if a <= -250: return 40
    if a <= -200: return 30
    if a <= -150: return 20
    if a <= -105: return 10  # -105..-149
    # Underdogs (positive): longer dogs require bigger reversal
    if a >= 300:  return 40
    if a >= 250:  return 30
    if a >= 200:  return 25
    if a >= 150:  return 20
    return 15  # +100..+149

def is_drop_from_peak_american(peak_dec: float, curr_dec: float) -> tuple[bool, int, int]:
    """
    Returns (dropped?, drop_cents, threshold_cents) using American odds.
    """
    a_peak = decimal_to_american(peak_dec)
    a_curr = decimal_to_american(curr_dec)
    if a_peak is None or a_curr is None:
        return False, 0, 999999
    dropped = a_curr < a_peak         # lower = worse price for you → a drop from peak
    drop_cents = (a_peak - a_curr) if dropped else 0
    threshold_cents = dynamic_cents_threshold(a_peak)
    return dropped, drop_cents, threshold_cents


def cents_threshold_for_decimal(d: float) -> float:
    if d < 1.7:  return 0.03
    if d < 2.1:  return 0.05
    if d < 3.0:  return 0.07
    return 0.10

def parse_move_delta(s: str):
    if not s: return None
    parts = [p.strip() for p in str(s).split("→")]
    nums = [re.search(r"[-+]?\d+", p) for p in parts]
    nums = [int(m.group()) for m in nums if m]
    if len(nums) >= 2: return nums[-1] - nums[0]
    return nums[-1] if nums else None

def gen_trade_id(away: str, home: str, label: str) -> str:
    ts_hex = format(int(time.time()*1000), "x").upper()
    lbl = "".join(ch for ch in label if ch.isalnum())[:3].upper() or "MKT"
    rand = secrets.token_hex(3).upper()
    return f"T{ts_hex}-{lbl}-{rand}"

def _json_default(o):
    if isinstance(o, datetime): return o.astimezone(timezone.utc).isoformat()
    return str(o)

def _read_json_file(path: str, default):
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception: pass
    return default

def _write_json_file(path: str, obj):
    try:
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    except Exception as e:
        logging.warning("Couldn't write %s: %s", path, e)

# State
tracked_games: Dict[str, Any] = _read_json_file(TRACKED_GAMES_FILE, {})
peak_prices: Dict[str, Any] = _read_json_file(PEAK_PRICES_FILE, {})
recent_picks: Dict[str, float] = _read_json_file(RECENT_PICKS_FILE, {})

def save_state():
    _write_json_file(TRACKED_GAMES_FILE, tracked_games)
    _write_json_file(PEAK_PRICES_FILE, peak_prices)
    _write_json_file(RECENT_PICKS_FILE, recent_picks)

def seen_recently(game_id: str, dedupe_key: str) -> bool:
    ts = recent_picks.get(f"{game_id}|{dedupe_key}")
    return bool(ts and (time.time() - ts) < SEEN_RECENT_SEC)

def mark_recent(game_id: str, dedupe_key: str):
    recent_picks[f"{game_id}|{dedupe_key}"] = time.time()



# ========= Strong config =========
def _load_json_field(path: str, key: str, default):
    try:
        if os.path.exists(path):
            return json.loads(open(path,"r").read()).get(key, default)
    except Exception:
        pass
    return default

STRONG_CONFIG_JSON = os.getenv("STRONG_CONFIG_JSON", "strong_config.json")
MIN_PROBA = float(_load_json_field(STRONG_CONFIG_JSON, "min_proba", 0.57))
MAX_PROBA = _load_json_field(STRONG_CONFIG_JSON, "max_proba", None)
ALLOWED_BINS = set(map(str, _load_json_field(STRONG_CONFIG_JSON, "odds_bin", ["Fav"])))
STRONG_SPORTS_SET = set(map(str, _load_json_field(STRONG_CONFIG_JSON, "sports", SPORTS_LIST)))
STRONG_MARKETS_SET = set(map(str, _load_json_field(STRONG_CONFIG_JSON, "markets",
                                                   ["H2H Home","H2H Away","Total Under"])))

# Load sport×market pairs if present
STRONG_PAIRS_FILE = os.getenv("STRONG_PAIRS_JSON", "strong_pairs.json")
PAIR_WHITELIST = set()
try:
    if os.path.exists(STRONG_PAIRS_FILE):
        pairs = json.loads(open(STRONG_PAIRS_FILE, "r").read())
        PAIR_WHITELIST = {(p.get("sport",""), p.get("market","")) for p in pairs}
except Exception:
    pass

def is_strong(sport_key: str, market_label: str) -> bool:
    base_ok = (sport_key in STRONG_SPORTS_SET) and (market_label in STRONG_MARKETS_SET)
    if PAIR_WHITELIST:
        return base_ok and ((sport_key, market_label) in PAIR_WHITELIST)
    return base_ok


# ========= Odds extraction helpers =========
def _get_book(data: dict, name: str):
    for b in data.get("bookmakers") or []:
        if (b.get("key") or "").lower() == name.lower():
            return b
    return None

def _extract_price_from_market(markets, market_key, side, *, home_name="", away_name="") -> Optional[float]:
    side_l = side.lower(); home_l = (home_name or "").lower(); away_l = (away_name or "").lower()
    for m in markets or []:
        if (m.get("key") or "").lower() != market_key: continue
        for o in (m.get("outcomes") or []):
            nm = (o.get("name") or o.get("participant") or o.get("description") or "").strip().lower()
            def _try(v): 
                try: return float(v)
                except: return None
            if market_key == "totals":
                if nm.startswith("over") and side_l=="over": return _try(o.get("price"))
                if nm.startswith("under") and side_l=="under": return _try(o.get("price"))
            else:
                if nm == side_l: return _try(o.get("price"))
                if side_l=="home" and home_l and nm == home_l: return _try(o.get("price"))
                if side_l=="away" and away_l and nm == away_l: return _try(o.get("price"))
                
                # after exact matches fail:
                if market_key in ("h2h","spreads"):
                    # two-outcome fallback: assume ordering is [home, away]
                    outs = [o for o in m.get("outcomes") or []]
                    if len(outs) == 2:
                        idx = 0 if side_l=="home" else 1 if side_l=="away" else None
                        if idx is not None:
                            try: return float(outs[idx].get("price"))
                            except: return None

    return None

def _extract_line_from_market(markets, market_key, which, *, home_name="", away_name="") -> Optional[float]:
    which_l = (which or "").lower()
    home_l = (home_name or "").lower()
    away_l = (away_name or "").lower()

    for m in (markets or []):
        if (m.get("key") or "").lower() != market_key:
            continue
        outs = m.get("outcomes") or []

        # Totals: any outcome carries the total "point"
        if market_key == "totals" and which_l == "total":
            for o in outs:
                try:
                    return float(o.get("point"))
                except:
                    pass
            return None

        # Spreads: we want the handicap ("point") for home/away
        if market_key == "spreads":
            for o in outs:
                nm = (o.get("name") or o.get("participant") or o.get("description") or "").strip().lower()
                if nm == which_l or (which_l == "home" and nm == home_l) or (which_l == "away" and nm == away_l):
                    try:
                        return float(o.get("point"))
                    except:
                        return None

            # fallback for two-outcome spreads: [home, away] order
            if len(outs) == 2 and which_l in ("home","away"):
                idx = 0 if which_l == "home" else 1
                try:
                    return float(outs[idx].get("point"))
                except:
                    return None

    return None



def _extract_bet_limit_max(bookmaker: dict) -> Optional[float]:
    """Return the maximum limit found anywhere under this bookmaker structure.
       Supports Odds API includeBetLimits shapes (direct keys, nested 'limits', 'maxStake')."""
    if not bookmaker:
        return None

    def _max_from_obj(obj):
        if not isinstance(obj, dict):
            return None
        vals = []
        for k in ("limit", "max", "maxLimit", "maxStake", "maximum"):
            try:
                v = float(obj.get(k))
                if v > 0:
                    vals.append(v)
            except Exception:
                pass
        return max(vals) if vals else None

    best = None

    # bookmaker-level direct keys
    v = _max_from_obj(bookmaker)
    if v is not None:
        best = v if best is None else max(best, v)

    # bookmaker-level "limits" map (e.g., {"h2h":{"max":...},"spreads":{"maxStake":...}})
    limits_map = bookmaker.get("limits")
    if isinstance(limits_map, dict):
        for _, lim_obj in limits_map.items():
            v = _max_from_obj(lim_obj if isinstance(lim_obj, dict) else {})
            if v is not None:
                best = v if best is None else max(best, v)

    # market-level + each market's "limits"
    for m in (bookmaker.get("markets") or []):
        v = _max_from_obj(m)
        if v is not None:
            best = v if best is None else max(best, v)
        lims = m.get("limits")
        if isinstance(lims, dict):
            v = _max_from_obj(lims)
            if v is not None:
                best = v if best is None else max(best, v)
        # outcomes-level
        for o in (m.get("outcomes") or []):
            v = _max_from_obj(o)
            if v is not None:
                best = v if best is None else max(best, v)

    return best



# ========= Parse one event =========
def parse_odds(game: dict) -> Optional[dict]:
    try:
        commence = parse_commence_utc(game.get("commence_time"))
        home, away = game["home_team"], game["away_team"]
    except:
        return None

    # choose book (Pinnacle preferred)
    markets, used_book = [], None
    for bk in [PRIMARY_BOOKMAKER] + [b for b in BOOKMAKERS if b != PRIMARY_BOOKMAKER]:
        b = _get_book(game, bk)
        if b and (b.get("markets") or []):
            markets, used_book = b.get("markets"), bk
            if used_book != "pinnacle":
                logging.debug("⚠️ Pinnacle missing or empty for %s@%s; used %s instead", away, home, used_book)
                _dump_pinnacle_markets(game, "pinnacle_missing")
            break

    if not markets:
        logging.debug("No markets available for %s@%s", away, home)
        _dump_pinnacle_markets(game, "no_markets")  # <— add this
        return None

    # prices + lines
    home_price  = _extract_price_from_market(markets, "h2h", "home",  home_name=home, away_name=away)
    away_price  = _extract_price_from_market(markets, "h2h", "away",  home_name=home, away_name=away)
    spread_home = _extract_price_from_market(markets, "spreads", "home", home_name=home, away_name=away)
    spread_away = _extract_price_from_market(markets, "spreads", "away", home_name=home, away_name=away)
    totals_over = _extract_price_from_market(markets, "totals", "over")
    totals_under= _extract_price_from_market(markets, "totals", "under")
    spread_line_home = _extract_line_from_market(markets, "spreads", "home", home_name=home, away_name=away)
    spread_line_away = _extract_line_from_market(markets, "spreads", "away", home_name=home, away_name=away)
    total_line       = _extract_line_from_market(markets, "totals",  "total")
    
    
    if used_book == "pinnacle":
        if home_price is None or away_price is None:
            logging.debug("🧩 [PIN] H2H nulls for %s@%s (home=%s away=%s) — dumping outcomes",
                          away, home, home_price, away_price)
            _dump_pinnacle_markets(game, "h2h_null")
        if spread_home is None or spread_away is None or spread_line_home is None or spread_line_away is None:
            logging.debug("🧩 [PIN] Spreads nulls for %s@%s (ph=%s pa=%s lph=%s lpa=%s) — dumping outcomes",
                          away, home, spread_home, spread_away, spread_line_home, spread_line_away)
            _dump_pinnacle_markets(game, "spreads_null")
        if totals_over is None or totals_under is None or total_line is None:
            logging.debug("🧩 [PIN] Totals partial/null for %s@%s (over=%s under=%s line=%s) — dumping outcomes",
                          away, home, totals_over, totals_under, total_line)
            _dump_pinnacle_markets(game, "totals_null")


    # backfill H2H from LV/BOL if missing
    if (not PINNACLE_ONLY_PEAKS) and (home_price is None or away_price is None):
        for bk in ("lowvig", "betonlineag"):
            b = _get_book(game, bk)
            if not b: continue
            mks = b.get("markets") or []
            if home_price is None:
                hp = _extract_price_from_market(mks, "h2h", "home", home_name=home, away_name=away)
                if hp is not None: home_price = hp
            if away_price is None:
                ap = _extract_price_from_market(mks, "h2h", "away", home_name=home, away_name=away)
                if ap is not None: away_price = ap


    # bet limit — prefer Pinnacle even if used_book != Pinnacle
    bet_limit = None
    try:
        pin = _get_book(game, "pinnacle")
        # one-time probe removed (no game_id here); simple debug is fine:
        if pin:
            logging.debug("[PIN limits probe] keys=%s ; limits=%s",
                          sorted(list(pin.keys())),
                          json.dumps(pin.get("limits", {}))[:300])
            pin_lim = _extract_bet_limit_max(pin)
        else:
            pin_lim = None

        if pin_lim is not None:
            bet_limit = pin_lim
        else:
            sel = _get_book(game, used_book) if used_book else None
            bet_limit = _extract_bet_limit_max(sel)
    except Exception as e:
        logging.debug("limit extract error: %s", e)




    # extras (snapshot)
    extras = {}
    for bk in ("lowvig", "betonlineag"):
        b = _get_book(game, bk)
        if not b: continue
        mks = b.get("markets") or []
        extras[bk] = {
            "home": _extract_price_from_market(mks, "h2h", "home", home_name=home, away_name=away),
            "away": _extract_price_from_market(mks, "h2h", "away", home_name=home, away_name=away),
        }

    result = {
        "home_team": home, "away_team": away, "commence_time": commence,
        "home_price": home_price, "away_price": away_price,
        "spread_home": spread_home, "spread_away": spread_away,
        "totals_home": totals_over, "totals_away": totals_under,
        "spread_line_home": spread_line_home, "spread_line_away": spread_line_away,
        "total_line": total_line, "extras": extras, "bet_limit": bet_limit,
        "book_used": used_book,                # <— RETURN THIS
    }
    return result

# ========= Reason builder =========
def _rule_based_reason(label: str, p_ml: float, decimal_odds: float, movement_str: str, limit_now, trend: str) -> str:
    bits = []
    if label.startswith("H2H"):
        if decimal_odds >= 2.0:
            bits.append("Underdog price suggests value.")
        bits.append("Reversal after a peak.")
    elif label.startswith("Total"):
        bits.append("Late reversal on the total.")
    elif label.startswith("Spread"):
        bits.append("Spread moved off the peak.")
    if isinstance(limit_now, (int, float)):
        bits.append(f"Limits {trend} into kickoff.")
    return " ".join(bits)


# ========= Graceful shutdown =========
_shutdown = False
def _sig_handler(signum, frame):
    global _shutdown; _shutdown = True
    hb_ping("")
try:
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
except Exception: pass

# ========= Main loop =========
async def track_games():
    global tracked_games, peak_prices
    async with aiohttp.ClientSession() as session:
        while not _shutdown:
            cycle_start = time.time()
            logging.info("⏳ Poll cycle start")
            hb_tick()

            try:
                now_utc = datetime.now(timezone.utc)
                logging.info("⏱ now_utc=%s", now_utc.isoformat())
                all_events = []
                
                # per-cycle counters
                pin_with_any = pin_h2h = pin_spread = pin_total = 0

                # fetch
                for sport in SPORTS_LIST:
                    url=(f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
                         f"?regions={REGION}&markets={','.join(MARKETS)}&bookmakers={','.join(BOOKMAKERS)}"
                         f"&apiKey={API_KEY}&includeBetLimits={'true' if INCLUDE_BET_LIMITS else 'false'}")
                    try:
                        data = []
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                logging.error("❌ Fetch odds failed %s: %s", sport, await resp.text()); continue
                            data = await resp.json()
                            logging.debug("%s: events=%d", sport, len(data) if isinstance(data,list) else -1)

                            # Immediately after: data = await resp.json()
                            if isinstance(data, dict):
                                # The Odds API sometimes returns an error object with HTTP 200
                                logging.error("❗ %s returned object instead of list: %s", sport, json.dumps(data)[:300])
                                continue
                            if not data:
                                logging.info("🛰️ %s fetched 0 events (empty list from API)", sport)
                                continue


                            logging.debug("🔧 %s raw events=%d (regions=%s, books=%s, limits=%s)",
                                          sport, len(data), REGION, ",".join(BOOKMAKERS), INCLUDE_BET_LIMITS)

                            # Peek ONE event to see if bookmakers/markets are there
                            if data:
                                eg = data[0]
                                bkeys = [b.get("key") for b in (eg.get("bookmakers") or [])]
                                logging.debug("🔧 %s eg: commence=%s | bookmakers=%s",
                                              sport, eg.get("commence_time"), bkeys)
                                              

                    except Exception as e:
                        logging.error("❌ HTTP error for %s: %s", sport, e); continue

                    
                    
                    
                    upcoming = 0
                    earliest = latest = None
                    for g in data:
                        g["sport_key"] = sport
                        try:
                            ct = parse_commence_utc(g.get("commence_time"))
                        except Exception:
                            continue

                        # Collect everything; filter past later.
                        all_events.append(g)

                        # Visibility counters + quick window telemetry
                        earliest = ct if earliest is None or ct < earliest else earliest
                        latest   = ct if latest   is None or ct > latest   else latest
                        if ct > now_utc:
                            upcoming += 1

                    logging.info("🛰️ %s fetched %d upcoming events | window: %s … %s (now=%s, Δmin earliest=%.1f, latest=%.1f)",
                                 sport, upcoming,
                                 earliest.isoformat() if earliest else "—",
                                 latest.isoformat()   if latest   else "—",
                                 now_utc.isoformat(),
                                 0.0 if earliest is None else (earliest - now_utc).total_seconds()/60.0,
                                 0.0 if latest   is None else (latest   - now_utc).total_seconds()/60.0)
                                        

                future_events = [
                    g for g in all_events
                    if parse_commence_utc(g.get("commence_time")) > now_utc
                ]
                logging.info("📣 cycle summary: all=%d | future=%d (first=%s, last=%s)",
                             len(all_events), len(future_events),
                             "-" if not future_events else parse_commence_utc(future_events[0]["commence_time"]).isoformat(),
                             "-" if not future_events else parse_commence_utc(future_events[-1]["commence_time"]).isoformat())
                
                

                # process
                alerts_fired = 0
                for game in all_events:
                    parsed = parse_odds(game)
                    if not parsed: continue
                    
                    
                    if parsed.get("book_used") == "pinnacle":
                        pin_with_any += 1
                        if parsed.get("home_price") is not None and parsed.get("away_price") is not None:
                            pin_h2h += 1
                        if parsed.get("spread_home") is not None and parsed.get("spread_away") is not None:
                            pin_spread += 1
                        if (parsed.get("totals_home") is not None and
                            parsed.get("totals_away") is not None and
                            parsed.get("total_line")  is not None):
                            pin_total += 1
         

                    game_id = f"{parsed['away_team']}@{parsed['home_team']}"
                    
                    game_time = parse_commence_utc(game.get("commence_time"))
                    if game_time <= now_utc:
                        continue


                    # init trackers
                    if game_id not in peak_prices:
                        peak_prices[game_id] = {
                            "home_team": parsed["home_team"],
                            "away_team": parsed["away_team"],
                            "commence_time": game_time,
                            "home_price": None, "away_price": None,
                            "spread_home": None, "spread_away": None,
                            "totals_home": None, "totals_away": None,
                            "spread_line_home": parsed.get("spread_line_home"),
                            "spread_line_away": parsed.get("spread_line_away"),
                            "total_line": parsed.get("total_line"),
                            "extras": parsed.get("extras", {}),
                            "bet_limit": parsed.get("bet_limit"),
                            "_limit_prev": None,
                            "_limit_prev_ts": None,
                            "pinnacle_opening_done": False,
                            "_peak_ts": {},     # NEW: when each market’s peak was set
                            "_alerted": {},     # NEW: one-time alert marker per market
                        }
 
 
                        
                        
                        # seed ONLY if Pinnacle
                        if parsed.get("book_used") == "pinnacle":
                            for k in ["home_price","away_price","spread_home","spread_away","totals_home","totals_away"]:
                                v = parsed.get(k)
                                if v is not None:
                                    peak_prices[game_id][f"opening_{k}"] = v
                                    peak_prices[game_id][k] = v
                            peak_prices[game_id]["pinnacle_opening_done"] = True                        
                                            

                    # update peaks/openings (even far from kickoff)

                    price_keys = {
                        "home_price":"H2H Home","away_price":"H2H Away",
                        "spread_home":"Spread Home","spread_away":"Spread Away",
                        "totals_home":"Total Over","totals_away":"Total Under",
                    }

                    time_until = (game_time - now_utc).total_seconds() / 60.0

                    # --- BLOCK non-Pinnacle from changing openings/peaks ---
                    if PINNACLE_ONLY_PEAKS and parsed.get("book_used") != "pinnacle":
                        pass
                    else:
                        # one-time overwrite of openings the first time Pinnacle shows up
                        if parsed.get("book_used") == "pinnacle" and not peak_prices[game_id].get("pinnacle_opening_done"):
                            for key in price_keys.keys():
                                curr = parsed.get(key)
                                if curr is not None:
                                    peak_prices[game_id][f"opening_{key}"] = curr
                                    peak_prices[game_id][key] = curr
                            peak_prices[game_id]["pinnacle_opening_done"] = True

                                                                
                        for key in price_keys.keys():
                            curr = parsed.get(key)
                            if curr is None:
                                continue
                            prev_peak = peak_prices[game_id].get(key)
                            if prev_peak is None or curr > prev_peak:
                                peak_prices[game_id][key] = curr
                                peak_prices[game_id].setdefault("_peak_ts", {})[key] = time.time()     # NEW
                                # New higher peak invalidates prior ‘alerted’ state for this market:
                                peak_prices[game_id].setdefault("_alerted", {}).pop(key, None)         # NEW

                    # ---- Rolling limit baseline/trend (every cycle) ----
                    limit_now = parsed.get("bet_limit")
                    if isinstance(limit_now, (int, float)):
                        EPS = float(os.getenv("LIMIT_TREND_EPS", "1.01"))  # 1% default
                        prev = peak_prices.get(game_id, {}).get("_limit_prev")
                        trend = "flat"
                        if isinstance(prev, (int, float)):
                            if limit_now >= prev * EPS:
                                trend = "up"
                            elif limit_now <= prev / EPS:
                                trend = "down"
                        peak_prices.setdefault(game_id, {})["_limit_prev"] = limit_now
                        peak_prices[game_id]["_limit_prev_ts"] = time.time()
                        peak_prices[game_id]["_limit_trend"] = trend

                                       

                    # alert window gate
                    if time_until > ALERT_WINDOW_MIN:
                        continue  # keep peaks updated, no alert
                        
                        
                    if PINNACLE_ONLY_PEAKS and parsed.get("book_used") != "pinnacle": 
                        continue

                    candidates_for_game = []
                    # evaluate reversal vs peak (per market)
                    for key, label in price_keys.items():
                        
                        
                        curr = parsed.get(key); openv = peak_prices[game_id].get(f"opening_{key}")
                        peakv = peak_prices[game_id].get(key)
                        if curr is None or openv is None or peakv is None:
                            continue
                        if curr > peakv:  # new peak, update and skip
                            peak_prices[game_id][key] = curr
                            continue

                        # move_dir = "Down" if curr < peakv else "Up"
                        # band = cents_threshold_for_decimal(curr)
                        # # require at least band reversal from peak (or within 60m allow any under opening)
                        # if not ((move_dir=="Down" and (peakv - curr) >= band) and (curr < openv or time_until <= 60)):
                            # continue
                            
                            
                        ap  = peak_prices[game_id].setdefault("_alerted", {})
                        pts = peak_prices[game_id].setdefault("_peak_ts", {})

                        # One-time per game×market
                        if ap.get(key):
                            continue

                        # Last-hour only (you already gate earlier, this is extra safety)
                        if not (0 < time_until <= 60):
                            continue

                        # Optional cool-off after the peak was set (avoid immediate whipsaw)
                        if MIN_MINUTES_SINCE_PEAK and (time.time() - pts.get(key, 0) < MIN_MINUTES_SINCE_PEAK * 60):
                            continue

                        # Dynamic American-cents drop from peak
                        dropped, drop_cents, threshold_cents = is_drop_from_peak_american(peakv, curr)
                        if not (dropped and drop_cents >= threshold_cents):
                            continue


                        # Build alert & log
                        market_name = "H2H" if label.startswith("H2H") else ("Totals" if label.startswith("Total") else "Spread")
                        decimal_odds = float(curr)
                        ml_direction = (
                            "Underdog" if label.startswith("H2H") and decimal_odds>=2.0 else
                            "Favorite" if label.startswith("H2H") else
                            "Over" if label.endswith("Over") else
                            "Under" if label.endswith("Under") else
                            ("Spread Underdog" if decimal_odds>=2.0 else "Spread Favorite")
                        )
                        
                        # Direction for the human-readable movement string
                        move_dir = "Down" if dropped else "Up"

                        movement_str = (
                            f"{decimal_to_american(openv)} → {decimal_to_american(peakv)} → "
                            f"{decimal_to_american(curr)} ({move_dir}, -{drop_cents}¢ vs peak, "
                            f"≥{threshold_cents}¢ req)"
                        )

                        game_time_fmt = game_time.astimezone(TIMEZONE).strftime("%b %d, %I:%M %p %Z").replace(" 0", " ")
                        bet_id = gen_trade_id(parsed["away_team"], parsed["home_team"], label)




                        trend = peak_prices[game_id].get("_limit_trend", "flat")



                        # ML with guard
                        # 1) Build features once so they’re easy to log
                        features = {
                            "sport": game.get("sport_key",""),
                            "market": market_name,
                            "Direction": ml_direction,
                            "Movement": parse_move_delta(movement_str),
                            "american_odds": decimal_to_american(curr),
                            "decimal_odds": decimal_odds,
                            "hours_to_start": max((game_time - datetime.now(TIMEZONE)).total_seconds()/3600.0, 0.0),
                            "bet_limit": limit_now if isinstance(limit_now,(int,float)) else None,
                            "limit_trend": trend,
                            "spread_line_home": parsed.get("spread_line_home"),
                            "spread_line_away": parsed.get("spread_line_away"),
                            "total_line": parsed.get("total_line"),
                            "Confidence": "High",
                            "Kelly": 0.0,
                            "Tags": "Reversal",
                        }

                        # 2) Predict with strong validation + logs
                        try:
                            raw = predict_win_prob(features)
                            p_ml = float(raw)
                            if not math.isfinite(p_ml):
                                raise ValueError(f"non-finite prob: {raw!r}")
                            if p_ml > 1.0:
                                p_ml /= 100.0
                            # clamp
                            p_ml = max(0.001, min(p_ml, 0.999))
                            logging.debug("ML ok p=%.4f feats=%s", p_ml, json.dumps(features)[:400])
                        except Exception as e:
                            logging.error("ML predict error: %s | feats=%s", e, json.dumps(features))
                            # heuristic fallback instead of 0.5 flat:
                            # push slightly above/below implied based on reversal strength
                            p_implied = 1.0 / decimal_odds
                            bump = min(0.10, max(0.02, (drop_cents / max(threshold_cents,1)) * 0.05))
                            if market_name == "H2H":
                                # if it’s an underdog reversal, lean > implied; for fav, lean < implied
                                p_ml = min(0.99, p_implied + bump) if decimal_odds >= 2.0 else max(0.01, p_implied - bump)
                            else:
                                # totals/spreads: small symmetric bump
                                p_ml = min(0.99, max(0.01, p_implied + (0.5 - p_implied) * 0.2))


                        # Kelly 1/4
                        b = decimal_odds - 1.0
                        
                        p_implied = 1.0 / decimal_odds
                        edge = p_ml - p_implied
                        MIN_EDGE = float(os.getenv("MIN_EDGE", "0.015"))  # start at ~1.5%
                        if edge < MIN_EDGE:
                            continue
                        
                        
                        q = 1.0 - p_ml
                        full_kelly = (b*p_ml - q)/b if b>0 else 0.0
                        if p_ml <= p_implied or full_kelly <= 0: full_kelly = 0.0
                        kelly_quarter = full_kelly/4.0
                        stake_amount = DEFAULT_BANKROLL * kelly_quarter
                        stake_label = f"¼ Kelly ({kelly_quarter:.2%})"

                        reason_txt = _rule_based_reason(label, p_ml, decimal_odds, movement_str, limit_now, trend)
                        predicted_value = (
                            parsed["home_team"] if label in ("H2H Home","Spread Home") else
                            parsed["away_team"] if label in ("H2H Away","Spread Away") else
                            ("Over" if label.endswith("Over") else "Under")
                        )


                        group_key = market_group_from_label(label)
                        
                        
                        
                        strong = (is_strong(game.get("sport_key",""), label)
                                  and (p_ml>=MIN_PROBA)
                                  and ((not ALLOWED_BINS) or (live_odds_bin(decimal_odds) in ALLOWED_BINS))
                                  and (MAX_PROBA is None or p_ml<MAX_PROBA))
                                                          
                        
                                                          
                        # ---- Collect candidate instead of sending now ----
                        candidates_for_game.append({
                            "game_id": game_id,
                            "group_key": group_key,
                            "market_key": key,           # H2H Home, Spread Away, etc.
                            "label": label,
                            "msg": None,                 # we'll render fresh when we send
                            "strong": strong,
                            "bets_row": None,            # we'll rebuild to ensure no stale values
                            "edge": edge,
                            "p_ml": p_ml,
                            "ml_direction": ml_direction, 
                            "kelly_quarter": kelly_quarter,
                            "stake_amount": stake_amount,
                            "stake_label": stake_label,
                            "predicted_value": predicted_value,
                            "movement_str": movement_str,
                            "decimal_odds": decimal_odds,
                            "p_implied": p_implied,
                            "limit_now": limit_now,
                            "trend": trend,
                            "game_time_fmt": game_time_fmt,
                            "bet_id": bet_id,
                            "parsed": parsed,
                            "sport_key": game.get("sport_key","unknown"),
                            "drop_cents": drop_cents,
                            "threshold_cents": threshold_cents,                         
                            
                        })



                    # update tracked snapshot
                    tracked_games[game_id] = parsed

                    # ---- BEST-OF-TWO (one per market group for THIS game) ----
                    # ---- BEST-OF-TWO (one per market group for THIS game) ----
                    if candidates_for_game:
                        by_group = {}
                        for c in candidates_for_game:
                            by_group.setdefault(c["group_key"], []).append(c)

                        for gkey, lst in by_group.items():
                            best = _choose_best(lst)
                            if not best:
                                continue

                            gid = best["game_id"]
                            ap_grouped = peak_prices[gid].setdefault("_alerted", {})
                            if ap_grouped.get(gkey) or seen_recently(gid, gkey):
                                continue

                            # rebuild message from 'best'
                            label           = best["label"]
                            parsed          = best["parsed"]
                            ml_direction    = best["ml_direction"]
                            limit_now       = best["limit_now"]
                            trend           = best["trend"]
                            edge            = best["edge"]
                            p_implied       = best["p_implied"]
                            decimal_odds    = best["decimal_odds"]
                            stake_amount    = best["stake_amount"]
                            stake_label     = best["stake_label"]
                            predicted_value = best["predicted_value"]
                            movement_str    = best["movement_str"]
                            game_time_fmt   = best["game_time_fmt"]
                            bet_id          = best["bet_id"]
                            strong          = best["strong"]

                            def esc(x):
                                return html.escape(str(x)) if x is not None else ""

                            line_info = ""
                            if label == "Spread Home":
                                line_info = f"Line: {esc(parsed.get('spread_line_home'))}"
                            elif label == "Spread Away":
                                line_info = f"Line: {esc(parsed.get('spread_line_away'))}"
                            elif label.startswith("Total"):
                                line_info = f"Total: {esc(parsed.get('total_line'))}"

                            limit_str = f"${int(limit_now):,}" if isinstance(limit_now, (int, float)) else "n/a"
                            strength_badge = "✅ STRONG" if strong else "ℹ️ Watchlist"
                            threshold_note = f"(min {MIN_PROBA:.0%})" if MIN_PROBA else ""

                        
                            # --- Tiering from tier_config.json ---
                            odds_bin = live_odds_bin(decimal_odds)
                            tier_code, tier_label = assign_tier(
                                proba=best["p_ml"],
                                sport=best["sport_key"],
                                market=best["label"],
                                odds_bin=odds_bin,
                            )


                            # Map tier to the existing badge for consistency
                            if tier_code == "A":
                                strength_badge = "✅ STRONG"
                            elif tier_code == "B":
                                strength_badge = "ℹ️ Watchlist"  # medium
                            elif tier_code == "C":
                                strength_badge = "ℹ️ Watchlist"  # low
                            else:
                                strength_badge = "ℹ️ Watchlist"


                            why_bits = [
                                ("Underdog price suggests value." if (label.startswith("H2H") and decimal_odds >= 2.0) else None),
                                "Reversal after a peak." if label.startswith("H2H") else
                                "Spread moved off the peak." if label.startswith("Spread") else
                                "Late reversal on the total." if label.startswith("Total") else None,
                                (f"Limits {trend} into kickoff." if isinstance(limit_now,(int,float)) else None),
                            ]
                            why_line = " ".join(b for b in why_bits if b)



                          
                            msg = (
                                f"📉 <b>Alert</b> ({esc(label)} — {esc(ml_direction)}) {strength_badge}\n"
                                f"{esc(parsed['away_team'])} vs {esc(parsed['home_team'])}\n"
                                f"🏷️ Tier: {tier_label}\n"                                   # <— NEW
                                f"{line_info}\n"
                                f"🏟️ Sport: {format_sport(best['sport_key'])}\n"
                                f"📅 Game Time: {esc(game_time_fmt)}\n"
                                f"💰 Movement: {esc(movement_str)}\n"
                                f"📊 Limit: {limit_str} (trend: {trend})\n"
                                f"📐 Edge: {edge:.2%} vs implied {p_implied:.1%}\n"
                                f"🏷️ Setup: Reversal\n"
                                f"💵 Stake: ${stake_amount:.2f} ({esc(stake_label)})\n"
                                f"🔮 Pick: <b>{esc(predicted_value)}</b>\n"
                                f"🧠 Why: Model {best['p_ml']:.1%} {threshold_note} vs implied {1/decimal_odds:.1%}. {why_line}\n"
                                f"🆔 bet_id: <code>{esc(bet_id)}</code>"
                            )
                            
                            
                            # --- Log to AllObservations when we actually send (RECORD_ON_SEND_ONLY=1) ---
                            if obs_sheet is not None and RECORD_ON_SEND_ONLY:
                                def am_or_blank(decval):
                                    a = decimal_to_american(decval)
                                    return "" if a is None else str(a)

                                gid         = best["game_id"]
                                key         = best["market_key"]              # e.g., "home_price", "totals_away" etc.
                                opening_val = peak_prices[gid].get(f"opening_{key}")
                                peak_val    = peak_prices[gid].get(key)
                                curr_dec    = best["decimal_odds"]
                                parsed      = best["parsed"]
                                lv          = parsed.get("extras",{}).get("lowvig",{})
                                bo          = parsed.get("extras",{}).get("betonlineag",{})

                                # Recompute tier so it’s logged exactly as messaged
                                odds_bin = live_odds_bin(curr_dec)
                                tier_code, tier_label = assign_tier(
                                    proba=best["p_ml"],
                                    sport=best["sport_key"],
                                    market=best["label"],
                                    odds_bin=odds_bin,
                                )


                                obs_map = {
                                    "Timestamp": datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                    "Sport": best["sport_key"],
                                    "Away": parsed["away_team"],
                                    "Home": parsed["home_team"],
                                    "Market": best["label"],
                                    "Direction": best["ml_direction"],
                                    "Movement": best["movement_str"],
                                    "Spread Line Home": parsed.get("spread_line_home") or "",
                                    "Spread Line Away": parsed.get("spread_line_away") or "",
                                    "Total Line": parsed.get("total_line") or "",
                                    "Odds (Am)": am_or_blank(curr_dec),
                                    "Reason Text": _rule_based_reason(best["label"], best["p_ml"], curr_dec, best["movement_str"], best["limit_now"], best["trend"]),
                                    "Opening Decimal": opening_val,
                                    "Peak Decimal": peak_val,
                                    "Current Decimal": float(curr_dec),
                                    "MinutesToStart": round((parse_commence_utc(parsed["commence_time"]) - datetime.now(timezone.utc)).total_seconds()/60.0, 1),
                                    "Actual Winner": "",
                                    "Predicted": best["predicted_value"],
                                    "Prediction Result": "",
                                    "Stake Amount": f"{best['stake_amount']:.2f}",
                                    "Game Time": best["game_time_fmt"],
                                    "American Odds": am_or_blank(curr_dec),
                                    "LowVig Home Odds (Am)": am_or_blank(lv.get("home")),
                                    "LowVig Away Odds (Am)": am_or_blank(lv.get("away")),
                                    "BetOnline Home Odds (Am)": am_or_blank(bo.get("home")),
                                    "BetOnline Away Odds (Am)": am_or_blank(bo.get("away")),
                                    "LowVig Home Decimal": lv.get("home") or "",
                                    "LowVig Away Decimal": lv.get("away") or "",
                                    "Decimal Odds (Current)": float(curr_dec),
                                    "ImpliedProb": f"{1/float(curr_dec):.6f}",
                                    "Edge": f"{best['edge']:.6f}",
                                    "Bet ID": best["bet_id"],
                                    # Make sure your sheet header has these two columns; if not, rename your old "Blank 3/Blank4"
                                    "Tier Code": tier_code,
                                    "Tier Label": tier_label,
                                    "Blank5": "", "Blank6": "",
                                    "ML Direction": best["ml_direction"],
                                    "Model Probability": f"{best['p_ml']:.4f}",
                                    "Bet Limit": (best["limit_now"] if isinstance(best["limit_now"], (int, float)) else ""),
                                    "Limit Trend": best["trend"],
                                    "Min Proba Threshold": f"{MIN_PROBA:.2f}",
                                    "Strong Flag (YES/NO)": "YES" if best["strong"] else "NO",
                                    "Blank7": "", "Blank8": "", "Bank9": "", "Blank10": "", "Blank11": "", "Blank12": "",
                                    "Blank13": "", "Blank14": "", "Blank15": "", "Blank16": "",
                                    "KalshiTicker": "", "KalshiSide": "", "KalshiPrice": "", "KalshiTIF": "",
                                    "KalshiQty": "", "SentToKalshi": "", "KalshiStatus": "", "KalshiOrderId": "",
                                    "KalshiTs": "", "KalshiError": "",
                                }
                                append_by_header(obs_sheet, obs_map)

                            
                            

                            # Send
                            send_pro(msg)
                            alerts_fired += 1
                            
                            
 

                            # mark dedupe by group only (prevents sibling side later)
                            ap_grouped[gkey] = True
                            mark_recent(gid, gkey)

                            # Strong → Enterprise + AllBets
                            if strong and sheet_bets is not None:

                                bets_row = [
                                    datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                    best["sport_key"],
                                    parsed["away_team"], parsed["home_team"],
                                    label, ml_direction, movement_str, predicted_value, game_time_fmt,
                                    parsed.get("spread_line_home") or "", parsed.get("spread_line_away") or "", parsed.get("total_line") or "",
                                    "Reversal", "", "", f"{stake_amount:.2f}", "¼ Kelly", "", "", "",
                                    decimal_odds,
                                    f"{edge:.6f}",  # Edge
                                    "", "", "", "", "", bet_id
                                ]
                                bets_row = _fit_to_headers(bets_row, sheet_bets)
                                try:
                                    send_enterprise(msg)
                                    sheet_bets.append_row(bets_row, value_input_option="USER_ENTERED")
                                except APIError as e:
                                    logging.warning("AllBets append failed: %s", e)



                # persist state
                save_state()
                logging.info("📈 [PIN] cycle completeness: events=%d h2h=%d spreads=%d totals=%d",
                             pin_with_any, pin_h2h, pin_spread, pin_total)
                logging.info("💾 Saved state: %d games tracked | peak_prices.json keys=%d | alerts this cycle=%d",
                             len(tracked_games), len(peak_prices), alerts_fired)
                logging.info("✅ Poll cycle done in %.1fs", time.time() - cycle_start)

            except Exception as e:
                logging.error("❌ Error during tracking loop: %s", e, exc_info=True)

            await asyncio.sleep(TRACK_BASE)

# ========= Entrypoint =========
if __name__ == "__main__":
    try:
        asyncio.run(track_games())
    except Exception as e:
        logging.error("❌ Unhandled exception: %s", e, exc_info=True)
        hb_ping("fail")
# Trigger ECR build

# Trigger ECR build again
