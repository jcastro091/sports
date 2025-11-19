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

# ===== ML hook =====
ENGINE_ROOT = r"C:\\Users\\Jcast\\Documents\\alpha_signal_engine"
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
from src.ml_predict import predict_win_prob  # returns probability 0..1


# ========= Argparse / Logging =========
parser = argparse.ArgumentParser()
parser.add_argument("--trace-pinnacle", action="store_true",
                    help="Dump Pinnacle market/outcome names when nulls occur")
parser.add_argument("--verbose", action="store_true",
                    help="Verbose logging")
args = parser.parse_args()

TRACE_PINNACLE = (os.getenv("TRACE_PINNACLE", "0").lower() in ("1","true")) or bool(args.trace_pinnacle)

# --- Logging to file + console ---
from logging.handlers import RotatingFileHandler

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "sports.log"

handlers = [
    logging.StreamHandler(sys.stdout),  # still see live output
    RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
]

logging.basicConfig(
    level=(logging.DEBUG if args.verbose else logging.INFO),
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=handlers
)

logging.info("✅ sports.py (peaks + Sheets + Telegram + ML + verbose) started")
logging.info("📂 Logging to %s", log_file)




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



# Use a small list first; add more when you confirm it’s running chatty
SPORTS_LIST: List[str] = [
    "baseball_mlb",
    "basketball_nba",
    "basketball_wnba",
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
ALERT_WINDOW_MIN = int(os.getenv("ALERT_WINDOW_MIN", "60"))
SEEN_RECENT_SEC  = int(os.getenv("SEEN_RECENT_SEC", "3600"))
MIN_MINUTES_SINCE_PEAK = int(os.getenv("MIN_MINUTES_SINCE_PEAK", "5"))  

TRACK_BASE = int(os.getenv("TRACK_BASE", "300"))
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
        if decimal_odds >= 2.0: bits.append("Underdog price suggests value.")
        bits.append("Reversal after a peak.")
    elif label.startswith("Total"):
        bits.append("Total showed a late reversal.")
    elif label.startswith("Spread"):
        bits.append("Spread moved off the peak.")
    if isinstance(limit_now,(int,float)): bits.append(f"Limits {trend} into kickoff.")
    try:
        implied = 1/float(decimal_odds); bits.append(f"Model {p_ml:.1%} vs implied {implied:.1%}.")
    except: bits.append(f"Model {p_ml:.1%}.")
    bits.append(f"Recent movement: {movement_str}")
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
                        # after: parsed = parse_odds(game); if not parsed: continue
                        # limit_now = parsed.get("bet_limit")
                        # if isinstance(limit_now, (int, float)):
                            # prev = peak_prices.setdefault(game_id, {}).get("_limit_prev")
                            # EPS = float(os.getenv("LIMIT_TREND_EPS", "1.01"))
                            # trend = "flat"
                            # if isinstance(prev, (int, float)):
                                # if limit_now >= prev * EPS:
                                    # trend = "up"
                                # elif limit_now <= prev / EPS:
                                    # trend = "down"
                            # peak_prices[game_id]["_limit_prev"] = limit_now
                            # peak_prices[game_id]["_limit_prev_ts"] = time.time()
                            # peak_prices[game_id]["_limit_trend"] = trend



                        trend = peak_prices[game_id].get("_limit_trend", "flat")



                        # ML with guard
                        p_ml = 0.5
                        try:
                            p = predict_win_prob({
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
                            })
                            p_ml = float(p)
                            if p_ml > 1: p_ml /= 100.0
                            p_ml = max(0.001, min(p_ml, 0.999))
                        except Exception as e:
                            logging.error("ML predict error: %s", e)

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
                                                          
                        
                        

                        # ===== Sheets: AllObservations =====
                        if obs_sheet is not None:
                            def am_or_blank(decval):
                                a = decimal_to_american(decval)
                                return "" if a is None else str(a)
                            lv = parsed.get("extras",{}).get("lowvig",{})
                            bo = parsed.get("extras",{}).get("betonlineag",{})
                            lowvig_home_am = am_or_blank(lv.get("home")); lowvig_away_am = am_or_blank(lv.get("away"))
                            bol_home_am    = am_or_blank(bo.get("home")); bol_away_am    = am_or_blank(bo.get("away"))
                            obs_row = [
                                datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                game.get("sport_key","unknown"),
                                parsed["away_team"],
                                parsed["home_team"],
                                label,
                                ml_direction,
                                movement_str,
                                parsed.get("spread_line_home") or "",
                                parsed.get("spread_line_away") or "",
                                parsed.get("total_line") or "",
                                decimal_to_american(curr) or "",
                                reason_txt,
                                peak_prices[game_id].get(f"opening_{key}"),
                                peak_prices[game_id].get(key),
                                curr,
                                round(time_until, 1),
                                "",
                                predicted_value,
                                "",
                                f"{stake_amount:.2f}",
                                game_time_fmt,
                                decimal_to_american(curr) or "",
                                lowvig_home_am, lowvig_away_am, bol_home_am, bol_away_am,
                                lv.get("home") or "", lv.get("away") or "",
                                decimal_odds,
                                "", "",
                                bet_id,
                                "", "", "", "",
                                ml_direction,
                                f"{p_ml:.4f}",
                                limit_now if limit_now is not None else "",
                                trend,
                            ]
                            obs_row += [f"{MIN_PROBA:.2f}", "YES" if strong else "NO"]
                            
                            if obs_sheet is not None and not RECORD_ON_SEND_ONLY:
                                try:
                                    obs_sheet.append_row(obs_row, value_input_option="USER_ENTERED")
                                except APIError as e:
                                    logging.warning("AllObservations append failed: %s", e)


                        # ===== Telegram =====
                        # def esc(x): return html.escape(str(x)) if x is not None else ""
                        # line_info = ""
                        # if label == "Spread Home": line_info = f"Line: {esc(parsed.get('spread_line_home'))}"
                        # elif label == "Spread Away": line_info = f"Line: {esc(parsed.get('spread_line_away'))}"
                        # elif label.startswith("Total"): line_info = f"Total: {esc(parsed.get('total_line'))}"
                        # limit_str = f"${int(limit_now):,}" if isinstance(limit_now,(int,float)) else "n/a"


                        # strength_badge = "✅ STRONG" if strong else "ℹ️ Watchlist"
                        # threshold_note = f"(min {MIN_PROBA:.0%})" if MIN_PROBA else ""
                        
                        
                        # msg = (
                            # f"📉 <b>Alert</b> ({esc(label)} — {esc(ml_direction)}) {strength_badge}\n"
                            # f"{esc(parsed['away_team'])} vs {esc(parsed['home_team'])}\n"
                            # f"{line_info}\n"
                            # f"📅 Game Time: {esc(game_time_fmt)}\n"
                            # f"💰 Movement: {esc(movement_str)}\n"
                            # f"📊 Limit: {limit_str} (trend: {trend})\n"
                            # f"📐 Edge: {edge:.2%} vs implied {p_implied:.1%}\n"
                            # f"🏷️ Setup: Reversal\n"
                            # f"💵 Stake: ${stake_amount:.2f} ({esc(stake_label)})\n"
                            # f"🔮 Pick: <b>{esc(predicted_value)}</b>\n"
                            # f"🧠 Why: Model {p_ml:.1%} {threshold_note} vs implied {1/decimal_odds:.1%}. {esc(reason_txt)}\n"
                            # f"🆔 bet_id: <code>{esc(bet_id)}</code>"
                        # )
                        
                        
                        
                        # send_pro(msg)
                        # alerts_fired += 1
                        # ap[key] = True
                        # peak_prices[game_id].setdefault("_alerted", {})[group_key] = True
                        # mark_recent(game_id, group_key)

                        # # If strong, also AllBets + Enterprise
                        # if strong:
                            # send_enterprise(msg)
                            # if sheet_bets is not None:
                                # bets_row = [
                                    # datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                    # game.get("sport_key","unknown"),
                                    # parsed["away_team"], parsed["home_team"],
                                    # label, ml_direction, movement_str, predicted_value, game_time_fmt,
                                    # parsed.get("spread_line_home") or "", parsed.get("spread_line_away") or "", parsed.get("total_line") or "",
                                    # "Reversal", "", "", f"{stake_amount:.2f}", "¼ Kelly", "", "", "", decimal_odds, "", "", "", "", "", bet_id
                                # ]
                                # try:
                                    # sheet_bets.append_row(bets_row, value_input_option="USER_ENTERED")
                                # except APIError as e:
                                    # logging.warning("AllBets append failed: %s", e)
                                    
                                    
                                    
                                    
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

                            msg = (
                                f"📉 <b>Alert</b> ({esc(label)} — {esc(ml_direction)}) {strength_badge}\n"
                                f"{esc(parsed['away_team'])} vs {esc(parsed['home_team'])}\n"
                                f"{line_info}\n"
                                f"📅 Game Time: {esc(game_time_fmt)}\n"
                                f"💰 Movement: {esc(movement_str)}\n"
                                f"📊 Limit: {limit_str} (trend: {trend})\n"
                                f"📐 Edge: {edge:.2%} vs implied {p_implied:.1%}\n"
                                f"🏷️ Setup: Reversal\n"
                                f"💵 Stake: ${stake_amount:.2f} ({esc(stake_label)})\n"
                                f"🔮 Pick: <b>{esc(predicted_value)}</b>\n"
                                f"🧠 Why: Model {best['p_ml']:.1%} {threshold_note} vs implied {1/decimal_odds:.1%}. "
                                f"{esc(_rule_based_reason(label, best['p_ml'], decimal_odds, movement_str, limit_now, trend))}\n"
                                f"🆔 bet_id: <code>{esc(bet_id)}</code>"
                            )

                            # Send
                            send_pro(msg)
                            alerts_fired += 1
                            
                            
                            
                            # If we only want to record what was actually sent, write one row now.
                            if obs_sheet is not None and RECORD_ON_SEND_ONLY:
                                def am_or_blank(decval):
                                    a = decimal_to_american(decval)
                                    return "" if a is None else str(a)

                                gid = best["game_id"]
                                mk  = best["market_key"]         # e.g., "Spread Home"
                                opening_val = peak_prices[gid].get(f"opening_{mk}")
                                peak_val    = peak_prices[gid].get(mk)
                                curr_dec    = best["decimal_odds"]
                                lv = best["parsed"].get("extras",{}).get("lowvig",{})
                                bo = best["parsed"].get("extras",{}).get("betonlineag",{})

                                obs_row = [
                                    datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                    best["sport_key"],
                                    best["parsed"]["away_team"],
                                    best["parsed"]["home_team"],
                                    label,
                                    ml_direction,
                                    movement_str,
                                    best["parsed"].get("spread_line_home") or "",
                                    best["parsed"].get("spread_line_away") or "",
                                    best["parsed"].get("total_line") or "",
                                    decimal_to_american(curr_dec) or "",
                                    _rule_based_reason(label, best["p_ml"], curr_dec, movement_str, limit_now, trend),
                                    opening_val,
                                    peak_val,
                                    curr_dec,
                                    round((parse_commence_utc(tracked_games[gid]["commence_time"]) - datetime.now(timezone.utc)).total_seconds()/60.0, 1),
                                    "",                               # blank column preserved
                                    predicted_value,
                                    "",                               # blank column preserved
                                    f"{stake_amount:.2f}",
                                    game_time_fmt,
                                    decimal_to_american(curr_dec) or "",
                                    am_or_blank(lv.get("home")), am_or_blank(lv.get("away")),
                                    am_or_blank(bo.get("home")), am_or_blank(bo.get("away")),
                                    lv.get("home") or "", lv.get("away") or "",
                                    curr_dec,
                                    "", "",                           # preserve two blanks
                                    bet_id,
                                    "", "", "", "",                   # preserve four blanks
                                    ml_direction,
                                    f"{best['p_ml']:.4f}",
                                    limit_now if isinstance(limit_now,(int,float)) else "",
                                    trend,
                                ]
                                # Add strong-config audit columns to match original schema
                                obs_row += [f"{MIN_PROBA:.2f}", "YES" if strong else "NO"]

                                try:
                                    obs_sheet.append_row(obs_row, value_input_option="USER_ENTERED")
                                except APIError as e:
                                    logging.warning("AllObservations append (on-send) failed: %s", e)


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
                                    decimal_odds, "", "", "", "", "", bet_id
                                ]
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