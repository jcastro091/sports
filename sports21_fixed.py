#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sports.py — adaptive polling + quota alerts + new football keys + bet limits
"""

# ---------- Stdlib ----------
import os, sys, re, json, math, time, html, asyncio, logging, secrets, csv
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

# ---------- Third-party ----------
import aiohttp
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Bot
from dateutil import parser
from gspread.exceptions import APIError

# ---------- ML import (package-safe) ----------
ENGINE_ROOT = r"C:\\Users\\Jcast\\Documents\\alpha_signal_engine"  # parent of 'src'
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
from src.ml_predict import predict_win_prob  # returns probability 0..1

# ---------- CONFIG ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4")

# Pro = watchlist of ALL observations; Enterprise = only STRONG bets
PRO_CHAT_ID = int(os.getenv("PRO_CHAT_ID", "-1002756836961"))
ENTERPRISE_CHAT_ID = int(os.getenv("ENTERPRISE_CHAT_ID", "-1002635869925"))

# Odds API
API_KEY = os.getenv("ODDS_API_KEY", "7e79a3f0859a50153761270b1d0e867f")
SPORTS_LIST: List[str] = [
    "baseball_mlb",
    "soccer_usa_mls",
    "basketball_wnba",
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "boxing_boxing",
    "mma_mixed_martial_arts",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "icehockey_nhl",
    "basketball_nba",
    "soccer_france_ligue_one",
    "soccer_epl",
]
PRIMARY_BOOKMAKER = "pinnacle"
BOOKMAKERS = ["pinnacle", "lowvig", "betonlineag"]
REGION = "us,eu"
MARKETS = ["h2h", "spreads", "totals"]
INCLUDE_BET_LIMITS = True

TIMEZONE = pytz.timezone("US/Eastern")
MAX_HOURS_AHEAD = int(os.getenv("MAX_HOURS_AHEAD", 12))
ALERT_WINDOW_MIN = int(os.getenv("ALERT_WINDOW_MIN", 90))
TRACK_BASE = int(os.getenv("TRACK_BASE", 180))  # 3 min

# (hours_to, poll_seconds) — sorted ascending in code for correct matching
POLL_PLAN = [(12,10800),(6,3600),(2,900),(1,300),(0,120)]

QUOTA_ALERT_THRESHOLD = int(os.getenv("QUOTA_ALERT_THRESHOLD", 1500))
QUOTA_HARD_STOP = int(os.getenv("QUOTA_HARD_STOP", 200))

CSV_LOG_FILE = "prediction_alert_log.csv"
ODDS_CACHE_FILE = "odds_cache.json"
PEAK_PRICES_FILE = "peak_prices.json"
TRACKED_GAMES_FILE = "tracked_games.json"
BET_LIMITS_CSV = "bet_limits_log.csv"

DEFAULT_BANKROLL = float(os.getenv("BANKROLL", 10000))

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logging.info("✅ sports.py (adaptive + limits) started")

# ---------- Telegram ----------
bot = Bot(token=TELEGRAM_BOT_TOKEN)

def send_pro(text: str):
    try:
        bot.send_message(chat_id=PRO_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logging.error("Telegram (Pro) send failed: %s", e)

def send_enterprise(text: str):
    try:
        bot.send_message(chat_id=ENTERPRISE_CHAT_ID, text=text, parse_mode="HTML")
    except Exception as e:
        logging.error("Telegram (Enterprise) send failed: %s", e)

# system/status pings go to Enterprise by default (e.g., low credits)
def ping(text: str):
    send_enterprise(text)

# ---------- Google Sheets ----------
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "ConfirmedBets")
TAB_BETS = os.getenv("TAB_BETS", "AllBets")
TAB_OBS  = os.getenv("TAB_OBS", "AllObservations")
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE", os.path.join(os.path.dirname(__file__), "telegrambetlogger-35856685bc29.json"))
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
gc = gspread.authorize(creds)
sheet_bets = gc.open(SPREADSHEET_NAME).worksheet(TAB_BETS)
obs_sheet = gc.open(SPREADSHEET_NAME).worksheet(TAB_OBS)

# ---- GSheet throttle + retry ----
SHEETS_SEM = asyncio.Semaphore(2)
GS_MAX_RETRIES = 5

async def gs_call(coro_factory, *, purpose=""):
    delay = 1.0
    for attempt in range(1, GS_MAX_RETRIES + 1):
        async with SHEETS_SEM:
            try:
                return await asyncio.get_event_loop().run_in_executor(None, coro_factory)
            except APIError as e:  # type: ignore
                msg = str(e)
                if "429" in msg:
                    logging.warning("Sheets 429 on %s (attempt %d) -> backoff %.1fs", purpose, attempt, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise

# ---------- State ----------
tracked_games: Dict[str, Any] = {}
peak_prices: Dict[str, Any] = {}
last_limit_for_game: Dict[str, float] = {}

sport_next_poll: Dict[str, datetime] = {s: datetime.min.replace(tzinfo=timezone.utc) for s in SPORTS_LIST}
sport_backoff_idle: Dict[str, int] = {s: 0 for s in SPORTS_LIST}
_last_quota_warn_ts: Optional[float] = None

# --- Strong filters ---
ENABLE_STRONG_FILTERS = os.getenv("ENABLE_STRONG_FILTERS", "true").lower() == "true"
STRONG_MARKETS = os.getenv("STRONG_MARKETS", "H2H Home,H2H Away,Total Over").split(",")
STRONG_SPORTS  = os.getenv("STRONG_SPORTS",
    "soccer_usa_mls,soccer_italy_serie_a,soccer_germany_bundesliga,americanfootball_ncaaf,basketball_wnba,boxing_boxing,mma_mixed_martial_arts"
).split(",")
STRONG_CONFIG_JSON = os.getenv("STRONG_CONFIG_JSON", "strong_config.json")

# ================================
# Helpers & utilities
# ================================

# --- JSON state I/O ---
def _json_default(o):
    if isinstance(o, datetime):
        return o.astimezone(timezone.utc).isoformat()
    return str(o)

def _read_json_file(path: str, default):
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _write_json_file(path: str, obj):
    try:
        Path(path).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8"
        )
    except Exception as e:
        logging.warning("Couldn't write %s: %s", path, e)

def load_tracked_games():
    return _read_json_file(TRACKED_GAMES_FILE, {})

def save_tracked_games(obj):
    _write_json_file(TRACKED_GAMES_FILE, obj)

def load_peak_prices():
    return _read_json_file(PEAK_PRICES_FILE, {})

def save_peak_prices(obj):
    _write_json_file(PEAK_PRICES_FILE, obj)

# --- CSV appender for bet limits ---
def _ensure_csv_has_header(path: Path, header: List[str]):
    exists = path.exists()
    if not exists:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

def log_limit_csv(ts_local: str, sport_key: str, game_id: str, market: str, limit_val):
    try:
        p = Path(BET_LIMITS_CSV)
        header = ["ts_local","sport","game_id","market","bet_limit"]
        _ensure_csv_has_header(p, header)
        with p.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts_local, sport_key, game_id, market, "" if limit_val is None else limit_val])
    except Exception as e:
        logging.warning("Couldn't write bet limit csv: %s", e)

# --- Odds conversions & bins ---
def decimal_to_american(dec) -> Optional[int]:
    try:
        d = float(dec)
    except Exception:
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    # favorite
    return int(round(-100 / (d - 1.0)))

def am_from_decimal(dec) -> Optional[int]:
    return decimal_to_american(dec)

def live_odds_bin(dec) -> str:
    try:
        d = float(dec)
        imp = 1.0 / d
    except Exception:
        return "unknown"
    if imp >= 0.70:  return "HeavyFav"
    if imp >= 0.55:  return "Fav"
    if imp >= 0.40:  return "Balanced"
    if imp >= 0.20:  return "Dog"
    return "Longshot"

def cents_threshold_for_decimal(curr_dec: float) -> float:
    """Reverse-movement band in *decimal* odds (≈ 10–20¢ American)."""
    try:
        d = float(curr_dec)
    except Exception:
        return 0.05
    if d < 1.7:  return 0.03   # heavy fav
    if d < 2.1:  return 0.05   # near even
    if d < 3.0:  return 0.07
    return 0.10

def next_poll_delay_sec(hours_to: float) -> int:
    plan = sorted(POLL_PLAN, key=lambda x: x[0])  # ascending by hours threshold
    for h, sec in plan:
        if hours_to <= h:
            return int(sec)
    return int(plan[-1][1])

# --- Event filtering / parsing from Odds API ---
def _get_book(data: dict, name: str):
    books = data.get("bookmakers") or []
    for b in books:
        if (b.get("key") or "").lower() == name.lower():
            return b
    return None

def _extract_price_from_market(markets: list, market_key: str, side: str) -> Optional[float]:
    # market_key: h2h | spreads | totals
    # side: 'home'/'away' for h2h, 'home'/'away' for spreads, 'over'/'under' for totals
    for m in markets:
        if (m.get("key") or "").lower() != market_key:
            continue
        outcomes = m.get("outcomes") or []
        for o in outcomes:
            nm = (o.get("name") or "").lower()
            if market_key == "totals" and side in ("over","under"):
                if nm == side:
                    try:
                        return float(o.get("price"))
                    except Exception:
                        return None
            elif side in ("home","away"):
                if nm == side:
                    try:
                        return float(o.get("price"))
                    except Exception:
                        return None
    return None

def _extract_line_from_market(markets: list, market_key: str, which: str) -> Optional[float]:
    # market_key: spreads|totals, which: 'home'/'away' for spreads, 'total' for totals
    for m in markets:
        if (m.get("key") or "").lower() != market_key:
            continue
        outcomes = m.get("outcomes") or []
        for o in outcomes:
            nm = (o.get("name") or "").lower()
            try:
                if market_key == "spreads" and which in ("home","away"):
                    if nm == which:
                        return float(o.get("point"))
                if market_key == "totals" and which == "total":
                    return float(o.get("point"))
            except Exception:
                continue
    return None

def filter_upcoming_events(data: list) -> list:
    out = []
    now = datetime.now(timezone.utc)
    for g in (data or []):
        ct = g.get("commence_time")
        try:
            when = parser.isoparse(ct).astimezone(timezone.utc)
        except Exception:
            continue
        if when > now:
            out.append(g)
    return out

def parse_odds(game: dict) -> Optional[dict]:
    """Return a normalized dict with teams, time, key prices & lines. Includes LowVig/BetOnline extras if present."""
    try:
        commence = parser.isoparse(game["commence_time"]).astimezone(timezone.utc)
        home = game["home_team"]; away = game["away_team"]
    except Exception:
        return None

    # primary book markets
    primary = _get_book(game, PRIMARY_BOOKMAKER)
    markets = primary.get("markets") if primary else []

    home_price  = _extract_price_from_market(markets, "h2h", "home")
    away_price  = _extract_price_from_market(markets, "h2h", "away")
    spread_home = _extract_price_from_market(markets, "spreads", "home")
    spread_away = _extract_price_from_market(markets, "spreads", "away")
    totals_over = _extract_price_from_market(markets, "totals", "over")
    totals_under= _extract_price_from_market(markets, "totals", "under")

    spread_line_home = _extract_line_from_market(markets, "spreads", "home")
    spread_line_away = _extract_line_from_market(markets, "spreads", "away")
    total_line       = _extract_line_from_market(markets, "totals",  "total")

    # bet limit (if provided by Odds API)
    bet_limit = None
    try:
        bet_limit = primary.get("markets", [{}])[0].get("limit")
    except Exception:
        pass

    # extras: LowVig & BetOnline head-to-head decimals
    extras = {}
    for book in ("lowvig", "betonlineag"):
        b = _get_book(game, book)
        if not b:
            continue
        mks = b.get("markets") or []
        extras[book] = {
            "home": _extract_price_from_market(mks, "h2h", "home"),
            "away": _extract_price_from_market(mks, "h2h", "away"),
        }

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "home_price": home_price,
        "away_price": away_price,
        "spread_home": spread_home,
        "spread_away": spread_away,
        "totals_home": totals_over,
        "totals_away": totals_under,
        "spread_line_home": spread_line_home,
        "spread_line_away": spread_line_away,
        "total_line": total_line,
        "extras": extras,
        "bet_limit": bet_limit,
    }

# --- Strong config loaders ---
def load_strong_overrides():
    try:
        if os.path.exists(STRONG_CONFIG_JSON):
            with open(STRONG_CONFIG_JSON, "r") as f:
                cfg = json.load(f)
                sports  = cfg.get("sports", STRONG_SPORTS)
                markets = cfg.get("markets", STRONG_MARKETS)
                return set(map(str.strip, sports)), set(map(str.strip, markets))
    except Exception as e:
        logging.warning("Couldn't load strong_config.json: %s", e)
    return set(map(str.strip, STRONG_SPORTS)), set(map(str.strip, STRONG_MARKETS))

def _load_json_field(path: str, key: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f)
                return cfg.get(key, default)
    except Exception:
        pass
    return default

def load_proba_thresholds():
    min_p = float(_load_json_field(STRONG_CONFIG_JSON, "min_proba", 0.0))
    max_p = _load_json_field(STRONG_CONFIG_JSON, "max_proba", None)
    try:
        max_p = None if max_p is None else float(max_p)
    except Exception:
        max_p = None
    return min_p, max_p

def load_allowed_bins():
    bins = _load_json_field(STRONG_CONFIG_JSON, "odds_bin", [])
    return set(map(str, bins))

MIN_PROBA, MAX_PROBA = load_proba_thresholds()
ALLOWED_BINS = load_allowed_bins()
STRONG_SPORTS_SET, STRONG_MARKETS_SET = load_strong_overrides()

logging.info(
    "📊 Strong config loaded | min_proba=%.2f%s | bins=%s | sports=%s | markets=%s",
    MIN_PROBA,
    f" <{MAX_PROBA:.2f}" if MAX_PROBA is not None else "",
    sorted(ALLOWED_BINS) if ALLOWED_BINS else "ALL",
    sorted(STRONG_SPORTS_SET),
    sorted(STRONG_MARKETS_SET),
)

def is_strong(sport_key: str, market_label: str) -> bool:
    if not ENABLE_STRONG_FILTERS:
        return True
    return (sport_key in STRONG_SPORTS_SET) and (market_label in STRONG_MARKETS_SET)

# ----- misc helpers -----
def parse_latest_american_odds(s: str):
    if not s: return None
    last = str(s).split("→")[-1]
    m = re.search(r"[-+]?\d+", last)
    return int(m.group()) if m else None

def parse_move_delta(s: str):
    if not s: return None
    parts = [p.strip() for p in str(s).split("→")]
    nums = [re.search(r"[-+]?\d+", p) for p in parts]
    nums = [int(m.group()) for m in nums if m]
    if len(nums) >= 2: return nums[-1] - nums[0]
    return nums[-1] if nums else None

def gen_trade_id(away_team: str, home_team: str, label: str) -> str:
    ts_hex = format(int(time.time() * 1000), "x").upper()
    lbl = "".join(ch for ch in label if ch.isalnum())[:3].upper() or "MKT"
    rand = secrets.token_hex(3).upper()
    return f"T{ts_hex}-{lbl}-{rand}"

# --- LLM-backed explanation (optional) ---
LLM_REASON_ENDPOINT = os.getenv("LLM_REASON_ENDPOINT", "").strip()
def _rule_based_reason(*, label, predicted_value, p_ml, movement_str, limit_now, trend, decimal_odds):
    bits = []
    if label.startswith("H2H"):
        if float(decimal_odds) >= 2.0:
            bits.append("Underdog price suggests value.")
        bits.append("Reversal after a peak in price.")
    elif label.startswith("Total"):
        bits.append("Total showed a late reversal.")
    elif label.startswith("Spread"):
        bits.append("Spread moved off the peak and pulled back.")
    if isinstance(limit_now, (int, float)):
        bits.append(f"Limits {('rising' if trend=='rising' else 'steady')} into kickoff.")
    try:
        implied = 1/float(decimal_odds)
        bits.append(f"Model {p_ml:.1%} vs implied {implied:.1%}.")
    except Exception:
        bits.append(f"Model {p_ml:.1%}.")
    bits.append(f"Recent movement: {movement_str}")
    return " ".join(bits)

def generate_pick_reason(payload: dict, meta: dict) -> str:
    if LLM_REASON_ENDPOINT:
        try:
            import urllib.request, json as _json
            body = _json.dumps({"payload": payload, "meta": meta}).encode("utf-8")
            req = urllib.request.Request(LLM_REASON_ENDPOINT, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=6) as r:
                obj = _json.loads(r.read().decode("utf-8"))
                txt = (obj.get("reason") or obj.get("text") or "").strip()
                if txt:
                    return txt
        except Exception:
            pass
    return _rule_based_reason(
        label = meta.get("label",""),
        predicted_value = meta.get("predicted_value",""),
        p_ml = meta.get("p_ml", 0.5),
        movement_str = meta.get("movement_str",""),
        limit_now = meta.get("limit_now"),
        trend = meta.get("trend","flat"),
        decimal_odds = meta.get("decimal_odds", 2.0),
    )

# ---------- Sheets helpers ----------
async def ensure_obs_columns() -> List[str]:
    """Returns the AllObservations header row; if absent, returns a safe default header."""
    try:
        header = await gs_call(lambda: obs_sheet.row_values(1), purpose="read header")
        if header:
            return header
    except Exception:
        pass
    # Fallback – covers columns we append in obs_bet_row
    return [
        "ts_local","sport","away","home","label","Direction","Movement",
        "spread_home","spread_away","total_line","odds_curr_am","reason",
        "opening_price","peak_price","curr_price","minutes_to_start",
        "notes","predicted_value","stake_pct","stake_amt","game_time",
        "final_am","lowvig_home_am","lowvig_away_am","bol_home_am","bol_away_am",
        "lowvig_home_dec","lowvig_away_dec","curr_dec","col30","col31",
        "bet_id","col33","col34","col35","col36",
        "Direction_dup","p_ml","bet_limit","limit_trend",
        "lowvig_home_dec_dup","lowvig_away_dec_dup","lowvig_home_am_dup","lowvig_away_am_dup",
        "bol_home_dec","bol_away_dec","bol_home_am_dup","bol_away_am_dup"
    ]

async def write_reason_cell(bet_id: str, reason_txt: str):
    """Find row by bet_id in AllObservations and write reason text into 'reason' column."""
    try:
        header = await ensure_obs_columns()
        try:
            all_rows = await gs_call(lambda: obs_sheet.get_all_values(), purpose="scan for bet_id")
        except Exception:
            return
        if not all_rows:
            return
        hdr = [h.strip().lower() for h in all_rows[0]]
        try:
            bet_idx = hdr.index("bet_id")
        except ValueError:
            bet_idx = len(hdr) - 1
        try:
            reason_idx = hdr.index("reason")
        except ValueError:
            reason_idx = None

        target_row = None
        for r_i, row in enumerate(all_rows[1:], start=2):
            if bet_idx < len(row) and row[bet_idx] == bet_id:
                target_row = r_i
                break
        if not target_row:
            return

        if reason_idx is None:
            await gs_call(lambda: obs_sheet.update_cell(1, len(hdr) + 1, "reason"), purpose="add reason header")
            reason_idx = len(hdr)

        await gs_call(lambda: obs_sheet.update_cell(target_row, reason_idx + 1, reason_txt), purpose="write reason")
    except Exception as e:
        logging.warning("write_reason_cell failed: %s", e)

# ---------- Setup & scheduling ----------
def determine_setup(prediction_entity: str, movement_str: str, game_time_dt: datetime, now_dt: datetime) -> str:
    """Minimal tagging; expand as needed."""
    try:
        mv = parse_move_delta(movement_str) or 0
    except Exception:
        mv = 0
    mins_to = max((game_time_dt - now_dt).total_seconds() / 60.0, 0.0)
    if mins_to <= 60 and mv < 0:
        return "Late Reversal"
    if mv < 0:
        return "Reversal"
    if mv > 0:
        return "Climb"
    return "Flat"

async def schedule_closing_odds_capture(bet_id: str, commence_time: datetime):
    """Stub – capture closing prices after start time (optional)."""
    try:
        now = datetime.now(timezone.utc)
        delay = max((commence_time - now).total_seconds() + 120, 0)  # 2m after start
        await asyncio.sleep(delay)
        # TODO: implement capture if you want to record close
    except Exception:
        pass

# ---------- Alert message builder (includes REASON) ----------
def _build_alert_msg(*, label, ml_direction, parsed, game_time_fmt, movement_str, limit_now, trend, setup_tag, stake_amount, stake_label, predicted_value, bet_id, strong: bool, reason_txt: str):
    def esc(x): return html.escape(str(x)) if x is not None else ""
    line_info = ""
    if label == "Spread Home": line_info = f"Line: {esc(parsed.get('spread_line_home'))}"
    elif label == "Spread Away": line_info = f"Line: {esc(parsed.get('spread_line_away'))}"
    elif label in ("Total Over", "Total Under"): line_info = f"Total: {esc(parsed.get('total_line'))}"
    limit_str = f"${int(limit_now):,}" if isinstance(limit_now,(int,float)) else "n/a"
    strength_badge = "✅ STRONG" if strong else "ℹ️ Watchlist"

    return (
        f"📉 <b>Alert</b> ({esc(label)} — {esc(ml_direction)}) {strength_badge}\n"
        f"{esc(parsed['away_team'])} vs {esc(parsed['home_team'])}\n"
        f"{line_info}\n"
        f"📅 Game Time: {esc(game_time_fmt)}\n"
        f"💰 Movement: {esc(movement_str)}\n"
        f"📊 Limit: {limit_str} ({trend})\n"
        f"🏷️ Setup: {esc(setup_tag)}\n"
        f"💵 Stake: ${stake_amount:.2f} ({esc(stake_label)})\n"
        f"🔮 Pick: <b>{esc(predicted_value)}</b>\n"
        f"🧠 Why: {esc(reason_txt)}\n"
        f"🆔 bet_id: <code>{esc(bet_id)}</code>"
    )

# ---------------- NEW: ML failure guard ----------------
PREDICTIONS_ENABLED = True

def _predict_with_guard(payload: dict) -> Optional[float]:
    """Call predict_win_prob once; on failure, disable predictions for the rest of the run."""
    global PREDICTIONS_ENABLED
    if not PREDICTIONS_ENABLED:
        return None
    try:
        p = predict_win_prob(payload)
        dbg = {k: payload.get(k) for k in ("sport","market","Direction","american_odds","decimal_odds","hours_to_start")}
        logging.info("🔎 ML payload %s -> raw_prob=%.4f", dbg, p)
        return float(p)
    except Exception as e:
        logging.error("ML not ready: %s. Disabling predictions until restart.", e, exc_info=True)
        PREDICTIONS_ENABLED = False
        return None

# ---------------- Main tracker ----------------
async def track_games():
    global tracked_games, peak_prices, _last_quota_warn_ts
    tracked_games = load_tracked_games()
    peak_prices = load_peak_prices()
    logging.info("✅ Loaded tracked state from disk.")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                all_events = []

                # ---- fetch loop ----
                for sport in SPORTS_LIST:
                    if now_utc < sport_next_poll.get(sport, datetime.min.replace(tzinfo=timezone.utc)):
                        continue
                    url = (
                        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
                        f"?regions={REGION}&markets={'%2C'.join(MARKETS)}&bookmakers={','.join(BOOKMAKERS)}"
                        f"&apiKey={API_KEY}"
                        f"&includeBetLimits={'true' if INCLUDE_BET_LIMITS else 'false'}"
                    )
                    async with session.get(url) as resp:
                        rem = resp.headers.get("x-requests-remaining")
                        used = resp.headers.get("x-requests-used")
                        try:
                            rem_i = int(rem) if rem is not None else None
                        except:
                            rem_i = None

                        if rem_i is not None and rem_i <= QUOTA_HARD_STOP:
                            if not _last_quota_warn_ts or time.time() - _last_quota_warn_ts > 3600:
                                ping(f"⚠️ <b>Odds API nearly exhausted</b> — remaining: {rem_i}. Switching to slow mode.")
                                _last_quota_warn_ts = time.time()
                            for s in SPORTS_LIST:
                                sport_next_poll[s] = now_utc + timedelta(hours=6)
                        elif rem_i is not None and rem_i <= QUOTA_ALERT_THRESHOLD:
                            if not _last_quota_warn_ts or time.time() - _last_quota_warn_ts > 7200:
                                ping(f"🔔 <b>Heads up</b>: Odds API credits low. Remaining: {rem_i} (used {used}).")
                                _last_quota_warn_ts = time.time()

                        if resp.status != 200:
                            txt = await resp.text()
                            logging.error("❌ Fetch odds failed %s: %s", sport, txt)
                            sport_backoff_idle[sport] = min(sport_backoff_idle[sport] + 1, 6)
                            sport_next_poll[sport] = now_utc + timedelta(minutes=5 * sport_backoff_idle[sport])
                            continue

                        data = await resp.json()
                        filtered = filter_upcoming_events(data)
                        for g in filtered:
                            g["sport_key"] = sport
                        all_events.extend(filtered)

                        if filtered:
                            soonest = min(parser.isoparse(g["commence_time"]).astimezone(timezone.utc) for g in filtered)
                            hours_to = max((soonest - now_utc).total_seconds() / 3600.0, 0.0)
                            delay = next_poll_delay_sec(hours_to)
                            sport_next_poll[sport] = now_utc + timedelta(seconds=delay)
                            sport_backoff_idle[sport] = 0
                            logging.info("⏱️ %s next poll in %dm (%.1fh to next game)", sport, delay//60, hours_to)
                        else:
                            sport_backoff_idle[sport] = min(sport_backoff_idle[sport] + 1, 6)
                            backoff = 30 * sport_backoff_idle[sport]
                            sport_next_poll[sport] = now_utc + timedelta(minutes=backoff)
                            logging.info("😴 %s idle — backoff to %dm", sport, backoff)

                # ---- process events ----
                for game in all_events:
                    parsed = parse_odds(game)
                    if not parsed:
                        continue

                    game_id = f"{parsed['away_team']}@{parsed['home_team']}"
                    game_time = parsed["commence_time"]
                    if game_time.astimezone(timezone.utc) < datetime.now(timezone.utc):
                        continue

                    limit_now = parsed.get("bet_limit")
                    ts_est = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
                    log_limit_csv(ts_est, game.get("sport_key",""), game_id, "all", limit_now)

                    prev_lim = last_limit_for_game.get(game_id)
                    trend = "rising" if (isinstance(limit_now, (int,float)) and isinstance(prev_lim,(int,float)) and limit_now > prev_lim) else "flat"
                    if prev_lim is None and isinstance(limit_now,(int,float)):
                        trend = "rising"
                    if isinstance(limit_now,(int,float)):
                        last_limit_for_game[game_id] = float(limit_now)

                    if game_id not in tracked_games:
                        tracked_games[game_id] = parsed
                        peak_prices[game_id] = parsed.copy()
                        for k in ["home_price","away_price","spread_home","spread_away","totals_home","totals_away"]:
                            peak_prices[game_id][f"opening_{k}"] = parsed.get(k)
                        continue

                    price_keys = {
                        "home_price":  "H2H Home",
                        "away_price":  "H2H Away",
                        "spread_home": "Spread Home",
                        "spread_away": "Spread Away",
                        "totals_home": "Total Over",
                        "totals_away": "Total Under",
                    }

                    for key, label in price_keys.items():
                        if game_id not in peak_prices:
                            peak_prices[game_id] = parsed.copy()
                        prev_peak = peak_prices[game_id].get(key)
                        opening_price = peak_prices[game_id].get(f"opening_{key}")
                        curr_price = parsed.get(key)
                        if prev_peak is None or curr_price is None or opening_price is None:
                            continue

                        time_until = (game_time - datetime.now(timezone.utc)).total_seconds() / 60.0
                        if time_until > ALERT_WINDOW_MIN:
                            continue

                        if curr_price > prev_peak:
                            peak_prices[game_id][key] = curr_price
                            continue

                        move_dir = "Down" if curr_price < prev_peak else "Up"
                        movement_str = f"{decimal_to_american(peak_prices[game_id].get(f'opening_{key}'))} → {decimal_to_american(prev_peak)} → {decimal_to_american(curr_price)} ({move_dir})"
                        bet_time = datetime.now(TIMEZONE)

                        if key == "home_price":
                            prediction_entity = parsed["home_team"]
                        elif key == "away_price":
                            prediction_entity = parsed["away_team"]
                        elif key == "spread_home":
                            prediction_entity = f"{parsed['home_team']} (Spread)"
                        elif key == "spread_away":
                            prediction_entity = f"{parsed['away_team']} (Spread)"
                        elif key == "totals_home":
                            prediction_entity = "Over"
                        elif key == "totals_away":
                            prediction_entity = "Under"
                        else:
                            prediction_entity = "N/A"

                        if label == "H2H Home":
                            predicted_value = parsed["home_team"]
                        elif label == "H2H Away":
                            predicted_value = parsed["away_team"]
                        elif label == "Spread Home":
                            predicted_value = parsed["home_team"]
                        elif label == "Spread Away":
                            predicted_value = parsed["away_team"]
                        elif label == "Total Over":
                            predicted_value = "Over"
                        elif label == "Total Under":
                            predicted_value = "Under"
                        else:
                            predicted_value = "N/A"

                        setup_tag = determine_setup(prediction_entity, movement_str, parsed["commence_time"], bet_time)
                        band = cents_threshold_for_decimal(curr_price)

                        reversal_trigger = (
                            (move_dir == "Down") and
                            (curr_price <= prev_peak - band) and
                            ((curr_price < opening_price) or (time_until <= 60))
                        )
                        if not reversal_trigger:
                            continue

                        try:
                            decimal_odds = float(curr_price)
                        except:
                            continue
                        if label in ("H2H Home", "H2H Away") and decimal_odds < 2.0:
                            continue

                        if label in ("H2H Home", "H2H Away"):
                            ml_direction = "Underdog" if decimal_odds >= 2.0 else "Favorite"
                        elif label in ("Total Over", "Total Under"):
                            ml_direction = "Over" if label.endswith("Over") else "Under"
                        elif label in ("Spread Home", "Spread Away"):
                            ml_direction = "Spread Underdog" if decimal_odds >= 2.0 else "Spread Favorite"
                        else:
                            ml_direction = "Favorite"

                        game_time_fmt = parsed["commence_time"].strftime("%b %d, %I:%M %p %Z").replace(" 0", " ")
                        odds_curr_am = decimal_to_american(curr_price)
                        bet_id = gen_trade_id(parsed["away_team"], parsed["home_team"], label)

                        market_name = (
                            "H2H" if label in ("H2H Home","H2H Away")
                            else "Spread" if label in ("Spread Home","Spread Away")
                            else "Totals" if label in ("Total Over","Total Under")
                            else label
                        )

                        try:
                            am_latest = parse_latest_american_odds(movement_str)
                        except Exception:
                            am_latest = None

                        hours_to_start = max((parsed["commence_time"] - datetime.now(TIMEZONE)).total_seconds() / 3600.0, 0.0)

                        payload = {
                            "sport": game.get("sport_key",""),
                            "market": market_name,
                            "Direction": (
                                "Underdog" if label in ("H2H Home","H2H Away") and float(decimal_odds) >= 2.0 else
                                "Favorite" if label in ("H2H Home","H2H Away") else
                                ("Over" if label.endswith("Over") else "Under") if label.startswith("Total") else
                                ("Spread Underdog" if float(decimal_odds) >= 2.0 else "Spread Favorite") if label.startswith("Spread") else
                                "Other"
                            ),
                            "Movement": parse_move_delta(movement_str),
                            "american_odds": am_latest,
                            "decimal_odds": float(decimal_odds),
                            "hours_to_start": hours_to_start,
                            "bet_limit": limit_now if isinstance(limit_now,(int,float)) else None,
                            "limit_trend": trend,
                            "spread_line_home": parsed.get("spread_line_home"),
                            "spread_line_away": parsed.get("spread_line_away"),
                            "total_line": parsed.get("total_line"),
                            "Confidence": "High",
                            "Kelly": 0.0,
                            "Tags": setup_tag or "Other",
                        }

                        # ---------------- ML (guarded) ----------------
                        raw_prob = _predict_with_guard(payload)
                        if raw_prob is None:
                            p_ml = 0.5
                        else:
                            try:
                                p_ml = float(raw_prob)
                                if p_ml > 1.0:
                                    p_ml /= 100.0
                                p_ml = max(0.001, min(p_ml, 0.999))
                            except Exception:
                                p_ml = 0.5

                        # --- Kelly ---
                        b = decimal_odds - 1.0
                        p_implied = 1.0 / decimal_odds
                        q = 1.0 - p_ml
                        full_kelly = (b * p_ml - q) / b if b > 0 else 0.0
                        if p_ml <= p_implied or full_kelly <= 0:
                            full_kelly = 0.0
                        kelly_quarter = full_kelly / 4.0
                        stake_amount = DEFAULT_BANKROLL * kelly_quarter
                        stake_label = f"¼ Kelly ({kelly_quarter:.2%})"

                        header = await ensure_obs_columns()

                        # --- LowVig & BetOnline snapshot ---
                        lv = parsed.get("extras", {}).get("lowvig", {})
                        bo = parsed.get("extras", {}).get("betonlineag", {})
                        def am_or_blank(dec):
                            a = am_from_decimal(dec if dec is None else str(dec))
                            return "" if a is None else str(a)
                        lowvig_home_dec = lv.get("home"); lowvig_away_dec = lv.get("away")
                        bol_home_dec = bo.get("home");   bol_away_dec = bo.get("away")
                        lowvig_home_am = am_or_blank(lowvig_home_dec)
                        lowvig_away_am = am_or_blank(lowvig_away_dec)
                        bol_home_am    = am_or_blank(bol_home_dec)
                        bol_away_am    = am_or_blank(bol_away_dec)

                        obs_bet_row = [
                            datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                            game.get("sport_key", "unknown"),
                            parsed["away_team"],
                            parsed["home_team"],
                            label,
                            ml_direction,
                            movement_str,
                            parsed.get("spread_line_home") or "",
                            parsed.get("spread_line_away") or "",
                            parsed.get("total_line") or "",
                            odds_curr_am or "",
                            "",
                            peak_prices[game_id].get(f"opening_{key}"),
                            prev_peak,
                            curr_price,
                            round(time_until, 1),
                            "",
                            predicted_value,
                            "",
                            f"{stake_amount:.2f}",
                            game_time_fmt,
                            odds_curr_am or "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            decimal_odds,
                            "", "",
                            bet_id,
                            "", "", "", "",
                            ml_direction,
                            f"{p_ml:.4f}",
                            limit_now if limit_now is not None else "",
                            trend,
                            lowvig_home_dec or "",
                            lowvig_away_dec or "",
                            lowvig_home_am,
                            lowvig_away_am,
                            bol_home_dec or "",
                            bol_away_dec or "",
                            bol_home_am,
                            bol_away_am,
                        ]

                        reason_meta = {
                            "label": label,
                            "predicted_value": predicted_value,
                            "p_ml": p_ml,
                            "movement_str": movement_str,
                            "limit_now": limit_now,
                            "trend": trend,
                            "decimal_odds": float(decimal_odds),
                        }
                        reason_txt = generate_pick_reason(payload, reason_meta)

                        row_len = len(header)
                        await gs_call(lambda: obs_sheet.append_row(obs_bet_row[:row_len]), purpose="append AllObs")
                        asyncio.create_task(write_reason_cell(bet_id, reason_txt))

                        sport_market_ok = is_strong(game.get("sport_key", "unknown"), label)
                        curr_bin = live_odds_bin(decimal_odds)
                        bin_ok = (not ALLOWED_BINS) or (curr_bin in ALLOWED_BINS)
                        proba_ok = (p_ml >= MIN_PROBA) and (MAX_PROBA is None or p_ml < MAX_PROBA)
                        strong = sport_market_ok and bin_ok and proba_ok

                        # -------- Telegram routing (includes reason) --------
                        msg = _build_alert_msg(
                            label=label,
                            ml_direction=ml_direction,
                            parsed=parsed,
                            game_time_fmt=game_time_fmt,
                            movement_str=movement_str,
                            limit_now=limit_now,
                            trend=trend,
                            setup_tag=setup_tag,
                            stake_amount=stake_amount,
                            stake_label=stake_label,
                            predicted_value=predicted_value,
                            bet_id=bet_id,
                            strong=strong,
                            reason_txt=reason_txt,
                        )

                        # 1) ALWAYS alert Pro with AllObservations (watchlist) message
                        send_pro(msg)

                        # 2) If STRONG, also alert Enterprise and write AllBets
                        if strong:
                            send_enterprise(msg)

                            log_data = [
                                datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                game.get("sport_key", "unknown"),
                                parsed["away_team"],
                                parsed["home_team"],
                                label,
                                ml_direction,
                                movement_str,
                                predicted_value,
                                game_time_fmt,
                                parsed.get("spread_line_home") or "",
                                parsed.get("spread_line_away") or "",
                                parsed.get("total_line") or "",
                                determine_setup(prediction_entity, movement_str, parsed["commence_time"], datetime.now(TIMEZONE)),
                                "", "", f"{stake_amount:.2f}", "¼ Kelly", "", "", "", decimal_odds, "", "", "", "", "", bet_id,
                            ]
                            await gs_call(lambda: sheet_bets.append_row(log_data), purpose="append AllBets")
                        else:
                            logging.info("🧪 Logged only (no strong bet): %s | %s", game.get("sport_key", ""), label)

                        # keep peak updated and schedule closing capture
                        peak_prices[game_id][key] = curr_price
                        asyncio.create_task(schedule_closing_odds_capture(bet_id, parsed["commence_time"]))

                    tracked_games[game_id] = parsed

                save_peak_prices(peak_prices)
                save_tracked_games(tracked_games)

            except Exception as e:
                logging.error("❌ Error during tracking loop: %s", e, exc_info=True)

            await asyncio.sleep(TRACK_BASE)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    try:
        asyncio.run(track_games())
    except Exception as e:
        logging.error("❌ Unhandled exception in main: %s", e, exc_info=True)
