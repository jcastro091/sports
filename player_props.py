#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, time, html, asyncio, logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import html

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

import pytz
import aiohttp
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Bot
from gspread.exceptions import APIError, WorksheetNotFound

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)-7s %(message)s")
logging.info("✅ player_props.py starting… (LOG_LEVEL=%s)", LOG_LEVEL)

# ---------- Reuse your engine model ----------
ENGINE_ROOT = r"C:\\Users\\Jcast\\Documents\\alpha_signal_engine"
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)
try:
    from src.ml_predict import predict_win_prob as predict_prop_prob
except Exception:
    logging.warning("⚠️ Could not import predict_prop_prob; using fallback 0.55")
    def predict_prop_prob(_row: Dict[str, Any]) -> float:
        return 0.55

# ---------- Config / Envs ----------
TIMEZONE = pytz.timezone("US/Eastern")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ENTERPRISE_CHAT_ID = int(os.getenv("ENTERPRISE_CHAT_ID", "-1002794017379"))
API_KEY = os.getenv("ODDS_API_KEY")
PRIMARY_BOOKMAKER = os.getenv("PRIMARY_BOOKMAKER", "pinnacle")
BOOKMAKERS = os.getenv("BOOKMAKERS", "pinnacle,lowvig,betonlineag").split(",")
REGION = os.getenv("REGION", "us,eu")
NBA = "basketball_nba"
NHL = "icehockey_nhl"
SPORTS = [NBA, NHL]

NBA_MARKETS = [
    "player_points","player_rebounds","player_assists",
    "player_threes","player_points_rebounds_assists"
]
NHL_MARKETS = [
    "player_shots_on_goal","player_points","player_goals","player_assists"
]
PROP_MARKETS_BY_SPORT = {NBA: NBA_MARKETS, NHL: NHL_MARKETS}

BALANCE_SIDES = os.getenv("BALANCE_SIDES", "false").lower() == "true"
ALT_PICK_STATE: Dict[str, str] = {}  # remembers last side picked per prop key for alternation


PROPS_STRONG_CONFIG_JSON = os.getenv("PROPS_STRONG_CONFIG_JSON", "props_strong_config.json")
ENABLE_STRONG_FILTERS = os.getenv("ENABLE_STRONG_FILTERS", "true").lower() == "true"
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "10000"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
TRACK_BASE = int(os.getenv("PROP_TRACK_BASE", "180"))
MAX_HOURS_AHEAD = int(os.getenv("MAX_HOURS_AHEAD", "12"))
STEAM_WINDOW_MIN = int(os.getenv("STEAM_WINDOW_MIN", "15"))

SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "ConfirmedBets")
TAB_OBS = os.getenv("TAB_PROPS_OBS", "PlayerPropsObservations")
TAB_BETS = os.getenv("TAB_PROPS_BETS", "PlayerProps")
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE", os.path.join(os.path.dirname(__file__), "telegrambetlogger-35856685bc29.json"))

OBS_BUFFER: List[list] = []
BETS_BUFFER: List[list] = []
LAST_FLUSH = 0.0
FLUSH_EVERY_SEC = 5   # tune: batch every 5s or when size is big
FLUSH_MIN_ROWS = 50   # tune: flush early if many rows queued

SEND_QUEUE: List[str] = []
LAST_SEND_TS = 0.0
TG_MIN_INTERVAL = 1.2    # seconds between messages
EDGE_MIN = float(os.getenv("EDGE_MIN", "0.005"))  # 0.5% min edge


PEAK_PROPS_FILE = "peak_props_props.json"
RECENT_PROPS_FILE = "recent_props.json"
SEEN_RECENT_SEC = int(os.getenv("SEEN_RECENT_SEC", "3600"))  # 1h dedupe across cycles
PINNACLE_ONLY_PEAKS = True  # only Pinnacle updates openings/peaks
ALERT_WINDOW_MIN = int(os.getenv("ALERT_WINDOW_MIN", "60"))  # last-hour alerts only

PEAK_PROPS_FILE = os.path.join(os.path.dirname(__file__), "peak_props_props.json")
RECENT_PROPS_FILE = os.path.join(os.path.dirname(__file__), "recent_props.json")

# ---- Local safe predictor that honors the model's feature list ----
import pickle
import pandas as pd
from joblib import load as joblib_load

MODEL_PATH = os.path.join(ENGINE_ROOT, "data", "results", "models", "model.joblib")
FEATURES_PATH = os.path.join(ENGINE_ROOT, "data", "results", "models", "model_features.pkl")

_MODEL = None
_FEATURES = None

def _load_model_and_features():
    global _MODEL, _FEATURES
    if _MODEL is None:
        logging.info("🔮 Model loaded via local fallback: %s", MODEL_PATH)
        _MODEL = joblib_load(MODEL_PATH)
    if _FEATURES is None:
        with open(FEATURES_PATH, "rb") as f:
            _FEATURES = pickle.load(f)
        logging.info("📐 Feature names loaded (local): %s", FEATURES_PATH)

_NUMERIC_DEFAULTS = {
    "open_imp": 0.0,
    "curr_imp": 0.0,
    "delta_imp_close_curr": 0.0,
    "delta_imp_close_open": 0.0,
    "delta_am_close_curr": 0.0,
    "delta_am_close_open": 0.0,
    "bet_limit_raw": 0.0,
    "bet_limit_log": 0.0,
    "bet_limit_z": 0.0,
}

_REQUIRED_NUMERIC_COLS = set(_NUMERIC_DEFAULTS.keys())


def predict_prop_prob_local(row: Dict[str, Any]) -> float:
    _load_model_and_features()

    # Union of model features and required numeric cols the pipeline expects
    all_cols = list(dict.fromkeys(list(_FEATURES) + list(_REQUIRED_NUMERIC_COLS)))

    feat_row = {}
    for col in all_cols:
        if col in row:
            feat_row[col] = row[col]
        elif col in _NUMERIC_DEFAULTS:
            feat_row[col] = _NUMERIC_DEFAULTS[col]
        else:
            feat_row[col] = np.nan  # safe default for categorical/text

    # sanitize Nones/NaNs on numerics we know
    for col in _NUMERIC_DEFAULTS:
        v = feat_row.get(col, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            feat_row[col] = _NUMERIC_DEFAULTS[col]

    X = pd.DataFrame([feat_row], columns=all_cols)
    logging.info("🧪 X.columns has %d cols; missing? %s",
                 len(X.columns),
                 set(['open_imp','curr_imp','delta_imp_close_curr','delta_imp_close_open',
                     'delta_am_close_curr','delta_am_close_open','bet_limit_raw',
                     'bet_limit_log','bet_limit_z']) - set(X.columns))

    proba = float(_MODEL.predict_proba(X)[0][1])
    return proba


def predict_prop_prob_unified(row: Dict[str, Any]) -> float:
    """
    Try engine's predict_win_prob; if it rejects due to missing columns,
    fall back to the local aligned predictor.
    """
    try:
        p = predict_prop_prob(row)
        return float(p)
    except Exception as e:
        if "columns are missing" in str(e):
            missing = str(e).split("columns are missing:")[-1].strip()
            logging.warning("⚠️ Engine predictor missing columns%s → using local fallback.", missing)
            return predict_prop_prob_local(row)
        # unknown error → also try local
        logging.warning("⚠️ Engine predictor error %s → using local fallback.", e)
        return predict_prop_prob_local(row)


# Startup env sanity
if not API_KEY or len(API_KEY.strip()) < 20:
    raise RuntimeError("ODDS_API_KEY not set or looks wrong. Put it in your .env or export it in the shell.")
logging.info("🔑 Using Odds API key prefix: %s*****", API_KEY[:6])
if not TELEGRAM_BOT_TOKEN:
    logging.warning("⚠️ TELEGRAM_BOT_TOKEN not set — alerts will be skipped.")

# ---------- Telegram ----------
bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
def ping(msg: str):
    if not bot: 
        logging.debug("Telegram disabled; message would be: %s", msg)
        return
    try:
        bot.send_message(chat_id=ENTERPRISE_CHAT_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        logging.error("Telegram error: %s", e)

async def drain_telegram_queue():
    global LAST_SEND_TS
    if not bot or not SEND_QUEUE:
        return
    while SEND_QUEUE:
        msg = SEND_QUEUE.pop(0)
        gap = TG_MIN_INTERVAL - (time.time() - LAST_SEND_TS)
        if gap > 0:
            await asyncio.sleep(gap)
        try:
            bot.send_message(chat_id=ENTERPRISE_CHAT_ID, text=msg, parse_mode="HTML")
            LAST_SEND_TS = time.time()
        except Exception as e:
            # Handle Flood control message gracefully by pausing
            txt = str(e)
            if "Flood control exceeded. Retry in" in txt:
                try:
                    # parse an integer like "... Retry in 36.0 seconds"
                    secs = int(float(txt.split("Retry in")[1].split("seconds")[0].strip()))
                except Exception:
                    secs = 30
                logging.warning("Telegram flood wait: sleeping %ss", secs)
                await asyncio.sleep(secs)
            else:
                logging.error("Telegram send failed: %s", e)


# ---------- GSheets ----------
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
gc = gspread.authorize(creds)
sh = gc.open(SPREADSHEET_NAME)

def open_or_create_worksheet(sheet, title: str, rows: int = 5000, cols: int = 60):
    try:
        return sheet.worksheet(title)
    except WorksheetNotFound:
        logging.info("📄 Creating worksheet: %s", title)
        return sheet.add_worksheet(title=title, rows=str(rows), cols=str(cols))

ws_obs  = open_or_create_worksheet(sh, TAB_OBS)
ws_bets = open_or_create_worksheet(sh, TAB_BETS)

SHEETS_SEM = asyncio.Semaphore(2)
async def gs_call(coro_factory, purpose=""):
    delay = 1.0
    for attempt in range(1, 6):
        async with SHEETS_SEM:
            try:
                return await asyncio.get_event_loop().run_in_executor(None, coro_factory)
            except APIError as e:
                if "429" in str(e):
                    logging.warning("Sheets 429 on %s (attempt %d) → backoff %.1fs", purpose, attempt, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 30.0); continue
                raise
                
                
async def flush_buffers(force: bool=False):
    global LAST_FLUSH
    now = time.time()
    if not force and (now - LAST_FLUSH < FLUSH_EVERY_SEC) and len(OBS_BUFFER) < FLUSH_MIN_ROWS and len(BETS_BUFFER) < FLUSH_MIN_ROWS:
        return
    try:
        if OBS_BUFFER:
            rows = OBS_BUFFER[:]
            OBS_BUFFER.clear()
            await gs_call(lambda: ws_obs.spreadsheet.values_append(
                TAB_OBS,
                params={"valueInputOption":"USER_ENTERED"},
                body={"values": rows}
            ), "append props obs batch")

        if BETS_BUFFER:
            rows = BETS_BUFFER[:]
            BETS_BUFFER.clear()
            await gs_call(lambda: ws_bets.spreadsheet.values_append(
                TAB_BETS,
                params={"valueInputOption":"USER_ENTERED"},
                body={"values": rows}
            ), "append props bet batch")

    except APIError as e:
        logging.warning("Batch append failed: %s", e)
    finally:
        LAST_FLUSH = now


# ensure headers
async def ensure_headers():
    header = await gs_call(lambda: ws_obs.row_values(1), "props read headers")
    needed = [
        "Timestamp (ET)","Sport","Game Id","Game (Away @ Home)","Event Start (ET)",
        "Market","Player","Side","Line","Price (Dec) @ Primary","Book @ Primary",
        "Price (Dec) @ Rec1","Book @ Rec1","Price (Dec) @ Rec2","Book @ Rec2",
        "Sharp Mid Prob","EV vs Rec1","EV vs Rec2","Steam Δ¢(N min)","Reversal?",
        "Bet Limit","Limit Trend","Mins to Start","AI Pred","AI Prob","bet_id"
    ]
    missing = [c for c in needed if c not in header]
    if missing:
        logging.info("🔧 Extending %s header with: %s", TAB_OBS, missing)
        start = len(header)+1
        a1 = f"{TAB_OBS}!{gspread.utils.rowcol_to_a1(1,start)}"
        await gs_call(lambda: ws_obs.spreadsheet.values_update(
            a1, params={"valueInputOption": "RAW"}, body={"values":[missing]}
        ), "props add headers")

    bheader = await gs_call(lambda: ws_bets.row_values(1), "props read bet headers")
    bneeded = [
        "Timestamp (ET)","Sport","Game (Away @ Home)","Event Start (ET)",
        "Market","Player","Side","Line","Odds Taken (Dec)","Odds Taken (Am)",
        "Book","AI Prob","Stake $","Kelly Frac","Bet Limit","bet_id"
    ]
    bmissing = [c for c in bneeded if c not in bheader]
    if bmissing:
        logging.info("🔧 Extending %s header with: %s", TAB_BETS, bmissing)
        start = len(bheader)+1
        a1 = f"{TAB_BETS}!{gspread.utils.rowcol_to_a1(1,start)}"
        await gs_call(lambda: ws_bets.spreadsheet.values_update(
            a1, params={"valueInputOption": "RAW"}, body={"values":[bmissing]}
        ), "props add bet headers")

# ---------- Helpers ----------


def _read_json(path, default):
    try:
        p = Path(path)
        return json.loads(p.read_text("utf-8")) if p.exists() else default
    except Exception:
        return default

def _write_json(path, obj):
    try:
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logging.warning("Couldn't write %s: %s", path, e)

peak_props = _read_json(PEAK_PROPS_FILE, {})     # key -> dict(opening, peak, last_peak_ts, alerted)
recent_props = _read_json(RECENT_PROPS_FILE, {}) # key|side -> ts

def save_state():
    _write_json(PEAK_PROPS_FILE, peak_props)
    _write_json(RECENT_PROPS_FILE, recent_props)

def seen_recently(key: str) -> bool:
    ts = recent_props.get(key)
    return bool(ts and (time.time() - ts) < SEEN_RECENT_SEC)

def mark_recent(key: str):
    recent_props[key] = time.time()


def gen_bet_id(sport: str, mk: str, player: str, line, side: str) -> str:
    key = f"{sport}|{mk}|{player}|{line}|{side}"
    return f"PP-{int(time.time())}-{abs(hash(key)) % 100000}"


def esc(value) -> str:
    if value is None:
        return ""
    # Always stringify before escaping to avoid float/None errors
    return html.escape(str(value))


def american_from_decimal(d: float | None) -> int | None:
    if d is None or d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2.0 else int(round(-100 / (d - 1)))

def _dynamic_cents_threshold(a_peak: int) -> int:
    if a_peak <= -300: return 50
    if a_peak <= -250: return 40
    if a_peak <= -200: return 30
    if a_peak <= -150: return 20
    if a_peak <= -105: return 10
    if a_peak >= 300:  return 40
    if a_peak >= 250:  return 30
    if a_peak >= 200:  return 25
    if a_peak >= 150:  return 20
    return 15  # +100..+149

def drop_from_peak_ok(peak_dec: float, curr_dec: float) -> tuple[bool, int, int]:
    a_peak = american_from_decimal(peak_dec)
    a_curr = american_from_decimal(curr_dec)
    if a_peak is None or a_curr is None:
        return False, 0, 10**9
    dropped = (a_curr < a_peak)     # current odds worse than peak for us
    drop_cents = (a_peak - a_curr) if dropped else 0
    return dropped, drop_cents, _dynamic_cents_threshold(a_peak)


def odds_bin(dec: float) -> str:
    try: d=float(dec)
    except: return "unknown"
    if d >= 3.0: return "Longshot"
    if d >= 2.3: return "Dog"
    if d >= 1.9: return "Balanced"
    return "Fav"

def remove_vig_prob(price_over: float, price_under: float) -> Optional[float]:
    try:
        p_over = 1.0/price_over
        p_under= 1.0/price_under
        z = p_over + p_under
        if z <= 0: return None
        return p_over / z
    except Exception:
        return None

def kelly_fraction(p_win: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    q = 1.0 - p_win
    if b <= 0: return 0.0
    f = (b*p_win - q)/b
    return max(0.0, f)
    
    
def choose_best_side(player: str, mk: str, line: float, candidates: list[dict], *, sport: Optional[str]=None) -> Optional[dict]:
    """
    Pick a side among candidates based on edge/kelly, with an optional alternation
    when both sides qualify (for testing/balance).
    """
    # Only consider those that clear edge + strong filters
    good = [c for c in candidates if (c["p_ml"] - c["p_implied"]) > EDGE_MIN and c["passed"]]
    if not good:
        return None

    # Sort by stake then model probability (original behavior)
    good.sort(key=lambda c: (c["stake"], c["p_ml"]), reverse=True)

    # If not balancing, keep original behavior
    if not BALANCE_SIDES:
        return good[0]

    # Balancing path only triggers when BOTH Over and Under qualify
    sides_present = {c["side"] for c in good}
    if {"Over", "Under"}.issubset(sides_present):
        key = f"{sport or ''}|{mk}|{player}|{line}"
        last = ALT_PICK_STATE.get(key)
        # flip-flop: if last was Over → choose Under; else choose Over
        want = "Under" if last == "Over" else "Over"
        alt = next((c for c in good if c["side"] == want), None)
        if alt is not None:
            ALT_PICK_STATE[key] = alt["side"]
            return alt
        # fallback to best if somehow missing
        ALT_PICK_STATE[key] = good[0]["side"]
        return good[0]

    # If only one side qualified, pick it (and remember)
    key = f"{sport or ''}|{mk}|{player}|{line}"
    ALT_PICK_STATE[key] = good[0]["side"]
    return good[0]




def load_props_config() -> Tuple[set,set,float,Optional[float],set]:
    sports = set(); markets=set(); bins=set(); min_p=0.0; max_p=None
    try:
        if os.path.exists(PROPS_STRONG_CONFIG_JSON):
            with open(PROPS_STRONG_CONFIG_JSON, "r") as f:
                cfg = json.load(f)
                sports  = set(map(str, cfg.get("sports", [])))
                markets = set(map(str, cfg.get("markets", [])))
                bins    = set(map(str, cfg.get("odds_bin", [])))
                min_p   = float(cfg.get("min_proba", 0.0))
                max_p   = cfg.get("max_proba", None)
                max_p   = None if max_p is None else float(max_p)
    except Exception as e:
        logging.warning("props_strong_config.json load failed: %s", e)
    return sports, markets, min_p, max_p, bins

STRONG_SPORTS, STRONG_MARKETS, MIN_PROBA, MAX_PROBA, ALLOWED_BINS = load_props_config()

def strong_gate(sport: str, market: str, p_ml: float, dec_odds: float) -> bool:
    if not ENABLE_STRONG_FILTERS: return True
    if STRONG_SPORTS and sport not in STRONG_SPORTS: return False
    if STRONG_MARKETS and market not in STRONG_MARKETS: return False
    if ALLOWED_BINS and odds_bin(dec_odds) not in ALLOWED_BINS: return False
    if p_ml < MIN_PROBA: return False
    if MAX_PROBA is not None and p_ml >= MAX_PROBA: return False
    return True

# steam store: (key) -> list[(ts, price_dec)]
STEAM: Dict[str, List[Tuple[float, float]]] = {}
def steam_delta_cents(key: str, now_ts: float, window_min: int) -> float:
    hist = STEAM.get(key, [])
    cutoff = now_ts - window_min*60
    hist = [(t,p) for (t,p) in hist if t >= cutoff]
    STEAM[key] = hist
    if not hist: return 0.0
    return (hist[-1][1] - hist[0][1]) * 100.0
def update_steam(key: str, price_dec: float):
    STEAM.setdefault(key, []).append((time.time(), price_dec))

# ---------- Odds API ----------
BASE = "https://api.the-odds-api.com/v4"

async def fetch_json(session, url: str) -> Any:
    async with session.get(url) as resp:
        if resp.status != 200:
            txt = await resp.text()
            logging.error("HTTP %s: %s", resp.status, txt)
            return None
        # log quota headers if present
        rem = resp.headers.get("x-requests-remaining")
        used= resp.headers.get("x-requests-used")
        if rem and used:
            logging.debug("OddsAPI credits → remaining=%s used=%s", rem, used)
        return await resp.json()

async def list_events(session, sport: str) -> List[Dict[str,Any]]:
    url = f"{BASE}/sports/{sport}/events?apiKey={API_KEY}"
    logging.info("📡 Listing events: %s", sport)
    data = await fetch_json(session, url)
    if not isinstance(data, list):
        logging.info("No events list for %s", sport)
        return []
    out=[]
    now_utc = datetime.now(timezone.utc)
    for e in data:
        ct = datetime.fromisoformat(e["commence_time"].replace("Z","+00:00"))
        if now_utc < ct <= now_utc + timedelta(hours=MAX_HOURS_AHEAD):
            out.append(e)
    logging.info("🗂️ %s upcoming events for %s (≤ %dh)", len(out), sport, MAX_HOURS_AHEAD)
    return out

async def event_props(session, sport: str, event_id: str, markets: List[str]) -> Dict[str,Any]:
    m = "%2C".join(markets)
    b = ",".join(BOOKMAKERS)
    url = f"{BASE}/sports/{sport}/events/{event_id}/odds?regions={REGION}&markets={m}&bookmakers={b}&apiKey={API_KEY}"
    logging.debug("→ Props fetch %s %s (markets=%d)", sport, event_id, len(markets))
    return await fetch_json(session, url)

# ---------- Main loop ----------
async def run_props():
    await ensure_headers()
    timeout = aiohttp.ClientTimeout(total=30, connect=8, sock_read=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            loop_start = time.time()
            try:
                total_events = 0
                total_props  = 0
                for sport in SPORTS:
                    events = await list_events(session, sport)
                    total_events += len(events)
                    wanted = PROP_MARKETS_BY_SPORT.get(sport, [])
                    for ev in events:
                        ev_id = ev["id"]
                        game = f"{ev['away_team']} @ {ev['home_team']}"
                        start_dt = datetime.fromisoformat(ev["commence_time"].replace("Z","+00:00")).astimezone(TIMEZONE)
                        start_lbl = start_dt.strftime("%Y-%m-%d %I:%M %p %Z").replace(" 0"," ")
                        mins_to = int((start_dt - datetime.now(TIMEZONE)).total_seconds()/60)

                        data = await event_props(session, sport, ev_id, wanted)
                        if not data or "bookmakers" not in data:
                            logging.debug("No props data for %s | %s", sport, game)
                            continue

                        primary = next((b for b in data["bookmakers"] if b.get("key")==PRIMARY_BOOKMAKER), None)
                        if not primary:
                            logging.debug("Primary %s not present for %s", PRIMARY_BOOKMAKER, game)
                            continue

                        # Build rec lookup
                        rec_prices: Dict[Tuple[str,str,str,float], List[Tuple[str,float]]] = {}
                        for bk in data["bookmakers"]:
                            for mkt in bk.get("markets", []):
                                mk = mkt.get("key")
                                for o in mkt.get("outcomes", []):
                                    side = o.get("name")  # "Over"/"Under"
                                    line = o.get("point")
                                    player = o.get("description") or o.get("player_name") or ""
                                    price = o.get("price")
                                    if not all([mk, side, line, player, price]): 
                                        continue
                                    rec_prices.setdefault((mk,player,side,float(line)), []).append((bk.get("key",""), float(price)))
                        per_event_count = 0
                        for mkt in primary.get("markets", []):
                            mk = mkt.get("key")
                            if mk not in wanted: 
                                continue
                            # group by (player,line)
                            by_pl: Dict[Tuple[str,float], Dict[str,float]] = {}
                            for o in mkt.get("outcomes", []):
                                side = o.get("name")
                                line = o.get("point")
                                player = o.get("description") or o.get("player_name") or ""
                                price = o.get("price")
                                if not all([player, side, line, price]): 
                                    continue
                                dline = float(line); dprice=float(price)
                                by_pl.setdefault((player,dline), {})[side] = dprice

                            seen_bets_for_event = set()


                            for (player, line), sides in by_pl.items():
                                over = sides.get("Over"); under = sides.get("Under")
                                if not (over and under): 
                                    continue
                                p_fair_over = remove_vig_prob(over, under)
                                if p_fair_over is None: 
                                    continue


                                prop_base = f"{sport}|{mk}|{player}|{float(line)}"


                                cands = []

                                for side, dec_price in [("Over", over), ("Under", under)]:
                                    key = f"{sport}|{mk}|{player}|{line}|{side}"
                                    update_steam(key, dec_price)
                                    d_cents = steam_delta_cents(key, time.time(), STEAM_WINDOW_MIN)
                                    reversal = (d_cents < -3.0)

                                    twins = rec_prices.get((mk, player, side, float(line)), [])
                                    rec1 = twins[0] if twins else ("", None)
                                    rec2 = twins[1] if len(twins) > 1 else ("", None)


                                    key_side = f"{prop_base}|{side}"
                                    now_min_to = mins_to  # already computed

                                    # Only let Pinnacle seed/update openings/peaks if PINNACLE_ONLY_PEAKS
                                    book_used = PRIMARY_BOOKMAKER  # because we picked from 'primary' above
                                    state = peak_props.setdefault(key_side, {
                                        "opening": None, "peak": None, "_peak_ts": 0.0, "_alerted": False
                                    })

                                    curr_dec = float(rec1[1] or dec_price)

                                    # Seed openings/peaks once
                                    if (state["opening"] is None) and (not PINNACLE_ONLY_PEAKS or book_used == "pinnacle"):
                                        state["opening"] = curr_dec
                                        state["peak"] = curr_dec
                                        state["_peak_ts"] = time.time()
                                        state["_alerted"] = False

                                    # Update peak upwards (only by Pinnacle if configured)
                                    if (not PINNACLE_ONLY_PEAKS or book_used == "pinnacle") and curr_dec is not None:
                                        if (state["peak"] is None) or (curr_dec > state["peak"]):
                                            state["peak"] = curr_dec
                                            state["_peak_ts"] = time.time()
                                            state["_alerted"] = False  # new peak clears prior alert

                                    peak_props[key_side] = state


                                    p_side = p_fair_over if side == "Over" else (1.0 - p_fair_over)
                                    def ev_vs(price: Optional[float]) -> Optional[float]:
                                        if not price: return None
                                        b = price - 1.0; q = 1.0 - p_side
                                        return (b * p_side - q)
                                    ev1 = ev_vs(rec1[1]); ev2 = ev_vs(rec2[1])

                                    # model prob                                  
                                    # --- add minimal features the model expects ---
                                    dec = float(rec1[1] or dec_price)
                                    curr_imp = 1.0 / dec if dec > 0 else None

                                    row_for_model = {
                                        "Sport": sport,
                                        "Market": mk,
                                        "Player": player,
                                        "Side": side,
                                        "Line": line,
                                        "PrimaryPriceDec": dec_price,
                                        "FairProb": p_side,
                                        "EV1": ev1 or 0.0,
                                        "EV2": ev2 or 0.0,
                                        "SteamCents": d_cents,
                                        "Reversal": int(reversal),
                                        "MinsToStart": mins_to,

                                        # --- newly added placeholders to satisfy the pipeline ---
                                        # If you don’t have true open/close yet, set open==curr so deltas == 0
                                        "open_imp": curr_imp,
                                        "curr_imp": curr_imp,
                                        "delta_imp_close_curr": 0.0,
                                        "delta_imp_close_open": 0.0,
                                        "delta_am_close_curr": 0.0,
                                        "delta_am_close_open": 0.0,

                                        # No limits available for props yet → zeros (your pipeline’s imputer/scaler can handle)
                                        "bet_limit_raw": 0.0,
                                        "bet_limit_log": 0.0,
                                        "bet_limit_z": 0.0,
                                    }

                                    if logging.getLogger().isEnabledFor(logging.INFO):
                                        logging.info("🧩 Row keys sample: %s", sorted(list(row_for_model.keys()))[:10])


                                    p_ml = predict_prop_prob_unified(row_for_model)
                        
                                    
                                    
                                    
                                    try:
                                        p_ml = float(p_ml)
                                        if p_ml > 1.0: p_ml /= 100.0
                                        p_ml = max(0.001, min(p_ml, 0.999))
                                    except:
                                        p_ml = 0.55

                                    # gating (use the price we will likely take: rec1 or primary)
                                    dec = float(rec1[1] or dec_price)
                                    passed = strong_gate(sport, mk, p_ml, dec)

                                    # Kelly
                                    kf = kelly_fraction(p_ml, dec) * KELLY_FRACTION
                                    stake = DEFAULT_BANKROLL * kf

                                    cands.append({
                                        "side": side, "dec": dec, "p_ml": p_ml,
                                        "p_implied": 1.0/dec if dec > 0 else 1.0,
                                        "stake": stake, "passed": passed,
                                        "rec1": rec1, "rec2": rec2,
                                        "d_cents": d_cents, "reversal": reversal,
                                        "ev1": ev1, "ev2": ev2,
                                    })

                                
                                per_event_count += len(cands)
                                total_props     += len(cands)
                                
                                # Debug: show Over/Under edges if both candidates exist
                                if len(cands) == 2:
                                    def _edge(c): return (c["p_ml"] - c["p_implied"])
                                    # Ensure order is Over then Under if present
                                    over_c = next((x for x in cands if x["side"] == "Over"), None)
                                    under_c = next((x for x in cands if x["side"] == "Under"), None)
                                    if over_c and under_c:
                                        logging.info(
                                            "🧮 %-28s | %-22s line=%s  Over: p_ml=%.4f imp=%.4f edge=%.4f | Under: p_ml=%.4f imp=%.4f edge=%.4f",
                                            mk, player, line,
                                            over_c["p_ml"], over_c["p_implied"], _edge(over_c),
                                            under_c["p_ml"], under_c["p_implied"], _edge(under_c),
                                        )

                                
                                # pick ONE
                                best = choose_best_side(player, mk, float(line), cands, sport=sport)

                                # Always write observation rows for BOTH sides (for analytics),
                                # but only write a bet row + Telegram for the chosen side.
                                for c in cands:
                                    orow = [
                                        datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                        sport, ev_id, game, start_lbl,
                                        mk, player, c["side"], line,
                                        over if c["side"] == "Over" else under, PRIMARY_BOOKMAKER,
                                        c["rec1"][1] or "", c["rec1"][0] or "", c["rec2"][1] or "", c["rec2"][0] or "",
                                        round((p_fair_over if c["side"]=="Over" else 1.0 - p_fair_over),4),
                                        round(c["ev1"],4) if c["ev1"] is not None else "",
                                        round(c["ev2"],4) if c["ev2"] is not None else "",
                                        round(c["d_cents"],1), "Y" if c["reversal"] else "N",
                                        "", "flat", mins_to,
                                        c["side"], f"{c['p_ml']:.4f}",
                                        gen_bet_id(sport, mk, player, line, c["side"]),
                                    ]
                                   # OBS_BUFFER.append(orow)
                                    
                                    # ----- ONLY CONSIDER ALERT/LOG IF last hour -----
                                    if 0 < now_min_to <= ALERT_WINDOW_MIN and state["opening"] and state["peak"] and curr_dec:
                                        dropped, drop_cents, thresh = drop_from_peak_ok(state["peak"], curr_dec)

                                        # one-time per side until a new peak forms; also dedupe in time window
                                        dedupe_key = f"{key_side}|obs"
                                        if dropped and (drop_cents >= thresh) and not state["_alerted"] and not seen_recently(dedupe_key):
                                            # >>> compute model p_ml, kelly, etc. (you already do this) <<<
                                            # keep your cands[] building as-is

                                            # Instead of always appending BOTH sides, append only when the condition triggers:
                                            OBS_BUFFER.append([
                                                datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                                sport, ev_id, game, start_lbl,
                                                mk, player, side, line,
                                                dec_price, PRIMARY_BOOKMAKER,
                                                c["rec1"][1] or "", c["rec1"][0] or "", c["rec2"][1] or "", c["rec2"][0] or "",
                                                round(p_side,4),
                                                round(ev1,4) if ev1 is not None else "",
                                                round(ev2,4) if ev2 is not None else "",
                                                round(c["d_cents"],1), "Y" if c["reversal"] else "N",
                                                "", "flat", now_min_to,
                                                side, f"{p_ml:.4f}",
                                                gen_bet_id(sport, mk, player, line, side),
                                            ])

                                            # Telegram + Bets row only for chosen side later (keep your choose_best_side)
                                            state["_alerted"] = True
                                            peak_props[key_side] = state
                                            mark_recent(dedupe_key)



                                # Decide and send only if best side passes the same last-hour + drop-from-peak gates
                                if best:
                                    best_key_side = f"{prop_base}|{best['side']}"
                                    best_state = peak_props.get(best_key_side, {"opening": None, "peak": None, "_alerted": False})
                                    best_curr_dec = float(best["dec"])

                                    ok_window = 0 < mins_to <= ALERT_WINDOW_MIN
                                    ok_peak = bool(best_state.get("opening") and best_state.get("peak") and best_curr_dec)
                                    dropped, drop_cents, thresh = (False, 0, 10**9)
                                    if ok_peak:
                                        dropped, drop_cents, thresh = drop_from_peak_ok(best_state["peak"], best_curr_dec)

                                    bet_dedupe_key = f"{best_key_side}|bet"

                                    if ok_window and ok_peak and dropped and (drop_cents >= thresh) and not best_state.get("_alerted", False) and not seen_recently(bet_dedupe_key):
                                        bet_id = gen_bet_id(sport, mk, player, line, best["side"])
                                        brow = [
                                            datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"),
                                            sport, game, start_lbl,
                                            mk, player, best["side"], line, best["dec"], american_from_decimal(best["dec"]) or "",
                                            best["rec1"][0] or PRIMARY_BOOKMAKER,
                                            f"{best['p_ml']:.4f}", f"{best['stake']:.2f}", f"{KELLY_FRACTION:.2f}", "", bet_id
                                        ]
                                        BETS_BUFFER.append(brow)
                                        logging.info("✅ BET %s | %s %s %s @ %.2f (p=%.3f stake=$%s, drop=%s¢≥%s¢) id=%s",
                                                     mk, player, best["side"], line, best["dec"], best["p_ml"], f"{best['stake']:.2f}",
                                                     drop_cents, thresh, bet_id)

                                        SEND_QUEUE.append((
                                            f"🎯 <b>Player Prop</b> ({esc(mk)})\n"
                                            f"{esc(player)} — {esc(best['side'])} {esc(f'{float(line):.1f}')}\n"
                                            f"{esc(game)}\n"
                                            f"🕒 {esc(start_lbl)}\n"
                                            f"🤖 AI: {best['p_ml']*100:.1f}% | 💵 {best['stake']:.2f} (¼ Kelly) @ {best['dec']:.2f}\n"
                                            f"📉 Off peak by ~{drop_cents}¢ (need ≥{thresh}¢)\n"
                                            f"id: <code>{esc(bet_id)}</code>"
                                        ))

                                        best_state["_alerted"] = True
                                        peak_props[best_key_side] = best_state
                                        mark_recent(bet_dedupe_key)


                            await flush_buffers()
                            await drain_telegram_queue()


                        logging.info("📊 Parsed %d prop rows for %s | %s", per_event_count, sport, game)

                took = time.time() - loop_start
                sleep_sec = max(5, TRACK_BASE - int(took))
                logging.info("⏲️ Loop done: events=%d props=%d | sleeping %ds", total_events, total_props, sleep_sec)
                
                
                await flush_buffers(force=True)
                await drain_telegram_queue()

                save_state()
                await asyncio.sleep(sleep_sec)

            except Exception as e:
                logging.error("props loop error: %s", e, exc_info=True)
                save_state()
                await asyncio.sleep(TRACK_BASE)

if __name__ == "__main__":
    try:
        asyncio.run(run_props())
    except KeyboardInterrupt:
        logging.info("🛑 player_props.py stopped by user")
