import re
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
import os, time, logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os, sys, time, logging, argparse, traceback, itertools, math
from datetime import datetime
from typing import Optional, Tuple, List
from telegram import Bot
import tweepy



_ET_TOKENS = re.compile(r"\b(?:ET|EDT|EST|E[DS]T)\b", re.IGNORECASE)

def _clean_dt_string(s: str) -> str:
    s = _ET_TOKENS.sub("", str(s)).strip()
    s = re.sub(r",[ ]*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_to_local(value, tz_name: str = "America/New_York"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    tz = ZoneInfo(tz_name)
    # Fast path: pandas can already parse many types
    try:
        if isinstance(value, (pd.Timestamp, datetime)):
            ts = pd.to_datetime(value, errors="coerce")
        else:
            s = _clean_dt_string(value)
            # Let pandas parse (naive -> naive)
            ts = pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        return pd.NaT

    if ts is pd.NaT or pd.isna(ts):
        return pd.NaT

    # Normalize tz: localize naive; convert aware
    if getattr(ts, "tzinfo", None) is None:
        try:
            ts = ts.tz_localize(tz) if hasattr(ts, "tz_localize") else ts.replace(tzinfo=tz)
        except Exception:
            # Fallback: construct via datetime
            ts = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=tz)
    else:
        ts = ts.tz_convert(tz) if hasattr(ts, "tz_convert") else ts.astimezone(tz)

    return ts

def is_same_local_day(ts: pd.Timestamp, day: datetime.date, tz_name: str = "America/New_York") -> bool:
    if ts is pd.NaT or pd.isna(ts):
        return False
    tz = ZoneInfo(tz_name)
    # Ensure same zone
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts.date() == day



# --- ensure repo root is on sys.path BEFORE internal imports ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



# heartbeat (internal)
from utils.heartbeat import touch as hb_touch, every as hb_every, ping as hb_ping

HB_FILE = os.getenv("HEARTBEAT_FILE_AUTOPOSTER", "/var/run/ss_autoposter_heartbeat")
HB_ENV  = "HEARTBEAT_URL_AUTOPOSTER"
_last_bucket = -1

from pick_service import fetch_recent_rows, normalize_row
from post_daily_picks import (
    build_caption,
    already_posted,
    mark_posted,
    post_text_only_to_x,
    _MIN_GAP_SEC,
)

# --- Logging Setup ---
# Allow optional --log-path even before normal args
import argparse, os, sys, logging
from logging.handlers import RotatingFileHandler

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--log-path", default=os.getenv("AUTOPOSTER_LOG_FILE", "autoposter.log"))
_pre_args, remaining_argv = _pre.parse_known_args()

LOG_PATH = _pre_args.log_path
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
abs_log = os.path.abspath(LOG_PATH)

handlers = [
    RotatingFileHandler(LOG_PATH, mode="a", maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
    logging.StreamHandler(sys.stdout),
]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=handlers,
    force=True,
)
logger = logging.getLogger("autoposter")
logger.info(f"📂 Logging to {abs_log}")






# ---------------- CLI ----------------

ap = argparse.ArgumentParser(parents=[_pre])

ap.add_argument("--bet_id", default="", help="post only this bet_id")
ap.add_argument("--once", action="store_true", help="exit after first post attempt")
ap.add_argument("--force", action="store_true", help="ignore posted_log dedupe for this run")
ap.add_argument("--verbose", action="store_true", help="extra logging")
ap.add_argument("--today-only", action="store_true", help="force only rows from 'today' (local TZ)")
ap.add_argument("--all-days", action="store_true", help="ignore 'today' filter and consider all rows")
args = ap.parse_args()
if args.verbose:
    logger.setLevel(logging.DEBUG)

# ------------- Env -------------
POLL_SEC               = int(os.getenv("BROADCAST_POLL_SEC", "30"))
DRY_RUN                = os.getenv("DRY_RUN", "false").lower() == "true"
TELEGRAM_ENABLED       = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
X_ENABLED              = os.getenv("X_ENABLED", "false").lower() == "true"
ENV_REQUIRE_ROW_TODAY  = os.getenv("REQUIRE_ROW_TODAY","false").lower() == "true"
TZ_NAME                = os.getenv("TZ","America/New_York")
MAX_X                  = int(os.getenv("X_MAX_CHARS", "280"))
RECENT_WINDOW          = int(os.getenv("RECENT_WINDOW", "600"))

# CLI overrides env for "today only"
if args.today_only:
    REQUIRE_ROW_TODAY = True
elif args.all_days:
    REQUIRE_ROW_TODAY = False
else:
    REQUIRE_ROW_TODAY = ENV_REQUIRE_ROW_TODAY

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
PRO_CHAT_ID        = int(os.getenv("PRO_CHAT_ID", "0"))
tg_bot = Bot(token=TELEGRAM_BOT_TOKEN) if (TELEGRAM_ENABLED and TELEGRAM_BOT_TOKEN) else None

# X (Twitter) client (v2)
X_API_KEY             = os.getenv("X_API_KEY", "")
X_API_SECRET          = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")
_X_LOCK_UNTIL = 0  # epoch when we can next attempt any X post
_LAST_TWEET_TS = 0 # for simple 1-per-65s throttle

# ---- Sheets logging (where to append successful posts) ----
GSHEET_ID_LOG   = os.getenv("GSHEET_ID_LOG", "")          # spreadsheet ID for logging (recommended)
GSHEET_TAB_LOG  = os.getenv("GSHEET_TAB_LOG", "PostedToX")# tab name for appended rows
GOOGLE_CREDS    = os.getenv("GOOGLE_CREDS_FILE", "")      # service-account json

# If you prefer to update the source sheet row instead of appending, set these:
GSOURCE_SHEET_ID   = os.getenv("GSOURCE_SHEET_ID", "")    # spreadsheet ID of source (AllObservations)
GSOURCE_TAB_NAME   = os.getenv("GSOURCE_TAB_NAME", "AllObservations")
GSOURCE_POSTED_COL = os.getenv("GSOURCE_POSTED_COL", "")  # e.g., "PostedToX" (leave empty to skip)

# --- Free Daily Pick gating (X only) ---
FREE_DAILY_MODE     = os.getenv("FREE_DAILY_MODE", "true").lower() == "true"
FREE_POSTS_PER_DAY  = int(os.getenv("FREE_POSTS_PER_DAY", "1"))
FREE_HEADER         = os.getenv("FREE_HEADER", "🔓 Free Daily Pick")
SUBSCRIBE_URL       = os.getenv("SUBSCRIBE_URL", "https://sharps-signal.com/subscribe")

def _today_key(tz_name: str = TZ_NAME) -> str:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d")

def _get_daily_count(state) -> int:
    tk = _today_key()
    daily = state.get("daily_free_posts", {})
    return int(daily.get(tk, 0) or 0)

def _inc_daily_count(state):
    tk = _today_key()
    daily = state.get("daily_free_posts", {})
    daily[tk] = int(daily.get(tk, 0) or 0) + 1
    # prune old keys (optional, keeps state tidy)
    if len(daily) > 7:
        for k in sorted(list(daily.keys()))[:-7]:
            daily.pop(k, None)
    state["daily_free_posts"] = daily


def _gspread_client():
    if not GOOGLE_CREDS:
        return None
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS, scope)
        return gspread.authorize(creds)
    except Exception as e:
        logger.warning("gspread init failed: %s", e)
        return None

_gs = _gspread_client()

import re

def _safe_hb_touch(path: str):
    # If no path provided, default to a temp file on Windows/Linux
    if not path:
        path = os.path.join(os.getenv("TEMP", "."), "ss_autoposter_heartbeat")

    p = str(path).strip()
    # Normalize backslashes so 'https:\\...' is treated as a URL too
    p_norm = p.replace("\\", "/")

    # Looks like a URL? skip touching the filesystem
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://', p_norm):
        logger.debug("HB_FILE looks like a URL; skipping hb_touch: %s", p)
        return

    try:
        hb_touch(p)
    except Exception as e:
        logger.debug("hb_touch failed (non-fatal): %s", e)
        
        
STATE_PATH = os.getenv("AUTOPOSTER_STATE_FILE", ".autoposter_state.json")

def load_state():
    try:
        import json
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_seen_iso": None, "last_tweet_ts": 0}

def save_state(state):
    try:
        import json, tempfile, os
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        logger.debug("save_state failed: %s", e)




def log_post_to_sheet(bet_id: str, tweet_id: str, caption: str, when_ts: int):
    """Append a row to GSHEET_ID_LOG/GSHEET_TAB_LOG so you see 'new rows' whenever X really posts."""
    if not _gs or not GSHEET_ID_LOG:
        return
    try:
        sh = _gs.open_by_key(GSHEET_ID_LOG)
        ws = sh.worksheet(GSHEET_TAB_LOG)
    except Exception:
        try:
            # create the tab if missing
            sh = _gs.open_by_key(GSHEET_ID_LOG)
            ws = sh.add_worksheet(title=GSHEET_TAB_LOG, rows=2000, cols=10)
            ws.append_row(["Timestamp", "bet_id", "tweet_id", "Caption"])
        except Exception as e:
            logger.warning("log_post_to_sheet: open/create failed: %s", e)
            return
    try:
        iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when_ts))
        ws.append_row([iso, bet_id, tweet_id, caption], value_input_option="USER_ENTERED")
    except Exception as e:
        logger.warning("log_post_to_sheet: append failed: %s", e)



def _is_cell_not_found(err: Exception) -> bool:
    msg = str(err).lower()
    return "cell not found" in msg or "no matches found" in msg or "unable to find" in msg

def mark_source_row_posted(bet_id: str, tweet_id: str):
    if not (_gs and GSOURCE_SHEET_ID and GSOURCE_POSTED_COL):
        return
    try:
        sh = _gs.open_by_key(GSOURCE_SHEET_ID)
        ws = sh.worksheet(GSOURCE_TAB_NAME)
        header = ws.row_values(1)
        if GSOURCE_POSTED_COL not in header:
            header.append(GSOURCE_POSTED_COL)
            ws.update("1:1", [header])
        col_idx = header.index(GSOURCE_POSTED_COL) + 1

        # Try to find the bet row (version-agnostic handling)
        try:
            cell = ws.find(bet_id)  # some versions raise a generic Exception on no match
            if cell:
                ws.update_cell(cell.row, col_idx, f"X:{tweet_id}")
        except Exception as e:
            if _is_cell_not_found(e):
                # quietly ignore if bet_id isn't present yet
                logger.debug("bet_id %s not found in %s — skipping stamp.", bet_id, GSOURCE_TAB_NAME)
            else:
                raise
    except Exception as e:
        logger.warning("mark_source_row_posted: failed: %s", e)




def _mask(s: str, show=4):
    if not s: return ""
    s = str(s)
    if len(s) <= show: return "*" * len(s)
    return s[:show] + "…" + "*" * max(0, len(s) - show - 1)

x_client = None
if X_ENABLED:
    try:
        x_client = tweepy.Client(
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET,
        )
        logger.info("X client initialized (keys=%s/%s tok=%s/%s)",
                    _mask(X_API_KEY), _mask(X_API_SECRET),
                    _mask(X_ACCESS_TOKEN), _mask(X_ACCESS_TOKEN_SECRET))
        # Health check: verify credentials and log handle
        try:
            me = x_client.get_me()
            handle = f"@{me.data.username}" if getattr(me, "data", None) else "(unknown)"
            logger.info("X healthcheck OK: posting as %s", handle)
        except Exception as e:
            logger.exception("X healthcheck failed; disabling X posting.")
            X_ENABLED = False
            x_client = None
    except Exception:
        logger.exception("Failed to init X client; X posting disabled.")
        X_ENABLED = False
        x_client = None
else:
    logger.info("X posting disabled by env (X_ENABLED=false).")

# ------------- Utility (trim for X) -------------
def trim_for_x(text: str, key_suffix: str) -> str:
    t = text
    if "\n\n🧠 Why:" in t and len(t) > MAX_X:
        t = t.split("\n\n🧠 Why:")[0].rstrip()
    if len(t) > MAX_X:
        t = t[:MAX_X-5].rstrip() + " …"
    suf = f" [{key_suffix[-6:]}]" if key_suffix else ""
    if len(t) + len(suf) <= MAX_X:
        t += suf
    return t

# ------------- Date helpers -------------
from zoneinfo import ZoneInfo
from dateutil import parser as dtparser

LOCAL_TZ = ZoneInfo(TZ_NAME)

def _parse_any_dt(val) -> Optional[pd.Timestamp]:
    """Parse many sheet formats → tz-aware pandas Timestamp in local tz, or None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    # strip stray ET/EDT/EST tokens to avoid pandas/dateutil warnings
    s = _ET_TOKENS.sub("", s).strip()
    if not s:
        return None
    # First try pandas (fast)
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt is pd.NaT or pd.isna(dt):
        # Fallback to dateutil (handles 'Sep 27, 2:15 PM EDT')
        try:
            py_dt = dtparser.parse(s)
        except Exception:
            return None
        dt = pd.Timestamp(py_dt)
    # Attach or convert tz
    try:
        if dt.tzinfo is None:
            dt = dt.tz_localize(LOCAL_TZ)
        else:
            dt = dt.tz_convert(LOCAL_TZ)
    except Exception:
        # Last resort: assume local
        dt = pd.Timestamp(dt.replace(tzinfo=None)).tz_localize(LOCAL_TZ)
    return dt


DATE_CANDIDATES: tuple[str, ...] = (
    "Timestamp",
    "Placed At (ET)",
    "Confirmed At",
    "Game Time",
)

ALLOW_TODAY_FALLBACK = os.getenv("ALLOW_TODAY_FALLBACK", "true").lower() == "true"

def _best_date_column(df: pd.DataFrame) -> Tuple[str, List[pd.Timestamp]]:
    """Return (column_name, parsed_series) for the date column with the most non-null parses."""
    best_col = ""
    best_series = None
    best_count = -1
    for col in DATE_CANDIDATES:
        if col in df.columns:
            parsed = df[col].apply(_parse_any_dt)
            cnt = sum(x is not None for x in parsed)
            if cnt > best_count:
                best_count = cnt
                best_col = col
                best_series = parsed
    if best_series is None:
        # No candidate columns; return all None
        best_series = [None] * len(df.index)
    return best_col, list(best_series)

def keep_most_recent(df: pd.DataFrame, window: int = 600) -> pd.DataFrame:
    """Sort by best date column desc and keep top N; no state or new-row logic here."""
    if df is None or df.empty:
        return df

    col, parsed = _best_date_column(df)
    fill = pd.Timestamp("1900-01-01", tz=LOCAL_TZ)
    sort_key = [p if p is not None else fill for p in parsed]
    df2 = df.assign(__sort_dt=sort_key)
    df2 = df2.sort_values("__sort_dt", ascending=False)
    recent = df2.head(window).drop(columns=["__sort_dt"])

    have_dates = [p for p in parsed if p is not None]
    lo = str(min(have_dates)) if have_dates else "n/a"
    hi = str(max(have_dates)) if have_dates else "n/a"
    logger.info("Recent-window=%d based on '%s' → [%s … %s]", window, col or "(none)", lo, hi)
    if col and not have_dates:
        logger.warning("'%s' found but none of its values parsed as datetimes.", col)
    return recent
    
def filter_new_since(df: pd.DataFrame, last_seen_iso: str | None) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Return only rows strictly newer than last_seen_iso (by best date col)."""
    if df is None or df.empty:
        return df, []
    col, parsed = _best_date_column(df)
    if not last_seen_iso:
        # First run: don’t backfill; treat newest timestamp as cursor baseline with empty result
        have = [p for p in parsed if p is not None]
        return df.iloc[0:0], have  # empty df, but we still return parsed for cursor init

    try:
        last_seen = pd.Timestamp(last_seen_iso).tz_convert(LOCAL_TZ)
    except Exception:
        last_seen = None

    if last_seen is None:
        return df, [p for p in parsed if p is not None]

    mask_new = [(p is not None and p > last_seen) for p in parsed]
    df2 = df[mask_new]
    logger.info("After new-row filter (> %s): %d rows remain", last_seen, len(df2))
    return df2, [p for p in parsed if p is not None]
    




# change signature
def filter_today_df(df: pd.DataFrame, allow_fallback: bool) -> pd.DataFrame:
    if not REQUIRE_ROW_TODAY or df.empty:
        return df
    col, parsed = _best_date_column(df)
    if not col:
        logger.warning("No candidate date columns found; skipping 'today' filter.")
        return df

    today_local = datetime.now(LOCAL_TZ).date()
    is_today_mask, unparsable = [], 0
    for ts in parsed:
        if ts is None:
            is_today_mask.append(False); unparsable += 1
        else:
            is_today_mask.append(ts.date() == today_local)

    kept = df[is_today_mask]
    logger.info("After 'today' filter (column=%s): kept=%d dropped=%d unparsable=%d (REQUIRE_ROW_TODAY=%s, TZ=%s)",
                col, len(kept), len(df) - len(kept), unparsable, REQUIRE_ROW_TODAY, TZ_NAME)

    if kept.empty and allow_fallback:
        logger.warning("No rows matched 'today' in %s. Falling back to ALL rows for this loop.", col)
        return df

    return kept




# ------------- Cooldown per bet (avoid hammering during lockout) -------------
_COOLDOWN = {}  # bet_id -> epoch when we may retry

def can_retry(bet_id: str) -> bool:
    return int(time.time()) >= _COOLDOWN.get(bet_id, 0)

def set_retry_after(bet_id: str, seconds: int):
    _COOLDOWN[bet_id] = int(time.time()) + int(seconds)
    
    
MIN_TWEET_GAP = int(os.getenv("MIN_TWEET_GAP_SEC", "65"))
_next_tweet_ok_at = int(time.time())
_next_poll_at = int(time.time()) + POLL_SEC




# ------------- Main loop -------------
if __name__ == "__main__":
    loop = 0
    logger.info("Broadcaster (text-only) started.")
    try:
        hb_ping(HB_ENV, payload={"svc": "autoposter", "status":"start","ts": int(time.time())}, suffix="start")
    except Exception:
        pass
    logger.info("Flags -> DRY_RUN=%s  TG=%s  X=%s  TODAY_ONLY=%s  POLL_SEC=%s  TZ=%s  RECENT_WINDOW=%s",
                DRY_RUN, bool(tg_bot), bool(x_client), REQUIRE_ROW_TODAY, POLL_SEC, TZ_NAME, RECENT_WINDOW)
                
                
    state = load_state()
    _LAST_TWEET_TS = int(state.get("last_tweet_ts", 0))
    
   

    while True:
        loop += 1
        logger.debug("——— Loop %d ———", loop)
        try:
            df = fetch_recent_rows(limit=200)
            if df is None:
                _next_poll_at = int(time.time()) + POLL_SEC
                time.sleep(POLL_SEC)
                continue



            logger.debug("Fetched dataframe columns: %s", list(df.columns) if not df.empty else [])
            df = keep_most_recent(df, window=RECENT_WINDOW)
            
            total_rows = len(df.index)
            logger.info("Pulled %d rows from sheet before filters", total_rows)

 
            if df.empty:
                _next_poll_at = int(time.time()) + POLL_SEC
                time.sleep(POLL_SEC)
                continue



            allow_fallback = (not args.today_only) and (os.getenv("ALLOW_TODAY_FALLBACK","true").lower()=="true")
            df2 = filter_today_df(df, allow_fallback) 
            logger.info("After 'today' filter: %d rows (REQUIRE_ROW_TODAY=%s)", len(df2), REQUIRE_ROW_TODAY)
            
            
            
            # NEW: only rows newer than the saved cursor
            df2, parsed_all = filter_new_since(df2, state.get("last_seen_iso"))
            
            # Seed the cursor on the very first run (no backfill)
            if state.get("last_seen_iso") is None and parsed_all:
                try:
                    state["last_seen_iso"] = str(max([p for p in parsed_all if p is not None]))
                    save_state(state)
                    logger.debug("Initialized last_seen_iso to %s (first run)", state["last_seen_iso"])
                except Exception:
                    pass

            
            if df2.empty:
                _next_poll_at = int(time.time()) + POLL_SEC
                time.sleep(POLL_SEC)
                continue
           

            # # bet_id column
            # betid_col = None
            # for c in df2.columns:
                # if c.lower() in ("bet id","bet_id","id"):
                    # betid_col = c; break
            # if not betid_col:
                # logger.warning("Sheet has no 'bet_id' column yet. Sleeping…")
                # _next_poll_at = int(time.time()) + POLL_SEC
                # time.sleep(POLL_SEC)
                # continue

                

            skipped_filter = skipped_dedupe = posted = attempted = 0
            
            broke_for_rate_limit = False


            for _, row in df2.iterrows():

                    
                n = normalize_row(row, df2.columns, TZ_NAME)

                # Fallback: if normalize_row didn't yield bet_id, pull it directly from the sheet
                betid_col = None
                for c in df2.columns:
                    if c.lower() in ("bet id","bet_id","id"):
                        betid_col = c; break

                bet_id = (n.get("bet_id") or str(row.get(betid_col) or "")).strip() if betid_col else (n.get("bet_id") or "").strip()

                if not bet_id:
                    logger.warning("Row missing bet_id after normalization and fallback; skipping. Columns present=%s", list(df2.columns))
                    continue

                # Ensure the normalized dict also carries bet_id for downstream functions
                n["bet_id"] = bet_id


                if args.bet_id and bet_id != args.bet_id:
                    skipped_filter += 1
                    logger.debug("Skipped by --bet_id filter: %s", bet_id)
                    continue

                if (not args.force) and already_posted(bet_id):
                    skipped_dedupe += 1
                    logger.debug("Skipped (already posted): %s", bet_id)
                    continue

                attempted += 1

                caption = build_caption(pd.Series({
                    "league":     n.get("league",""),
                    "market":     n.get("market",""),
                    "pick":       n.get("pick",""),
                    "odds":       n.get("odds",""),
                    "home":       n.get("home",""),
                    "away":       n.get("away",""),
                    "total_line": n.get("total_line",""),
                    "spread_line":n.get("spread_line",""),
                    "__game_dt":  n.get("game_time_dt"),
                }))
                if n.get("reason"):
                    caption = f"{caption}\n\n🧠 Why: {n['reason']}"

                logger.info("Prepared text post for bet_id=%s", bet_id)
                if args.verbose:
                    logger.debug("Caption:\n%s", caption)

                if DRY_RUN:
                    logger.info("DRY_RUN=true — not posting. (bet_id=%s)", bet_id)
                    if args.once:
                        logger.info("Done (single run)."); raise SystemExit(0)
                    continue

                tg_ok = False
                x_ok  = False

                # Telegram Pro
                if tg_bot and PRO_CHAT_ID and TELEGRAM_ENABLED:
                    try:
                        tg_bot.send_message(chat_id=PRO_CHAT_ID, text=caption)
                        logger.info("✅ Posted text alert to Telegram Pro (bet_id=%s)", bet_id)
                        tg_ok = True
                    except Exception as e:
                        logger.error("Telegram Pro post failed (bet_id=%s): %s", bet_id, e)
                        logger.debug("Traceback:\n%s", traceback.format_exc())

                # X (text-only) with cooldown
                
                # --- global lockout / throttle ---
                now = int(time.time())
                if now < _X_LOCK_UNTIL:
                    logger.info("X in global lockout for %ss. Skipping X for bet_id=%s.", _X_LOCK_UNTIL - now, bet_id)
                    continue
                    
                if now < _next_tweet_ok_at:
                    logger.info("Rate limiter active for %ss; deferring remaining posts.", _next_tweet_ok_at - now)
                    broke_for_rate_limit = True
                    break

                # Gate X by daily free-post limit
                if X_ENABLED and x_client:
                    if FREE_DAILY_MODE:
                        posted_today = _get_daily_count(state)
                        if posted_today >= FREE_POSTS_PER_DAY:
                            logger.info("Daily free-post cap reached (%d). Skipping X for bet_id=%s.", posted_today, bet_id)
                            # We *do* want to advance the cursor later to avoid retrying these rows.
                            # Continue to next row without X attempt.
                            continue

                # Build X caption with header + subscribe CTA
                x_caption = trim_for_x(caption, bet_id)
                x_caption = f"{FREE_HEADER}\n\n{x_caption}\n\nSubscribe to get all picks & live alerts: {SUBSCRIBE_URL}"


                                                
                                

                tweet_id = ""
                if X_ENABLED and x_client and can_retry(bet_id) and (int(time.time()) >= _X_LOCK_UNTIL):
                    try:
                        tweet_id = post_text_only_to_x(x_caption, x_client) or ""

                        # After a successful post:
                        if tweet_id:
                            logger.info("✅ Posted to X (tweet_id=%s, bet_id=%s)", tweet_id, bet_id)
                            x_ok = True
                            set_retry_after(bet_id, _MIN_GAP_SEC)
                            _LAST_TWEET_TS = int(time.time())
                            _next_tweet_ok_at = _LAST_TWEET_TS + MIN_TWEET_GAP
                            # ✅ Count toward daily free-post quota ONLY on success
                            try:
                                _inc_daily_count(state)
                                save_state(state)
                            except Exception:
                                logger.debug("Failed updating daily free-post counter.", exc_info=True)
                        else:
                            # If you want to treat duplicates as success, detect by error codes/text and mark_posted here.
                            logger.error("X post returned empty tweet_id (bet_id=%s). Applying global lockout.", bet_id)
                            set_retry_after(bet_id, 5 * 60)
                            _X_LOCK_UNTIL = int(time.time()) + 15 * 60



                    except tweepy.TweepyException as e:
                        reset_after = 15 * 60
                        rst = None
                        
                        try:
                            msg = getattr(e, "response", None)
                            if msg and hasattr(msg, "text"):
                                logger.error("X error body: %s", msg.text)
                        except Exception:
                            pass                        
                        
                        
                        
                        
                        try:
                            if getattr(e, "response", None):
                                h = getattr(e.response, "headers", {}) or {}
                                rem = int(h.get("x-rate-limit-remaining", "1"))
                                rst = int(h.get("x-rate-limit-reset", "0"))  # epoch seconds
                                if rem <= 0 and rst > 0:
                                    reset_after = max(5, rst - int(time.time()))
                        except Exception:
                            pass
                        logger.error("X post failed (bet_id=%s): %s — global backoff %ss", bet_id, e, reset_after)
                        set_retry_after(bet_id, 5 * 60)  # per-bet cooldown
                        _X_LOCK_UNTIL = max(
                            _X_LOCK_UNTIL,
                            (rst + 5) if isinstance(rst, int) and rst > 0 else int(time.time()) + reset_after
                        )

                    except Exception as e:
                        logger.error("X post failed (bet_id=%s): %s (global 10m backoff)", bet_id, e)
                        set_retry_after(bet_id, 5 * 60)
                        _X_LOCK_UNTIL = int(time.time()) + 10 * 60

                elif X_ENABLED and x_client and not can_retry(bet_id):
                    wait = _COOLDOWN.get(bet_id, 0) - int(time.time())
                    logger.info("Skipping X retry for %s — cooldown %ss remaining.", bet_id, max(0, wait))


                # ✅ Only mark as posted if X succeeded
                #    or if X is disabled and TG succeeded (so we don't spam TG).
                if x_ok or (tg_ok and not X_ENABLED):
                    mark_posted(bet_id, tweet_id)
                    posted += 1
                    logger.debug("Marked as posted (bet_id=%s, tweet_id=%s)", bet_id, tweet_id)
                    try:
                        log_post_to_sheet(bet_id, tweet_id, caption, int(time.time()))
                        mark_source_row_posted(bet_id, tweet_id)  # no-op unless configured
                    except Exception as e:
                        logger.warning("Post success logged but sheet write failed: %s", e)
                else:
                    logger.info("Not marking as posted yet for %s (will retry).", bet_id)

                if args.once:
                    logger.info("Done (single run)."); raise SystemExit(0)


            # Advance cursor if we didn't hit a rate-limit break and we *processed* the batch
            # (even if daily cap blocked X posts), so we don't retry the same rows forever.
            if not broke_for_rate_limit and parsed_all and attempted > 0:
                try:
                    state["last_seen_iso"] = str(max([p for p in parsed_all if p is not None]))
                except Exception:
                    pass

            state["last_tweet_ts"] = _LAST_TWEET_TS
            save_state(state)

            logger.info(
                "Loop summary: rows_before=%d rows_after=%d attempted=%d posted=%d skipped_by_filter=%d skipped_by_dedupe=%d",
                total_rows, len(df2), attempted, posted, skipped_filter, skipped_dedupe
            )
  
        except Exception:
            logger.exception("Broadcaster loop error")
            try:
                hb_ping(HB_ENV, payload={"svc": "autoposter", "status":"fail","ts": int(time.time())}, suffix="fail")
            except Exception:
                pass

        # Heartbeat once per minute + touch a local file for watchdogs
        bucket = hb_every(1)
        if bucket != _last_bucket:
            _last_bucket = bucket
            _safe_hb_touch(HB_FILE)
            hb_ping(HB_ENV, payload={"svc": "autoposter", "ts": int(time.time())})

        # simple non-blocking pacing
        now = int(time.time())
        if now >= _next_poll_at:
            _next_poll_at = now + POLL_SEC
        else:
            time.sleep(0.5)
