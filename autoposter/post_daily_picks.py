import os, io, json, time, uuid, shutil, subprocess, tempfile, logging, requests
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser, tz
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import tweepy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s  %(message)s")

# Load env once
load_dotenv()

# -------- Config --------
TZ = os.getenv("TZ", "America/New_York")
LOCAL_TZ = tz.gettz(TZ)

SHEET_ENABLED = os.getenv("SHEET_ENABLED", "false").lower() == "true"
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "./service_account.json")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
GOOGLE_WORKSHEET_NAME = os.getenv("GOOGLE_WORKSHEET_NAME", "")
PICKS_CSV = os.getenv("PICKS_CSV", "./sample_picks.csv")

ASSETS_DIR = os.getenv("ASSETS_DIR", "./assets"); os.makedirs(ASSETS_DIR, exist_ok=True)

POST_WINDOW_MIN = int(os.getenv("POST_WINDOW_MIN", "15"))
POST_WINDOW_MAX = int(os.getenv("POST_WINDOW_MAX", "60"))
CONFIDENCE_FILTER = [s.strip().lower() for s in os.getenv("CONFIDENCE_FILTER", "High,Medium").split(",")]
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# X
X_ENABLED = os.getenv("X_ENABLED", "false").lower() == "true"
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")

SUBSCRIBE_URL = os.getenv("SUBSCRIBE_URL", "https://sharps-signal.com/subscribe")


# TikTok (kept but cleaned)
TT_ENABLED        = os.getenv("TT_ENABLED", "false").lower() == "true"
TT_CLIENT_KEY     = os.getenv("TT_CLIENT_KEY", "")
TT_CLIENT_SECRET  = os.getenv("TT_CLIENT_SECRET", "")
TT_ACCESS_TOKEN   = os.getenv("TT_ACCESS_TOKEN", "")
TT_REFRESH_TOKEN  = os.getenv("TT_REFRESH_TOKEN", "")
TT_PRIVACY_LEVEL  = (os.getenv("TT_PRIVACY_LEVEL","SELF_ONLY").strip().upper() or "SELF_ONLY")
if TT_PRIVACY_LEVEL not in {"SELF_ONLY","MUTUAL_FOLLOW_FRIENDS","PUBLIC_TO_EVERYONE"}:
    TT_PRIVACY_LEVEL = "SELF_ONLY"

# Branding
def hex2rgb(h: str):
    h = (h or "").strip().lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
BRAND_BG      = hex2rgb(os.getenv("BRAND_BG", "#0B0F14"))
BRAND_PANEL   = hex2rgb(os.getenv("BRAND_PANEL", "#11151C"))
BRAND_CARD    = hex2rgb(os.getenv("BRAND_CARD", "#0F131A"))
BRAND_TEXT    = hex2rgb(os.getenv("BRAND_TEXT", "#F2F5F9"))
BRAND_SUBTLE  = hex2rgb(os.getenv("BRAND_SUBTLE", "#A9B3C9"))
BRAND_ACCENT  = hex2rgb(os.getenv("BRAND_ACCENT", "#4FD1C5"))
FONT_BOLD_TTF = os.getenv("BRAND_FONT_BOLD", "./assets/fonts/Inter-SemiBold.ttf")
FONT_REG_TTF  = os.getenv("BRAND_FONT_REG",  "./assets/fonts/Inter-Regular.ttf")
LOGO_PATH     = os.getenv("BRAND_LOGO", "./assets/brand/logo.png")
ICONS_DIR     = os.getenv("ICONS_DIR", "./assets/icons")

POST_LOG = "./posted_log.csv"

# ---------------- posted_log helpers ----------------
def already_posted(key: str) -> bool:
    """Treat as posted only if tweet_id is non-empty (X actually succeeded)."""
    if not os.path.exists(POST_LOG): return False
    try:
        df = pd.read_csv(POST_LOG)
        df = df[df["tweet_id"].astype(str).str.len() > 0]
        return key in set(df["key"])
    except Exception:
        return False

def mark_posted(key: str, tweet_id: str):
    row = pd.DataFrame([{"key": key, "tweet_id": tweet_id, "ts": time.time()}])
    if os.path.exists(POST_LOG):
        row.to_csv(POST_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(POST_LOG, index=False)

# ---------------- fonts/helpers ----------------
def load_font(path: str, size: int, fallback: str="arialbd.ttf"):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            return ImageFont.load_default()

def format_local_time(dt_obj):
    if os.name == "nt":
        return dt_obj.strftime("%#I:%M %p")
    return dt_obj.strftime("%-I:%M %p")

def prettify_league(s: str) -> str:
    s = (s or "").replace("_"," ").strip()
    m = s.lower()
    if "mixed martial arts" in m or "mma" in m:            return "MMA"
    if "ncaaf" in m or "college" in m or "american" in m:  return "College Football • NCAAF"
    if "mlb" in m or "baseball" in m:                      return "Baseball • MLB"
    if "nba" in m or "basketball" in m:                    return "Basketball • NBA"
    if "soccer" in m or "mls" in m:                        return "Soccer • MLS"
    if "nhl" in m or "hockey" in m:                        return "Hockey • NHL"
    if "boxing" in m:                                      return "Boxing"
    return s.title()

def sport_key_for(league: str) -> str:
    m = (league or "").lower()
    if "mma" in m: return "mma"
    if "ncaaf" in m or "college" in m or "american" in m: return "football"
    if "mlb" in m or "baseball" in m: return "baseball"
    if "nba" in m or "basketball" in m: return "basketball"
    if "soccer" in m or "mls" in m: return "soccer"
    if "nhl" in m or "hockey" in m: return "hockey"
    if "boxing" in m: return "boxing"
    if "tennis" in m: return "tennis"
    return ""

# ---------------- X quota/rate-limit helpers ----------------
_MIN_GAP_SEC = int(os.getenv("X_POST_MIN_GAP_SEC", "60"))  # min spacing between posts
_QUOTA_PATH = os.path.join(tempfile.gettempdir(), "x_quota.json")

def _quota_read():
    try:
        with open(_QUOTA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"next_post_epoch": 0}

def _quota_write(state):
    try:
        with open(_QUOTA_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

@contextmanager
def poster_lock(name="x_poster.lock"):
    lock_path = os.path.join(tempfile.gettempdir(), name)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

def _sleep_until_reset(resp):
    """Parse X rate headers; persist reset, then sleep until window opens."""
    try:
        headers = getattr(resp, "headers", {}) or {}
        remaining = headers.get("x-rate-limit-remaining")
        reset_ts = headers.get("x-rate-limit-reset")
        limit = headers.get("x-rate-limit-limit")
        if reset_ts:
            reset_epoch = int(reset_ts)
            now = int(time.time())
            wait_s = max(reset_epoch - now, 0) + 2
            reset_iso = datetime.fromtimestamp(reset_epoch, tz=timezone.utc).isoformat()

            # persist next allowed posting time
            st = _quota_read()
            st["next_post_epoch"] = reset_epoch + 2
            _quota_write(st)

            log.warning(
                f"X rate limit hit. remaining={remaining} limit={limit} "
                f"reset_epoch={reset_ts} (~{wait_s}s) reset_time_utc={reset_iso}"
            )
            time.sleep(min(wait_s, 20 * 60))
            return True
    except Exception as e:
        log.exception(f"Failed to parse rate-limit headers: {e}")
    log.warning("X rate limit hit but no reset header; sleeping 120s as fallback.")
    time.sleep(120)
    return True

def post_text_only_to_x(caption: str, client: tweepy.Client):
    """
    Posts a text-only tweet with basic rate-limit handling + jitter + single-flight lock,
    respecting a persisted next_post_epoch and enforcing a minimum inter-post gap.
    Returns tweet id string on success, else None.
    """
    import random
    with poster_lock():
        # Respect persisted quota window (from prior success/429)
        st = _quota_read()
        now = int(time.time())
        if st.get("next_post_epoch", 0) > now:
            wait_s = st["next_post_epoch"] - now
            log.info(f"Waiting {wait_s}s for X quota window.")
            time.sleep(min(wait_s, 20 * 60))

        # Enforce min gap ahead of time (with small jitter)
        next_gap_epoch = now + _MIN_GAP_SEC + random.randint(0, 6)
        st["next_post_epoch"] = max(st.get("next_post_epoch", 0), next_gap_epoch)
        _quota_write(st)

        # Short jitter to avoid thundering herd
        time.sleep(random.uniform(1.0, 6.0))

        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                resp = client.create_tweet(text=caption)
                tweet_id = str(resp.data.get("id")) if getattr(resp, "data", None) else None
                h = getattr(resp, "headers", {}) or {}
                log.info(
                    f"Posted to X id={tweet_id} "
                    f"rl_rem={h.get('x-rate-limit-remaining')} "
                    f"rl_lim={h.get('x-rate-limit-limit')} "
                    f"rl_reset={h.get('x-rate-limit-reset')}"
                )

                # After success, push the next allowed time forward by min gap
                st = _quota_read()
                st["next_post_epoch"] = max(st.get("next_post_epoch", 0), int(time.time()) + _MIN_GAP_SEC)
                _quota_write(st)

                return tweet_id
            except tweepy.errors.TooManyRequests as e:
                # Persist reset from headers and sleep until then, then retry
                _sleep_until_reset(getattr(e, "response", None))
                continue
            except tweepy.Forbidden as e:
                log.error(f"X Forbidden (possible duplicate or perms): {e}")
                return None
            except Exception as e:
                log.exception(f"Unexpected X post error: {e}")
                time.sleep(5 * attempts)
                continue
        log.error("Failed to post to X after retries.")
        return None

# ---------------- Data loading + normalization ----------------
REQUIRED_COLS = ["game_time","league","market","pick","odds","confidence"]
CONF_HIGH = float(os.getenv("CONF_HIGH_THRESHOLD","0.60"))
CONF_MED  = float(os.getenv("CONF_MED_THRESHOLD","0.53"))

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_lookup = {c.lower(): c for c in df.columns}
    ALIASES = {
        "game_time": ["game_time","game time","kickoff","start time","time","Game Time"],
        "league":    ["league","sport","competition","Sport"],
        "market":    ["market","bet type","type","Market"],
        "pick":      ["pick","selection","side","team","bet","Predicted"],
        "odds":      ["odds","price","american odds","line","Odds (Am)","Odds Taken (AM)","Closing Odds (Am)","Opening Price"],
        "confidence":["confidence","conf","ai conf","ai confidence","rating","confidence level","AI Prob"],
        "notes":     ["notes","comment","reason","details"],
        "minutes_to_start": ["MinutesToStart","minutes_to_start","mins_to_start"],
        "timestamp": ["Timestamp","timestamp"],
    }
    def find_alias(key: str):
        for alias in ALIASES.get(key, []):
            if alias.lower() in lower_lookup:
                return lower_lookup[alias.lower()]
        return None

    out = pd.DataFrame()
    ts_col = find_alias("timestamp")

    for key in ["game_time","league","market"]:
        col = find_alias(key)
        if not col:
            raise ValueError(f"Missing required column: {key}. Columns found: {', '.join(df.columns)}")
        out[key] = df[col]

    pick_col = find_alias("pick")
    if not pick_col:
        raise ValueError("Missing required column: pick.")
    out["pick"] = df[pick_col]

    odds_col = find_alias("odds")
    if odds_col:
        out["odds"] = df[odds_col]
        if out["odds"].fillna("").eq("").mean() > 0.5:
            for alt in ["Closing Odds (Am)","Opening Price"]:
                if alt.lower() in lower_lookup:
                    out["odds"] = df[lower_lookup[alt.lower()]]
                    break
    else:
        for alt in ["Closing Odds (Am)","Opening Price"]:
            if alt.lower() in lower_lookup:
                out["odds"] = df[lower_lookup[alt.lower()]]
                break
        if "odds" not in out:
            raise ValueError("Missing required column: odds.")

    conf_col = find_alias("confidence")
    if conf_col and "ai prob" in conf_col.lower():
        probs = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0)
        def to_bucket(p):
            if p >= CONF_HIGH: return "High"
            if p >= CONF_MED:  return "Medium"
            return "Low"
        out["confidence"] = probs.map(to_bucket)
        out["ai_prob_raw"] = probs
    elif conf_col:
        out["confidence"] = df[conf_col].astype(str)
        out["ai_prob_raw"] = pd.NA
    else:
        out["confidence"] = "Medium"
        out["ai_prob_raw"] = pd.NA

    notes_col = find_alias("notes")
    out["notes"] = df[notes_col] if notes_col else ""

    m2s_col = find_alias("minutes_to_start")
    if m2s_col:
        out["MinutesToStart"] = pd.to_numeric(df[m2s_col], errors="coerce")

    if ts_col:
        out["RowTimestamp"] = df[ts_col]

    return out

def read_picks_from_sheet() -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_JSON, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    ws = sh.worksheet(GOOGLE_WORKSHEET_NAME)
    values = ws.get_all_values()
    if not values:
        raise ValueError("Worksheet is empty.")
    header_idx = None
    for i, row in enumerate(values):
        if sum(1 for c in row if str(c).strip()) >= 3:
            header_idx = i; break
    if header_idx is None:
        raise ValueError("Could not locate a header row with enough columns.")
    headers = [str(c).strip() for c in values[header_idx]]
    data = values[header_idx + 1:]
    df = pd.DataFrame(data, columns=headers).dropna(how="all")
    return _normalize_columns(df)

def read_picks_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)

def read_picks() -> pd.DataFrame:
    if SHEET_ENABLED:
        logging.info("Reading picks from Google Sheets…")
        return read_picks_from_sheet()
    logging.info("Reading picks from CSV…")
    return read_picks_from_csv(PICKS_CSV)

# ---------------- pick selection + caption ----------------
def to_datetime_safe(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        dt = dateparser.parse(str(x), tzinfos={"EDT": LOCAL_TZ, "EST": LOCAL_TZ})
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=LOCAL_TZ)
        return dt.astimezone(LOCAL_TZ)
    except Exception:
        return None

def choose_postworthy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__game_dt"] = df["game_time"].apply(to_datetime_safe)
    df["__confidence"] = df["confidence"].astype(str).str.strip().str.lower()
    if "MinutesToStart" in df.columns:
        now_local = datetime.now(LOCAL_TZ)
        need_fill = df["__game_dt"].isna() & df["MinutesToStart"].notna()
        df.loc[need_fill, "__game_dt"] = df.loc[need_fill, "MinutesToStart"].map(
            lambda m: now_local + timedelta(minutes=float(m))
        )
    if "RowTimestamp" in df.columns:
        df["__row_ts"] = df["RowTimestamp"].apply(to_datetime_safe)
    else:
        df["__row_ts"] = pd.NaT

    now = datetime.now(LOCAL_TZ)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end   = today_start + timedelta(days=1)

    if df["__row_ts"].notnull().any():
        df = df[(df["__row_ts"].notnull()) & (df["__row_ts"].between(today_start, today_end))]
    df = df[(df["__game_dt"].notnull()) & (df["__game_dt"].between(today_start, today_end))]

    lo = now + timedelta(minutes=POST_WINDOW_MIN)
    hi = now + timedelta(minutes=POST_WINDOW_MAX)
    mask = (df["__game_dt"].between(lo, hi)) & (df["__confidence"].isin(CONFIDENCE_FILTER))
    return df[mask].copy().sort_values("__game_dt")

def build_caption(row: pd.Series) -> str:
    league = prettify_league(str(row.get("league","")))
    market = str(row.get("market","")).strip()
    pick   = str(row.get("pick","")).strip()
    odds   = str(row.get("odds","")).strip()
    home   = str(row.get("home","")).strip()
    away   = str(row.get("away","")).strip()
    total_line  = str(row.get("total_line","")).strip()
    spread_line = str(row.get("spread_line","")).strip()
    gdt    = row.get("__game_dt")
    gtime = format_local_time(gdt) if isinstance(gdt, (datetime, pd.Timestamp)) else ""
    matchup = f"{away} @ {home}" if home and away else ""
    if "total" in market.lower() and total_line:
        pickline = f"{pick} {total_line} @ {odds}"
    elif "spread" in market.lower() and spread_line:
        pickline = f"{pick} {spread_line} @ {odds}"
    else:
        pickline = f"{pick} @ {odds}"
    hashtags = "#SportsBetting #Picks #SharpsSignal"
    return (
        f"🎯 {league} • {market}\n"
        f"🆚 {matchup}\n"
        f"📊 Pick: {pickline}\n"
        f"🕒 Kickoff: {gtime}\n\n"
        f"Subscribe to get all picks & live alerts: {SUBSCRIBE_URL}\n{hashtags}"

    )

# ---------------- Image card + optional X/TikTok posting (unchanged) ----------------
def fit_text(draw: ImageDraw.ImageDraw, text: str, preferred_font_path: str, max_w: int, start_size=110, min_size=48, bold=True):
    fallback = "arialbd.ttf" if bold else "arial.ttf"
    size = start_size
    while size >= min_size:
        font = load_font(preferred_font_path, size, fallback=fallback)
        if draw.textbbox((0,0), text, font=font)[2] <= max_w:
            return font
        size -= 4
    return load_font(preferred_font_path, min_size, fallback=fallback)

def render_card(row: pd.Series, out_dir: str) -> str:
    W, H = 1080, 1920
    PAD = 64
    img = Image.new("RGB", (W, H), BRAND_BG)
    draw = ImageDraw.Draw(img)

    font_title   = load_font(FONT_BOLD_TTF, 76)
    font_league  = load_font(FONT_BOLD_TTF, 64)
    font_label   = load_font(FONT_BOLD_TTF, 54)
    font_text    = load_font(FONT_REG_TTF,  52)
    def th(t, f): return draw.textbbox((0,0), t, font=f)[3]

    raw_league = str(row.get("league",""))
    league = prettify_league(raw_league)
    market = str(row.get("market","")).replace("_"," ").title()
    pick   = str(row.get("pick","")).strip()
    odds   = str(row.get("odds","")).strip()
    conf   = str(row.get("confidence","")).title()
    gdt    = row.get("__game_dt")
    gtime  = gdt.astimezone(LOCAL_TZ).strftime("%a • %b %d • %I:%M %p") if isinstance(gdt, datetime) else ""
    conf_emoji = "✅" if conf.lower()=="high" else ("🟡" if conf.lower()=="medium" else "⚪")

    # Header
    hdr_h = 200
    draw.rounded_rectangle([(PAD, PAD), (W-PAD, PAD+hdr_h)], radius=24, fill=BRAND_PANEL)
    x = PAD + 28
    y = PAD + 32
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        ratio = 64 / logo.height
        logo = logo.resize((int(logo.width*ratio), 64), Image.LANCZOS)
        img.paste(logo, (x, y), logo)
        x += logo.width + 20
    except Exception:
        pass
    header_text = os.getenv("CARD_HEADER", "SharpsSignal • Pick of the Day")
    draw.text((x, PAD + 56), header_text, font=font_title, fill=BRAND_TEXT)

    # Main panel
    top = PAD + hdr_h + 32
    draw.rounded_rectangle([(PAD, top), (W-PAD, H-240)], radius=28, fill=BRAND_CARD)

    # League
    draw.text((PAD+40, top+56), league, font=font_league, fill=BRAND_ACCENT)

    # Headline
    y_ptr = top + 150
    headline = f"{market}: {pick}".strip(": ")
    draw.text((PAD+40, y_ptr), headline, font=font_title, fill=BRAND_TEXT)
    y_ptr += th(headline, font_title) + 24

    # Pills
    pill_h = 112
    inner  = 26
    gap    = 24
    x_ptr  = PAD + 40
    def draw_pill(text):
        nonlocal x_ptr
        if not text: return
        w = draw.textbbox((0,0), text, font=font_label)[2]
        draw.rounded_rectangle([(x_ptr, y_ptr), (x_ptr + w + inner*2, y_ptr + pill_h)], radius=22, fill=BRAND_PANEL)
        draw.text((x_ptr + inner, y_ptr + (pill_h - th(text, font_label))//2), text, font=font_label, fill=BRAND_TEXT)
        x_ptr += w + inner*2 + gap

    if odds:
        draw_pill(f"Odds: {odds}")
    draw_pill(f"Confidence: {conf_emoji} {conf}")

    y_ptr += pill_h + 28
    gt_text = f"Game Time: {gtime}" if gtime else "Game Time: —"
    gw = draw.textbbox((0,0), gt_text, font=font_label)[2]
    left_x = PAD + 40
    right_x = W - PAD - 40
    draw.rounded_rectangle([(left_x, y_ptr), (min(left_x + gw + inner*2, right_x), y_ptr + pill_h)], radius=22, fill=BRAND_PANEL)
    draw.text((left_x + inner, y_ptr + (pill_h - th(gt_text, font_label))//2), gt_text, font=font_label, fill=BRAND_TEXT)

    # Footer
    foot_y = H - 220
    draw.text((PAD+40, foot_y-64), "Free picks & realtime alerts:", font=font_text, fill=BRAND_SUBTLE)
    draw.text((PAD+40, foot_y), os.getenv("SUBSCRIBE_URL", "https://sharps-signal.com/subscribe"), font=font_label, fill=BRAND_TEXT)


    os.makedirs(out_dir, exist_ok=True)
    base = f"{prettify_league(raw_league)}_{market}_{pick}".replace(" ", "_").replace("/", "-")
    out_path = os.path.join(out_dir, f"{base}_{int(time.time())}.jpg")
    img.save(out_path, "JPEG", quality=95)
    return out_path

# ---------------- Optional image+social runner (unchanged behavior) ----------------
def post_to_x(image_path: str, caption: str) -> str:
    if not X_ENABLED:
        logging.info("X posting disabled; skipping.")
        return ""
    try:
        # v1.1: media upload
        auth = tweepy.OAuth1UserHandler(
            X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
        )
        api = tweepy.API(auth)
        media = api.media_upload(filename=image_path)
        # v2: create tweet with media
        client = tweepy.Client(
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET,
        )
        resp = client.create_tweet(text=caption, media_ids=[media.media_id_string])
        tweet_id = str(resp.data["id"])
        logging.info(f"Posted to X: https://x.com/i/web/status/{tweet_id}")
        return tweet_id
    except Exception:
        logging.exception("Failed to post to X")
        return ""

# --- TikTok helpers (kept concise) ---
def _tt_refresh():
    global TT_ACCESS_TOKEN, TT_REFRESH_TOKEN
    if not TT_REFRESH_TOKEN:
        return False
    r = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        data={
            "client_key": TT_CLIENT_KEY,
            "client_secret": TT_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": TT_REFRESH_TOKEN,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30
    )
    j = r.json()
    if "access_token" in j:
        TT_ACCESS_TOKEN  = j["access_token"]
        TT_REFRESH_TOKEN = j.get("refresh_token", TT_REFRESH_TOKEN)
        logging.info("TikTok: refreshed access token.")
        return True
    logging.warning(f"TikTok: refresh failed: {j}")
    return False

def post_photo_to_tiktok(image_path: str, caption: str) -> str:
    if not TT_ENABLED:
        logging.info("TikTok disabled; skipping.")
        return ""
    if not TT_ACCESS_TOKEN:
        logging.error("TikTok: missing TT_ACCESS_TOKEN; run get_tt_token.py first.")
        return ""
    headers = {"Authorization": f"Bearer {TT_ACCESS_TOKEN}", "Content-Type": "application/json"}

    # init
    init = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers=headers,
        json={
            "post_info": {"title": caption[:2200], "privacy_level": TT_PRIVACY_LEVEL},
            "source_info": {"source": "FILE_UPLOAD", "video_size": os.path.getsize(image_path), "chunk_size": os.path.getsize(image_path), "total_chunk_count": 1},
        },
        timeout=60
    )
    if init.status_code == 401 and _tt_refresh():
        headers["Authorization"] = f"Bearer {TT_ACCESS_TOKEN}"
        init = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers=headers,
            json={
                "post_info": {"title": caption[:2200], "privacy_level": TT_PRIVACY_LEVEL},
                "source_info": {"source": "FILE_UPLOAD", "video_size": os.path.getsize(image_path), "chunk_size": os.path.getsize(image_path), "total_chunk_count": 1},
            },
            timeout=60
        )
    js = init.json()
    upload_url = ((js.get("data") or {}).get("upload_url"))
    publish_id = ((js.get("data") or {}).get("publish_id"))
    if not upload_url or not publish_id:
        logging.error(f"TikTok init failed: {js}")
        return ""

    # upload
    with open(image_path, "rb") as f:
        up = requests.put(upload_url, data=f, headers={"Content-Type": "video/mp4"}, timeout=300)
    if up.status_code not in (200, 204):
        logging.error(f"TikTok upload failed: {up.status_code} {up.text[:200]}")
        return ""

    # status poll
    status_url = "https://open.tiktokapis.com/v2/post/publish/status/"
    for _ in range(24):
        time.sleep(5)
        st = requests.post(status_url, headers=headers, json={"publish_id": publish_id}, timeout=30)
        if st.status_code == 401 and _tt_refresh():
            headers["Authorization"] = f"Bearer {TT_ACCESS_TOKEN}"
            st = requests.post(status_url, headers=headers, json={"publish_id": publish_id}, timeout=30)
        js = st.json()
        state = ((js.get("data") or {}).get("status") or "").upper()
        if state == "PUBLISH_COMPLETED":
            logging.info(f"TikTok publish complete (publish_id={publish_id})")
            return publish_id
        if state == "PUBLISH_FAILED":
            logging.error(f"TikTok publish failed: {js}")
            return ""
    logging.warning(f"TikTok publish timed out (publish_id={publish_id})")
    return publish_id

# ---------------- Simple runner (optional) ----------------
def run_once():
    logging.info(f"Flags -> DRY_RUN={DRY_RUN}  X_ENABLED={X_ENABLED}")
    try:
        df = read_picks()
    except Exception:
        logging.exception("Failed to read picks")
        return
    chosen = choose_postworthy(df)
    chosen = chosen.dropna(subset=["pick"]).query("pick.str.strip() != ''", engine="python")
    chosen = chosen.sort_values(["__game_dt"]).drop_duplicates(subset=["league","market","pick"], keep="last")
    if chosen.empty:
        logging.info("No picks in the posting window; nothing to post.")
        return
    for _, row in chosen.iterrows():
        key = f'{row.get("league")}|{row.get("market")}|{row.get("pick")}|' \
              f'{row.get("__game_dt"):%Y-%m-%d %H:%M}' if isinstance(row.get("__game_dt"), datetime) else ""
        if already_posted(key):
            logging.info(f"Skip (already posted): {key}")
            continue
        image_path = render_card(row, ASSETS_DIR)
        caption = build_caption(row)
        logging.info(f"Rendered: {image_path}")
        logging.info(f"Caption:\n{caption}")
        if not DRY_RUN:
            tweet_id = post_to_x(image_path, caption)
            if tweet_id:
                mark_posted(key, tweet_id)

def run_loop():
    logging.info("Starting loop (checks every 15 minutes). Ctrl+C to stop.")
    try:
        while True:
            run_once()
            time.sleep(15 * 60)
    except KeyboardInterrupt:
        logging.info("Stopped.")

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    interval_min = int(os.getenv("LOOP_INTERVAL_MIN", "10"))
    if args.once:
        run_once(); sys.exit(0)
    if args.loop:
        logging.info(f"Loop mode: polling every {interval_min} min")
        while True:
            try:
                run_once()
            except Exception:
                logging.exception("Loop iteration failed")
            time.sleep(interval_min * 60)
    else:
        run_once()
