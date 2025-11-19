#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_line_engine.py — SharpsSignal live line gating (API-conscious, sport-agnostic)

Phase-1 behavior:
- Line-shops live odds (Caesars/BetMGM/FanDuel by default).
- De-vigs two-way markets (ML / Totals / Spreads).
- Requires agreement with your pregame model pick AND positive CLV vs pregame close.
- Emits alerts (CSV + optional Telegram + optional Google Sheets).

Event ID convention (canonicalized to avoid name mismatches):
    "{SPORT}|{canon(Home)}|{canon(Away)}"
"""

import os
import sys
import time
import json
import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import unicodedata

# -------- optional deps --------
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None


# ======================= CONFIG DEFAULTS =======================

CONFIG = {
    # Odds API
    "ODDS_API_KEY": os.getenv("ODDS_API_KEY", ""),

    # Regions/markets depend on your provider; adjust as needed
    "regions": ["us", "us2"],

    # YOUR THREE BOOKS by default (override via --books)
    "bookmakers": ["caesars", "betmgm", "fanduel"],

    # Provider sport keys
    "SPORT_KEYS": {
        "NFL": "americanfootball_nfl",
        "WNBA": "basketball_wnba",
        "NBA": "basketball_nba",
        "MLB": "baseball_mlb",
        "Soccer": "soccer_usa_mls",
        "MMA": "mma_mixed_martial_arts",
    },

    # Markets to evaluate
    "MARKETS": ["ML", "Total", "Spread"],

    # Polling cadence (seconds) — API-conscious
    "POLLING": {"idle": 30, "live": 12, "window": 6, "error_backoff": [30, 60, 120, 300]},

    # Gating thresholds
    "STALE_SEC": 8,         # reject quotes older than this during live (indirect via dispersion)
    "MIN_BOOKS": 2,         # need at least N books with both sides to de-vig
    "DISPERSION_BPS": 75,   # max dispersion of implied prob across books (basis points)
    "REQUIRE_POSITIVE_CLV": True,

    # Time remaining guards (seconds)
    "TIME_LEFT_THRESH": {"NFL": 120, "WNBA": 90, "NBA": 90, "MLB": 9999, "Soccer": 600, "MMA": 9999},

    # Per-event cooldown after an alert (seconds)
    "ALERT_COOLDOWN": 240,

    # Logging
    "CSV_PATH": "AllLiveAlerts.csv",

    # Optional Telegram sink
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),

    # Optional Google Sheets sink
    "GSHEETS_JSON": os.getenv("GSHEETS_JSON", ""),
    "GSHEET_NAME": "ConfirmedBets",
    "GSHEET_TAB": "AllLiveAlerts",
}


# ======================= HELPERS / NORMALIZATION =======================

def _model_debug_blob(model_store, close_store, event_id, market, sel_key):
    mc = getattr(model_store, "_cache", {})
    cc = getattr(close_store, "_cache", {})
    e_has = event_id in mc
    e_markets = list(mc.get(event_id, {}).keys()) if e_has else []
    m_keys = list(mc.get(event_id, {}).get(market, {}).keys()) if market in mc.get(event_id, {}) else []
    c_has = event_id in cc
    c_markets = list(cc.get(event_id, {}).keys()) if c_has else []
    c_keys = list(cc.get(event_id, {}).get(market, {}).keys()) if market in cc.get(event_id, {}) else []
    return {
        "model": {"has_event": e_has, "markets": e_markets, "keys_for_market": m_keys},
        "close": {"has_event": c_has, "markets": c_markets, "keys_for_market": c_keys},
        "sel_key_checked": sel_key
    }


def canon_name(s: str) -> str:
    """
    Canonicalize team/fighter names so Sheet and feed match (strip accents, suffixes, punctuation).
    Examples: "CF Montréal" -> "cf montreal", "LA Galaxy" -> "la galaxy", "St. Louis City SC" -> "st louis city"
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # drop common soccer suffixes/prefixes that vary across feeds
    for junk in [" football club", " futbol club", " club de foot", " club", " fc", " sc", ".", ","]:
        s = s.replace(junk, "")
    s = s.replace("&", "and")
    return " ".join(s.split())  # collapse whitespace


def american_to_prob(odds: int) -> float:
    """American odds → implied probability (with vig)."""
    return 100.0 / (odds + 100.0) if odds >= 100 else -odds / (-odds + 100.0)


def devig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    """Remove overround for a two-way book; return no-vig probabilities for side A and B."""
    s = p1 + p2
    return (0.0, 0.0) if s == 0 else (p1 / s, p2 / s)


def dispersion_bps(values: List[float]) -> float:
    """Simple dispersion (max-min) in probability space → basis points (0.01%)."""
    if not values:
        return 9999.0
    return (max(values) - min(values)) * 10000.0


# ======================= DATA MODELS =======================

@dataclass
class BookQuote:
    book: str
    market: str            # "ML" / "Total" / "Spread"
    selection: str         # "HOME", "AWAY", "OVER 42.5", "HOME -3.5", etc.
    line: Optional[float]  # spread/total number; None for ML
    odds: int              # American
    ts: float              # epoch seconds
    suspended: bool = False
    limit: Optional[float] = None


@dataclass
class CloseLine:
    market: str
    selection: str
    line: Optional[float]
    odds: int


@dataclass
class ModelPick:
    market: str
    selection: str
    confidence: Optional[float] = None


@dataclass
class EventState:
    sport: str
    event_id: str
    league: str
    status: str              # "pregame", "live", "halftime", "post"
    clock_sec_left: int
    window: bool             # betting window (stoppage) — MVP uses True
    raw: Dict[str, Any] = field(default_factory=dict)


# ======================= PROVIDER (The Odds API v4) =======================

def norm_sel_ml(team: str, home: str, away: str) -> str:
    return "HOME" if team == home else ("AWAY" if team == away else team.upper())


def norm_sel_total(name: str, point: float) -> str:
    return f"{name.upper()} {float(point):.1f}"


def norm_sel_spread(team: str, point: float, home: str, away: str) -> str:
    who = "HOME" if team == home else ("AWAY" if team == away else team.upper())
    return f"{who} {float(point):+.1f}"  # NOTE: +.1f (no space)


class OddsProvider:
    """Odds API adapter (single call per sport; returns board: event_id -> quotes)."""

    def __init__(self, api_key: str, regions: List[str], bookmakers: List[str]):
        self.api_key = api_key
        self.regions = regions
        self.bookmakers = bookmakers

    async def fetch_event_board(
        self,
        session: "aiohttp.ClientSession",
        sport_key: str,
        sport_label: str,
        live_only: bool = False
    ) -> Dict[str, List[BookQuote]]:
        if not self.api_key:
            return {}

        base = f"https://api.the-odds-api.com/v4/sports/{sport_key}"
        endpoint = f"{base}/odds-live" if live_only else f"{base}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ",".join(self.regions),
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": ",".join(self.bookmakers),
            "dateFormat": "iso",
        }

        board: Dict[str, List[BookQuote]] = {}
        try:
            async with session.get(endpoint, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
        except Exception:
            return {}

        now = time.time()
        for ev in data:
            # Guard: ignore games that haven't started when live_only (allow within 5m of start)
            commence = ev.get("commence_time")
            try:
                t0 = datetime.fromisoformat(str(commence).replace("Z", "+00:00")).timestamp()
            except Exception:
                t0 = None
            if live_only and (t0 is None or now + 5*60 < t0):
                continue

            home_raw = ev.get("home_team")
            away_raw = ev.get("away_team")
            if not home_raw or not away_raw:
                continue  # skip malformed events

            # Canonical names for ID and ML role mapping
            home_c = canon_name(home_raw)
            away_c = canon_name(away_raw)
            event_id = f"{sport_label}|{home_c}|{away_c}"

            quotes: List[BookQuote] = []
            for bk in (ev.get("bookmakers") or []):
                book = (bk.get("key") or "").lower()
                for m in (bk.get("markets") or []):
                    mk = m.get("key")               # 'h2h' | 'spreads' | 'totals'
                    outcomes = m.get("outcomes") or []
                    last = m.get("last_update") or bk.get("last_update") or commence
                    try:
                        ts = datetime.fromisoformat(str(last).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        ts = now

                    if mk == "h2h":
                        for oc in outcomes:
                            team, price = oc.get("name"), oc.get("price")
                            if team in (home_raw, away_raw) and isinstance(price, (int, float)):
                                quotes.append(BookQuote(book, "ML", norm_sel_ml(team, home_raw, away_raw), None, int(price), ts))

                    elif mk == "totals":
                        for oc in outcomes:
                            name, price, point = oc.get("name"), oc.get("price"), oc.get("point")
                            if name and point is not None and isinstance(price, (int, float)):
                                quotes.append(BookQuote(book, "Total", norm_sel_total(name, float(point)), float(point), int(price), ts))

                    elif mk == "spreads":
                        for oc in outcomes:
                            team, price, point = oc.get("name"), oc.get("price"), oc.get("point")
                            if team and point is not None and isinstance(price, (int, float)):
                                quotes.append(BookQuote(book, "Spread", norm_sel_spread(team, float(point), home_raw, away_raw), float(point), int(price), ts))

            if quotes:
                board[event_id] = quotes

        return board


# ======================= STORES (closes + model picks) =======================

class CloseStore:
    """Read pregame closes from pregame_closes.json"""

    def __init__(self, path: str = "pregame_closes.json"):
        self.path = path
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def get_close(self, event_id: str, market: str, selection_key: str) -> Optional[CloseLine]:
        rec = (self._cache.get(event_id, {}).get(market, {})).get(selection_key)
        if not rec:
            return None
        return CloseLine(market, rec.get("selection", selection_key), rec.get("line"), int(rec.get("odds", 0)))


class ModelStore:
    """Read pregame model picks from model_picks.json"""

    def __init__(self, path: str = "model_picks.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def get_pick(self, event_id: str, market: str) -> Optional[ModelPick]:
        recs = self._cache.get(event_id, {}).get(market)
        if not recs:
            return None
        sel, conf = None, None
        for k, v in recs.items():
            c = v.get("confidence")
            if conf is None or (c is not None and c > conf):
                sel, conf = k, c
        return None if sel is None else ModelPick(market, sel, conf)


# ======================= MODEL AGREEMENT HELPERS =======================

def normalize_model_selection_for_event(event_id: str, market: str, model_sel: str) -> Optional[str]:
    """
    Map model selection (which might be team/fighter names) to engine keys.
    event_id format: 'SPORT|home_canon|away_canon'
    Returns normalized selection key for comparison:
      - ML: 'HOME'/'AWAY'
      - Total: 'OVER'/'UNDER' (role-only ok; line is matched via selection_key later)
      - Spread: 'HOME'/'AWAY' (role-only ok; line matched via selection_key)
    """
    try:
        _, home_c, away_c = event_id.split("|", 2)
    except ValueError:
        home_c = away_c = ""

    t = (model_sel or "").strip()
    up = t.upper()

    if market == "ML":
        if up in ("HOME", "AWAY"):
            return up
        if canon_name(t) == home_c:
            return "HOME"
        if canon_name(t) == away_c:
            return "AWAY"
        return None

    if market == "Total":
        return up if up in ("OVER", "UNDER") else None

    if market == "Spread":
        if up in ("HOME", "AWAY"):
            return up
        if canon_name(t) == home_c:
            return "HOME"
        if canon_name(t) == away_c:
            return "AWAY"
        return None

    return None


# ======================= GATING HELPERS =======================

def normalize_selection_key(_: str, selection: str) -> str:
    return selection.strip().upper()


def compute_no_vig_prob_for_selection(book_side_odds: Dict[str, Tuple[int, int]]) -> Optional[float]:
    """Average the no-vig probability across books for the target selection."""
    probs = []
    for _, (odds_a, odds_b) in book_side_odds.items():
        p_a, p_b = american_to_prob(odds_a), american_to_prob(odds_b)
        nv_a, _ = devig_two_way(p_a, p_b)
        probs.append(nv_a)
    if not probs:
        return None
    return sum(probs) / len(probs)


def pick_best_display_price(book_side_odds: Dict[str, Tuple[int, int]]) -> Tuple[str, int]:
    """Pick the best bettor-facing price for display (highest EV)."""
    best_book, best_odds = None, -10 ** 9
    for book, (odds_a, _) in book_side_odds.items():
        val = odds_a if odds_a > 0 else -1_000_000 + odds_a  # prefer larger +, less negative −
        if best_book is None or val > best_odds:
            best_book, best_odds = book, odds_a
    return best_book, best_odds


def should_alert(
    event: EventState,
    market: str,
    selection_key: str,
    live_pairs: Dict[str, Dict[str, Tuple[int, int]]],
    close_line: Optional[CloseLine],
    model_pick: Optional[ModelPick],
    cfg=CONFIG
) -> Tuple[bool, str, Dict[str, Any]]:

    if model_pick is None:
        return False, "Model disagrees or missing", {}

    # Exact match first
    if normalize_selection_key(market, model_pick.selection) == selection_key:
        pass
    else:
        # Allow role-only model picks:
        sel_up = selection_key.upper()
        mp_up = (model_pick.selection or "").upper()
        ok_role = False
        if market == "ML" and mp_up in ("HOME", "AWAY"):
            ok_role = mp_up in sel_up
        elif market == "Total" and mp_up in ("OVER", "UNDER"):
            ok_role = sel_up.startswith(mp_up + " ")
        elif market == "Spread" and mp_up in ("HOME", "AWAY"):
            ok_role = sel_up.startswith(mp_up + " ")
        if not ok_role:
            return False, "Model disagrees or missing", {}

    # Enough books to de-vig
    side_books = live_pairs.get(selection_key, {})
    if len(side_books) < cfg["MIN_BOOKS"]:
        return False, "Not enough books", {}

    # Live no-vig probability
    p_live_novig = compute_no_vig_prob_for_selection(side_books)
    if p_live_novig is None:
        return False, "No-vig calc failed", {}

    # Closing price (no-vig proxy via sharp close)
    if close_line is None:
        return False, "No closing line", {}
    p_close_novig = american_to_prob(close_line.odds)  # If you store both sides at close, devig them here.

    # Positive CLV vs close (lower prob now = better price for us)
    clv_close_prob = p_close_novig - p_live_novig
    if cfg["REQUIRE_POSITIVE_CLV"] and clv_close_prob <= 0:
        return False, "No positive CLV vs close", {}

    # Consensus dispersion guard
    imps = [american_to_prob(o[0]) for o in side_books.values()]
    if dispersion_bps(imps) > cfg["DISPERSION_BPS"]:
        return False, "Consensus too loose (dispersion)", {}

    # Time remaining guard
    min_left = cfg["TIME_LEFT_THRESH"].get(event.sport, 60)
    if event.clock_sec_left is not None and event.clock_sec_left < min_left:
        return False, "Too little time left", {}

    # Window guard (MVP: engine sets window=True; wire actual stoppages later)
    if not event.window:
        return False, "Not in betting window", {}

    best_book, best_display_odds = pick_best_display_price(side_books)
    return True, "OK", {
        "best_book": best_book,
        "best_odds": best_display_odds,
        "p_live_novig": p_live_novig,
        "p_close_novig": p_close_novig,
        "clv_close_prob": clv_close_prob,
        "selection_key": selection_key,
    }


# ======================= SINKS (CSV / Telegram / Sheets) =======================

def ts_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv_row(path: str, row: Dict[str, Any]):
    header = [
        "ts_alert", "sport", "league", "event_id", "market", "selection_key",
        "model_selection", "model_confidence", "best_book", "best_odds",
        "p_live_novig", "p_close_novig", "clv_close_prob", "clock_sec_left", "window_type",
        "status_fired", "reason_if_blocked", "books_snapshot_json"
    ]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        vals = [
            row.get("ts_alert", ""),
            row.get("sport", ""),
            row.get("league", ""),
            row.get("event_id", ""),
            row.get("market", ""),
            row.get("selection_key", ""),
            row.get("model_selection", ""),
            row.get("model_confidence", ""),
            row.get("best_book", ""),
            str(row.get("best_odds", "")),
            f'{row.get("p_live_novig",""):.6f}' if isinstance(row.get("p_live_novig", None), (int, float)) else "",
            f'{row.get("p_close_novig",""):.6f}' if isinstance(row.get("p_close_novig", None), (int, float)) else "",
            f'{row.get("clv_close_prob",""):.6f}' if isinstance(row.get("clv_close_prob", None), (int, float)) else "",
            str(row.get("clock_sec_left", "")),
            row.get("window_type", ""),
            row.get("status_fired", ""),
            row.get("reason_if_blocked", ""),
            json.dumps(row.get("books_snapshot_json", {}), ensure_ascii=False),
        ]
        f.write(",".join(map(lambda s: str(s).replace(",", ";"), vals)) + "\n")


async def send_telegram(msg: str, cfg=CONFIG):
    token = cfg.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = cfg.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id or aiohttp is None:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        await session.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"})


def append_gsheet(row: Dict[str, Any], cfg=CONFIG):
    if not cfg.get("GSHEETS_JSON") or gspread is None or ServiceAccountCredentials is None:
        return
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(cfg["GSHEETS_JSON"], scope)
        client = gspread.authorize(creds)
        sh = client.open(cfg["GSHEET_NAME"])
        ws = sh.worksheet(cfg["GSHEET_TAB"])
        ws.append_row([
            row.get("ts_alert", ""), row.get("sport", ""), row.get("league", ""), row.get("event_id", ""),
            row.get("market", ""), row.get("selection_key", ""),
            row.get("model_selection", ""), row.get("model_confidence", ""),
            row.get("best_book", ""), row.get("best_odds", ""),
            row.get("p_live_novig", ""), row.get("p_close_novig", ""),
            row.get("clv_close_prob", ""), row.get("clock_sec_left", ""), row.get("window_type", ""),
            row.get("status_fired", ""), row.get("reason_if_blocked", ""),
            json.dumps(row.get("books_snapshot_json", {}), ensure_ascii=False)
        ], value_input_option="RAW")
    except Exception as e:
        print(f"[warn] Google Sheets append failed: {e}", file=sys.stderr)


# ======================= ENGINE =======================

class LiveEngine:
    def __init__(self, sports: List[str], markets: List[str], dry_run: bool = False):
        self.cfg = CONFIG
        self.sports = sports
        self.markets = markets
        self.dry_run = dry_run
        self.provider = OddsProvider(self.cfg["ODDS_API_KEY"], self.cfg["regions"], self.cfg["bookmakers"])
        self.close_store = CloseStore()
        self.model_store = ModelStore()
        self.cooldowns: Dict[str, float] = {}  # event_id -> next_allowed_ts
        self.live_only: bool = False
        self.debug: bool = False

    async def run(self):
        if aiohttp is None:
            print("[error] aiohttp not installed. `pip install aiohttp`", file=sys.stderr)
            return

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            backoff_idx = 0
            while True:
                try:
                    await self.tick_all(session)
                    backoff_idx = 0
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[error] main loop: {e}", file=sys.stderr)
                    delay = self.cfg["POLLING"]["error_backoff"][min(backoff_idx, len(self.cfg["POLLING"]["error_backoff"]) - 1)]
                    backoff_idx += 1
                    await asyncio.sleep(delay)

    async def tick_all(self, session: "aiohttp.ClientSession"):
        for sport in self.sports:
            sport_key = self.cfg["SPORT_KEYS"].get(sport, "")
            if not sport_key:
                continue

            # Fetch a whole board with ONE call per sport (API-conscious)
            board = await self.provider.fetch_event_board(session, sport_key, sport, live_only=self.live_only)

            # MVP: treat as a betting window; wire actual stoppages later.
            now_win = True

            for event_id, quotes in board.items():
                event = EventState(sport, event_id, sport, "live", 9999, now_win, {})
                await self.process_event(event, quotes)

    async def process_event(self, event: EventState, quotes: List[BookQuote]):
        now_ts = time.time()
        if now_ts < self.cooldowns.get(event.event_id, 0):
            return

        # Build per-book maps and pair opposites for devig
        market_map: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {"ML": {}, "Total": {}, "Spread": {}}
        by_book_market: Dict[Tuple[str, str], Dict[str, BookQuote]] = {}

        for q in quotes:
            if q.market not in self.markets:
                continue
            key = normalize_selection_key(q.market, q.selection)
            by_book_market.setdefault((q.book, q.market), {})[key] = q

        for (book, market), sel_map in by_book_market.items():
            if market not in market_map:
                continue
            for sel, q_sel in list(sel_map.items()):
                opp = None
                if market == "ML":
                    if "HOME" in sel:
                        opp = sel.replace("HOME", "AWAY")
                    elif "AWAY" in sel:
                        opp = sel.replace("AWAY", "HOME")

                elif market == "Total":
                    if sel.startswith("OVER "):
                        opp = sel.replace("OVER", "UNDER", 1)
                    elif sel.startswith("UNDER "):
                        opp = sel.replace("UNDER", "OVER", 1)

                elif market == "Spread":
                    try:
                        mag = float(sel.split()[-1])
                        if "HOME" in sel:
                            opp = f"AWAY {(-mag):+.1f}"
                        elif "AWAY" in sel:
                            opp = f"HOME {(-mag):+.1f}"
                    except Exception:
                        opp = None

                if opp and opp in sel_map:
                    q_opp = sel_map[opp]
                    sel_key = normalize_selection_key(market, sel)
                    market_map[market].setdefault(sel_key, {})[book] = (q_sel.odds, q_opp.odds)
                
                
        for market in self.markets:
            pairs = market_map.get(market, {})
            for sel_key, book_side_odds in pairs.items():
                model_pick = self.model_store.get_pick(event.event_id, market)
                if model_pick:
                    norm = normalize_model_selection_for_event(event.event_id, market, model_pick.selection)
                    model_pick = ModelPick(market, norm, model_pick.confidence) if norm else None

                close_line = self.close_store.get_close(event.event_id, market, sel_key)

                ok, reason, meta = should_alert(event, market, sel_key, market_map[market], close_line, model_pick, self.cfg)
                await self.log_and_maybe_alert(event, market, sel_key, model_pick, ok, reason, meta, book_side_odds)
           
           
           
          

                # if ok:
                    # # Per-event cooldown to avoid spamming
                    # self.cooldowns[event.event_id] = time.time() + self.cfg["ALERT_COOLDOWN"]

    async def log_and_maybe_alert(
        self,
        event: EventState,
        market: str,
        sel_key: str,
        model_pick: Optional[ModelPick],
        ok: bool,
        reason: str,
        meta: Dict[str, Any],
        book_side_odds: Dict[str, Tuple[int, int]],
    ):
        row = {
            "ts_alert": ts_iso(),
            "sport": event.sport,
            "league": event.league,
            "event_id": event.event_id,
            "market": market,
            "selection_key": sel_key,
            "model_selection": (model_pick.selection if model_pick else ""),
            "model_confidence": (model_pick.confidence if model_pick else ""),
            "best_book": meta.get("best_book", ""),
            "best_odds": meta.get("best_odds", ""),
            "p_live_novig": meta.get("p_live_novig", ""),
            "p_close_novig": meta.get("p_close_novig", ""),
            "clv_close_prob": meta.get("clv_close_prob", ""),
            "clock_sec_left": event.clock_sec_left,
            "window_type": ("stoppage" if event.window else "live"),
            "status_fired": ("Y" if ok else "N"),
            "reason_if_blocked": ("" if ok else reason),
            "books_snapshot_json": {bk: {"odds_sel": o[0], "odds_opp": o[1]} for bk, o in book_side_odds.items()},
        }

        write_csv_row(self.cfg["CSV_PATH"], row)
        append_gsheet(row, self.cfg)

        if ok:
            msg = (
                f"🔔 <b>{event.sport} Live</b> — {market} | {sel_key}\n"
                f"<b>Best</b>: {meta.get('best_book')} {meta.get('best_odds')}\n"
                f"<b>CLV vs Close</b>: +{meta.get('clv_close_prob')*100:.2f} bps (prob)\n"
                f"<b>Model</b>: {row['model_selection']} (conf={row['model_confidence']})\n"
                f"<b>Time left</b>: ~{event.clock_sec_left}s"
            )
            print(msg)
            await send_telegram(msg, self.cfg)
        else:
            if getattr(self, "debug", False):
                dbg = {
                    "has_model_event": event.event_id in getattr(self.model_store, "_cache", {}),
                    "model_markets": list(getattr(self.model_store, "_cache", {}).get(event.event_id, {}).keys()),
                    "model_for_market": list(getattr(self.model_store, "_cache", {}).get(event.event_id, {}).get(market, {}).keys()) if getattr(self.model_store, "_cache", {}).get(event.event_id, {}).get(market) else [],
                    "has_close_event": event.event_id in getattr(self.close_store, "_cache", {}),
                    "close_markets": list(getattr(self.close_store, "_cache", {}).get(event.event_id, {}).keys()),
                    "close_for_market": list(getattr(self.close_store, "_cache", {}).get(event.event_id, {}).get(market, {}).keys()) if getattr(self.close_store, "_cache", {}).get(event.event_id, {}).get(market) else [],
                    "sel_key": sel_key,
                }
                print(f"skip {event.event_id} {market} {sel_key}: {reason} | dbg={json.dumps(dbg)}")

            else:
                # ALWAYS print a diagnostic blob so you can see exactly what's missing
                dbg = _model_debug_blob(self.model_store, self.close_store, event.event_id, market, sel_key)
                print(f"skip {event.event_id} {market} {sel_key}: {reason} | dbg={json.dumps(dbg)}")



# ======================= CLI =======================

def parse_args():
    p = argparse.ArgumentParser(description="SharpsSignal live line gating (books' live odds + model + CLV).")
    p.add_argument("--sports", type=str, default="NFL,MLB,WNBA,Soccer,MMA",
                   help="Comma list: NFL,MLB,WNBA,Soccer,MMA")
    p.add_argument("--markets", type=str, default="ML,Total,Spread",
                   help="Comma list: ML,Total,Spread")
    p.add_argument("--books", type=str, default="caesars,betmgm,fanduel",
                   help="Comma list of bookmaker keys (default: caesars,betmgm,fanduel)")
    p.add_argument("--live-only", action="store_true",
                   help="Use live odds endpoint and ignore pregame events.")
    p.add_argument("--debug", action="store_true",
                   help="Verbose diagnostics for skipped events.")
    p.add_argument("--dry-run", action="store_true",
                   help="Disable Telegram/Sheets sinks (CSV still logs).")
    return p.parse_args()


def main():
    args = parse_args()
    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    if args.books:
        CONFIG["bookmakers"] = [b.strip().lower() for b in args.books.split(",") if b.strip()]

    if args.dry_run:
        CONFIG["TELEGRAM_BOT_TOKEN"] = ""
        CONFIG["TELEGRAM_CHAT_ID"] = ""
        CONFIG["GSHEETS_JSON"] = ""

    engine = LiveEngine(sports, markets, dry_run=args.dry_run)
    engine.live_only = args.live_only
    engine.debug = args.debug

    # Clean asyncio runner (works well on Windows)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
