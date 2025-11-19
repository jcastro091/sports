# tools/find_kalshi_markets.py
from __future__ import annotations
from pathlib import Path
import sys, re

# --- Path shim: allow running as script or module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from execution.adapters.kalshi_adapter import KalshiAdapter

def load_cfg() -> dict:
    cfg_path = ROOT / "execution" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def gspread_open(cfg: dict):
    sh_cfg = cfg["sheets"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds  = ServiceAccountCredentials.from_json_keyfile_name(sh_cfg["service_account_json"], scopes)
    gc     = gspread.authorize(creds)
    sh     = gc.open_by_key(sh_cfg["spreadsheet_id"])
    ws_obs = sh.worksheet(sh_cfg["obs_worksheet"])
    return ws_obs

def find_event_and_markets(kalshi: KalshiAdapter, home: str, away: str):
    # 1) Find the event first (narrow search)
    q = f"{home} vs {away}"
    ev = kalshi._request("GET", "/events", params={"search": q})
    ev.raise_for_status()
    ev_js = ev.json() or {}
    events = ev_js.get("events") or ev_js.get("data") or []
    # Heuristic: choose the first event whose title/name contains both team names
    home_l, away_l = home.lower(), away.lower()
    target_ev = None
    for e in events:
        title = (e.get("name") or e.get("title") or e.get("display_name") or "").lower()
        if home_l in title and away_l in title:
            target_ev = e
            break
    if not target_ev:
        print(f"No event matched '{q}'. Got {len(events)} candidates.")
        for e in events[:10]:
            print("-", e.get("name") or e.get("title") or e.get("display_name"))
        return None, []

    ev_ticker = target_ev.get("ticker") or target_ev.get("event_ticker") or target_ev.get("id")
    print(f"Event: {target_ev.get('name') or target_ev.get('title')}  | event_ticker={ev_ticker}")

    # 2) Pull markets for that event
    mk = kalshi._request("GET", "/markets", params={"event_ticker": ev_ticker})
    mk.raise_for_status()
    mk_js = mk.json() or {}
    markets = mk_js.get("markets") or mk_js.get("data") or []

    # 3) Filter to Winner / H2H-style markets (heuristics on name/category)
    def is_h2h(m):
        txt = " ".join([
            str(m.get("category","")),
            str(m.get("name","")),
            str(m.get("title","")),
            str(m.get("display_name","")),
            str(m.get("subtitle","")),
        ]).lower()
        return any(k in txt for k in ["winner", "moneyline", "h2h", "reg time"]) and not any(
            k in txt for k in ["total", "spread", "over", "under"]
        )

    h2h = [m for m in markets if is_h2h(m)]
    if not h2h:
        print(f"No H2H markets found for event {ev_ticker}. Showing first 10 markets for debug:")
        for m in markets[:10]:
            print("-", m.get("ticker"), "|", m.get("name") or m.get("title") or m.get("display_name"))
        return target_ev, []

    # 4) Try to classify each H2H market as Paris / Lorient / Tie
    picks = []
    for m in h2h:
        name = (m.get("name") or m.get("title") or m.get("display_name") or "").lower()
        ticker = m.get("ticker") or m.get("market_ticker")
        outcome = ""
        if home_l in name:
            outcome = "HOME"
        elif away_l in name:
            outcome = "AWAY"
        elif re.search(r"\btie\b|\bdraw\b", name):
            outcome = "TIE"
        picks.append({"ticker": ticker, "label": outcome, "name": name})

    print("H2H candidates:")
    for p in picks:
        print(f"- {p['ticker']}  [{p['label']}]  {p['name']}")

    return target_ev, picks

def write_test_row(ws_obs, ticker: str, side_yes: bool, price_cents: int = 1, qty: int = 1, notes: str = "kalshi test (dry run)"):
    # Assumes your watcher expects these headers; create if missing:
    headers = [h.strip() for h in (ws_obs.get_all_values()[0] if ws_obs.row_count else [])]
    need = ["KalshiTicker","KalshiSide","KalshiPrice","KalshiQty","Notes"]
    if not headers:
        ws_obs.append_row(need)
        headers = need
    # ensure missing headers are appended
    for h in need:
        if h not in headers:
            ws_obs.update_cell(1, len(headers)+1, h)
            headers.append(h)

    row = [""] * len(headers)
    def setv(col, val):
        idx = headers.index(col)  # 0-based
        if idx >= len(row):
            row.extend([""] * (idx - len(row) + 1))
        row[idx] = str(val)

    setv("KalshiTicker", ticker)
    setv("KalshiSide",   "YES" if side_yes else "NO")
    setv("KalshiPrice",  price_cents)  # 1..99
    setv("KalshiQty",    qty)
    setv("Notes",        notes)

    ws_obs.append_row(row)
    print("Wrote test row to AllObservations.")

def main():
    cfg = load_cfg()

    kalshi = KalshiAdapter(
        base_url=cfg["kalshi"]["base_url"],
        key_id=cfg["kalshi"]["key_id"],
        private_key_file=cfg["kalshi"]["private_key_file"],
    )

    # Adjust team names to match Kalshi naming best you can
    home, away = "Paris", "Lorient"
    ev, markets = find_event_and_markets(kalshi, home, away)
    if not markets:
        return

    # Pick the HOME entry (Paris) for "Buy Yes · Paris"
    home_pick = next((m for m in markets if m["label"] == "HOME"), None)
    if not home_pick:
        print("Couldn't find a HOME (Paris) market from the candidates above.")
        return

    print(f"Chosen ticker: {home_pick['ticker']}")

    # OPTIONAL: write a fresh row to AllObservations for the watcher (dry run)
    ws_obs = gspread_open(cfg)
    write_test_row(ws_obs, ticker=home_pick["ticker"], side_yes=True, price_cents=1, qty=1)

if __name__ == "__main__":
    main()
