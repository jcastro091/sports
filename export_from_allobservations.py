#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_from_allobservations.py
Builds:
  - model_picks.json
  - pregame_closes.json

Reads your Google Sheet (ConfirmedBets / AllObservations) and exports:
- Event IDs canonicalized EXACTLY like live_line_engine:
    "{SPORT}|{canon(Home)}|{canon(Away)}"
- Selection keys in engine format:
    ML     -> "HOME" / "AWAY"
    Total  -> "OVER x.x" / "UNDER x.x"
    Spread -> "HOME ±x.x" / "AWAY ±x.x"

Usage:
  set GSHEETS_JSON=/path/to/service_account.json
  python export_from_allobservations.py --sheet-id "<SHEET_ID>" --tab "AllObservations" --out-dir .

Columns (flexible names supported):
  Sport, Home, Away, Market, Direction, Total Line,
  Spread Line Home, Spread Line Away,
  Closing Odds (Am)  (or Closing Odds (AM) / Close Am / closing_odds_am),
  Predicted (or Prediction)
  Confidence (optional)
"""

import os
import re
import json
import argparse
import unicodedata
from typing import Any, Dict, List, Optional

# ----- Google Sheets auth (service account) -----
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None
    ServiceAccountCredentials = None


# ============== helpers ==============

SPORT_ALIASES = {
    # sheet value        -> engine value
    "baseball_mlb": "MLB",
    "americanfootball_nfl": "NFL",
    "basketball_wnba": "WNBA",
    "basketball_nba": "NBA",
    "soccer_usa_mls": "Soccer",
    # pass through “nice” labels too
    "mlb": "MLB",
    "nfl": "NFL",
    "wnba": "WNBA",
    "nba": "NBA",
    "soccer": "Soccer",
}

def norm_sport(s: str) -> str:
    s = (s or "").strip()
    low = s.lower()
    return SPORT_ALIASES.get(low, s if s in SPORT_ALIASES.values() else s.upper() if low == s else s)

def canon_name(s: str) -> str:
    """Canonicalize names exactly like engine: strip accents, punctuation, common soccer suffixes, lowercase."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    for junk in [" football club", " futbol club", " club de foot", " club", " fc", " sc", ".", ","]:
        s = s.replace(junk, "")
    s = s.replace("&", "and")
    return " ".join(s.split())

def first(row: Dict[str, Any], names: List[str]) -> Any:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None

def to_float(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        # handle '+3.5', '−110', etc.
        return float(str(x).replace("−", "-").strip())
    except Exception:
        return None

def to_int_american(x: Any) -> Optional[int]:
    f = to_float(x)
    if f is None:
        return None
    return int(round(f))


def ml_role_from_pred(pred: str, home_c: str, away_c: str) -> Optional[str]:
    t = (pred or "").strip()
    up = t.upper()
    if up in ("HOME", "AWAY"):
        return up
    t_c = canon_name(t)
    if t_c == home_c:
        return "HOME"
    if t_c == away_c:
        return "AWAY"
    return None


# ============== export logic ==============

def export(sheet_id: str, tab: str, out_dir: str) -> None:
    # auth
    if gspread is None or ServiceAccountCredentials is None:
        raise RuntimeError("gspread / oauth2client not installed. `pip install gspread oauth2client`")

    creds_path = os.getenv("GSHEETS_JSON", "")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError("Set GSHEETS_JSON to your service account JSON path.")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(tab)

    rows = ws.get_all_records()  # list[dict]

    mp: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}  # event -> market -> sel -> info
    pc: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    # header name variants
    H_SPORT      = ["Sport"]
    H_HOME       = ["Home"]
    H_AWAY       = ["Away"]
    H_MARKET     = ["Market"]
    H_DIRECTION  = ["Direction"]
    H_TOTAL_LINE = ["Total Line", "TotalLine"]
    H_SPR_HOME   = ["Spread Line Home", "Spread Home", "SpreadLineHome"]
    H_SPR_AWAY   = ["Spread Line Away", "Spread Away", "SpreadLineAway"]
    H_CLOSE_AM   = ["Closing Odds (Am)", "Closing Odds (AM)", "Close Am", "closing_odds_am"]
    H_PRED       = ["Predicted", "Prediction"]
    H_CONF       = ["Confidence", "Model Confidence"]

    add_mp = 0
    add_pc = 0
    seen_events = set()

    for r in rows:
        sport_raw = first(r, H_SPORT)
        home_raw  = first(r, H_HOME)
        away_raw  = first(r, H_AWAY)
        market_raw = (first(r, H_MARKET) or "").strip().lower()

        if not sport_raw or not home_raw or not away_raw or not market_raw:
            continue

        sport = norm_sport(str(sport_raw))
        home_c = canon_name(str(home_raw))
        away_c = canon_name(str(away_raw))
        event_id = f"{sport}|{home_c}|{away_c}"
        seen_events.add(event_id)

        direction = (first(r, H_DIRECTION) or "").strip().lower()
        total_line = to_float(first(r, H_TOTAL_LINE))
        spread_home = to_float(first(r, H_SPR_HOME))
        spread_away = to_float(first(r, H_SPR_AWAY))
        close_am = to_int_american(first(r, H_CLOSE_AM))
        pred = first(r, H_PRED) or ""
        conf = first(r, H_CONF)

        # Build MODEL picks
        mp.setdefault(event_id, {})

        if market_raw.startswith("h2h") or "ml" in market_raw or "moneyline" in market_raw:
            role = ml_role_from_pred(str(pred), home_c, away_c)
            if role:
                mp[event_id].setdefault("ML", {})[role] = {"confidence": conf}
                add_mp += 1

        elif market_raw.startswith("total"):
            if direction in ("over", "under") and total_line is not None:
                sel = f"{direction.upper()} {total_line:.1f}"
                mp[event_id].setdefault("Total", {})[sel] = {"confidence": conf}
                add_mp += 1

        elif market_raw.startswith("spread"):
            # Prefer explicit "Spread Line Home/Away"; otherwise infer role from prediction
            if "home" in market_raw and spread_home is not None:
                sel = f"HOME {spread_home:+.1f}"
                mp[event_id].setdefault("Spread", {})[sel] = {"confidence": conf}
                add_mp += 1
            elif "away" in market_raw and spread_away is not None:
                sel = f"AWAY {spread_away:+.1f}"
                mp[event_id].setdefault("Spread", {})[sel] = {"confidence": conf}
                add_mp += 1
            else:
                role = ml_role_from_pred(str(pred), home_c, away_c)
                if role == "HOME" and spread_home is not None:
                    sel = f"HOME {spread_home:+.1f}"
                    mp[event_id].setdefault("Spread", {})[sel] = {"confidence": conf}
                    add_mp += 1
                elif role == "AWAY" and spread_away is not None:
                    sel = f"AWAY {spread_away:+.1f}"
                    mp[event_id].setdefault("Spread", {})[sel] = {"confidence": conf}
                    add_mp += 1

        # Build PREGAME CLOSES
        pc.setdefault(event_id, {})
        if close_am is not None:
            if market_raw.startswith("h2h") or "ml" in market_raw or "moneyline" in market_raw:
                # store under the role we know (from prediction) if possible; otherwise store both if we know nothing
                role = ml_role_from_pred(str(pred), home_c, away_c)
                pc[event_id].setdefault("ML", {})
                if role:
                    pc[event_id]["ML"][role] = {"selection": role, "line": None, "odds": close_am}
                    add_pc += 1
                else:
                    # if we cannot infer role, write both — engine will only read the one it needs
                    pc[event_id]["ML"]["HOME"] = {"selection": "HOME", "line": None, "odds": close_am}
                    pc[event_id]["ML"]["AWAY"] = {"selection": "AWAY", "line": None, "odds": close_am}
                    add_pc += 2

            elif market_raw.startswith("total") and total_line is not None and direction in ("over", "under"):
                sel = f"{direction.upper()} {total_line:.1f}"
                pc[event_id].setdefault("Total", {})[sel] = {"selection": sel, "line": total_line, "odds": close_am}
                add_pc += 1

            elif market_raw.startswith("spread"):
                pc[event_id].setdefault("Spread", {})
                if "home" in market_raw and spread_home is not None:
                    sel = f"HOME {spread_home:+.1f}"
                    pc[event_id]["Spread"][sel] = {"selection": sel, "line": spread_home, "odds": close_am}
                    add_pc += 1
                elif "away" in market_raw and spread_away is not None:
                    sel = f"AWAY {spread_away:+.1f}"
                    pc[event_id]["Spread"][sel] = {"selection": sel, "line": spread_away, "odds": close_am}
                    add_pc += 1
                else:
                    role = ml_role_from_pred(str(pred), home_c, away_c)
                    if role == "HOME" and spread_home is not None:
                        sel = f"HOME {spread_home:+.1f}"
                        pc[event_id]["Spread"][sel] = {"selection": sel, "line": spread_home, "odds": close_am}
                        add_pc += 1
                    elif role == "AWAY" and spread_away is not None:
                        sel = f"AWAY {spread_away:+.1f}"
                        pc[event_id]["Spread"][sel] = {"selection": sel, "line": spread_away, "odds": close_am}
                        add_pc += 1

    # write files
    os.makedirs(out_dir, exist_ok=True)
    mp_path = os.path.join(out_dir, "model_picks.json")
    pc_path = os.path.join(out_dir, "pregame_closes.json")

    with open(mp_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, ensure_ascii=False, separators=(",", ":"))
    with open(pc_path, "w", encoding="utf-8") as f:
        json.dump(pc, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[ok] Wrote {mp_path} (events: {len(mp)})  | model entries added: {add_mp}")
    print(f"[ok] Wrote {pc_path} (events: {len(pc)})  | close entries added: {add_pc}")
    # quick hint for debugging joins
    sample = next(iter(mp.keys()), None)
    if sample:
        print(f"[hint] sample event id: {sample}")


def parse_args():
    p = argparse.ArgumentParser(description="Export model picks and pregame closes from Google Sheet.")
    p.add_argument("--sheet-id", required=True, help="Google Sheet ID (the long id in the URL)")
    p.add_argument("--tab", default="AllObservations", help="Worksheet/tab name (default: AllObservations)")
    p.add_argument("--out-dir", default=".", help="Output directory (default: current dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.sheet_id, args.tab, args.out_dir)
