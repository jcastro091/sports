# telegram_confirm_logger_trades.py — Robust TV webhook → Google Sheet + Telegram
# - Accepts JSON, form-encoded, or raw text payloads from TradingView
# - One row per trade (Entry/Refresh keeps row Open; Exit updates same row and closes)
# - Infers entry/exit from pos/price if explicit event is missing
# - Clear Telegram messages for Entry and Exit
# - Healthchecks pings (optional)

import os, json, logging, time, re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

from flask import Flask, request, jsonify
import requests
import threading

from oauth2client.service_account import ServiceAccountCredentials
import gspread

from dateutil import parser as dtparser
import pytz

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN",  "")
TELEGRAM_CHAT_ID     = int(os.getenv("TELEGRAM_CHAT_ID", "0") or 0)

GOOGLE_CREDS_FILE    = os.getenv("GOOGLE_CREDS_FILE",   "telegrambetlogger-35856685bc29.json")
SPREADSHEET_KEY      = os.getenv("SPREADSHEET_KEY", "").strip()     # << use this if you have a key
SPREADSHEET_NAME     = os.getenv("SPREADSHEET_NAME",    "ConfirmedBets")
TAB_TRADES           = os.getenv("TAB_TRADES",          "ConfirmedTrades2")  # or set to ConfirmedBets if you prefer

# Slightly wider tolerance so near-identical entry prints are merged as a refresh (not a new row)
ENTRY_TOLERANCE      = float(os.getenv("ENTRY_TOLERANCE", "0.002"))


# -------- Healthchecks (HC) --------
HC_URL  = os.getenv("HEARTBEAT_URL_TG_TRADES", "").strip()
HC_FILE = os.getenv("HEARTBEAT_FILE_TG_TRADES", "").strip()
_last_hb_bucket = -1

def _hb_touch():
    if not HC_FILE: return
    try:
        Path(HC_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(HC_FILE).write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        pass

def _hb_ping(suffix: str = "", payload: dict | None = None):
    if not HC_URL: return
    u = HC_URL.rstrip("/")
    if suffix: u = f"{u}/{suffix.lstrip('/')}"
    try:
        if payload: requests.post(u, json=payload, timeout=5)
        else:       requests.get(u, timeout=5)
    except Exception:
        pass

def _hb_tick():
    global _last_hb_bucket
    b = int(time.time() // 60)
    if b != _last_hb_bucket:
        _last_hb_bucket = b
        _hb_ping("", {"svc":"tg_trades","ts": int(time.time())})
        _hb_touch()

_hb_ping("start", {"svc":"tg_trades","status":"start","ts":int(time.time())})
_hb_touch()
threading.Thread(target=lambda: ([_hb_tick() or time.sleep(60) for _ in iter(int,1)]), daemon=True).start()

# ------------------ LOGGING / TZ ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
EASTERN = pytz.timezone("US/Eastern")

# ------------------ UTIL ------------------
def to_est_string(iso_utc: str) -> str:
    try:
        dt_utc = dtparser.isoparse(iso_utc)
        dt_est = dt_utc.astimezone(EASTERN)
        return dt_est.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_utc

def near(a, b, tol=ENTRY_TOLERANCE):
    try:
        return abs(float(a) - float(b)) < tol
    except Exception:
        return False
        
        
# Top-level (near other globals)
_recent_entries = {}  # {trade_id: last_ts_epoch}

def seen_recent(trade_id: str, window_sec: int = 15) -> bool:
    now = time.time()
    last = _recent_entries.get(trade_id, 0)
    if now - last < window_sec:
        return True
    _recent_entries[trade_id] = now
    return False

        
        
# Add near the top with other helpers
def _to_float_safe(v):
    if v is None: 
        return None
    s = str(v).strip().rstrip(",} ]")
    try: 
        return float(s.replace(",", ""))
    except Exception:
        return None


def send_telegram(text: str, parse_mode: str = "Markdown"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

# ------------------ GOOGLE SHEETS ------------------
def _open_sheet_retry():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
    gc = gspread.authorize(creds)
    for i in range(3):
        try:
            sh = gc.open_by_key(SPREADSHEET_KEY) if SPREADSHEET_KEY else gc.open(SPREADSHEET_NAME)
            ws = sh.worksheet(TAB_TRADES)
            return ws
        except Exception as e:
            logging.warning(f"Open sheet attempt {i+1}/3 failed: {e}")
            time.sleep(1.0 + i*0.5)
    raise RuntimeError("Failed to open Google Sheet/tab after retries")

def get_sheet(): return _open_sheet_retry()

def get_header(ws): return ws.row_values(1)

def idx(header, name) -> Optional[int]:
    try: return header.index(name)
    except ValueError: return None

def a1(r, c_zero_idx):  # 1-based row, 0-based col -> A1
    return gspread.utils.rowcol_to_a1(r, c_zero_idx + 1)
    
    
def normalize_short_only(out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map TradingView fields to short-only:
      - Use sl_short/tp_short/rr_short → sl/tp/rr
      - Ignore sl_long/tp_long/rr_long
      - Force 'side=short' for non-exit events
    """
    o = dict(out)

    # Map short-only risk params
    if "sl_short" in o and o["sl_short"] not in (None, ""):
        o["sl"] = o["sl_short"]
    if "tp_short" in o and o["tp_short"] not in (None, ""):
        o["tp"] = o["tp_short"]
    if "rr_short" in o and o["rr_short"] not in (None, ""):
        o["rr"] = o["rr_short"]

    # Drop long-only noise
    for k in ("sl_long", "tp_long", "rr_long"):
        if k in o: o.pop(k)

    # Force short on entry-like events
    ev = (o.get("event") or "").lower()
    if ev not in ("tp","sl","exit"):
        o["side"] = "short"

    return o


# ------------------ MATCHING ------------------
def find_open_row(header, rows, symbol: str, side: str, entry: float,
                  trade_id: Optional[str],
                  c_trade_id: Optional[int], c_symbol: Optional[int], c_side: Optional[int],
                  c_entry: Optional[int], c_exit: Optional[int], c_status: Optional[int]) -> Optional[int]:
    if trade_id and c_trade_id is not None:
        for r_ix, r in enumerate(rows, start=2):
            try:
                if r[c_trade_id] == trade_id:
                    if (c_exit is None or r[c_exit] == "") and (c_status is None or r[c_status].lower() != "closed"):
                        return r_ix
                    break
            except Exception: continue

    for r_ix, r in enumerate(rows, start=2):
        try:
            side_ok = True
            if c_side is not None and side:
                side_ok = (r[c_side].strip().lower() == side)
            if (
                (c_symbol is None or r[c_symbol] == symbol) and
                side_ok and
                (c_exit   is not None and r[c_exit] == "") and
                (c_entry  is None or near(r[c_entry], entry)) and
                (c_status is None or r[c_status].lower() != "closed")
            ):
                return r_ix
        except Exception: continue
    return None

# ------------------ RESULT ------------------
def compute_result(side: str, entry: float, exit_: float, event: str) -> str:
    e = (event or "").lower()
    s = (side or "").lower()
    if e in ("tp", "sl"): return "Win" if e == "tp" else "Loss"
    if s == "long":  return "Win" if float(exit_) > float(entry) else "Loss"
    if s == "short": return "Win" if float(exit_) < float(entry) else "Loss"
    return ""

# ------------------ INBOUND PARSING ------------------
def parse_payload(flask_request) -> Tuple[Dict[str, Any], str]:
    """
    Returns (data_dict, raw_text_for_debug)
    Accepts JSON, form-data, or raw text lines like 'key=value' or 'key: value'
    """
    raw = flask_request.get_data(as_text=True) or ""
    ctype = (flask_request.headers.get("Content-Type") or "").lower()

    # 1) JSON body
    try:
        data_json = flask_request.get_json(force=False, silent=True)
        if isinstance(data_json, dict) and data_json:
            return data_json, raw
    except Exception:
        pass

    # 2) Form-encoded
    try:
        if "application/x-www-form-urlencoded" in ctype and flask_request.form:
            return {k: v for k, v in flask_request.form.items()}, raw
    except Exception:
        pass

    # 3) Try to parse raw text as JSON
    try:
        data_json2 = json.loads(raw)
        if isinstance(data_json2, dict):
            return data_json2, raw
    except Exception:
        pass

    # 4) Parse key=value or key: value lines
    data: Dict[str, Any] = {}
    for line in raw.splitlines():
        m = re.match(r'\s*"?([^"=:\s]+)"?\s*[:=]\s*"?([^"\n]+?)"?\s*$', line.strip())
        if m:
            k, v = m.group(1), m.group(2)
            data[k] = v
    return data, raw

def coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)

    # --- SYMBOL / TICKER NORMALIZATION ---
    sym_candidates = []
    for k in ("symbol","ticker","Ticker","SYMBOL","TICKER"):
        v = out.get(k)
        if v is not None and str(v).strip():
            sym_candidates.append(str(v).strip())

    sym = ""
    if sym_candidates:
        # pick the first non-empty and strip exchange prefix e.g. "BATS:TSLA" -> "TSLA"
        s0 = sym_candidates[0]
        # guard against unrendered templates like "{{ticker}}"
        if "{{" in s0 and "}}" in s0:
            s0 = ""
        if s0 and ":" in s0:
            s0 = s0.split(":")[-1].strip()
        sym = s0

    out["symbol"] = sym

    # Numerics (robust to commas/trailing chars)
    for k in ("entry","exit","price","pos","sl","tp","rr","contracts",
              "sl_short","tp_short","rr_short"):
        if k in out and out[k] not in (None, ""):
            out[k] = _to_float_safe(out[k])

    # Clean strings (remove trailing braces/commas that leak from TV templates)
    def _clean_str(v: Any) -> str:
        s = str(v).strip()
        s = s.rstrip("}, ")
        return s

    for k in ("side","interval","tag","trade_id","event","time","timestamp"):
        if k in out and out[k] is not None:
            out[k] = _clean_str(out[k])

    # Extra: if interval like "3" or "3m", keep just number for Sheets
    iv = out.get("interval")
    if iv:
        ivs = str(iv).strip().lower()
        if ivs.endswith("m") or ivs.endswith("min"):
            ivs = re.sub(r"[^0-9]", "", ivs)
        out["interval"] = ivs

    return out



def derive_missing_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)

    # side from pos if missing
    side = (out.get("side") or "").lower()
    pos  = float(out.get("pos") or 0.0)
    if side not in ("long","short","buy","sell"):
        if pos > 0: side = "long"
        elif pos < 0: side = "short"
    # Normalize buy/sell to long/short (just in case)
    if side == "buy":  side = "long"
    if side == "sell": side = "short"
    out["side"] = side

    # event from pos if missing
    event = (out.get("event") or "").lower()
    price_now = float(out.get("price") or 0.0)
    if not event:
        if pos == 0 and price_now > 0:
            event = "exit"
        out["event"] = event

    # fill entry/exit from price if needed
    if event in ("tp","sl","exit"):
        if not out.get("exit") and price_now > 0: out["exit"] = price_now
    else:
        if not out.get("entry") and price_now > 0: out["entry"] = price_now

    # 🔒 Strategy is SHORT-only on entries → enforce for non-exit events
    if event not in ("tp","sl","exit"):
        out["side"] = "short"

    # Ensure symbol exists (final fallback)
    if not out.get("symbol"):
        for k in ("ticker", "Ticker", "SYMBOL", "TICKER"):
            if k in out and out[k]:
                out["symbol"] = str(out[k]).strip()
                break

    # synthetic trade_id if missing (now includes symbol)
    if not out.get("trade_id"):
        ent = out.get("entry") or 0
        side_for_id = out.get("side") or "na"
        iv = out.get("interval") or "n/a"
        
        # Ensure symbol present; try from trade_id if needed (e.g., "TSLA-short-432.1-3")
        sym = (out.get("symbol") or "").strip()
        if not sym:
            for k in ("ticker","Ticker","SYMBOL","TICKER"):
                if out.get(k):
                    sym = str(out[k]).strip()
                    break
        if not sym:
            tid_probe = (out.get("trade_id") or "")
            m = re.match(r'([A-Za-z0-9._:-]+)-', tid_probe)
            if m:
                sym = m.group(1)
                if ":" in sym:
                    sym = sym.split(":")[-1]
        out["symbol"] = sym


    # default time
    if not out.get("time") and not out.get("timestamp"):
        out["time"] = datetime.utcnow().isoformat() + "Z"
        
        
    # Ensure symbol is present using multiple fallbacks
    sym = (out.get("symbol") or "").strip()
    if not sym:
        for k in ("ticker","Ticker","SYMBOL","TICKER"):
            if out.get(k):
                sym = str(out[k]).strip()
                break
    out["symbol"] = sym

    # Always include symbol in trade_id (even if TV provided one)
    tid = (out.get("trade_id") or "").strip()
    if sym and tid and not tid.startswith(sym + "-"):
        tid = f"{sym}-{tid.lstrip('-')}"
    elif not tid:
        ent = out.get("entry") or 0
        side_for_id = out.get("side") or "na"
        iv = out.get("interval") or "n/a"
        tid = f"{sym}-{side_for_id}-{round(float(ent),2)}-{iv}"
    out["trade_id"] = tid


    return out



# ------------------ CORE UPDATE ------------------
def update_or_append(ws, data: dict) -> bool:
    header = get_header(ws)
    rows = ws.get_all_values()[1:]  # without header

    c_ts       = idx(header, "Timestamp")
    c_symbol   = idx(header, "Symbol")
    c_side     = idx(header, "Trade Direction")
    c_entry    = idx(header, "Entry Price")
    c_exit     = idx(header, "Exit Price")
    c_exit_t   = idx(header, "Exit Time")
    c_qty      = idx(header, "Contracts")
    c_conf     = idx(header, "Confidence")
    c_result   = idx(header, "Result")
    c_status   = idx(header, "Status")
    c_posted   = idx(header, "Posted?")
    c_sl       = idx(header, "SL")
    c_tp       = idx(header, "TP")
    c_rr       = idx(header, "RR")
    c_tag      = idx(header, "Tag")
    c_pnl      = idx(header, "P&L")
    c_comments = idx(header, "Comments")
    c_interval = idx(header, "Interval")
    c_trade_id = idx(header, "Trade ID")

    symbol   = (data.get("symbol") or "").strip()
    side     = (data.get("side") or "").strip().lower()
    entry    = float(data.get("entry") or 0)
    exit_px  = float(data.get("exit")  or 0)
    qty      = data.get("contracts", 0)
    sl       = data.get("sl")
    tp       = data.get("tp")
    rr       = data.get("rr")
    tag      = data.get("tag")
    interval = data.get("interval", "")
    when_iso = data.get("time") or data.get("timestamp") or (datetime.utcnow().isoformat() + "Z")
    event    = (data.get("event") or "").lower()
    trade_id = (data.get("trade_id") or "").strip()

    ts_est = to_est_string(when_iso)

    # EXIT
    is_exit = event in ("tp","sl","exit")
    if is_exit and exit_px > 0 and c_exit is not None:
        target_row = find_open_row(header, rows, symbol, side, entry, trade_id,
                                   c_trade_id, c_symbol, c_side, c_entry, c_exit, c_status)
        if target_row is not None:
            row_vals = ws.row_values(target_row)
            updates: List[Dict[str, Any]] = [{"range": a1(target_row, c_exit), "values": [[exit_px]]}]
            if c_exit_t is not None:
                updates.append({"range": a1(target_row, c_exit_t), "values": [[ts_est]]})
            if c_result is not None:
                try:
                    ent = float(row_vals[c_entry]) if c_entry is not None and row_vals[c_entry] != "" else float(entry or 0)
                    result = compute_result(side, ent, exit_px, event)
                except Exception:
                    result = ""
                updates.append({"range": a1(target_row, c_result), "values": [[result]]})
            if c_pnl is not None:
                try:
                    q = float(row_vals[c_qty]) if c_qty is not None and row_vals[c_qty] != "" else 1.0
                    ent = float(row_vals[c_entry]) if c_entry is not None and row_vals[c_entry] != "" else float(entry or 0)
                    pl  = (exit_px - ent) * (1 if side == "long" else -1) * q
                    updates.append({"range": a1(target_row, c_pnl), "values": [[round(pl,2)]]})
                except Exception:
                    pass
            if c_status is not None:
                updates.append({"range": a1(target_row, c_status), "values": [["Closed"]]})
            ws.batch_update(updates)
            logging.info(f"Updated EXIT row {target_row}: {symbol} {side} -> {exit_px} ({event})")
            return True

    # ENTRY or REFRESH
    target_row = find_open_row(header, rows, symbol, side, entry, trade_id,
                               c_trade_id, c_symbol, c_side, c_entry, c_exit, c_status)
    if target_row is not None:
        updates = []
        if c_ts is not None:       updates.append({"range": a1(target_row, c_ts), "values": [[ts_est]]})
        if c_symbol is not None:   updates.append({"range": a1(target_row, c_symbol), "values": [[symbol]]})
        if c_side is not None:     updates.append({"range": a1(target_row, c_side), "values": [[side]]})
        if c_entry is not None and entry:
            updates.append({"range": a1(target_row, c_entry), "values": [[entry]]})
        if c_sl is not None and sl is not None:   updates.append({"range": a1(target_row, c_sl), "values": [[sl]]})
        if c_tp is not None and tp is not None:   updates.append({"range": a1(target_row, c_tp), "values": [[tp]]})
        if c_rr is not None and rr is not None:   updates.append({"range": a1(target_row, c_rr), "values": [[rr]]})
        if c_tag is not None and tag is not None: updates.append({"range": a1(target_row, c_tag), "values": [[tag]]})
        if c_interval is not None and interval:   updates.append({"range": a1(target_row, c_interval), "values": [[interval]]})
        if c_trade_id is not None and trade_id:   updates.append({"range": a1(target_row, c_trade_id), "values": [[trade_id]]})
        if c_status is not None:   updates.append({"range": a1(target_row, c_status), "values": [["Open"]]})
        if updates: ws.batch_update(updates)
        logging.info(f"Refreshed OPEN row {target_row}: {symbol} {side} @ {entry} ({interval})")
        return True
        
        
    # De-dupe fast repeats by trade_id (Sheets eventual consistency guard)
    if trade_id:
        if seen_recent(trade_id):
            logging.info(f"Skipped duplicate within window for trade_id={trade_id}")
            return True
        # Double-check directly in sheet by searching Trade ID cell text
        try:
            cell = ws.find(trade_id)
            if cell and cell.row >= 2:
                logging.info(f"Found existing row via find() for trade_id={trade_id}; converting to REFRESH")
                target_row = cell.row
                # minimal refresh update to set symbol/side/entry/SL/TP/RR/status etc.
                updates = []
                if c_symbol is not None: updates.append({"range": a1(target_row, c_symbol), "values": [[symbol]]})
                if c_side   is not None: updates.append({"range": a1(target_row, c_side),   "values": [[side]]})
                if c_entry  is not None and entry: updates.append({"range": a1(target_row, c_entry), "values": [[entry]]})
                if c_sl is not None and sl is not None: updates.append({"range": a1(target_row, c_sl), "values": [[sl]]})
                if c_tp is not None and tp is not None: updates.append({"range": a1(target_row, c_tp), "values": [[tp]]})
                if c_rr is not None and rr is not None: updates.append({"range": a1(target_row, c_rr), "values": [[rr]]})
                if c_interval is not None and interval: updates.append({"range": a1(target_row, c_interval), "values": [[interval]]})
                if c_status is not None: updates.append({"range": a1(target_row, c_status), "values": [["Open"]]})
                if updates: ws.batch_update(updates)
                return True
        except Exception:
            pass
     
        

    # Append new OPEN row
    new_row = [""] * len(header)
    def set_if(ci, v): 
        if ci is not None: new_row[ci] = v

    set_if(c_ts, ts_est)
    set_if(c_symbol, symbol)
    set_if(c_side, side)
    set_if(c_entry, entry)
    set_if(c_qty,   qty)
    set_if(c_conf,  ("" if data.get("confidence") is None else data.get("confidence")))
    set_if(c_posted, data.get("posted", ""))
    set_if(c_sl, sl); set_if(c_tp, tp); set_if(c_rr, rr); set_if(c_tag, tag)
    set_if(c_interval, interval); set_if(c_trade_id, trade_id)
    set_if(c_status, "Open")

    ws.append_row(new_row, value_input_option="USER_ENTERED")
    logging.info(f"Appended OPEN row: {symbol} {side} @ {entry} ({interval})")
    return True

# ------------------ FLASK ------------------
app = Flask(__name__)

@app.route("/tv-webhook", methods=["POST"])
def tv_webhook():
    try:
        _hb_tick()
        data_raw, raw_text = parse_payload(request)
        logging.info(f"Webhook Content-Type: {(request.headers.get('Content-Type') or '').lower()}")
        logging.info(f"Webhook RAW (first 400): {raw_text[:400]}")
        if not data_raw:
            logging.warning("Empty webhook payload; raw text: %r", raw_text[:200])
            send_telegram("⚠️ *Trade Webhook Warning*\nNo fields were parsed from the alert. Please check your TradingView alert message template.", "Markdown")
            return jsonify({"ok": False, "error": "empty payload"}), 200

        data = coerce_types(data_raw)
        data = derive_missing_fields(data)
        data = normalize_short_only(data)
        
        logging.info(f"[After parse] symbol={repr(data.get('symbol'))} "
                     f"ticker_raw={repr(data_raw.get('ticker'))} "
                     f"side={repr(data.get('side'))} entry={repr(data.get('entry'))} "
                     f"sl={repr(data.get('sl'))} tp={repr(data.get('tp'))} rr={repr(data.get('rr'))} "
                     f"interval={repr(data.get('interval'))} trade_id={repr(data.get('trade_id'))}")



        logging.info(f"Webhook payload (parsed): {json.dumps(data)}")

        # Write to sheet
        ws = get_sheet()
        ok = update_or_append(ws, data)

        # Build Telegram message
        when    = data.get("time") or data.get("timestamp") or (datetime.utcnow().isoformat()+"Z")
        ts_est  = to_est_string(when)
        symbol  = data.get("symbol","")
        side    = (data.get("side") or "").title() or "N/A"
        iv      = data.get("interval") or "n/a"
        entry   = data.get("entry") or data.get("price") or ""
        exit_px = data.get("exit")  or data.get("price") or ""
        sl      = data.get("sl","")
        tp      = data.get("tp","")
        rr      = data.get("rr","")
        tag     = data.get("tag","")
        event   = (data.get("event") or "").upper()

        if event in ("TP","SL","EXIT"):
            msg = (
                f"✅ *Trade Update*\n"
                f"────────────────────\n"
                f"*{symbol}* | {side} | ⏱ {iv}\n\n"
                f"• Exit Price: `{exit_px}`\n"
                f"• Event: *{event}*\n"
                f"• NY Time: `{ts_est}`"
            )
        else:
            msg = (
                f"📈 *Trade Alert*\n"
                f"────────────────────\n"
                f"*{symbol}* | {side} | ⏱ {iv}\n\n"
                f"• Entry: `{entry}`\n"
                f"• SL: `{sl}`   TP: `{tp}`   RR: `{rr}`\n"
                f"• Tag: `{tag}`\n"
                f"• NY Time: `{ts_est}`"
            )
        
        if not symbol:
            send_telegram("⚠️ *Trade Webhook Warning*\n`symbol` was empty after parsing. "
                          "Check alert JSON keys (`symbol`,`ticker`) and see server logs for details.")
                
        
        
        send_telegram(msg)
        return jsonify({"ok": ok}), 200

    except Exception as e:
        logging.exception("tv_webhook error")
        _hb_ping("fail", {"svc":"tg_trades","status":"fail","ts":int(time.time()),"where":"tv-webhook"})
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    _hb_tick()
    return "OK", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
