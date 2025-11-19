# telegram_confirm_logger_trades.py
# --- SharpSignal: ConfirmedTrades2 logger (entry + exit) ---
# Enhancements:
# - Keep ONE row per trade: entry creates/refreshes an OPEN row; exit updates that SAME row
# - Exit sets Exit Price, Exit Time, Result (Win/Loss), optional P&L, and Status=Closed
# - Optional exact matching via Trade ID column (preferable); fallback to symbol/side/near(entry)
# - Converts inbound UTC time to New York time for Timestamp/Exit Time
# - Records Interval, Tag, RR, SL, TP if present

import os
import json
import logging
from datetime import datetime
from typing import Optional, Tuple

from flask import Flask, request, jsonify
import requests

from oauth2client.service_account import ServiceAccountCredentials
import gspread

from dateutil import parser as dtparser
import pytz

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN",  "8413259557:AAFzdJT1woazNK9i8Hr8Ty_6V4TVqnT8GrU")
TELEGRAM_CHAT_ID    = int(os.getenv("TELEGRAM_CHAT_ID", "7502913977")) 
GOOGLE_CREDS_FILE   = os.getenv("GOOGLE_CREDS_FILE",   "telegrambetlogger-35856685bc29.json")
SPREADSHEET_NAME    = os.getenv("SPREADSHEET_NAME",    "ConfirmedBets")
TAB_TRADES          = os.getenv("TAB_TRADES",          "ConfirmedTrades2")

# How close two floats need to be to be considered the "same" entry
ENTRY_TOLERANCE = float(os.getenv("ENTRY_TOLERANCE", "0.01"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Timezones
EASTERN = pytz.timezone("US/Eastern")

# Flask
app = Flask(__name__)

# ------------------ GOOGLE SHEETS ------------------
def get_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
    gc = gspread.authorize(creds)
    sh = gc.open(SPREADSHEET_NAME)
    ws = sh.worksheet(TAB_TRADES)
    return ws

def get_header(ws):
    return ws.row_values(1)

def idx(header, name) -> Optional[int]:
    try:
        return header.index(name)
    except ValueError:
        return None

def near(a, b, tol=ENTRY_TOLERANCE):
    try:
        return abs(float(a) - float(b)) < tol
    except Exception:
        return False

def to_est_string(iso_utc: str) -> str:
    """UTC ISO8601 → 'YYYY-MM-DD HH:MM:SS' in US/Eastern."""
    dt_utc = dtparser.isoparse(iso_utc)  # handles 'Z' or '+00:00'
    dt_est = dt_utc.astimezone(EASTERN)
    return dt_est.strftime("%Y-%m-%d %H:%M:%S")

# ------------------ TELEGRAM ------------------
def send_telegram(text: str, parse_mode: str = "Markdown"):
    token = TELEGRAM_BOT_TOKEN
    chat_id = str(TELEGRAM_CHAT_ID).strip()
    if not token or not chat_id:
        logging.warning("Telegram not configured; skipping send.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

# ------------------ RESULT LOGIC ------------------
def compute_result(side: str, entry: float, exit_: float, event: str) -> str:
    """Decide Win/Loss based on event or price vs. entry."""
    event = (event or "").lower()
    side = (side or "").lower()
    if event in ("tp", "sl"):
        return "Win" if event == "tp" else "Loss"
    # Generic exit: price comparison
    if side == "long":
        return "Win" if float(exit_) > float(entry) else "Loss"
    if side == "short":
        return "Win" if float(exit_) < float(entry) else "Loss"
    return ""  # unknown

# ------------------ ROW MATCHING ------------------
def find_open_row(
    header, rows, symbol: str, side: str, entry: float, trade_id: Optional[str],
    c_trade_id: Optional[int], c_symbol: Optional[int], c_side: Optional[int],
    c_entry: Optional[int], c_exit: Optional[int], c_status: Optional[int]
) -> Optional[int]:
    """
    Return 1-based sheet row index (including header) for the open trade row to update.
    Preference:
      1) Exact Trade ID match (if provided and column exists, and row is not closed)
      2) Fallback to (symbol, side, ~entry) with empty Exit (and Status not Closed if column exists)
    """
    # Prefer Trade ID
    if trade_id and c_trade_id is not None:
        for r_ix, r in enumerate(rows, start=2):
            if r[c_trade_id] == trade_id:
                # If has Exit but Status not Closed, still allow (some users may clear Exit manually)
                if (c_exit is None or r[c_exit] == "") and (c_status is None or r[c_status].lower() != "closed"):
                    return r_ix
                # If already closed, treat as found but closed (skip); continue to fallback search
                break

    # Fallback: symbol+side+near(entry) & no Exit
    for r_ix, r in enumerate(rows, start=2):
        try:
            if (
                (c_symbol is None or r[c_symbol] == symbol) and
                (c_side   is None or r[c_side].strip().lower() == side) and
                (c_exit   is not None and r[c_exit] == "") and
                (c_entry  is None or near(r[c_entry], entry)) and
                (c_status is None or r[c_status].lower() != "closed")
            ):
                return r_ix
        except Exception:
            continue
    return None

def find_duplicate_entry_row(
    header, rows, symbol: str, side: str, entry: float, trade_id: Optional[str],
    c_trade_id: Optional[int], c_symbol: Optional[int], c_side: Optional[int],
    c_entry: Optional[int], c_exit: Optional[int], c_status: Optional[int]
) -> Optional[int]:
    """
    For ENTRY payloads, detect if a matching OPEN row already exists.
    If so, return that row so we update it instead of appending a duplicate.
    """
    return find_open_row(
        header, rows, symbol, side, entry, trade_id,
        c_trade_id, c_symbol, c_side, c_entry, c_exit, c_status
    )

# ------------------ CORE UPDATE LOGIC ------------------
def update_or_append(ws, data: dict):
    """
    Keep one row per trade.
    Expected columns (create if missing in your sheet):
      Timestamp | Symbol | Trade Direction | Entry Price | Exit Price | Exit Time | Contracts
      | Confidence | Result | Status | Posted? | SL | TP | RR | Tag | P&L | Comments
      | Interval | Trade ID
    All are optional except we strongly recommend: Timestamp, Symbol, Trade Direction, Entry Price.
    """
    header = get_header(ws)
    rows = ws.get_all_values()[1:]  # without header

    # Column indices (0-based)
    c_timestamp   = idx(header, "Timestamp")
    c_symbol      = idx(header, "Symbol")
    c_side        = idx(header, "Trade Direction")
    c_entry       = idx(header, "Entry Price")
    c_exit        = idx(header, "Exit Price")
    c_exit_time   = idx(header, "Exit Time")
    c_contracts   = idx(header, "Contracts")
    c_confidence  = idx(header, "Confidence")
    c_result      = idx(header, "Result")
    c_status      = idx(header, "Status")
    c_posted      = idx(header, "Posted?")
    c_sl          = idx(header, "SL")
    c_tp          = idx(header, "TP")
    c_rr          = idx(header, "RR")
    c_tag         = idx(header, "Tag")
    c_pnl         = idx(header, "P&L")
    c_comments    = idx(header, "Comments")
    c_interval    = idx(header, "Interval")
    c_trade_id    = idx(header, "Trade ID")

    # Inbound fields (with sane defaults)
    symbol    = (data.get("symbol") or "").strip()
    side      = (data.get("side") or "").strip().lower()  # "long" / "short"
    entry     = float(data.get("entry") or 0)
    exit_px   = float(data.get("exit")  or 0)
    contracts = data.get("contracts", 0)
    sl        = data.get("sl")
    tp        = data.get("tp")
    rr        = data.get("rr")
    tag       = data.get("tag")
    interval  = data.get("interval", "")
    when_iso  = data.get("time") or data.get("timestamp") or datetime.utcnow().isoformat() + "Z"
    event     = (data.get("event") or "").lower()  # "", "tp", "sl", "exit"
    trade_id  = (data.get("trade_id") or "").strip()

    # NY time strings
    try:
        ts_est = to_est_string(when_iso)
    except Exception:
        ts_est = when_iso

    # Create batch updates helper
    def cell(r, c):
        # convert zero-based header index to A1 for given 1-based row r
        return gspread.utils.rowcol_to_a1(r, c + 1)

    # If EXIT event, try to UPDATE the open row
    is_exit = event in ("tp", "sl", "exit")
    if is_exit and exit_px > 0 and c_exit is not None:
        target_row = find_open_row(
            header, rows, symbol, side, entry, trade_id,
            c_trade_id, c_symbol, c_side, c_entry, c_exit, c_status
        )
        if target_row is not None:
            row_vals = ws.row_values(target_row)
            updates = []

            # Exit price
            updates.append({"range": cell(target_row, c_exit), "values": [[exit_px]]})

            # Exit time
            if c_exit_time is not None:
                updates.append({"range": cell(target_row, c_exit_time), "values": [[ts_est]]})

            # Result
            if c_result is not None and c_entry is not None:
                try:
                    ent = float(row_vals[c_entry])
                    result = compute_result(side, ent, exit_px, event)
                except Exception:
                    result = ""
                updates.append({"range": cell(target_row, c_result), "values": [[result]]})

            # P&L (optional)
            if c_pnl is not None and c_entry is not None:
                try:
                    qty = float(row_vals[c_contracts]) if c_contracts is not None and row_vals[c_contracts] != "" else 1.0
                    ent = float(row_vals[c_entry])
                    pl  = (exit_px - ent) * (1 if side == "long" else -1) * qty
                    updates.append({"range": cell(target_row, c_pnl), "values": [[round(pl, 2)]]})
                except Exception:
                    pass

            # Status → Closed
            if c_status is not None:
                updates.append({"range": cell(target_row, c_status), "values": [["Closed"]]})

            # Apply batch
            if updates:
                ws.batch_update([{"range": u["range"], "values": u["values"]} for u in updates])
            logging.info(f"Updated EXIT row {target_row}: {symbol} {side} -> {exit_px} ({event})")
            return True

    # ENTRY or non-exit payload:
    # If an OPEN row already exists for this trade, refresh it instead of appending (prevents duplicates).
    target_row_for_entry = find_duplicate_entry_row(
        header, rows, symbol, side, entry, trade_id,
        c_trade_id, c_symbol, c_side, c_entry, c_exit, c_status
    )
    if target_row_for_entry is not None:
        updates = []
        # Update/refresh fields for the existing OPEN row
        if c_timestamp is not None:
            updates.append({"range": cell(target_row_for_entry, c_timestamp), "values": [[ts_est]]})
        if c_symbol is not None:
            updates.append({"range": cell(target_row_for_entry, c_symbol), "values": [[symbol]]})
        if c_side is not None:
            updates.append({"range": cell(target_row_for_entry, c_side), "values": [[side]]})
        if c_entry is not None and entry:
            updates.append({"range": cell(target_row_for_entry, c_entry), "values": [[entry]]})
        if c_sl is not None and sl is not None:
            updates.append({"range": cell(target_row_for_entry, c_sl), "values": [[sl]]})
        if c_tp is not None and tp is not None:
            updates.append({"range": cell(target_row_for_entry, c_tp), "values": [[tp]]})
        if c_rr is not None and rr is not None:
            updates.append({"range": cell(target_row_for_entry, c_rr), "values": [[rr]]})
        if c_tag is not None and tag is not None:
            updates.append({"range": cell(target_row_for_entry, c_tag), "values": [[tag]]})
        if c_interval is not None and interval:
            updates.append({"range": cell(target_row_for_entry, c_interval), "values": [[interval]]})
        if c_trade_id is not None and trade_id:
            updates.append({"range": cell(target_row_for_entry, c_trade_id), "values": [[trade_id]]})
        if c_status is not None:
            updates.append({"range": cell(target_row_for_entry, c_status), "values": [["Open"]]})

        if updates:
            ws.batch_update([{"range": u["range"], "values": u["values"]} for u in updates])
        logging.info(f"Refreshed OPEN row {target_row_for_entry}: {symbol} {side} @ {entry} ({interval})")
        return True

    # Otherwise, APPEND a brand new OPEN row
    new_row = [""] * len(header)
    def set_if(col_idx, value):
        if col_idx is not None:
            new_row[col_idx] = value

    set_if(c_timestamp, ts_est)
    set_if(c_symbol, symbol)
    set_if(c_side, side)
    set_if(c_entry, entry)
    set_if(c_contracts, contracts)
    set_if(c_confidence, data.get("confidence", ""))
    set_if(c_posted, data.get("posted", ""))
    set_if(c_sl, sl)
    set_if(c_tp, tp)
    set_if(c_rr, rr)
    set_if(c_tag, tag)
    set_if(c_interval, interval)
    set_if(c_trade_id, trade_id)
    set_if(c_status, "Open")

    ws.append_row(new_row, value_input_option="USER_ENTERED")
    logging.info(f"Appended OPEN row: {symbol} {side} @ {entry} ({interval})")
    return True

# ------------------ ROUTES ------------------
@app.route("/tv-webhook", methods=["POST"])
def tv_webhook():
    try:
        data = request.get_json(force=True, silent=True) or {}
        logging.info(f"Webhook payload: {json.dumps(data)}")

        ws = get_sheet()
        ok = update_or_append(ws, data)

        # Compose Telegram message
        when   = data.get("time") or data.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        try:
            ts_est = to_est_string(when)
        except Exception:
            ts_est = when

        symbol   = data.get("symbol", "")
        side     = (data.get("side") or "").title()
        interval = data.get("interval", "")
        entry    = data.get("entry", "")
        exit_px  = data.get("exit", "")
        sl       = data.get("sl", "")
        tp       = data.get("tp", "")
        rr       = data.get("rr", "")
        tag      = data.get("tag", "")
        event    = (data.get("event") or "").upper()

        if event in ("TP","SL","EXIT"):
            msg = (
                f"✅ *Trade Update*\n"
                f"────────────────────\n"
                f"*{symbol}* | {side} | ⏱ {interval or 'n/a'}\n\n"
                f"• Exit Price: `{exit_px}`\n"
                f"• Event: *{event}*\n"
                f"• NY Time: `{ts_est}`"
            )
        else:
            msg = (
                f"📈 *Trade Alert*\n"
                f"────────────────────\n"
                f"*{symbol}* | {side} | ⏱ {interval or 'n/a'}\n\n"
                f"• Entry: `{entry}`\n"
                f"• SL: `{sl}`   TP: `{tp}`   RR: `{rr}`\n"
                f"• Tag: `{tag}`\n"
                f"• NY Time: `{ts_est}`"
            )


        send_telegram(msg)
        return jsonify({"ok": ok}), 200
    except Exception as e:
        logging.exception("tv_webhook error")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
