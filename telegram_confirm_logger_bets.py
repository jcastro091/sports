
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import html
import logging
import re
from datetime import datetime
import pytz
from typing import Optional, Tuple, List

from telegram import ParseMode, Update
from telegram.error import TelegramError
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ========= CONFIG =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4")

GOOGLE_CREDS_FILE = os.getenv(
    "GOOGLE_CREDS_FILE",
    os.path.join(os.path.dirname(__file__), "telegrambetlogger-35856685bc29.json"),
)

SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "ConfirmedBets")
TAB_BETS = os.getenv("TAB_BETS", "AllBets")
TAB_OBS  = os.getenv("TAB_OBS",  "AllObservations")

CHANNELS = {
    "starter":    int(os.getenv("STARTER_CHAT_ID", "-1000000000000")),
    "pro":        int(os.getenv("PRO_CHAT_ID",     "-1000000000000")),
    "enterprise": int(os.getenv("ENTERPRISE_CHAT_ID", "-1002635869925")),
}

CONFIRM_WORDS = {"✅", "high", "medium", "low", "hi", "med", "lo", "y", "yes"}

EASTERN = pytz.timezone("US/Eastern")

# ========= SETUP =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_FILE, scope)
gc = gspread.authorize(creds)
sheet_bets = gc.open(SPREADSHEET_NAME).worksheet(TAB_BETS)
sheet_obs  = gc.open(SPREADSHEET_NAME).worksheet(TAB_OBS)

# ========= HELPERS =========
def ensure_obs_columns():
    header = sheet_obs.row_values(1)
    needed = ["Bet Placed?", "Placed Channel", "Confirmed By", "Confirmed At",
              "AI Pred", "AI Prob", "Odds Taken (AM)", "Odds Taken (Dec)", "Sportsbook", "Placed At (ET)"]
    for name in needed:
        if name not in header:
            sheet_obs.update_cell(1, len(header) + 1, name)
            header = sheet_obs.row_values(1)
    return header

def find_row_by_bet_id(ws, bet_id: str) -> Tuple[Optional[int], List[str]]:
    header = ws.row_values(1)
    if "bet_id" not in header:
        return None, header
    col_idx = header.index("bet_id") + 1
    col_vals = ws.col_values(col_idx)[1:]
    for i, v in enumerate(col_vals, start=2):
        if str(v).strip() == str(bet_id).strip():
            return i, header
    return None, header

def get_header_indexes(header: List[str], names: List[str]) -> dict:
    return {n: (header.index(n) + 1 if n in header else None) for n in names}

def parse_reply_text(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if "✅" in t: return "High"
    if t in CONFIRM_WORDS:
        if t in {"high", "hi"}: return "High"
        if t in {"medium", "med"}: return "Medium"
        if t in {"low", "lo"}: return "Low"
        if t in {"y", "yes"}: return "High"
    return None

def esc(x): return html.escape(str(x)) if x is not None else ""

def am_to_decimal(am: int | str | None) -> Optional[float]:
    if am is None: return None
    try: a = int(str(am).strip())
    except: return None
    return 1 + (a / 100) if a > 0 else (1 + 100 / abs(a))

# Regex: "-120 at BetMGM", "MGM -120", "(-115) DraftKings"
AM_BOOK = re.compile(
    r"(?:^|\s|[\(\[])(?P<am>[+\-]\d{2,4})(?:[\)\]]|\s|$|,)*(?:\s*(?:at|@)?\s*(?P<book>[A-Za-z][\w .\-/&]*))?",
    re.IGNORECASE,
)

def _extract_bet_id_from_original_message(msg: Optional[str]) -> Optional[str]:
    if not msg:
        return None
    # try both plain and <code> wrapped variants
    m = re.search(r"bet_id[:=]\s*(?:<code>)?([A-Za-z0-9\-]+)", msg, re.IGNORECASE)
    return m.group(1) if m else None

def _safe_username(update: Update) -> str:
    # Handles channel posts where from_user can be None
    user = update.effective_user
    if user is None:
        return "Unknown"
    handle = getattr(user, "username", None)
    full = getattr(user, "full_name", None)
    return (handle and f"@{handle}") or (full or "Unknown")

# ========= TELEGRAM HANDLER =========
def on_message(update: Update, context: CallbackContext):
    msg = update.effective_message
    if not msg or not msg.reply_to_message:
        return

    username = _safe_username(update)

    # Prefer text_html when present (alerts often include <code>)
    orig = getattr(msg.reply_to_message, "text_html", None) or getattr(msg.reply_to_message, "text", "") or ""
    bet_id = _extract_bet_id_from_original_message(orig)

    # --- Try parse odds + book from reply ---
    m = AM_BOOK.search(msg.text or "")
    if m and bet_id:
        try:
            am = int(m.group("am"))
        except Exception:
            am = None
        book = (m.group("book") or "").strip()

        # Normalize common short codes
        book_map = {"dk":"DraftKings","fd":"FanDuel","mgm":"BetMGM","caesars":"Caesars","pinnacle":"Pinnacle"}
        bl = book.lower()
        if bl in book_map: book = book_map[bl]
        elif bl.startswith("dk"): book = "DraftKings"
        elif bl in ("fd","fanduel","@fd"): book = "FanDuel"
        elif bl in ("mgm","betmgm"): book = "BetMGM"

        row_o, header_o = find_row_by_bet_id(sheet_obs, bet_id)
        if not row_o:
            msg.reply_text(f"Could not find row for {bet_id}.")
            return

        to_write = {
            "Odds Taken (AM)": str(am) if am is not None else "",
            "Odds Taken (Dec)": f"{am_to_decimal(am) or ''}",
            "Sportsbook": book,
            "Placed At (ET)": datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S"),
            "Bet Placed?": "Yes",
            "Placed Channel": "Telegram",
            "Confirmed By": username,
            "Confirmed At": datetime.now(EASTERN).strftime("%Y-%m-%d %I:%M %p"),
        }
        for col, val in to_write.items():
            if col in header_o:
                sheet_obs.update_cell(row_o, header_o.index(col)+1, val)

        msg.reply_text(f"Logged: {am} at {book} for {bet_id}")
        return

    # --- Otherwise, fall back to confidence handling ---
    confidence = parse_reply_text(msg.text or "")
    if not confidence:
        return

    logging.info("Confirmation from %s: %s", username, confidence)

    # Find AllBets row by bet_id; if missing, fallback to today's last unposted row
    header_b = sheet_bets.row_values(1)
    row_b = None
    if bet_id:
        row_b, header_b = find_row_by_bet_id(sheet_bets, bet_id)

    if not row_b:
        idx = get_header_indexes(header_b, ["Posted?","Timestamp"])
        rows = sheet_bets.get_all_values()[1:]
        today = datetime.now(EASTERN).date()
        for i in range(len(rows), 0, -1):
            r = rows[i-1]
            try:
                ts = r[idx["Timestamp"]-1]
                if not ts:
                    continue
                dt = EASTERN.localize(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                if dt.date() != today:
                    continue
                posted = (r[idx["Posted?"]-1] or "").strip()
                if not posted:
                    row_b = i+1
                    break
            except Exception:
                continue

    if not row_b:
        msg.reply_text("Couldn't locate the bet in AllBets to log this confirmation.")
        return

    # Compose and broadcast confirmation (unchanged behavior)
    original_txt = msg.reply_to_message.text or ""
    final_html = f"{esc(original_txt)}\n\n🌟 <b>Confidence:</b> {esc(confidence)}"

    posted_list = list(CHANNELS.keys())
    for plan in posted_list:
        chat_id = CHANNELS[plan]
        try:
            context.bot.send_message(chat_id=chat_id, text=final_html, parse_mode=ParseMode.HTML)
        except TelegramError as e:
            logging.warning("Failed to send to %s: %s", plan, e)
            try:
                context.bot.send_message(chat_id=chat_id, text=html.unescape(final_html))
            except TelegramError:
                pass

    # Update AllBets with confidence + Posted?
    header_b = sheet_bets.row_values(1)
    idx_b = get_header_indexes(header_b, ["Confidence", "Posted?"])
    if idx_b.get("Confidence"):
        sheet_bets.update_cell(row_b, idx_b["Confidence"], confidence)
    if idx_b.get("Posted?"):
        sheet_bets.update_cell(row_b, idx_b["Posted?"], ", ".join([p.capitalize() for p in posted_list]))

    # Mirror to AllObservations
    try:
        ensure_obs_columns()
        if not bet_id and "bet_id" in header_b:
            bet_id = sheet_bets.cell(row_b, header_b.index("bet_id")+1).value

        if bet_id:
            row_o, header_o = find_row_by_bet_id(sheet_obs, bet_id)
            if row_o:
                idx_o = get_header_indexes(header_o, ["Bet Placed?","Placed Channel","Confirmed By","Confirmed At"])
                if idx_o.get("Bet Placed?"):
                    sheet_obs.update_cell(row_o, idx_o["Bet Placed?"], "Yes")
                if idx_o.get("Placed Channel"):
                    sheet_obs.update_cell(row_o, idx_o["Placed Channel"], ", ".join([p.capitalize() for p in posted_list]))
                if idx_o.get("Confirmed By"):
                    sheet_obs.update_cell(row_o, idx_o["Confirmed By"], username)
                if idx_o.get("Confirmed At"):
                    sheet_obs.update_cell(row_o, idx_o["Confirmed At"], datetime.now(EASTERN).strftime("%Y-%m-%d %I:%M %p"))
    except Exception as e:
        logging.error("Mirror to AllObservations failed: %s", e)

# ========= MAIN =========
def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), on_message))
    updater.start_polling(drop_pending_updates=True)
    logging.info("telegram_confirm_logger_bets running…")
    updater.idle()

if __name__ == "__main__":
    main()
