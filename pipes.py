# pipes.py
import os, json, time, logging
from typing import Dict, Any

log = logging.getLogger("pipes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s  %(message)s")

# === TELEGRAM ===
import requests

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

def send_telegram(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TG_BOT_TOKEN or TG_CHAT_ID not set")
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True})
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

# === GOOGLE SHEETS (AllBets) ===
# Uses gspread + service account json (path or inline json via env).
import gspread

SHEETS_CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")  # inline JSON
SHEETS_CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")       # path to service-account.json
ALLBETS_SHEET_KEY = os.getenv("ALLBETS_SHEET_KEY")  # Google Sheet ID for the ConfirmedBets spreadsheet
ALLBETS_TAB_NAME  = os.getenv("ALLBETS_TAB_NAME", "AllBets")

def _get_gspread_client():
    if SHEETS_CREDS_JSON:
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(json.loads(SHEETS_CREDS_JSON), scopes=scopes)
        return gspread.authorize(creds)
    if SHEETS_CREDS_PATH:
        return gspread.service_account(filename=SHEETS_CREDS_PATH)
    raise RuntimeError("Missing Google creds: set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_JSON")

def append_allbets(row: Dict[str, Any]) -> None:
    """
    Minimal columns; extra keys are ignored. Adjust the order to match your sheet.
    """
    if not ALLBETS_SHEET_KEY:
        raise RuntimeError("ALLBETS_SHEET_KEY not set")

    gc = _get_gspread_client()
    sh = gc.open_by_key(ALLBETS_SHEET_KEY)
    ws = sh.worksheet(ALLBETS_TAB_NAME)

    # Map dict -> list in your preferred column order:
    columns = [
        "timestamp","bet_id","sport","league","market","side",
        "team","opponent","american_odds","decimal_odds","bin","proba",
        "reason","stake","is_canary"
    ]
    values = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        row.get("bet_id",""),
        row.get("sport",""),
        row.get("league",""),
        row.get("market",""),
        row.get("side",""),
        row.get("team",""),
        row.get("opponent",""),
        row.get("american_odds",""),
        row.get("decimal_odds",""),
        row.get("bin",""),
        row.get("proba",""),
        row.get("reason",""),
        row.get("stake",0),
        row.get("is_canary",0),
    ]
    ws.append_row(values, value_input_option="USER_ENTERED")

# === X (Twitter) autopost via webhook (recommended) ===
# Have your existing autoposter listen to POST {text, is_canary}
X_WEBHOOK_URL = os.getenv("X_WEBHOOK_URL", "")  # e.g., your FastAPI/Cloudflare Worker endpoint

def autopost_to_x(text: str, is_canary: bool = False) -> None:
    if not X_WEBHOOK_URL:
        raise RuntimeError("X_WEBHOOK_URL not set")
    r = requests.post(X_WEBHOOK_URL, json={"text": text, "is_canary": bool(is_canary)})
    if r.status_code >= 300:
        raise RuntimeError(f"X webhook error {r.status_code}: {r.text}")

# === Helper to build a row like your runner does ===
def build_allbets_row_from_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bet_id":        pred.get("event_id"),
        "sport":         pred.get("sport"),
        "league":        pred.get("league"),
        "market":        pred.get("market"),
        "side":          pred.get("side"),
        "team":          pred.get("team"),
        "opponent":      pred.get("opponent"),
        "american_odds": pred.get("american_odds"),
        "decimal_odds":  pred.get("decimal_odds"),
        "bin":           pred.get("bin"),
        "proba":         pred.get("proba"),
        "reason":        pred.get("reason"),
        "stake":         pred.get("stake", 0.0),
        "is_canary":     pred.get("is_canary", 0),
    }
