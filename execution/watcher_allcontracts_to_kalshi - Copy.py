# watcher_allcontracts_to_kalshi.py
from __future__ import annotations
import os, time, logging
from datetime import datetime, timezone
from pathlib import Path
import gspread, yaml
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import APIError
import uuid, time

from .router import Order
from .adapters.kalshi_adapter import KalshiAdapter

# ---------- config ----------
def load_cfg(path: str | None = None) -> dict:
    here = Path(__file__).resolve().parent
    path = path or (here / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- sheets helpers ----------
def _open_ws(spreadsheet_id: str, worksheet: str, creds_json: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(spreadsheet_id).worksheet(worksheet)

def _safe_update(ws, a1_range, values, value_input_option="USER_ENTERED"):
    for wait in (0, 1, 2, 4, 8, 16):
        try:
            return ws.update(a1_range, values, value_input_option=value_input_option)
        except APIError as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                logging.warning("Sheets quota hit; retrying %s in %ss …", a1_range, wait)
                time.sleep(wait); continue
            raise
    raise RuntimeError(f"Exceeded retry budget for Sheets update: {a1_range}")

def _parse_row(headers, vals, row_idx: int):
    row = {(headers[i] or "").strip().lower(): (vals[i] if i < len(vals) else "")
           for i in range(len(headers))}
    ticker = row.get("kalshiticker","").strip()
    side   = row.get("kalshiside","").strip().upper()
    price  = row.get("kalshiprice","").strip()
    qty    = row.get("kalshiqty","").strip()
    tif    = (row.get("kalshitif","") or "GTC").strip().upper()
    if not (ticker and side in ("YES","NO")): return None
    try:
        price_cents = int(round(float(price)))   # 50 -> 50 cents
        qty_int     = max(1, int(float(qty)))
    except ValueError:
        return None
    sent = row.get("senttokalshi","").lower() in ("true","1","yes")
    oid  = row.get("kalshiorderid","")
    return {"row_idx": row_idx, "ticker": ticker, "side": side,
            "price_cents": price_cents, "qty": qty_int, "tif": tif,
            "already_sent": sent or bool(oid)}

# ---------- kalshi helpers ----------
# --- add near the top imports ---
import json
import requests

# --- replace _startup_probe with a stronger one ---
def _startup_probe(kalshi: KalshiAdapter):
    """
    Auth/health check using documented endpoints that exist on both hosts.
    Prefer /portfolio/balance & a tiny markets query. Avoid /me (not always present).
    """
    import requests
    try:
        # 1) Balance
        r_bal = kalshi._request("GET", "/trade-api/v2/portfolio/balance")
        logging.info("auth: /portfolio/balance status=%s body=%s", r_bal.status_code, (r_bal.text or "")[:300])
        r_bal.raise_for_status()
        js = r_bal.json() or {}
        bal = float(js.get("balance") or js.get("portfolio_balance") or js.get("cash") or 0)

        # 2) Markets sanity (pull 1)
        r_mkts = kalshi._request("GET", "/trade-api/v2/markets", params={"limit": 1, "state": "all"})
        logging.info("auth: /markets status=%s body=%s", r_mkts.status_code, (r_mkts.text or "")[:180])
        r_mkts.raise_for_status()

        logging.info("✅ Kalshi connected (%s) | balance=$%.2f", kalshi.base_url, bal)
        return True
    except requests.exceptions.RequestException as e:
        # HTTP errors with body
        try:
            body = getattr(getattr(e, "response", None), "text", "") or ""
            logging.error("Kalshi startup probe failed: %s | %s", e, body[:600])
        except Exception:
            logging.error("Kalshi startup probe failed: %s", e)
        return False
    except Exception as e:
        logging.error("Kalshi startup probe failed: %s", e)
        return False


def _pick_base_url(cfg_base_url: str):
    """
    Try cfg_base_url first; if DNS fails or 404 on balance, fall back to elections host (read-only).
    """
    primary = (cfg_base_url or "").strip() or "https://trading-api.kalshi.com"
    fallback = "https://api.elections.kalshi.com"
    return primary, fallback



def _fetch_market_meta(kalshi, q: str):
    qU = (q or "").strip().upper()
    # direct
    for cand in (q, qU):
        r = kalshi._request("GET", f"/trade-api/v2/markets/{cand}")
        

        if r.status_code == 200:
            js = r.json() or {}

            # 1) Wrapped list
            if isinstance(js, dict) and isinstance(js.get("markets"), list) and js["markets"]:
                m = js["markets"][0]
                return m if m.get("ticker") else None

            # 2) Wrapped single
            if isinstance(js, dict) and isinstance(js.get("market"), dict):
                m = js["market"]
                return m if m.get("ticker") else None

            # 3) Bare market object
            if isinstance(js, dict) and js.get("ticker"):
                return js
            # else: not a usable market → fall through
                
        if r.status_code != 404: r.raise_for_status()
    # tickers param
    r = kalshi._request("GET", "/trade-api/v2/markets", params={"tickers": qU})
    if r.status_code == 200:
        js = r.json() or {}; mkts = js.get("markets", js if isinstance(js, list) else [])
        for m in mkts:
            if (m.get("ticker") or "").upper() == qU: return m
    elif r.status_code != 404: r.raise_for_status()
    # search by slug/title
    r = kalshi._request("GET", "/trade-api/v2/markets", params={"search": q, "state": "all", "limit": 100})
    if r.status_code == 200:
        js = r.json() or {}; mkts = js.get("markets", js if isinstance(js, list) else [])
        ql = (q or "").lower()
        for m in mkts:
            if ql in (m.get("slug","").lower(), (m.get("ticker","") or "").lower()): return m
        for m in mkts:
            if ql in (m.get("slug","").lower() or "") or ql in (m.get("title","").lower() or ""): return m
    elif r.status_code != 404: r.raise_for_status()
    return None

def _resolve_tradeable_ticker(kalshi, q: str):
    logging.info("resolve: start q=%s", q)
    m = _fetch_market_meta(kalshi, q)
    if m and m.get("ticker"):
        ticker = m["ticker"]
        logging.info("resolve: markets hit -> %s | state=%s | title=%s",
                     ticker, (m.get("status") or m.get("state")), m.get("title"))
        return ticker
    # fallback via events with nested markets (handles event tickers)
    # r = kalshi._request("GET", "/trade-api/v2/events",
                        # params={"tickers": q.upper(), "with_nested_markets": "true"})
    # logging.info("resolve: events status=%s", r.status_code)
    # if r.status_code == 200:
        # evs = (r.json() or {}).get("events", [])
        # if evs and evs[0].get("markets"):
            # for mk in evs[0]["markets"]:
                # st = (mk.get("state") or mk.get("status") or "").lower()
                # if st in ("active","open","trading"):
                    # logging.info("resolve: events hit -> %s (state=%s title=%s)",
                                 # mk.get("ticker"), st, mk.get("title"))
                    # return mk.get("ticker")
            # t = evs[0]["markets"][0].get("ticker")
            # logging.info("resolve: events fallback first -> %s", t)
            # return t
    return None


# raw order (bypass adapter) for truthy server messages
def _place_raw_limit(kalshi, ticker, side, price_cents, count, tif="GTC"):
    tif_norm = (tif or "GTC").upper()
    if tif_norm in ("IOC", "IMMEDIATE_OR_CANCEL"):
        time_in_force = "immediate_or_cancel"
    elif tif_norm in ("FOK", "FILL_OR_KILL"):
        time_in_force = "fill_or_kill"
    else:
        time_in_force = None

    client_order_id = f"{ticker}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    payload = {
        "ticker": ticker,
        "action": "buy",
        "side": side.lower(),           # "yes" | "no"
        "count": int(count),            # contracts
        "client_order_id": client_order_id,   # <<< REQUIRED NOW
    }
    if side.upper() == "YES":
        payload["yes_price"] = int(price_cents)
    else:
        payload["no_price"]  = int(price_cents)

    if time_in_force:
        payload["time_in_force"] = time_in_force

    logging.info("raw-order: POST /portfolio/orders payload=%s", payload)
    r = kalshi._request("POST", "/trade-api/v2/portfolio/orders", json=payload)
    logging.info("raw-order: status=%s body[:400]=%r", r.status_code, (r.text or "")[:400])
    r.raise_for_status()
    return r.json()



# ---------- core ----------
def _place_and_record(ws_src, ws_out, kalshi: KalshiAdapter, order: dict, dry_run: bool):
    resolved = _resolve_tradeable_ticker(kalshi, order["ticker"])
    if not resolved:
        raise RuntimeError(f"resolve: no tradeable market for {order['ticker']}")

    now_iso = datetime.now(timezone.utc).isoformat()
    errtxt, ord_id, status, fill = "", "", "PENDING", 0

    try:
        if dry_run:
            ord_id, status = f"DRYRUN-{int(time.time())}", "SIMULATED"
        else:
            use_raw = os.getenv("KALSHI_USE_RAW","0") in ("1","true","TRUE","yes","YES")
            if use_raw:
                resp = _place_raw_limit(kalshi, resolved, order["side"], order["price_cents"], order["qty"], order.get("tif","GTC"))
                ord_id = str(resp.get("id") or resp.get("order_id") or "")
                status = resp.get("status","SUBMITTED") or "SUBMITTED"
                fill   = int(resp.get("filled") or resp.get("count") or 0)
            else:
                price_dollars = order["price_cents"] / 100.0
                size_dollars  = max(1, order["qty"]) * price_dollars
                o = Order(row_id=str(order["row_idx"]), ticker=resolved, side=order["side"],
                          price=price_dollars, size=size_dollars, price_floor=price_dollars,
                          tif=order.get("tif","GTC"), notes=f"src={order['ticker']}")
                resp  = kalshi.place(o)
                ord_id = getattr(resp, "order_id", "") or getattr(resp, "id", "")
                status = getattr(resp, "status", "SUBMITTED")
                fill   = int(getattr(resp, "filled", 0) or getattr(resp, "count", 0))
        ok = True
    except Exception as e:
        ok, status = False, "REJECTED"
        try:
            import requests
            if isinstance(e, requests.HTTPError) and e.response is not None:
                errtxt = f"{e} | {e.response.text[:600]}"
            else:
                errtxt = str(e)
        except Exception:
            errtxt = str(e)

    # Update source row (BF..BJ): Sent, Status, OrderId, Ts, Error
    hdrs = [h.strip() for h in ws_src.row_values(1)]
    def ensure_col(name):
        try: return hdrs.index(name)+1
        except ValueError:
            ws_src.update_cell(1, len(hdrs)+1, name); hdrs.append(name); return len(hdrs)

    row = order["row_idx"]
    for col in ("SentToKalshi","KalshiStatus","KalshiOrderId","KalshiTs","KalshiError"):
        ensure_col(col)
    _safe_update(ws_src, f"BF{row}:BJ{row}", [[ "TRUE" if ok else "FALSE", status, ord_id, now_iso, errtxt ]])

    # Append receipt to out sheet (ensure headers)
    out_hdrs = [h.strip() for h in ws_out.row_values(1)]
    need = ["Timestamp","Ticker","Side","PriceCents","Qty","OrderId","Status","FilledQty","DryRun","Error","ResolvedTicker"]
    if not out_hdrs:
        ws_out.append_row(need); out_hdrs = need[:]
    else:
        for col in need:
            if col not in out_hdrs:
                ws_out.update_cell(1, len(out_hdrs)+1, col); out_hdrs.append(col)

    ws_out.append_row([now_iso, order["ticker"], order["side"], order["price_cents"], order["qty"],
                       ord_id, status, fill, "TRUE" if dry_run else "FALSE", errtxt, resolved])

def main():
    cfg = load_cfg()
    logging.basicConfig(level=getattr(logging, cfg.get("logging",{}).get("level","INFO")))
    sh_cfg = cfg["sheets"]
    ws_src = _open_ws(sh_cfg["spreadsheet_id"], sh_cfg["obs_worksheet"], sh_cfg["service_account_json"])
    ws_out = _open_ws(sh_cfg["spreadsheet_id"], sh_cfg["out_worksheet"], sh_cfg["service_account_json"])

    kalshi_cfg = cfg["kalshi"]
    kalshi_cfg = cfg["kalshi"]

    # Allow env override and add fallback if DNS/404
    cfg_base = os.getenv("KALSHI_BASE_URL", kalshi_cfg.get("base_url", "https://api.kalshi.com"))
    primary, fallback = _pick_base_url(cfg_base)

    def _make_adapter(base):
        return KalshiAdapter(base_url=base, key_id=kalshi_cfg["key_id"], private_key_file=kalshi_cfg["private_key_file"])

    kalshi = _make_adapter(primary)
    if not _startup_probe(kalshi):
        logging.warning("Primary host failed (%s). Trying fallback host …", primary)
        kalshi = _make_adapter(fallback)
        if not _startup_probe(kalshi):
            raise SystemExit(2)


    dry_run  = bool(cfg.get("dry_run", True))
    poll_sec = int(cfg.get("watcher",{}).get("poll_sec", 5))
    logging.info("Watcher started | env=%s src=%s out=%s dry_run=%s poll=%ss",
                 cfg.get("env","demo"), sh_cfg["obs_worksheet"], sh_cfg["out_worksheet"], dry_run, poll_sec)

    while True:
        try:
            rows = ws_src.get_all_values()
            if not rows or len(rows) < 2: time.sleep(poll_sec); continue
            headers = [h.strip() for h in rows[0]]
            processed = 0
            for i, row in enumerate(rows[1:], start=2):
                if processed >= 3: break
                po = _parse_row(headers, row, i)
                if not po or po["already_sent"]: continue
                logging.info("candidate: row=%s ticker=%s side=%s px=%s qty=%s", po["row_idx"], po["ticker"], po["side"], po["price_cents"], po["qty"])
                _place_and_record(ws_src, ws_out, kalshi, po, dry_run)
                processed += 1
                time.sleep(0.5)
        except Exception as e:
            logging.exception("Watcher loop error: %s", e)
        time.sleep(poll_sec)

if __name__ == "__main__":
    main()
