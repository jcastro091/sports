#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, numpy as np, pandas as pd, requests, smtplib
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Config / ENV
# =========================
CSV_PATH = os.getenv("ALL_OBS_CSV", "ConfirmedBets - AllObservations.csv")
STAKE_DEFAULT = float(os.getenv("STAKE_DEFAULT", "100"))
TZ            = os.getenv("TZ", "UTC")

TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHATS     = [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]
TG_PARSEMODE = os.getenv("TELEGRAM_PARSE_MODE", "Markdown").strip()

EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "").lower()
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_TO       = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "465"))
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

OUTDIR = "out"; os.makedirs(OUTDIR, exist_ok=True)
logging.basicConfig(filename="weekly_recap.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# =========================
# Helpers
# =========================
def pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def american_profit(odds_am, stake):
    if pd.isna(odds_am): return 0.0
    o = float(odds_am);  return stake * (o/100.0 if o > 0 else 100.0/abs(o))

def parse_closing(movement):
    if pd.isna(movement): return np.nan
    s = str(movement).replace("→","->").replace("=>","->")
    vals=[]
    for p in s.split("->"):
        p=p.strip().replace("+","")
        try: vals.append(float(p))
        except: pass
    return vals[-1] if vals else np.nan

def nice_money(x): return f"{'-' if x<0 else ''}${abs(x):,.2f}"
def pct(x):       return f"{x*100:0.1f}%" if x==x else "0.0%"

# =========================
# Load CSV
# =========================
def load_allobservations_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)

# =========================
# Metrics
# =========================
def compute_week(df):
    ts_col    = pick(df, ["Timestamp","ts","Time","Date"])
    res_col   = pick(df, ["Prediction Result","Result"])
    odds_col  = pick(df, ["Odds Taken","Odds (Am)","Odds"])
    risk_col  = pick(df, ["Stake Amount", "Risk", "Stake", "stake"])
    sport_col = pick(df, ["Sport"])
    market_col= pick(df, ["Market","Bet Type"])
    move_col  = pick(df, ["Movement"])
    if not ts_col or not res_col:
        raise ValueError("Required: Timestamp & Prediction Result")

    work = df.copy()
    work["ts"] = pd.to_datetime(work[ts_col], errors="coerce", utc=True)

    # --- last completed Mon→Sun in local TZ ---
    now = pd.Timestamp.now(tz="UTC")
    today_local = now.tz_convert(TZ).date()  # local date (Mon=0..Sun=6)
    # Start of "current" week (Monday)
    this_mon = today_local - pd.Timedelta(days=today_local.weekday())
    # If it's Monday today, we want the *previous* full week.
    end_sun = (this_mon - pd.Timedelta(days=1)) if today_local == this_mon else (this_mon + pd.Timedelta(days=6))
    start_mon = end_sun - pd.Timedelta(days=6)

    local_dates = work["ts"].dt.tz_convert(TZ).dt.date
    mask = (local_dates >= start_mon) & (local_dates <= end_sun)
    work = work[mask].copy()
    if work.empty: 
        return work, now, start_mon, {}

    # --- graded only (0/1) ---
    raw = work[res_col].astype(str).str.strip().str.lower()
    keep = raw.isin(["0","1","0.0","1.0","win","lose","w","l"])
    work = work[keep].copy()
    if work.empty: return work, now, now - pd.Timedelta(days=7), {}
    res = raw[keep].map({"1":1,"1.0":1,"win":1,"w":1,"0":0,"0.0":0,"lose":0,"l":0}).astype(int)

    # --- odds/CLV ---
    if odds_col: work[odds_col] = pd.to_numeric(work[odds_col], errors="coerce")
    if move_col:
        work["Closing Odds"] = work[move_col].apply(parse_closing)
    else:
        work["Closing Odds"] = np.nan
    work["use_odds"] = work["Closing Odds"].where(pd.notna(work["Closing Odds"]), work.get(odds_col, np.nan))
    work["CLV"] = (work["Closing Odds"] - work[odds_col]) if odds_col in work.columns else np.nan

    # --- risk: zeros/NaN -> STAKE_DEFAULT ---
    base_risk = pd.to_numeric(work[risk_col], errors="coerce") if risk_col else np.nan
    work["Risk_used"] = base_risk.where(base_risk > 0, np.nan).fillna(STAKE_DEFAULT)

    # --- P&L (risk-$100-at-close model) ---
    work["is_win"]  = (res == 1).astype(int)
    work["is_loss"] = (res == 0).astype(int)
    def row_pl(r):
        o = r["use_odds"]
        if r["is_win"] == 1:
            return r["Risk_used"] * (o/100.0 if o > 0 else 100.0/abs(o))
        if r["is_loss"] == 1:
            return -r["Risk_used"]
        return 0.0
    work["P&L"] = work.apply(row_pl, axis=1)

    total   = int(len(work))
    wins    = int(work["is_win"].sum())
    losses  = int(work["is_loss"].sum())
    pushes  = 0
    net     = float(work["P&L"].sum())
    risk_in = float(work["Risk_used"].sum())
    roi     = (net/risk_in) if risk_in else 0.0
    wr      = (wins/total) if total else 0.0
    avg_clv = float(work["CLV"].mean()) if "CLV" in work.columns and work["CLV"].notna().any() else None

    # caption bits
    metrics = dict(
        start=str(start_mon), end=str(end_sun),
        total=total, wins=wins, losses=losses, pushes=pushes,
        net=net, roi=roi, wr=wr, avg_clv=avg_clv,
        top_sports="n/a", top_markets="n/a",
    )

    for colname, key in [(sport_col,"top_sports"), (market_col,"top_markets")]:
        if colname and colname in work.columns:
            g = work.groupby(colname)["P&L"].agg(["count","sum"]).reset_index().sort_values("sum", ascending=False).head(3)
            metrics[key] = ", ".join([f"{r[colname]} ({int(r['count'])} bets, {nice_money(r['sum'])})" for _,r in g.iterrows()]) or "n/a"
    return work, now, now - pd.Timedelta(days=7), metrics

# =========================
# Charts
# =========================
def chart_roi_by_day(df, tz, path):
    d=df.copy(); d["day"]=d["ts"].dt.tz_convert(tz).dt.date
    agg=d.groupby("day")["P&L"].sum().reset_index()
    plt.figure(figsize=(8,4.5), dpi=200); plt.bar(agg["day"].astype(str), agg["P&L"])
    plt.title("Weekly P&L by Day"); plt.xlabel("Day"); plt.ylabel("P&L (USD)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def chart_clv_hist(df, path):
    d=df.copy(); d=d[pd.notna(d["CLV"])]
    plt.figure(figsize=(8,4.5), dpi=200)
    if len(d)==0:
        plt.text(0.5,0.5,"CLV not available this week", ha="center", va="center"); plt.axis("off")
    else:
        plt.hist(d["CLV"], bins=20); plt.title("CLV Distribution (Closing Odds – Taken)")
        plt.xlabel("CLV (American odds)"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

# =========================
# Captions (+ Lessons)
# =========================
def lessons_learned(metrics, week_df):
    pts=[]; roi=metrics.get("roi",0.0); wr=metrics.get("wr",0.0); clv=metrics.get("avg_clv",None)
    if roi < -0.15: pts.append("Weekly windows are volatile; one week isn’t the long-run signal.")
    if clv is not None:
        if clv > 0:  pts.append(f"Positive CLV (+{round(clv,1)}) shows we beat the close despite results.")
        elif clv < 0: pts.append(f"Negative CLV ({round(clv,1)}) means we paid above fair; tighten entries.")
    cold=[]
    for col in ["Market","Sport"]:
        if col in week_df.columns:
            g=week_df.groupby(col)["P&L"].sum().sort_values().head(2)
            for name,pnl in g.items():
                if pnl<0: cold.append(f"{name} ({nice_money(pnl)})")
    if cold: pts.append("Cold areas: " + ", ".join(cold) + " — temporarily de-prioritized.")
    actions=[
        "Reduce size to 1%/bet on underperforming markets.",
        "Only take prices with non-negative CLV trend into close.",
        "Skip late drifts against us; keep entry discipline."
    ]
    return "📚 Lessons Learned\n" + "\n".join([f"- {p}" for p in pts]) + "\n\n🔧 Actions\n" + "\n".join([f"- {a}" for a in actions])

def caption_telegram(m, ll_text=""):
    base = "\n".join([
        f"📊 Weekly Recap ({m['start']} → {m['end']})",
        f"Total: {m['total']} | ✅ {m['wins']} / ❌ {m['losses']} / 🤝 {m['pushes']}",
        f"Win Rate: {pct(m['wr'])} | ROI: {pct(m['roi'])} | Net: {nice_money(m['net'])}",
        f"Avg CLV: {('+' if m['avg_clv'] and m['avg_clv']>0 else '') + str(round(m['avg_clv'],1)) if m['avg_clv'] is not None else 'n/a'}",
        f"Top Sports: {m['top_sports']}",
        f"Top Markets: {m['top_markets']}",
    ])
    tail = "\n\n" + ll_text if ll_text else ""
    # no underscores-as-italics, no asterisks at all
    return base + tail + "\n\nSmarter Bets, Stronger Signals"


def caption_ig(m, ll_text=""):
    lines = [
        f"📊 Weekly Recap ({m['start']} → {m['end']})",
        f"Total: {m['total']} | ✅ {m['wins']} / ❌ {m['losses']} / 🤝 {m['pushes']}",
        f"Win Rate: {pct(m['wr'])} | ROI: {pct(m['roi'])} | Net: {nice_money(m['net'])}",
        f"Avg CLV: {('+' if m['avg_clv'] and m['avg_clv']>0 else '') + str(round(m['avg_clv'],1)) if m['avg_clv'] is not None else 'n/a'}",
        f"Top Sports: {m['top_sports']}",
        f"Top Markets: {m['top_markets']}",
    ]
    if ll_text: lines += ["", ll_text]
    lines += ["", "#SharpSignal #SportsBetting #ROI #CLV #Picks"]
    return "\n".join(lines)

def caption_x(m, roi_bad=False):
    blurb = "Lessons: tighten entries, size down." if roi_bad else "Focus: top sports/markets."
    out = (
        f"📊 Weekly Recap {m['start']}→{m['end']} | "
        f"Total {m['total']} | ✅{m['wins']} ❌{m['losses']} 🤝{m['pushes']} | "
        f"Win {pct(m['wr'])} | ROI {pct(m['roi'])} | Net {nice_money(m['net'])}. "
        f"{blurb} #SharpSignal"
    )
    return out[:279]

# =========================
# Telegram send + pin (photo + caption) with error logging
# =========================
MAX_TG_LEN = 4000

def telegram_send_and_pin(text, photo_path=None):
    if not TG_TOKEN or not TG_CHATS:
        print("⚠️ Telegram not configured (token or chat IDs missing).")
        return []

    # never send parse_mode; we’ll keep it plain text
    results = []
    text = text if len(text) <= MAX_TG_LEN else text[:MAX_TG_LEN-1] + "…"

    def _send(chat_id, txt):
        if photo_path and os.path.exists(photo_path):
            files = {"photo": open(photo_path, "rb")}
            data  = {"chat_id": chat_id, "caption": txt}
            return requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                                 data=data, files=files, timeout=30)
        else:
            data  = {"chat_id": chat_id, "text": txt}
            return requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                                 json=data, timeout=30)

    for chat_id in TG_CHATS:
        try:
            r = _send(chat_id, text)
            if r.status_code != 200:
                body = r.text
                print(f"❌ Telegram send failed ({chat_id}): {r.status_code} {body}")

                # Fallback: strip any stray markup & underscores that might look like entities
                if "can't parse entities" in body.lower():
                    clean = (text.replace("*", "")
                                 .replace("_", " ")
                                 .replace("`", ""))
                    r2 = _send(chat_id, clean[:MAX_TG_LEN])
                    if r2.status_code != 200:
                        print(f"❌ Telegram retry failed ({chat_id}): {r2.status_code} {r2.text}")
                        continue
                    r = r2  # use the successful retry response

            msg = r.json().get("result", {})
            message_id = msg.get("message_id")
            if message_id:
                pr = requests.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/pinChatMessage",
                    json={"chat_id": chat_id, "message_id": message_id, "disable_notification": True},
                    timeout=30
                )
                if pr.status_code != 200:
                    print(f"⚠️ Pin failed ({chat_id}): {pr.status_code} {pr.text}")
            results.append((chat_id, message_id))
        except Exception as e:
            logging.exception("Telegram error for chat %s", chat_id)
            print(f"❌ Telegram exception for {chat_id}: {e}")
    return results

# =========================
# Email (optional)
# =========================
def send_email(subject, body):
    if not EMAIL_TO or not EMAIL_SENDER: return
    if EMAIL_PROVIDER == "sendgrid":
        import json
        headers={"Authorization": f"Bearer {SENDGRID_API_KEY}","Content-Type":"application/json"}
        data={
            "personalizations":[{"to":[{"email":e} for e in EMAIL_TO]}],
            "from":{"email": EMAIL_SENDER},"subject": subject,
            "content":[{"type":"text/plain","value": body}]
        }
        r=requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, data=json.dumps(data))
        r.raise_for_status(); return
    msg=MIMEText(body,"plain","utf-8"); msg["Subject"]=subject; msg["From"]=EMAIL_SENDER; msg["To"]=",".join(EMAIL_TO)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(EMAIL_SENDER, EMAIL_PASS); server.sendmail(EMAIL_SENDER, EMAIL_TO, msg.as_string())

# =========================
# Main
# =========================
def main():
    try:
        df = load_allobservations_csv()
    except Exception as e:
        logging.exception("Failed to read CSV"); print(f"❌ Could not read CSV: {e}"); return

    week_df, now, week_ago, metrics = compute_week(df)
    if not metrics:
        print("No data for the last 7 days — nothing to report."); return

    roi_path = os.path.join(OUTDIR, "weekly_roi_by_day.png")
    clv_path = os.path.join(OUTDIR, "weekly_clv_hist.png")
    chart_roi_by_day(week_df, TZ, roi_path); chart_clv_hist(week_df, clv_path)

    roi_bad = metrics["roi"] < -0.15
    ll_text = lessons_learned(metrics, week_df) if roi_bad else ""

    t_msg = caption_telegram(metrics, ll_text)
    ig_msg = caption_ig(metrics, ll_text)
    x_msg  = caption_x(metrics, roi_bad)

    open(os.path.join(OUTDIR,"weekly_caption_telegram.txt"),"w",encoding="utf-8").write(t_msg)
    open(os.path.join(OUTDIR,"weekly_caption_ig.txt"),"w",encoding="utf-8").write(ig_msg)
    open(os.path.join(OUTDIR,"weekly_caption_x.txt"),"w",encoding="utf-8").write(x_msg)

    # Send to Telegram as photo+caption and pin
    telegram_send_and_pin(t_msg, photo_path=roi_path)

    # Optional email: use the fuller IG body
    try:
        subject=f"SharpSignal Weekly Recap ({metrics['start']} → {metrics['end']})"
        send_email(subject, ig_msg)
    except Exception: logging.exception("Email send failed")

    print("✅ Weekly recap saved to 'out/' and sent to Telegram (if configured).")

if __name__ == "__main__":
    main()
