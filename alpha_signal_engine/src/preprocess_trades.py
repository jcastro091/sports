# src/preprocess_trades.py
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

ET = pytz.timezone("US/Eastern")

# Model columns
NUM_COLS = [
    "Entry Price", "Exit Price", "SL", "TP", "RR", "Contracts", "P&L",
    "Risk_abs", "Reward_abs", "RR_computed", "DistToSL_pct", "DistToTP_pct",
    "HoldTimeMin",
]
CAT_COLS = ["Symbol", "Trade Direction", "Confidence", "Tag", "Session"]

# ---------------- helpers ----------------
def _to_float(x):
    if pd.isna(x): 
        return np.nan
    s = str(x).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def _parse_timestamp(ts):
    if pd.isna(ts):
        return None
    s = str(ts).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return ET.localize(datetime.strptime(s, fmt)) if "UTC" not in s else datetime.fromisoformat(s)
        except Exception:
            pass
    try:
        return pd.to_datetime(s).tz_localize(ET)
    except Exception:
        return None

def _session_from_et_hour(dt):
    if not dt:
        return None
    h = dt.hour
    if 2 <= h < 9:   return "Asia"
    if 9 <= h < 16:  return "London"
    return "NY"

# --------------- main prep ----------------
def clean_and_prepare_trades(csv_path: str):
    # 1) Read
    df = pd.read_csv(csv_path)

    # 2) Normalize header whitespace/case
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    # 3) Auto-map common variants → canonical names
    def _pick(*candidates):
        for name in candidates:
            if name in df.columns:
                return name
            low = name.lower()
            if low in cols_lower:
                return cols_lower[low]
        return None

    auto_map = {}
    pairs = {
        "Symbol": ["Symbol", "Ticker", "Pair", "Asset"],
        "Trade Direction": ["Trade Direction", "Side", "Direction", "Order Side"],
        "Entry Price": ["Entry Price", "Entry", "EntryPrice"],
        "SL": ["SL", "Stop Loss", "Stop", "StopLoss"],
        "TP": ["TP", "Take Profit", "Target", "TakeProfit"],
        "Exit Price": ["Exit Price", "Exit", "ExitPrice"],
        "P&L": ["P&L", "PnL", "Pnl", "Profit", "Profit/Loss"],
        "Confidence": ["Confidence", "Conf", "Conf Level"],
        "Tag": ["Tag", "Setup", "Strategy Tag"],
        "Timestamp": ["Timestamp", "Time", "Datetime", "Date"],
    }
    for canon, candidates in pairs.items():
        found = _pick(*candidates)
        if found and found != canon:
            auto_map[found] = canon
    if auto_map:
        df = df.rename(columns=auto_map)

    # 3b) Synthesize Symbol if missing (bets-like files with Away/Home)
    if "Symbol" not in df.columns:
        if {"Away", "Home"}.issubset(df.columns):
            df["Symbol"] = df["Away"].astype(str).str.strip() + "@" + df["Home"].astype(str).str.strip()
        elif "Market" in df.columns:
            df["Symbol"] = df["Market"].astype(str).str.strip()

    # 4) Normalize numeric fields (create columns if missing)
    for c in ["Entry Price", "Exit Price", "SL", "TP", "RR", "Contracts", "P&L"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_float)
        else:
            df[c] = np.nan

    # 5) Require the columns we need to build features
    required = ["Symbol", "Trade Direction", "Entry Price", "SL", "TP"]

    # ✳️ Diagnostics: show how many non-null rows per required column
    diag = {c: int(df[c].notna().sum()) if c in df.columns else 0 for c in required}
    total = len(df)
    print(f"[preprocess_trades] Rows total: {total} | non-nulls → " + ", ".join(f"{k}:{v}" for k,v in diag.items()))

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"[preprocess_trades] Missing required columns: {missing_cols}. "
            f"Present columns: {list(df.columns)}"
        )

    df_req = df.dropna(subset=required)
    if len(df_req) == 0:
        raise RuntimeError(
            "[preprocess_trades] 0 rows have ALL of: "
            f"{required}. Check your sheet: you need at least one trade with direction + entry + SL + TP."
        )

    df = df_req

    # 6) Target y (prefer explicit Result; else derive from P&L sign)
    if "Result" in df.columns:
        res = df["Result"].astype(str).str.strip().str.lower()
        y = res.map({"win": 1, "won": 1, "lose": 0, "loss": 0})
    else:
        y = None

    if y is None or y.isna().all():
        pnl = df["P&L"] if "P&L" in df.columns else pd.Series([np.nan] * len(df))
        y = np.where(pnl.notna(), (pnl >= 0).astype(int), np.nan)

    labeled_mask = pd.Series(~pd.isna(y))
    df_labeled   = df[labeled_mask].copy()
    y            = y[labeled_mask].astype(int)

    if len(df_labeled) == 0:
        raise RuntimeError(
            "[preprocess_trades] 0 labeled rows. "
            "Either add a 'Result' column (Win/Lose) or fill 'P&L' so labels can be derived."
        )

    # 7) Feature engineering
    e  = df_labeled["Entry Price"]
    sl = df_labeled["SL"]
    tp = df_labeled["TP"]

    df_labeled["Risk_abs"]     = (e - sl).abs()
    df_labeled["Reward_abs"]   = (tp - e).abs()
    df_labeled["RR_computed"]  = df_labeled["Reward_abs"] / df_labeled["Risk_abs"].replace(0, np.nan)
    df_labeled["DistToSL_pct"] = df_labeled["Risk_abs"] / e.replace(0, np.nan)
    df_labeled["DistToTP_pct"] = df_labeled["Reward_abs"] / e.replace(0, np.nan)
    df_labeled["HoldTimeMin"]  = np.nan

    ts = df_labeled["Timestamp"].apply(_parse_timestamp) if "Timestamp" in df_labeled.columns else pd.Series([None]*len(df_labeled))
    df_labeled["Session"] = [_session_from_et_hour(t) for t in ts]

    # 8) Ensure all model columns exist
    for c in NUM_COLS:
        if c not in df_labeled.columns: df_labeled[c] = np.nan
    for c in CAT_COLS:
        if c not in df_labeled.columns: df_labeled[c] = ""

    X = df_labeled[NUM_COLS + CAT_COLS].copy()

    # 9) Preprocessor
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])

    X_pre = pre.fit_transform(X)
    return X_pre, y, pre, NUM_COLS, CAT_COLS
