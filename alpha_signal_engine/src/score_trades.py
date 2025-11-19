# src/score_trades.py
import joblib, pandas as pd, numpy as np
from pathlib import Path
from .preprocess_trades import NUM_COLS, CAT_COLS, _to_float, _parse_timestamp, _session_from_et_hour



BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODEL = DATA / "results" / "models" / "baseline_trades.pkl"

def prepare_for_scoring(df: pd.DataFrame):
    # Normalize numerics
    for c in ["Entry Price", "Exit Price", "SL", "TP", "RR", "Contracts", "P&L"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_float)
        else:
            df[c] = np.nan

    # Engineer the same features
    e, sl, tp = df["Entry Price"], df["SL"], df["TP"]
    df["Risk_abs"]     = (e - sl).abs()
    df["Reward_abs"]   = (tp - e).abs()
    df["RR_computed"]  = df["Reward_abs"] / df["Risk_abs"].replace(0, np.nan)
    df["DistToSL_pct"] = df["Risk_abs"] / e.replace(0, np.nan)
    df["DistToTP_pct"] = df["Reward_abs"] / e.replace(0, np.nan)
    df["HoldTimeMin"]  = np.nan
    ts = df["Timestamp"].apply(_parse_timestamp) if "Timestamp" in df.columns else [None]*len(df)
    df["Session"] = [ _session_from_et_hour(t) for t in ts ]

    # Ensure all columns exist
    for c in NUM_COLS:
        if c not in df.columns: df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns: df[c] = ""

    X = df[NUM_COLS + CAT_COLS].copy()
    return X

def main(in_csv="today_trades.csv", out_csv="today_trades_scored.csv"):
    bundle = joblib.load(MODEL)
    model, pre = bundle["model"], bundle["pre"]

    df = pd.read_csv(DATA / in_csv)
    X = prepare_for_scoring(df)
    X_pre = pre.transform(X)

    proba = model.predict_proba(X_pre)[:,1]
    df["Pred_Prob_Win"] = proba
    df.sort_values("Pred_Prob_Win", ascending=False).to_csv(DATA / out_csv, index=False)
    print(f"Scored {len(df)} trades → {DATA / out_csv}")

if __name__ == "__main__":
    main()
