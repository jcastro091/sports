import pandas as pd

def summarize(df: pd.DataFrame) -> dict:
    out = {}
    if "y_win" in df.columns and df["y_win"].notna().any():
        out["win_rate"] = float(df["y_win"].mean(skipna=True))
    if "decimal_odds" in df.columns:
        stake = df.get("stake", pd.Series(1.0, index=df.index))
        wins = df.get("y_win", pd.Series(0, index=df.index)).fillna(0)
        # PnL per bet assuming decimal odds
        pnl = wins*stake*(df["decimal_odds"]-1) - (1-wins)*stake
        out["roi"] = float(pnl.sum()/stake.sum())
    if "kelly" in df.columns:
        out["avg_kelly"] = float(df["kelly"].dropna().mean()) if df["kelly"].notna().any() else None
    if "stake" in df.columns:
        out["avg_stake"] = float(df["stake"].mean())
    out["n_bets"] = int(len(df))
    return out
