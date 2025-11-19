import pandas as pd

AMERICAN_RANGE = (101, 20000)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "odds_taken" in df.columns:
        df["implied_prob"] = df["odds_taken"].apply(_implied_prob)
        df["decimal_odds"] = df["odds_taken"].apply(_american_to_decimal)
    if "confidence" in df.columns:
        df["conf_num"] = df["confidence"].map({"✅ High":3,"🟡 Medium":2,"⚪️ Low":1}).fillna(0)
    # example: favorite/underdog tag from odds
    if "odds_taken" in df.columns:
        df["fav_flag"] = df["odds_taken"].apply(lambda o: 1 if _is_favorite(o) else 0)


    # === NEW: odds_edge ===
    if {"decimal_odds","implied_prob"}.issubset(df.columns):
        df["odds_edge"] = (1.0 / df["decimal_odds"]) - df["implied_prob"]

    # === NEW: market flags (one-hots) ===
    if "market" in df.columns:
        m = df["market"].str.upper().fillna("")
        df["market_h2h"]    = (m.str.contains("H2H") | m.eq("MONEYLINE")).astype(int)
        df["market_spread"] = m.str.contains("SPREAD").astype(int)
        df["market_totals"] = (m.str.contains("TOTAL") | m.str.contains("O/U") | m.str.contains("OVER/UNDER")).astype(int)

    return df
    
    
def add_odds_velocity(df: pd.DataFrame, group_cols=("sport","market")) -> pd.DataFrame:
    # If you have a time series of odds for same game, compute deltas.
    # Here, we stub with NA unless there are multiple rows per game.
    df = df.copy()
    # If you have unique game id + chronological rows, you can do:
    # df["odds_change"] = df.groupby(group_cols)["odds_taken"].apply(lambda s: s.diff()).reset_index(level=group_cols, drop=True)
    df["odds_change"] = pd.NA
    return df

def _is_favorite(o):
    try:
        o = float(o)
    except Exception:
        return False
    # American odds: favorite if negative
    if abs(o) >= AMERICAN_RANGE[0] and abs(o) <= AMERICAN_RANGE[1]:
        return o < 0
    # Decimal odds: favorite if < 2.0
    if 1.01 <= o <= 20:
        return o < 2.0
    return False

def _implied_prob(o):
    try:
        o = float(o)
    except Exception:
        return None
    # American
    if abs(o) >= AMERICAN_RANGE[0] and abs(o) <= AMERICAN_RANGE[1]:
        return (100/(o+100)) if o>0 else (abs(o)/(abs(o)+100))
    # Decimal
    if 1.01 <= o <= 20:
        return 1.0/o
    return None

def _american_to_decimal(o):
    try:
        o = float(o)
    except Exception:
        return None
    if abs(o) >= AMERICAN_RANGE[0] and abs(o) <= AMERICAN_RANGE[1]:
        return 1 + (o/100 if o>0 else 100/abs(o))
    if 1.01 <= o <= 20:
        return o
    return None
