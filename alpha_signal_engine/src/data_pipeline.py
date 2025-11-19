import os
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime
from .db import get_con

def load_settings(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

RENAME_MAP = {
    "Timestamp":"timestamp","Sport":"sport","Away Team":"away_team","Home Team":"home_team",
    "Market":"market","Direction":"direction","Movement":"movement","Predicted":"predicted",
    "Game Time":"game_time","Spread Line Home":"spread_line_home","Spread Line Away":"spread_line_away",
    "Total Line":"total_line","Tags":"tags","Actual Winner":"actual_winner",
    "Prediction Result":"prediction_result","Risk":"risk","Kelly":"kelly","Confidence":"confidence",
    "Posted?":"posted","Plan":"plan","Odds Taken":"odds_taken","Confirmed By":"confirmed_by",
    "Confirmed At":"confirmed_at"
}

def _read_any(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def _to_number(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):               # e.g. "1.5%" -> 0.015
        try:
            return float(s[:-1]) / 100.0
        except:
            return pd.NA
    try:
        return float(s)
    except:
        return pd.NA

def ingest_allbets():
    s = load_settings()
    csv_path = s["paths"]["allbets_csv"]
    db_path  = s["storage"]["duckdb_file"]

    df = _read_any(csv_path)
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # --- clean timestamps ---
    for col in ("timestamp", "game_time", "confirmed_at"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\s(EDT|EST)$", "", regex=True)
            )
            # parse like "Jul 15, 08:00 PM"
            parsed = pd.to_datetime(df[col], format="%b %d, %I:%M %p", errors="coerce")
            df[col] = parsed
            # if you want UTC, uncomment:
            # df[col] = df[col].dt.tz_localize("US/Eastern").dt.tz_convert("UTC").dt.tz_localize(None)

    # --- numeric hygiene BEFORE writing to DB ---
    for col in ("kelly", "risk", "odds_taken"):
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    con = get_con(db_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("DROP TABLE IF EXISTS core_allbets;")
    con.register("allbets_view", df)
    con.execute("CREATE TABLE core_allbets AS SELECT * FROM allbets_view;")

    con.execute("""
        CREATE OR REPLACE VIEW core_allbets_clean AS
        SELECT
          *,
          CASE
            WHEN lower(prediction_result) IN ('win','won','w') THEN 1
            WHEN lower(prediction_result) IN ('lose','lost','l') THEN 0
            ELSE NULL
          END AS y_win
        FROM core_allbets;
    """)

    print(f"[ingest_allbets] Ingested {len(df)} rows into core_allbets at {db_path}")

if __name__ == "__main__":
    ingest_allbets()
