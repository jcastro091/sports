# src/preprocess_infer.py
import numpy as np
import pandas as pd
from .preprocess import clean_direction, clean_movement  # clean_tags not needed here

def transform_for_inference(payload: dict, feature_columns: list) -> pd.DataFrame:
    row = {}
    # categoricals that training OHEs
    row["sport"] = payload.get("Sport") or ""
    row["market"] = payload.get("Market") or ""
    row["direction_clean"] = clean_direction(payload.get("Direction"))

    mts = payload.get("MinutesToStart")
    row["time_bin"] = ("unknown" if mts is None
                       else ("0–15m" if float(mts) <= 15 else
                             "15–60m" if float(mts) <= 60 else
                             "1–3h" if float(mts) <= 180 else
                             "3–12h" if float(mts) <= 720 else "12h+"))

    # limits (match training feature names)
    row["limit_trend"] = (str(payload.get("Limit Trend") or "").strip() or "unknown")
    bl = pd.to_numeric(payload.get("Bet Limit"), errors="coerce")
    row["bet_limit_raw"] = bl
    row["bet_limit_log"] = np.log1p(bl) if pd.notna(bl) else np.nan
    row["bet_limit_z"]   = np.nan  # unknown online

    # numbers we might have at inference
    row["movement"] = pd.to_numeric(clean_movement(payload.get("Movement")), errors="coerce")
    row["minutes_to_start"] = pd.to_numeric(payload.get("MinutesToStart"), errors="coerce")

    # odds/spread/total placeholders (training imputers will handle NaN)
    for k in ["close_imp","open_imp","curr_imp",
              "delta_am_close_open","delta_am_close_curr",
              "delta_imp_close_open","delta_imp_close_curr",
              "spread_home","spread_away","spread_home_abs","spread_away_abs",
              "spread_abs_min","spread_abs_max","home_fav_flag","away_fav_flag",
              "spread_dist_3","spread_dist_7","total_line"]:
        row[k] = np.nan

    df = pd.DataFrame([row])
    return df.reindex(columns=feature_columns, fill_value=np.nan)


