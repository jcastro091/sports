# alpha_signal_engine/src/thresholds_tuner.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.api.types import is_numeric_dtype

def _is_binary_numeric(series):
    if not is_numeric_dtype(series):
        return False
    vals = set(series.dropna().unique())
    # allow {0,1} or {0.0,1.0}
    return vals.issubset({0, 1, 0.0, 1.0})
    

def _guess_binary_result_col(df, preferred=None, fallbacks=None):
    """
    Pick a numeric 0/1 column to use as the true result.
    We ignore string columns like 'Actual Winner'.
    """
    fallbacks = fallbacks or []

    # 1) Preferred name if it exists and is binary numeric
    if preferred and preferred in df.columns and _is_binary_numeric(df[preferred]):
        print(f"[thresholds_tuner] Using {preferred!r} as binary result column")
        return preferred

    # 2) Fallback names if any are valid
    for fb in fallbacks:
        if fb in df.columns and _is_binary_numeric(df[fb]):
            print(f"[thresholds_tuner] Using {fb!r} as binary result column")
            return fb

    # 3) Search all numeric columns that look 0/1
    candidates = []
    for c in df.columns:
        if _is_binary_numeric(df[c]):
            candidates.append(c)

    if not candidates:
        raise KeyError(
            f"[thresholds_tuner] Could not find a numeric 0/1 result column. "
            f"Preferred={preferred!r}, fallbacks={fallbacks!r}, available={list(df.columns)!r}"
        )

    # 4) Rank candidates by name (prefer ones that look like a label)
    name_keys = ["result", "outcome", "label", "target", "win", "correct", "is_"]
    def score(col):
        lc = col.lower()
        hits = [lc.find(k) for k in name_keys if k in lc]
        return min(hits) if hits else 999

    candidates.sort(key=score)
    chosen = candidates[0]
    print(f"[thresholds_tuner] Guessed {chosen!r} as binary result column")
    return chosen


def _guess_column(df, preferred, fallbacks, kind):
    """
    Try to resolve a usable column name for probabilities / result / odds.
    - preferred: the name the caller asked for (e.g. 'ModelProba')
    - fallbacks: list of alternative names
    - kind: 'proba' | 'result' | 'odds'
    """
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # 1) exact match
    if preferred and preferred in df.columns:
        print(f"[thresholds_tuner] Using {preferred!r} as {kind} column")
        return preferred

    # 2) exact match on fallbacks
    for fb in fallbacks:
        if fb in df.columns:
            print(f"[thresholds_tuner] Using {fb!r} as {kind} column")
            return fb
        if fb.lower() in cols_lower:
            actual = cols_lower[fb.lower()]
            print(f"[thresholds_tuner] Using {actual!r} as {kind} column")
            return actual

    # 3) fuzzy match by substring
    for c in cols:
        lc = c.lower()
        if kind == "proba" and any(k in lc for k in ["proba", "prob", "pred"]):
            print(f"[thresholds_tuner] Guessed {c!r} as {kind} column")
            return c
        if kind == "result" and any(k in lc for k in ["result", "outcome", "label", "target", "win"]):
            print(f"[thresholds_tuner] Guessed {c!r} as {kind} column")
            return c
        if kind == "odds" and "odds" in lc:
            print(f"[thresholds_tuner] Guessed {c!r} as {kind} column")
            return c

    raise KeyError(
        f"[thresholds_tuner] Could not find a {kind} column. "
        f"Preferred={preferred!r}, fallbacks={fallbacks!r}, available={cols!r}"
    )


def _compute_profit_from_row(row, stake_col="Stake", odds_col="decimal_odds", result_col="Result"):
    """
    Compute profit assuming:
      - stake (unit) in stake_col (fallback = 1)
      - decimal odds in odds_col
      - result_col = 1 if win, 0 if loss
    """
    stake = float(row.get(stake_col, 1.0) or 1.0)
    dec = float(row[odds_col])
    win = int(row[result_col])

    # Profit for a single bet with 'stake' units at decimal odds:
    # win: stake * (dec - 1), loss: -stake
    return stake * (dec - 1.0) * win - stake * (1 - win)


def evaluate_cutoffs(df,
                     proba_col="ModelProba",
                     result_col="Result",
                     odds_col="decimal_odds",
                     stake_col="Stake"):
    """
    Given a dataframe with model probabilities + results + odds,
    evaluate a grid of probability cutoffs and compute ROI stats.
    """
    df = df.copy()

    # --- Probability column ---
    # For our pipeline, 'proba' is the real model probability column.
    if "proba" in df.columns and is_numeric_dtype(df["proba"]):
        proba_col = "proba"
        print("[thresholds_tuner] Using 'proba' as proba column")
    else:
        proba_col = _guess_column(
            df,
            preferred=proba_col,
            fallbacks=[
                "ModelProba",
                "model_proba",
                "PredProba",
                "Pred_Proba",
                "Model Probability",
                "probability",
            ],
            kind="proba",
        )

    if not is_numeric_dtype(df[proba_col]):
        raise TypeError(
            f"[thresholds_tuner] Probability column {proba_col!r} is not numeric. "
            f"Available dtypes: {df.dtypes.to_dict()}"
        )

    # --- Result column (must be numeric 0/1) ---
    # We ignore string columns like 'Actual Winner' here.
    result_col = _guess_binary_result_col(
        df,
        preferred=result_col,
        fallbacks=[
            "actual",        # <-- NEW (this is the column you have)
            "Result",
            "result",
            "y_true",
            "y",
            "Label",
            "Target",
            "is_win",
            "win_flag",
            "correct",
            "is_correct",
            "hit",
        ],
    )


    # Ensure it's numeric float/int
    df[result_col] = df[result_col].astype(float)

    # --- Odds column ---
    try:
        odds_col = _guess_column(
            df,
            preferred=odds_col,
            fallbacks=["decimal_odds", "DecimalOdds", "odds", "pinnacle_odds"],
            kind="odds",
        )
    except KeyError:
        print("[thresholds_tuner] No odds column; assuming decimal_odds = 2.0 for all rows.")
        odds_col = odds_col or "decimal_odds"
        df[odds_col] = 2.0

    # If odds column exists but is not numeric (e.g. 'odds_bin'), fall back to 2.0
    if not is_numeric_dtype(df[odds_col]):
        print(
            f"[thresholds_tuner] Odds column {odds_col!r} is not numeric; "
            "assuming decimal_odds = 2.0 for all rows."
        )
        odds_col = "decimal_odds"
        df[odds_col] = 2.0

    # --- Stake column ---
    if stake_col not in df.columns:
        print(f"[thresholds_tuner] No stake column {stake_col!r}; defaulting all stakes to 1.0.")
        df[stake_col] = 1.0

    # Drop rows missing any of the required fields
    df = df.dropna(subset=[proba_col, result_col, odds_col])
    
        # --- Compute per-row profit in "unit_profit" ---
    df["unit_profit"] = df.apply(
        _compute_profit_from_row,
        axis=1,
        stake_col=stake_col,
        odds_col=odds_col,
        result_col=result_col,
    )

    rows = []
    # grid from ~coin-flip up to strong favorite
    for cutoff in np.arange(0.50, 0.71, 0.01):   # 0.50, 0.51, ..., 0.70
        sub = df[df[proba_col] >= cutoff]
        n = len(sub)
        if n == 0:
            rows.append(
                {
                    "cutoff": cutoff,
                    "n_bets": 0,
                    "win_rate": None,
                    "avg_profit": None,
                    "roi": None,
                }
            )
            continue

        total_profit = sub["unit_profit"].sum()
        total_stake = sub[stake_col].sum()
        roi = total_profit / total_stake if total_stake > 0 else 0.0
        win_rate = sub[result_col].mean()

        rows.append(
            {
                "cutoff": cutoff,
                "n_bets": n,
                "win_rate": float(win_rate),
                "avg_profit": float(total_profit / max(n, 1)),
                "roi": float(roi),
            }
        )

    summary = pd.DataFrame(rows)
    summary["n_bets"] = summary["n_bets"].fillna(0).astype(int)
    return summary


def choose_thresholds(summary: pd.DataFrame, min_bets_A=30, min_bets_B=60, min_bets_C=100):
    """
    Given summary from evaluate_cutoffs, pick A/B/C thresholds.

    Heuristic:
      - A: highest cutoff with ROI > 3% and at least min_bets_A
      - B: highest cutoff below A with ROI > 1% and min_bets_B
      - C: highest cutoff below B with ROI >= 0 and min_bets_C
    """
    df = summary.dropna(subset=["roi"]).copy()

    # A tier
    A_df = df[(df["roi"] > 0.03) & (df["n_bets"] >= min_bets_A)]
    A_cut = A_df["cutoff"].max() if not A_df.empty else 0.60

    # B tier
    B_df = df[(df["cutoff"] < A_cut) & (df["roi"] > 0.01) & (df["n_bets"] >= min_bets_B)]
    B_cut = B_df["cutoff"].max() if not B_df.empty else min(A_cut - 0.02, 0.57)

    # C tier
    C_df = df[(df["cutoff"] < B_cut) & (df["roi"] >= 0.0) & (df["n_bets"] >= min_bets_C)]
    C_cut = C_df["cutoff"].max() if not C_df.empty else min(B_cut - 0.02, 0.53)

    # clamp + ordering
    cuts = sorted([A_cut, B_cut, C_cut], reverse=True)
    A_cut, B_cut, C_cut = cuts[0], cuts[1], cuts[2]

    thresholds = {
        "A": {"proba": round(float(A_cut), 4)},
        "B": {"proba": round(float(B_cut), 4)},
        "C": {"proba": round(float(C_cut), 4)},
    }
    return thresholds


def tune_and_save_thresholds(
    df: pd.DataFrame,
    output_path: Path,
    proba_col: str = "ModelProba",
    result_col: str = "Result",
    odds_col: str = "decimal_odds",
    stake_col: str = "Stake",
):
    """
    Main entrypoint to be called from train_model.py / train_sagemaker.py.
    """
    summary = evaluate_cutoffs(
        df,
        proba_col=proba_col,
        result_col=result_col,
        odds_col=odds_col,
        stake_col=stake_col,
    )
    thresholds = choose_thresholds(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    return thresholds, summary
