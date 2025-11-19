#!/usr/bin/env python3
"""
evaluate.py — Model evaluation for sports picks

Outputs:
- AUC (if both classes present)
- Overall win rate, ROI
- ROI & win rate by confidence buckets (quantiles)
- Optional CSV of bucket metrics

Assumes:
- proba: predicted probability that the PICK wins (0..1)
- label: 1 if the pick won, 0 if it lost
- odds:  American odds taken (e.g., -110, +145) [optional]
- stake: Units risked [optional -> defaults to 1]

Usage examples:
  python evaluate.py --input predictions.csv
  python evaluate.py --input predictions.csv --proba-col Pred --label-col Result --odds-col "Odds Taken" --buckets 10 --out buckets_today.csv
"""

import argparse
import sys
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
df = pd.read_csv("data/ConfirmedBets - ConfirmedTrades2.csv")
print(df.columns.tolist())


try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def american_to_decimal(american_odds: float) -> Optional[float]:
    """Convert American odds to Decimal odds. Returns None if nan/invalid."""
    try:
        if pd.isna(american_odds):
            return None
        o = float(american_odds)
        if o < 0:
            return 1.0 + (100.0 / abs(o))
        elif o > 0:
            return 1.0 + (o / 100.0)
        else:
            return None
    except Exception:
        return None


def profit_from_bet(label: int, dec_odds: Optional[float], stake: float = 1.0) -> float:
    """
    Profit given outcome and decimal odds.
      - If win: profit = stake * (dec_odds - 1)
      - If loss: profit = -stake
    If dec_odds missing -> assume even odds (2.0).
    """
    if dec_odds is None or not np.isfinite(dec_odds):
        dec_odds = 2.0  # even-odds proxy
    return stake * (dec_odds - 1.0) if label == 1 else -stake


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """Compute AUC if both classes are present and sklearn is available."""
    if roc_auc_score is None:
        return None, "sklearn not available; AUC skipped."
    uniq = np.unique(y_true[~pd.isna(y_true)])
    if len(uniq) < 2:
        return None, "Only one class present; AUC undefined."
    try:
        return float(roc_auc_score(y_true, y_score)), None
    except Exception as e:
        return None, f"AUC error: {e}"


def make_buckets(df: pd.DataFrame, proba_col: str, buckets: int) -> pd.Series:
    """
    Quantile buckets on predicted probability.
    Highest probabilities -> highest bucket index.
    """
    # Rank so that higher proba -> higher bucket
    # Use qcut to balance counts per bucket; handle duplicates via rank jitters
    # Add tiny noise to break ties deterministically by index
    eps = (np.arange(len(df)) % 997) * 1e-12
    probs = df[proba_col].to_numpy(dtype=float) + eps
    return pd.qcut(probs, q=buckets, labels=[f"B{i}" for i in range(1, buckets + 1)])


def evaluate(df: pd.DataFrame, proba_col: str, label_col: str,
             odds_col: Optional[str], stake_col: Optional[str],
             buckets: int) -> pd.DataFrame:
    """
    Returns a bucket summary DataFrame with:
      bucket, n, mean_proba, win_rate, avg_odds_dec, roi, cum_roi, hitrate_edge
    """
    work = df[[proba_col, label_col] + ([odds_col] if odds_col else []) + ([stake_col] if stake_col else [])].copy()

    # Basic cleaning
    work = work.dropna(subset=[proba_col, label_col])
    work[proba_col] = pd.to_numeric(work[proba_col], errors="coerce")

    # If label is blank, treat as 0 (loss) so evaluation runs, but warn
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int)

    # Stakes default to 1 (handle 'Stake' or 'Stake ' automatically)
    candidate_stake_cols = [stake_col] if stake_col else []
    candidate_stake_cols += ["Stake", "Stake "]  # tolerate either header
    stake_used = next((c for c in candidate_stake_cols if c in work.columns), None)
    if stake_used:
        work["stake_eval"] = pd.to_numeric(work[stake_used], errors="coerce").fillna(1.0).clip(lower=0.0)
    else:
        work["stake_eval"] = 1.0

    # Decimal odds if present
    if odds_col and odds_col in work.columns:
        work["dec_odds"] = work[odds_col].apply(american_to_decimal)
    else:
        work["dec_odds"] = np.nan

    # ---- Vectorized profit (no apply) ----
    dec = work["dec_odds"].fillna(2.0)  # assume even odds if missing
    win = work[label_col] == 1
    work["profit"] = np.where(win, work["stake_eval"] * (dec - 1.0), -work["stake_eval"])
    

    # Overall metrics
    total_stake = work["stake_eval"].sum()
    total_profit = work["profit"].sum()
    overall_roi = (total_profit / total_stake) if total_stake > 0 else np.nan
    overall_wr = work[label_col].mean() if len(work) else np.nan

    # AUC
    auc, auc_msg = safe_auc(work[label_col].to_numpy(), work[proba_col].to_numpy())

    # Buckets
    try:
        work["bucket"] = make_buckets(work, proba_col, buckets)
    except Exception:
        # Fall back to single bucket if too few rows
        work["bucket"] = "B1"
        buckets = 1

    by = work.groupby("bucket", observed=True, sort=False).agg(
        n=("profit", "size"),
        mean_proba=(proba_col, "mean"),
        win_rate=(label_col, "mean"),
        avg_odds_dec=("dec_odds", "mean"),
        stake=("stake_eval", "sum"),
        profit=("profit", "sum"),
    ).reset_index()

    by["roi"] = by["profit"] / by["stake"]
    by["hitrate_edge"] = by["win_rate"] - 0.5  # quick view vs even odds
    # Order buckets from highest confidence to lowest (last quantile is highest)
    by = by.sort_values("mean_proba", ascending=False, kind="mergesort").reset_index(drop=True)
    by["cum_profit"] = by["profit"].cumsum()
    by["cum_stake"] = by["stake"].cumsum()
    by["cum_roi"] = by["cum_profit"] / by["cum_stake"]

    # Attach headline metrics as attrs
    by.attrs["overall_n"] = int(len(work))
    by.attrs["overall_wr"] = float(overall_wr) if overall_wr == overall_wr else None
    by.attrs["overall_roi"] = float(overall_roi) if overall_roi == overall_roi else None
    by.attrs["auc"] = auc
    by.attrs["auc_msg"] = auc_msg
    return by


def print_summary(by: pd.DataFrame):
    overall_n = by.attrs.get("overall_n")
    overall_wr = by.attrs.get("overall_wr")
    overall_roi = by.attrs.get("overall_roi")
    auc = by.attrs.get("auc")
    auc_msg = by.attrs.get("auc_msg")

    print("\n===== MODEL EVALUATION =====")
    if auc is not None:
        print(f"AUC: {auc:0.3f}")
    else:
        print(f"AUC: (skipped) {auc_msg or ''}".strip())
    print(f"Samples: {overall_n}")
    if overall_wr is not None:
        print(f"Overall Win Rate: {overall_wr*100:0.1f}%")
    if overall_roi is not None:
        print(f"Overall ROI: {overall_roi*100:0.2f}%")

    print("\n-- ROI by Confidence Bucket (high → low) --")
    disp = by[["bucket", "n", "mean_proba", "win_rate", "roi", "cum_roi"]].copy()
    disp["mean_proba"] = (disp["mean_proba"] * 100.0).round(1)
    disp["win_rate"] = (disp["win_rate"] * 100.0).round(1)
    disp["roi"] = (disp["roi"] * 100.0).round(2)
    disp["cum_roi"] = (disp["cum_roi"] * 100.0).round(2)
    print(disp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Evaluate model: AUC + ROI by confidence bucket.")
    ap.add_argument("--input", required=True, help="Path to CSV with predictions and results.")
    
    ap.add_argument("--proba-col", default="Predicted", help="Predicted win probability column name.")
    ap.add_argument("--label-col", default="Prediction Result", help="Ground truth 1=win, 0=loss column name.")
    ap.add_argument("--odds-col", default="Odds (Am)", help="American odds column name (optional).")

    
    ap.add_argument("--stake-col", default="stake", help="Stake column name (optional).")
    ap.add_argument("--buckets", type=int, default=10, help="Number of confidence buckets (quantiles).")
    ap.add_argument("--out", default=None, help="Optional path to write bucket metrics CSV.")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read {args.input}: {e}", file=sys.stderr)
        sys.exit(2)

    for c in [args.proba_col, args.label_col]:
        if c not in df.columns:
            print(f"Missing required column: '{c}'", file=sys.stderr)
            print(f"Columns available: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)

    by = evaluate(
        df=df,
        proba_col=args.proba_col,
        label_col=args.label_col,
        odds_col=args.odds_col if args.odds_col in df.columns else None,
        stake_col=args.stake_col if args.stake_col in df.columns else None,
        buckets=args.buckets,
    )

    print_summary(by)

    if args.out:
        try:
            by.to_csv(args.out, index=False)
            print(f"\nSaved bucket metrics → {args.out}")
        except Exception as e:
            print(f"Could not write {args.out}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
