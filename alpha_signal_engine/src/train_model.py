# src/train_model.py
# -------------------------------------------------------
# Train baseline + boosted models from CSV (or DuckDB if available).
# - Defaults to "ConfirmedBets - AllObservations.csv"
# - Rich features from opening/current/closing odds, deltas, time-to-start,
#   spreads/totals (NaN-safe), and categorical encodings.
# - NEW: LowVig & BOL sportsbook odds ingestion + consensus/outlier features
# - NEW: Odds sweet-spot binning (odds_bin) for longshots vs favorites
# - Models: LogisticRegression, RandomForest, HistGradientBoosting
# - Threshold tuning (F1) and model selection (AUC)
# - Saves segment reports, feature importances, and per-row predictions
# -------------------------------------------------------

from __future__ import annotations

import os
import argparse
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import joblib
import logging

import datetime
import boto3

from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.decomposition import PCA

# ---------- Paths ----------
# Engine root = parent directory of this src/ folder
ENGINE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ENGINE_ROOT / "data"
MODEL_DIR = DATA_DIR / "results" / "models"
MODEL_PKL = MODEL_DIR / "baseline_winloss.pkl"
FEATS_PKL = MODEL_DIR / "model_features.pkl"
SPORTS_REPO_ROOT = Path(os.getenv(
    "SPORTS_REPO_ROOT",
    r"C:\Users\Jcast\OneDrive\Documents\sports-repo"
))
TIER_CONFIG_OUT = SPORTS_REPO_ROOT / "tier_config.json"


# ---------- Helpers ----------
S3_LOGGER = logging.getLogger("s3_upload")


def _get_s3_client():
    """
    Returns a boto3 S3 client or None if not configured.
    """
    try:
        return boto3.client("s3")
    except Exception as e:
        S3_LOGGER.warning("Could not create S3 client: %s", e)
        return None


def upload_to_s3(local_path: str, prefix_env: str, default_prefix: str = ""):
    """
    Upload a local file to S3 using env vars:

      ML_BUCKET           - bucket name (required)
      <prefix_env>        - folder/prefix under the bucket (optional)

    Files are stored under: s3://ML_BUCKET/<prefix>/<UTC-timestamp>/<filename>
    """
    bucket = os.getenv("ML_BUCKET")
    if not bucket:
        S3_LOGGER.info("ML_BUCKET not set; skipping S3 upload for %s", local_path)
        return

    if not os.path.exists(local_path):
        S3_LOGGER.warning("Local file does not exist, skip upload: %s", local_path)
        return

    s3 = _get_s3_client()
    if s3 is None:
        return

    prefix = os.getenv(prefix_env, default_prefix).strip("/")
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = os.path.basename(local_path)

    key_parts = [p for p in [prefix, ts, filename] if p]
    key = "/".join(key_parts)

    try:
        s3.upload_file(local_path, bucket, key)
        S3_LOGGER.info("Uploaded %s -> s3://%s/%s", local_path, bucket, key)
    except Exception as e:
        S3_LOGGER.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, key, e)



def _resolve_data_file(name: str) -> Path:
    return DATA_DIR / name

def _print_header(title: str):
    print("\n" + title)
    print("=" * len(title))
    
    
# === PATCH D1: helper ===
from sklearn.metrics import roc_auc_score

def _safe_auc(y_true, y_score):
    # Works for binary targets; returns None if only one class present
    try:
        if getattr(y_true, "nunique", None):
            if y_true.nunique() < 2:
                return None
        else:
            # Fallback if y_true is a plain array/list
            if len(set(y_true)) < 2:
                return None
        return roc_auc_score(y_true, y_score)
    except Exception:
        return None
# === /PATCH D1 ===

    
# ---- Feature name utilities ----
# ---- Telegram alerting (optional) ----
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DRIFT_ALERT_CHAT_ID = os.getenv("DRIFT_ALERT_CHAT_ID", os.getenv("PRO_CHAT_ID", "-1002794017379")).strip()

def _tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not DRIFT_ALERT_CHAT_ID:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": DRIFT_ALERT_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[warn] telegram send failed: {e}")


def _make_ohe():
    """Create OneHotEncoder that outputs dense, compatible with all sklearn versions."""
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older sklearn
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def get_transformed_feature_names(pre: "ColumnTransformer", num_feats: list[str], cat_feats: list[str]) -> list[str]:
    """
    Build column names *after* the preprocessor (num + OHE cat).
    Works because we configured OHE for dense output.
    """
    out = []

    # numeric block names are passed through scaler but name preserved
    out.extend(num_feats)

    # categorical block -> OneHot categories
    # Find the 'cat' pipeline
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            ohe = None
            if hasattr(trans, "steps"):
                for step_name, step in trans.steps:
                    if step_name == "ohe":
                        ohe = step
                        break
            if ohe is None:
                break
            cats = ohe.categories_
            for feat_name, cat_vals in zip(cat_feats, cats):
                for cat in cat_vals:
                    out.append(f"{feat_name}={cat}")
            break
    return out
    
def _save_learning_curve_csv(estimator, preprocessor, X, y, out_path, cv=5):
    """Saves train/test AUC vs. train size to CSV for an estimator wrapped with your preprocessor."""
    try:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("pre", preprocessor), ("est", estimator)])
        sizes, train_scores, test_scores = learning_curve(
            pipe, X, y, cv=cv, scoring="roc_auc",
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True, random_state=42
        )
        df_lc = pd.DataFrame({
            "train_size": sizes,
            "train_auc_mean": train_scores.mean(axis=1),
            "train_auc_std":  train_scores.std(axis=1),
            "val_auc_mean":   test_scores.mean(axis=1),
            "val_auc_std":    test_scores.std(axis=1),
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_lc.to_csv(out_path, index=False)
        print(f"📝 Saved learning curve → {out_path}")
    except Exception as e:
        print(f"[warn] learning curve failed: {e}")



# ============================
# CSV loader with rich features (NaN-safe)
# ============================
LEAKAGE_COLS = {
    "model probability", "edge",
    "tier code", "tier label",
    "strong flag", "strong flag (yes/no)"
}
def _drop_leakage(df_):
    cols_l = {c.lower(): c for c in df_.columns}
    to_drop = [cols_l[c] for c in cols_l if c in LEAKAGE_COLS]
    if to_drop:
        print(f"[leakage-guard] Dropping columns: {to_drop}")
        return df_.drop(columns=to_drop)
    return df_



def _export_weekly_metrics(dates, y_true, proba, out_dir):
    d = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce", utc=True),
        "y": y_true.astype(int).to_numpy(),
        "proba": proba.astype(float),
    }).dropna(subset=["date", "y", "proba"])
    if d.empty:
        print("[weekly] no valid rows")
        return
    d["week_start"] = d["date"].dt.to_period("W").apply(lambda r: r.start_time)
    rows = []
    for w, g in d.groupby("week_start"):
        try:
            auc = roc_auc_score(g["y"], g["proba"]) if g["y"].nunique() > 1 else np.nan
        except Exception:
            auc = np.nan
        pred = (g["proba"] >= 0.5).astype(int)
        acc = accuracy_score(g["y"], pred)
        rows.append({"week_start": w, "n": int(len(g)), "auc": float(auc) if auc==auc else None, "accuracy": float(acc)})
    out = out_dir / "weekly_metrics.csv"
    pd.DataFrame(rows).sort_values("week_start").to_csv(out, index=False)
    print(f"📆 Saved weekly metrics → {out}")


def _psi(a, b, bins=10):
    """Population Stability Index between two probability distributions (0..1)."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 50 or len(b) < 50:
        return np.nan
    qs = np.linspace(0, 1, bins + 1)
    cuts = a.quantile(qs).unique()
    a_hist = pd.cut(a, cuts, include_lowest=True)
    b_hist = pd.cut(b, cuts, include_lowest=True)
    a_pct = a_hist.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    b_pct = b_hist.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    return float(np.sum((a_pct - b_pct) * np.log(a_pct / b_pct)))

def run_drift_checks(model, X_train, y_train, X_test, y_test, out_dir: Path, alert=True):
    """Compute recent AUC vs prior, PSI on proba, and emit Telegram alerts on degradation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts = []

    # Current test metrics / probabilities
    try:
        proba_test = model.predict_proba(X_test)[:, 1]
        auc_test = _safe_auc(y_test, proba_test)
    except Exception:
        proba_test, auc_test = None, None

    # Train set probabilities (to compare distribution shift)
    try:
        proba_train = model.predict_proba(X_train)[:, 1]
        auc_train = _safe_auc(y_train, proba_train)
    except Exception:
        proba_train, auc_train = None, None

    # PSI between train and test probas
    psi = _psi(proba_train, proba_test) if (proba_train is not None and proba_test is not None) else np.nan

    # Weekly trend (if exists)
    weekly_csv = out_dir / "weekly_metrics.csv"
    week_drop = None
    if weekly_csv.exists():
        w = pd.read_csv(weekly_csv, parse_dates=["week_start"])
        w = w.sort_values("week_start")
        if "auc" in w.columns and w["auc"].notna().sum() >= 3:
            last3 = w["auc"].dropna().tail(3).tolist()
            if len(last3) == 3:
                week_drop = float(last3[-1] - last3[-3])  # AUC change over ~ 3 weeks

    # Thresholds (tune to taste)
    if psi is not np.nan and psi >= 0.2:
        alerts.append(f"PSI high: {psi:.3f} (≥ 0.20)")
    if auc_test is not None and auc_train is not None and (auc_train - auc_test) >= 0.07:
        alerts.append(f"Generalization gap large: train AUC {auc_train:.3f} → test AUC {auc_test:.3f} (Δ ≥ 0.07)")
    if week_drop is not None and week_drop <= -0.05:
        alerts.append(f"Recent weekly AUC drop: {week_drop:.3f} (≤ -0.05)")

    # Persist a tiny health file
    pd.DataFrame([{
        "auc_train": auc_train, "auc_test": auc_test, "psi_train_vs_test": psi, "weekly_auc_delta_3w": week_drop
    }]).to_csv(out_dir / "model_health_latest.csv", index=False)

    if alerts and alert:
        msg = "⚠️ <b>Model health alert</b>\n" + "\n".join(f"• {a}" for a in alerts)
        _tg_send(msg)
    else:
        print("✅ Model health OK")


def build_from_csv(csv_name: str = "ConfirmedBets - AllObservations.csv"):
    """
    CSV loader with richer features:
      Numeric (odds/time):
        - implied probs & decimal odds from opening/current/closing American odds
        - deltas: closing-open, closing-current (cents and implied-prob deltas)
        - movement (if present), minutes_to_start (+ time_bin categorical)
      Numeric (spread/total):
        - spread_home, spread_away (+ abs, min/max)
        - favorite flags from spread signs
        - distance to key numbers 3 and 7 (football-like sports only)
        - total_line
      Categorical:
        - sport, market, direction_clean, time_bin, limit_trend
      Betting limits:
        - raw, log, and per-sport/market z-score
      NEW Sportsbook signals:
        - LowVig & BOL open/curr/close odds -> implied probs
        - cross-book deltas vs baseline, consensus, dispersion, outlier flag
      Target:
        - Prefer 'Prediction Result' (0/1); else derive from 'Actual Winner' == 'Predicted'.
    All steps are NaN-safe; preprocessor imputes missing values.
    """
    def to_float(x):
        try:
            if isinstance(x, str):
                x = x.strip().replace('%', '')
            return float(x)
        except Exception:
            return np.nan

    def american_to_decimal(am):
        try:
            am = float(am)
        except Exception:
            return np.nan
        if am > 0:
            return 1.0 + (am / 100.0)
        if am < 0:
            return 1.0 + (100.0 / abs(am))
        return np.nan

    def implied_prob_from_american(am):
        try:
            am = float(am)
        except Exception:
            return np.nan
        if am > 0:
            return 100.0 / (am + 100.0)
        if am < 0:
            return abs(am) / (abs(am) + 100.0)
        return np.nan

    def clean_direction(s):
        if not isinstance(s, str):
            return ""
        t = s.strip().lower()
        if t.startswith("home"):
            return "Home"
        if t.startswith("away"):
            return "Away"
        if "spread favorite" in t:
            return "Spread Favorite"
        if "spread underdog" in t:
            return "Spread Underdog"
        if t == "underdog":
            return "Underdog"
        if t == "over":
            return "Over"
        if t == "under":
            return "Under"
        return "Other"

    def time_bin(mins):
        try:
            m = float(mins)
        except Exception:
            return "unknown"
        if m < 0:    return "<0m"
        if m <= 15:  return "0–15m"
        if m <= 60:  return "15–60m"
        if m <= 180: return "1–3h"
        if m <= 720: return "3–12h"
        return "12h+"

    def is_football_like(sport):
        if not isinstance(sport, str): return False
        t = sport.lower()
        return ("football" in t) or ("nfl" in t) or ("cfb" in t) or ("ncaa fb" in t)

    # ---- NEW: sportsbook helpers ----
    def find_col_case_insensitive(df, *cands):
        cols = {c.lower(): c for c in df.columns}
        for c in cands:
            if c and c.lower() in cols:
                return cols[c.lower()]
        return None

    def add_book_odds(df_target, source_df, book_name: str):
        """
        Attach book-specific odds features (open/curr/close) for a sportsbook.
        Returns the list of newly created columns.
        """
        open_cand = find_col_case_insensitive(source_df,
            f"{book_name} Opening Odds (Am)", f"{book_name} Opening Odds",
            f"{book_name} Opening Price", f"{book_name} Open (Am)",
            f"{book_name}_opening_odds_am", f"{book_name}_open_am"
        )
        curr_cand = find_col_case_insensitive(source_df,
            f"{book_name} Current Odds (Am)", f"{book_name} Current Odds",
            f"{book_name} Curr (Am)", f"{book_name}_current_odds_am", f"{book_name}_curr_am"
        )
        close_cand = find_col_case_insensitive(source_df,
            f"{book_name} Closing Odds (Am)", f"{book_name} Closing Odds",
            f"{book_name} Close (Am)", f"{book_name}_closing_odds_am", f"{book_name}_close_am"
        )

        new_cols = []
        def mk_odds(series):
            dec = series.apply(american_to_decimal)
            imp = series.apply(implied_prob_from_american)
            return dec, imp

        def add_one(kind, colname):
            base = f"{book_name.lower()}_{kind}"
            if colname is None:
                df_target[f"{base}_am"]  = np.nan
                df_target[f"{base}_dec"] = np.nan
                df_target[f"{base}_imp"] = np.nan
            else:
                df_target[f"{base}_am"]  = pd.to_numeric(source_df[colname], errors="coerce")
                dec, imp = mk_odds(df_target[f"{base}_am"])
                df_target[f"{base}_dec"] = dec
                df_target[f"{base}_imp"] = imp
            new_cols.extend([f"{base}_am", f"{base}_dec", f"{base}_imp"])

        add_one("open",  open_cand)
        add_one("curr",  curr_cand)
        add_one("close", close_cand)
        return new_cols

    def best_imp(row, pref_list):
        """Pick the first non-null implied prob from a preference list of columns."""
        for c in pref_list:
            v = row.get(c, np.nan)
            if pd.notna(v):
                return v
        return np.nan

    # Odds sweet-spot binning (use implied probability)
    def odds_bin(imp):
        if pd.isna(imp): return "unknown"
        if imp >= 0.70:  return "HeavyFav"   # e.g., -233 and shorter
        if imp >= 0.55:  return "Fav"        # about -122 to -233
        if imp >= 0.40:  return "Balanced"   # -150 to +150 area
        if imp >= 0.20:  return "Dog"        # about +150 to +400
        return "Longshot"                     # > +400 (e.g., +1200)

    # --- load ---
    csv_path = _resolve_data_file(csv_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _drop_leakage(df)

    # case-insensitive column fetcher
    cols = {c.lower(): c for c in df.columns}
    def col(*cands):
        for c in cands:
            k = c.lower()
            if k in cols:
                return cols[k]
        return None

    # --- target y ---
    y_col = col("Prediction Result", "prediction_result", "Predicted Result")
    if y_col and df[y_col].notna().any():
        y = pd.to_numeric(df[y_col], errors="coerce").dropna().astype(int)
    else:
        actual_col = col("Actual Winner", "actual_winner")
        pred_col   = col("Predicted", "predicted")
        if not actual_col or not pred_col:
            raise ValueError("No labeled rows: need 'Prediction Result' OR both 'Actual Winner' and 'Predicted'.")
        act = df[actual_col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        prd = df[pred_col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        y = (act == prd).astype("float").dropna().astype(int)

    # --- raw columns ---
    sport_col  = col("Sport", "sport")
    market_col = col("Market", "market")
    dir_col    = col("Direction", "direction", "Direction_clean", "direction_clean")

    close_col  = col("Closing Odds (Am)", "Odds (Am)", "Closing Odds", "closing_odds_am")
    open_col   = col("Opening Price", "Opening Odds (Am)", "Opening Odds", "opening_odds_am")
    curr_col   = col("Curr", "Current Odds (Am)", "Current Odds", "current_odds_am")

    move_col   = col("Movement", "movement")
    mts_col    = col("MinutesToStart", "minutes_to_start", "minutesToStart")

    spread_home_col = col("Spread Line Home", "home spread", "spread line home", "spread_home")
    spread_away_col = col("Spread Line Away", "away spread", "spread line away", "spread_away")
    total_col       = col("Total Line", "total line", "total", "total_line")
    
    limit_col      = col("Bet Limit", "bet_limit", "Limit")
    limit_trend_col= col("Limit Trend", "limit_trend")

    dfm = df.loc[y.index].copy()

    # --- categoricals ---
    dfm["sport"] = dfm[sport_col] if sport_col else ""
    dfm["market"] = dfm[market_col] if market_col else ""
    dfm["direction_clean"] = (dfm[dir_col].apply(clean_direction) if dir_col else "")
    dfm["time_bin"] = (dfm[mts_col].apply(time_bin) if mts_col else "unknown")
    
    dfm["bet_limit_raw"] = pd.to_numeric(dfm[limit_col], errors="coerce") if limit_col else np.nan
    # log scale to tame heavy tails
    dfm["bet_limit_log"] = np.log1p(dfm["bet_limit_raw"])

    # per sport+market normalization (liquidity is sport/market dependent)
    grp = dfm.groupby([dfm["sport"].astype(str), dfm["market"].astype(str)])[["bet_limit_raw"]]
    # avoid division by zero for std
    grp_mean = grp.transform("mean")
    grp_std = grp.transform("std").replace(0, np.nan)
    dfm["bet_limit_z"] = (dfm["bet_limit_raw"] - grp_mean["bet_limit_raw"]) / grp_std["bet_limit_raw"]

    # categorical trend (e.g., "Rising", "Flat", "Falling")
    dfm["limit_trend"] = (dfm[limit_trend_col].astype(str) if limit_trend_col else "unknown")

    # --- odds features ---
    def mk_odds_base(series):
        dec = series.apply(american_to_decimal)
        imp = series.apply(implied_prob_from_american)
        return dec, imp

    if close_col:
        dfm["close_am"] = pd.to_numeric(dfm[close_col], errors="coerce")
        dfm["close_dec"], dfm["close_imp"] = mk_odds_base(dfm["close_am"])
    else:
        dfm["close_am"] = np.nan; dfm["close_dec"] = np.nan; dfm["close_imp"] = np.nan

    if open_col:
        dfm["open_am"] = pd.to_numeric(dfm[open_col], errors="coerce")
        dfm["open_dec"], dfm["open_imp"] = mk_odds_base(dfm["open_am"])
    else:
        dfm["open_am"] = np.nan; dfm["open_dec"] = np.nan; dfm["open_imp"] = np.nan

    if curr_col:
        dfm["curr_am"] = pd.to_numeric(dfm[curr_col], errors="coerce")
        dfm["curr_dec"], dfm["curr_imp"] = mk_odds_base(dfm["curr_am"])
    else:
        dfm["curr_am"] = np.nan; dfm["curr_dec"] = np.nan; dfm["curr_imp"] = np.nan

    dfm["movement"] = (dfm[move_col].apply(to_float) if move_col else np.nan)

    dfm["delta_am_close_open"]  = dfm["close_am"] - dfm["open_am"]
    dfm["delta_am_close_curr"]  = dfm["close_am"] - dfm["curr_am"]
    dfm["delta_imp_close_open"] = dfm["close_imp"] - dfm["open_imp"]
    dfm["delta_imp_close_curr"] = dfm["close_imp"] - dfm["curr_imp"]

    dfm["minutes_to_start"] = pd.to_numeric(dfm[mts_col], errors="coerce") if mts_col else np.nan

    # ---- NEW: add LowVig & BOL sportsbook features ----
    _ = add_book_odds(dfm, df.loc[y.index], "LowVig")
    _ = add_book_odds(dfm, df.loc[y.index], "BOL")

    # Cross-book deltas vs baseline (your baseline is open_imp/curr_imp/close_imp)
    for book in ["lowvig", "bol"]:
        for kind in ["open", "curr", "close"]:
            book_imp = f"{book}_{kind}_imp"
            base_imp = f"{kind}_imp"
            book_am = f"{book}_{kind}_am"
            base_am = f"{kind}_am"
            if book_imp in dfm.columns and base_imp in dfm.columns:
                dfm[f"delta_imp_{book}_{kind}_vs_base"] = dfm[book_imp] - dfm[base_imp]
            if book_am in dfm.columns and base_am in dfm.columns:
                dfm[f"delta_cents_{book}_{kind}_vs_base"] = dfm[book_am] - dfm[base_am]

    # Consensus across available books (baseline + LowVig + BOL)
    imp_cols_for_consensus = [c for c in ["close_imp", "lowvig_close_imp", "bol_close_imp"] if c in dfm.columns]
    if len(imp_cols_for_consensus) == 0:
        dfm["consensus_close_imp"] = np.nan
    else:
        dfm["consensus_close_imp"] = dfm[imp_cols_for_consensus].median(axis=1, skipna=True)

    # Outlier flags: is any book > X abs-diff from consensus?
    def outlier_flag(row, cols, thresh=0.05):  # 5 percentage points in implied prob
        vals = [row.get(c, np.nan) for c in cols]
        vals = [v for v in vals if pd.notna(v)]
        if len(vals) < 2:
            return 0.0
        cons = row.get("consensus_close_imp", np.nan)
        if pd.isna(cons):
            return 0.0
        return 1.0 if any(abs(v - cons) >= thresh for v in vals) else 0.0

    dfm["book_outlier_flag"] = dfm.apply(
        lambda r: outlier_flag(r, imp_cols_for_consensus, thresh=0.05),
        axis=1
    )

    # Dispersion between books at close (signal of disagreement)
    if len(imp_cols_for_consensus) >= 2:
        dfm["book_imp_range_close"] = (dfm[imp_cols_for_consensus].max(axis=1) - 
                                       dfm[imp_cols_for_consensus].min(axis=1))
    else:
        dfm["book_imp_range_close"] = np.nan

    # ---- NEW: odds sweet-spot bin, choose the best available close implied prob on each row
    dfm["chosen_close_imp_for_bin"] = dfm.apply(
        lambda r: best_imp(r, ["close_imp", "lowvig_close_imp", "bol_close_imp", "curr_imp", "open_imp"]),
        axis=1
    )
    dfm["odds_bin"] = dfm["chosen_close_imp_for_bin"].apply(odds_bin)

    # --- spread / total features ---
    def to_spread(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"pk", "pick", "pickem", "pick'em"}:
                return 0.0
        return to_float(x)

    dfm["spread_home"] = dfm[spread_home_col].apply(to_spread) if spread_home_col else np.nan
    dfm["spread_away"] = dfm[spread_away_col].apply(to_spread) if spread_away_col else np.nan

    dfm["spread_home_abs"] = np.abs(dfm["spread_home"])
    dfm["spread_away_abs"] = np.abs(dfm["spread_away"])

    # row-wise min/max without warnings
    if spread_home_col or spread_away_col:
        pair = pd.concat([dfm["spread_home_abs"], dfm["spread_away_abs"]], axis=1)
        dfm["spread_abs_min"] = pair.min(axis=1, skipna=True)
        dfm["spread_abs_max"] = pair.max(axis=1, skipna=True)
    else:
        dfm["spread_abs_min"] = np.nan
        dfm["spread_abs_max"] = np.nan

    # favorite flags (negative spread -> favorite)
    dfm["home_fav_flag"] = (dfm["spread_home"] < 0).astype(float)
    dfm["away_fav_flag"] = (dfm["spread_away"] < 0).astype(float)

    # distance to 3 & 7 for football-like sports; else NaN
    is_fb = dfm["sport"].apply(is_football_like)
    dfm.loc[is_fb, "spread_dist_3"] = np.abs(dfm.loc[is_fb, "spread_abs_min"] - 3.0)
    dfm.loc[is_fb, "spread_dist_7"] = np.abs(dfm.loc[is_fb, "spread_abs_min"] - 7.0)
    dfm.loc[~is_fb, ["spread_dist_3", "spread_dist_7"]] = np.nan

    # totals
    dfm["total_line"] = dfm[total_col].apply(to_float) if total_col else np.nan

    # (optional) gate features by market
    # If Market doesn't contain "spread", blank spread features; if not "total", blank totals.
    mkt = dfm["market"].astype(str).str.lower()
    is_spread = mkt.str.contains("spread")
    is_total  = mkt.str.contains("total")
    spread_cols = ["spread_home","spread_away","spread_home_abs","spread_away_abs",
                   "spread_abs_min","spread_abs_max","home_fav_flag","away_fav_flag",
                   "spread_dist_3","spread_dist_7"]
    for c in spread_cols:
        dfm.loc[~is_spread, c] = np.nan
    if "total_line" in dfm.columns:
        dfm.loc[~is_total, "total_line"] = np.nan

    # --- feature lists ---
    num_feats = [
        # odds/implied
        "close_imp", "open_imp", "curr_imp",
        "delta_am_close_open", "delta_am_close_curr",
        "delta_imp_close_open", "delta_imp_close_curr",
        # time/movement
        "movement", "minutes_to_start",
        # spread
        "spread_home", "spread_away",
        "spread_home_abs", "spread_away_abs",
        "spread_abs_min", "spread_abs_max",
        "home_fav_flag", "away_fav_flag",
        "spread_dist_3", "spread_dist_7",
        # totals
        "total_line",
        # Betting Limits
        "bet_limit_raw", "bet_limit_log", "bet_limit_z",
        # NEW: consensus/dispersion & book deltas
        "consensus_close_imp", "book_imp_range_close", "book_outlier_flag",
        # NEW: raw book implied probs (so model can directly learn levels)
        "lowvig_open_imp", "lowvig_curr_imp", "lowvig_close_imp",
        "bol_open_imp",    "bol_curr_imp",    "bol_close_imp",
        # NEW: cross-book deltas vs baseline
        "delta_imp_lowvig_open_vs_base", "delta_imp_lowvig_curr_vs_base", "delta_imp_lowvig_close_vs_base",
        "delta_imp_bol_open_vs_base",    "delta_imp_bol_curr_vs_base",    "delta_imp_bol_close_vs_base",
        "delta_cents_lowvig_open_vs_base","delta_cents_lowvig_curr_vs_base","delta_cents_lowvig_close_vs_base",
        "delta_cents_bol_open_vs_base",   "delta_cents_bol_curr_vs_base",   "delta_cents_bol_close_vs_base",
    ]
    cat_feats = ["sport", "market", "direction_clean", "time_bin", "limit_trend",
                 # NEW categorical: odds sweet-spot bin
                 "odds_bin"]

    # Drop numeric columns that are entirely NaN
    num_feats = [c for c in num_feats if c in dfm.columns and dfm[c].notna().any()]
    if not num_feats:
        dfm["num_dummy"] = 0.0
        num_feats = ["num_dummy"]

    # Ensure categoricals are strings
    for c in cat_feats:
        dfm[c] = dfm[c].astype(str).fillna("")

    X = dfm[num_feats + cat_feats]
    y = y.loc[X.index]

    # Preprocessor (dense output for all models)
    # === PATCH B (final): single preprocessor with mean-centered numerics + dense OHE ===
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    cat_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe()),  # dense OHE
    ])

    pre = ColumnTransformer([
        ("num", numeric_transformer, num_feats),
        ("cat", cat_transformer,     cat_feats),
    ])
    # =====================================



    return X, y, pre, num_feats, cat_feats

# ============================
# Threshold tuning (F1 by default)
# ============================
def tune_threshold(y_true, y_proba, objective="f1"):
    """
    Scan thresholds and return the best by objective ('f1' or 'accuracy').
    """
    best = {"thr": 0.5, "f1": -1, "acc": -1, "prec": -1, "rec": -1}
    for thr in np.linspace(0.30, 0.70, 41):  # 0.30 .. 0.70 step 0.01
        y_hat = (y_proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="weighted", zero_division=0)
        
        acc = accuracy_score(y_true, y_hat)
        if (objective == "f1" and f1 > best["f1"]) or (objective == "accuracy" and acc > best["acc"]):
            best.update({"thr": float(thr), "f1": float(f1), "acc": float(acc), "prec": float(prec), "rec": float(rec)})
    return best
    
    
# ============================
# Hyperparameter tuning (RandomizedSearchCV, AUC)
# ============================
def build_search_spaces():
    # Logistic Regression (tries L2, L1, and ElasticNet)
    lr_space = {
        "clf__penalty": ["l2", "l1", "elasticnet"],
        "clf__C": loguniform(1e-3, 1e2),
        "clf__l1_ratio": uniform(0, 1),  # only used for elasticnet; ignored otherwise
    }

    # Random Forest
    rf_space = {
        "rf__n_estimators": randint(200, 801),
        "rf__max_depth": [None, 5, 10, 15, 20, 30],
        "rf__min_samples_split": randint(2, 11),
        "rf__min_samples_leaf": randint(1, 11),
        "rf__max_features": uniform(0.3, 0.7),  # fraction of features
    }

    # HistGradientBoosting
    hgb_space = {
        "hgb__learning_rate": loguniform(1e-3, 3e-1),
        "hgb__max_depth": [None, 3, 6, 12],
        "hgb__max_iter": randint(200, 1001),
        "hgb__min_samples_leaf": randint(10, 61),
        "hgb__l2_regularization": loguniform(1e-8, 1e-1),
    }

    return lr_space, rf_space, hgb_space


def hyperparameter_tune(X, y, pre, n_iter=25, cv_splits=5, random_state=42):
    """
    Returns tuned (best_estimator_) models: dict(name -> pipeline).
    Uses AUC for ranking. Falls back gracefully if sample size is small.
    """
    logreg, rf, hgb = Pipeline([("pre", pre), ("clf", LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced"))]), \
                      Pipeline([("pre", pre), ("rf", RandomForestClassifier(class_weight="balanced", random_state=random_state))]), \
                      Pipeline([("pre", pre), ("hgb", HistGradientBoostingClassifier(random_state=random_state))])

    lr_space, rf_space, hgb_space = build_search_spaces()

    if isinstance(cv_splits, int) and cv_splits >= 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    else:
        cv = 3  # minimal fallback

    searches = [
        ("Logistic Regression", logreg, lr_space),
        ("Random Forest", rf, rf_space),
        ("HistGradientBoosting (GBT)", hgb, hgb_space),
    ]

    tuned = {}
    for name, pipe, space in searches:
        try:
            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=space,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
            print(f"\n🔎 Tuning {name} ...")
            rs.fit(X, y)
            print(f"✅ {name} best AUC: {rs.best_score_:.3f}")
            print(f"   Best params: {rs.best_params_}")
            tuned[name] = rs.best_estimator_
        except Exception as e:
            print(f"[warn] Tuning {name} skipped: {e}")
    return tuned


# ============================
# Train / Evaluate / Select model
# ============================
def train_models(dataset_name: str | None = None):
    # Build from CSV (we keep DuckDB optional; CSV is robust)
    csv_name = dataset_name or os.getenv("DATASET_NAME", "ConfirmedBets - AllObservations.csv")
    X, y, pre, num_feats, cat_feats = build_from_csv(csv_name)

    # === PATCH A: make target strictly binary BEFORE counts/split ===
    y = y.replace({2: 1}).astype(int)
    # =====================================

    # Diagnostics
    counts = Counter(y)
    
    # === Split logic (chronological-first, else stratified/random) ===========
    print(f"[train_model] Class counts: {dict(counts)}")

    DO_CHRONO = os.getenv("DO_CHRONO", "0") == "1"
    X_train = X_test = y_train = y_test = None
    cv_splits = None  # will set later depending on split

    def _find_col_ci(df, *names):
        m = {c.lower(): c for c in df.columns}
        for n in names:
            if n and n.lower() in m:
                return m[n.lower()]
        return None

    chrono_done = False
    chrono_info = None

    if DO_CHRONO:
        # Re-open the same CSV to fetch a date/time column aligned to X/y indices.
        csv_name_local = dataset_name or os.getenv("DATASET_NAME", "ConfirmedBets - AllObservations.csv")
        csv_path_local = _resolve_data_file(csv_name_local)
        try:
            df_all_rows = pd.read_csv(csv_path_local)
            date_col = _find_col_ci(df_all_rows, "Timestamp", "Date", "Game Time", "GameTime")
            if date_col is None:
                print("[warn] Chronological split requested but no Timestamp/Date/Game Time column found; falling back to random split.")
            else:
                dt_series = pd.to_datetime(df_all_rows.loc[X.index, date_col], errors="coerce", utc=True, infer_datetime_format=True)
                if dt_series.notna().sum() < max(100, int(0.5 * len(dt_series))):
                    print("[warn] Chronological split requested but too many missing/invalid dates; falling back to random split.")
                else:
                    order_idx = dt_series.sort_values(kind="mergesort").index  # stable
                    n_total = len(order_idx)
                    cut = int(0.7 * n_total)
                    train_idx, test_idx = order_idx[:cut], order_idx[cut:]

                    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

                    tr_start, tr_end = dt_series.loc[train_idx].min(), dt_series.loc[train_idx].max()
                    te_start, te_end = dt_series.loc[test_idx].min(), dt_series.loc[test_idx].max()
                    chrono_info = (tr_start, tr_end, te_start, te_end, len(train_idx), len(test_idx))
                    chrono_done = True
        except Exception as e:
            print(f"[warn] Chronological split failed ({e}); falling back to random split.")

    if chrono_done:
        print(f"\n⏱️  Chronological split enabled:")
        print(f"   Train: {chrono_info[4]} rows  | {chrono_info[0]} → {chrono_info[1]}")
        print(f"   Test : {chrono_info[5]} rows  | {chrono_info[2]} → {chrono_info[3]}")
        # For CV, use only training data; set folds by smallest class count in y_train
        counts_train = Counter(y_train)
        if len(set(y_train)) > 1:
            cv_splits = min(5, min(counts_train.values()))
        else:
            cv_splits = None
    else:
        # Fallback: stratified 70/30 if possible, else 80/20 random
        min_class = min(counts.values()) if len(counts) > 0 else 0
        use_stratify = min_class >= 2
        cv_splits = min(5, min_class) if min_class >= 2 else None

        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        else:
            print("[warn] Minority class < 2; disabling stratify; using 80/20 split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, random_state=42
            )
    # =========================================================================

    
    # === Pipelines (all share the same preprocessor 'pre') ===
    def make_pipelines(pre):
        logreg = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                solver="saga",
                max_iter=5000,
                class_weight="balanced"
            )),
        ])

        rf = Pipeline([
            ("pre", pre),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                random_state=42,
                class_weight="balanced"
            )),
        ])

        hgb = Pipeline([
            ("pre", pre),
            ("hgb", HistGradientBoostingClassifier(
                learning_rate=0.07,
                max_depth=None,
                max_iter=600,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                scoring="roc_auc",                
                random_state=42
            )),
        ])
        
        # Optional: pre-train learning curves (5-fold AUC) for baselines
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _save_learning_curve_csv(
            estimator=logreg.named_steps["clf"], preprocessor=pre,
            X=X, y=y, out_path=MODEL_DIR / "logreg_learning_curve.csv"
        )
        _save_learning_curve_csv(
            estimator=rf.named_steps["rf"], preprocessor=pre,
            X=X, y=y, out_path=MODEL_DIR / "rf_learning_curve.csv"
        )

        
        
        
        return logreg, rf, hgb

    logreg, rf, hgb = make_pipelines(pre)

    # CLI-controlled toggles
    do_tune = bool(int(os.getenv("DO_TUNE", "0")))  # default off; enabled via --tune
    do_pca  = bool(int(os.getenv("DO_PCA", "0")))   # default off; enabled via --pca

    if do_tune:
        # number of CV splits we calculated earlier (cv_splits) is re-used here
        tuned_models = hyperparameter_tune(X_train, y_train, pre, n_iter=25, cv_splits=cv_splits or 3)
        # fall back to defaults if any are missing
        logreg = tuned_models.get("Logistic Regression", logreg)
        rf     = tuned_models.get("Random Forest", rf)
        hgb    = tuned_models.get("HistGradientBoosting (GBT)", hgb)
    else:
        logreg.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        hgb.fit(X_train, y_train)


    # Evaluate function
    def eval_model(name, model):
        print(f"\n🔍 {name}:")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = None
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC: {auc:.3f}")
        except Exception:
            pass
        print(f"Accuracy: {acc:.3f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
        tuned = None
        if y_proba is not None:
            tuned = tune_threshold(y_test, y_proba, objective="f1")
            print(f"🔧 Best F1 threshold: {tuned['thr']:.2f} | "
                  f"F1 {tuned['f1']:.3f} | Acc {tuned['acc']:.3f} | "
                  f"P {tuned['prec']:.3f} | R {tuned['rec']:.3f}")
        return {"name": name, "model": model, "auc": auc, "acc": acc, "tuned": tuned}

    res_lr  = eval_model("Logistic Regression", logreg)
    res_rf  = eval_model("Random Forest", rf)
    res_hgb = eval_model("HistGradientBoosting (GBT)", hgb)

    # Stratified CV (optional)
    if not DO_CHRONO and cv_splits and cv_splits >= 2:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        try:
            cv_lr = cross_val_score(logreg, X, y, cv=skf, scoring="roc_auc")
            print(f"\n🧪 CV Acc (LogReg): {cv_lr.mean():.3f} ± {cv_lr.std():.3f}")
        except Exception as e:
            print(f"[warn] CV(LogReg) skipped: {e}")
        try:
            cv_rf = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")
            print(f"🧪 CV Acc (RF)    : {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
        except Exception as e:
            print(f"[warn] CV(RF) skipped: {e}")
        try:
            cv_hgb = cross_val_score(hgb, X, y, cv=skf, scoring="roc_auc")
            print(f"🧪 CV Acc (HGB)   : {cv_hgb.mean():.3f} ± {cv_hgb.std():.3f}")
        except Exception as e:
            print(f"[warn] CV(HGB) skipped: {e}")
    else:
        print("\n🧪 Skipping CV: not enough minority samples for stratified folds.")
          
    # Select best model by AUC (fallback to accuracy)
    results = [res for res in [res_hgb, res_rf, res_lr]]
    results_sorted = sorted(
        results,
        key=lambda r: ((r["auc"] if r["auc"] is not None else 0.0), r["acc"]),
        reverse=True
    )
    best = results_sorted[0]
    
    try:
        run_drift_checks(best["model"], X_train, y_train, X_test, y_test, MODEL_DIR, alert=True)
    except Exception as e:
        print(f"[warn] drift checks failed: {e}")

    # Safe threshold display (handles None)
    thr_str = f"{best['tuned']['thr']:.2f}" if (best.get("tuned") and best["tuned"] is not None) else "0.50"
    auc_str = f"{best['auc']:.3f}" if best["auc"] is not None else "n/a"

    print(f"\n🏆 Selected model: {best['name']} (AUC={auc_str} Acc={best['acc']:.3f} Thr={thr_str})")
    
    # ---- Segment diagnostics: how the best model performs by Market and Sport ----
    try:
        def segment_report(label, seg_series, X_eval, y_eval, model):
            """
            Build per-segment metrics on an arbitrary eval set (test or full).
            seg_series must be aligned to X (same index).
            """
            y_pred = model.predict(X_eval)
            try:
                y_prob = model.predict_proba(X_eval)[:, 1]
            except Exception:
                y_prob = None

            seg = seg_series.loc[X_eval.index].astype(str)

            rows = []
            for seg_val, idx in seg.groupby(seg).groups.items():
                seg_idx = list(idx)
                y_true_seg = y_eval.loc[seg_idx]
                y_pred_seg = pd.Series(y_pred, index=y_eval.index).loc[seg_idx]
                acc = accuracy_score(y_true_seg, y_pred_seg)
                        
                # === PATCH D (final): guard tiny segments
                auc = None
                if y_prob is not None:
                    y_prob_seg = pd.Series(y_prob, index=y_eval.index).loc[seg_idx]
                    auc = _safe_auc(y_true_seg, y_prob_seg)
                # === /PATCH D ===

                rows.append({
                    label: seg_val,
                    "n": len(seg_idx),
                    "acc": round(acc, 3),
                    "auc": round(auc, 3) if auc is not None else None
                })
            rep = pd.DataFrame(rows).sort_values("n", ascending=False)
            return rep

        seg_market_all = X["market"].astype(str) if "market" in X.columns else pd.Series("", index=X.index)
        seg_sport_all  = X["sport"].astype(str)  if "sport"  in X.columns else pd.Series("", index=X.index)
        
        # Find a datetime column for the test indices
        df_all = pd.read_csv(_resolve_data_file(csv_name))
        date_col = None
        for cand in ["Timestamp", "Date", "Game Time"]:
            if cand in df_all.columns:
                date_col = cand; break

        if date_col is not None:
            proba_test = best["model"].predict_proba(X_test)[:, 1]
            _export_weekly_metrics(
                dates=df_all.loc[y_test.index, date_col],
                y_true=y_test,
                proba=proba_test,
                out_dir=MODEL_DIR
            )


        print("\n📈 Segment report by Market (test split):")
        rep_mkt_test = segment_report("Market", seg_market_all, X_test, y_test, best["model"])
        print(rep_mkt_test.to_string(index=False))

        print("\n📈 Segment report by Sport (test split):")
        rep_spt_test = segment_report("Sport", seg_sport_all, X_test, y_test, best["model"])
        print(rep_spt_test.to_string(index=False))

        print("\n📈 Segment report by Market (full dataset):")
        rep_mkt_full = segment_report("Market", seg_market_all, X, y, best["model"])
        print(rep_mkt_full.to_string(index=False))

        print("\n📈 Segment report by Sport (full dataset):")
        rep_spt_full = segment_report("Sport", seg_sport_all, X, y, best["model"])
        print(rep_spt_full.to_string(index=False))

        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            rep_mkt_test.to_csv(MODEL_DIR / "segments_market_test.csv", index=False)
            rep_spt_test.to_csv(MODEL_DIR / "segments_sport_test.csv", index=False)
            rep_mkt_full.to_csv(MODEL_DIR / "segments_market_full.csv", index=False)
            rep_spt_full.to_csv(MODEL_DIR / "segments_sport_full.csv", index=False)
            print(f"📝 Saved segment reports → {MODEL_DIR}")
        except Exception as e:
            print(f"[warn] Could not save segment CSVs: {e}")

        # ---- S3 uploads for model registry ----
        try:
            # NOTE: update these paths if you ever change where you save files
            model_dir = Path("data/results/models")

            best_model_path = model_dir / "baseline_winloss.pkl"
            feature_file_path = model_dir / "model_features.pkl"
            tier_config_path = Path("tier_config.json")  # saved in repo root
            model_card_path = model_dir / "model_card.json"
            metrics_path = model_dir / "weekly_metrics.csv"

            # Upload core model artifacts
            upload_to_s3(str(best_model_path), "ML_MODELS_PREFIX")
            upload_to_s3(str(feature_file_path), "ML_MODELS_PREFIX")

            # Upload “metadata” / docs
            upload_to_s3(str(tier_config_path), "ML_METADATA_PREFIX")
            upload_to_s3(str(model_card_path), "ML_METADATA_PREFIX")
            upload_to_s3(str(metrics_path), "ML_METADATA_PREFIX")

        except Exception as e:
            S3_LOGGER.error("Error while uploading artifacts to S3: %s", e)




    except Exception as e:
        print(f"[warn] Could not compute segment reports: {e}")

    # ---- Feature importance / coefficients (print top 15) ----
    print("\n🔎 Top features driving predictions (best model):")
    model = best["model"]

    pre_in_model = model.named_steps["pre"]
    transformed_names = get_transformed_feature_names(pre_in_model, num_feats, cat_feats)

    def _safe_top_importances(importances: np.ndarray, names: list[str], k: int = 15):
        imp = pd.DataFrame({"feature": names, "importance": importances})
        imp = imp.sort_values("importance", ascending=False)
        print(imp.head(k).to_string(index=False))

    printed = False
    # RandomForest
    if "rf" in model.named_steps:
        importances = model.named_steps["rf"].feature_importances_
        _safe_top_importances(importances, transformed_names)
        printed = True
    # HistGradientBoosting
    elif "hgb" in model.named_steps and hasattr(model.named_steps["hgb"], "feature_importances_"):
        importances = model.named_steps["hgb"].feature_importances_
        _safe_top_importances(importances, transformed_names)
        printed = True
    # LogisticRegression (coef_ for class 1)
    elif "clf" in model.named_steps and hasattr(model.named_steps["clf"], "coef_"):
        coef = model.named_steps["clf"].coef_.ravel()
        _safe_top_importances(np.abs(coef), transformed_names)
        printed = True

    if not printed:
        print(" (Model doesn’t expose importances; skipping.)")
        importances = None
        
    # ---- Persist feature importances to CSV for tracking week-to-week ----
    try:
        imp_df = None
        if printed and importances is not None:
            imp_df = pd.DataFrame({"feature": transformed_names, "importance": importances})
        if imp_df is None:
            perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
            imp_df = pd.DataFrame({"feature": list(X.columns), "importance": perm.importances_mean})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_path = MODEL_DIR / "feature_importances.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"📊 Saved feature importances → {imp_path}")
    except Exception as e:
        print(f"[warn] Could not save feature importances: {e}")


    # ---- Descriptive: win-rate by odds_bin (and by odds_bin × market) ----
    try:
        odds_bin_series = X["odds_bin"].astype(str) if "odds_bin" in X.columns else pd.Series("", index=X.index)
        # Align
        y_all = y.copy()
        df_desc = pd.DataFrame({"odds_bin": odds_bin_series, "y": y_all})
        table_bin = df_desc.groupby("odds_bin").agg(n=("y", "size"), win_rate=("y", "mean")).reset_index().sort_values("n", ascending=False)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        table_bin.to_csv(MODEL_DIR / "descriptive_winrate_by_odds_bin.csv", index=False)
        print("\n📊 Win rate by odds_bin (full dataset):")
        print(table_bin.to_string(index=False))

        # odds_bin × market (full dataset)
        if "market" in X.columns:
            df_desc2 = pd.DataFrame({"odds_bin": odds_bin_series, "market": X["market"].astype(str), "y": y_all})
            table_cross = (df_desc2.groupby(["odds_bin", "market"])
                .agg(n=("y", "size"), win_rate=("y", "mean"))
                .reset_index().sort_values(["odds_bin", "n"], ascending=[True, False]))
            table_cross.to_csv(MODEL_DIR / "descriptive_winrate_by_odds_bin_market.csv", index=False)

        # ============ NEW: test-split only ============ #
        if "odds_bin" in X.columns:
            odds_bin_test = X.loc[X_test.index, "odds_bin"].astype(str)
            y_test_series = y_test.copy()
            df_desc_test = pd.DataFrame({"odds_bin": odds_bin_test, "y": y_test_series})
            table_bin_test = (df_desc_test.groupby("odds_bin")
                              .agg(n=("y", "size"), win_rate=("y", "mean"))
                              .reset_index().sort_values("n", ascending=False))
            table_bin_test.to_csv(MODEL_DIR / "descriptive_winrate_by_odds_bin_test.csv", index=False)
            print("\n📊 Win rate by odds_bin (test split):")
            print(table_bin_test.to_string(index=False))
            
            
            # === AUTO-GENERATE tier_config.json FOR sports.py ===
            try:
                # Use test-split (odds_bin × market) performance as guidance
                # We only need odds_bin + market here; sports we take from X["sport"].
                unique_sports = sorted(X["sport"].astype(str).unique().tolist())

                # Fallback if table_cross_test wasn't created
                if "table_cross_test" in locals():
                    df_tiers_src = table_cross_test.copy()
                else:
                    df_tiers_src = None

                tier_rules = []

                if df_tiers_src is not None and not df_tiers_src.empty:
                    # Heuristics:
                    # - Require minimum sample size so we don't overfit tiny buckets
                    MIN_N_A = 40
                    MIN_N_B = 30
                    MIN_N_C = 20

                    # Define bands on *win_rate* to decide A/B/C
                    # (you can tweak these later)
                    def bucket_for_row(n, win):
                        if n >= MIN_N_A and win >= 0.60:
                            return "A"
                        if n >= MIN_N_B and win >= 0.55:
                            return "B"
                        if n >= MIN_N_C and win >= 0.50:
                            return "C"
                        return None

                    for _, row in df_tiers_src.iterrows():
                        odds_bin = str(row["odds_bin"])
                        market   = str(row["market"])
                        n        = int(row["n"])
                        win_rate = float(row["win_rate"])

                        tier_code = bucket_for_row(n, win_rate)
                        if tier_code is None:
                            continue

                        # Map tier_code to label + model-prob thresholds used at runtime
                        if tier_code == "A":
                            min_proba, max_proba = 0.60, 1.00
                            label = "Tier A"
                        elif tier_code == "B":
                            min_proba, max_proba = 0.55, 1.00
                            label = "Tier B"
                        else:  # "C"
                            min_proba, max_proba = 0.50, 1.00
                            label = "Tier C"

                        tier_rules.append({
                            "code": tier_code,
                            "label": label,
                            "sports": unique_sports,      # allow all sports for now
                            "markets": [market],          # specific Market, e.g. "H2H Away"
                            "odds_bin": [odds_bin],       # e.g. "Fav", "Dog"
                            "min_proba": float(min_proba),
                            "max_proba": float(max_proba),
                        })

                # Fallback: if nothing qualified, keep a safe default
                if not tier_rules:
                    tier_rules = [
                        {
                            "code": "A",
                            "label": "Tier A",
                            "sports": unique_sports,
                            "markets": ["H2H Home", "H2H Away", "Spread Home", "Spread Away"],
                            "odds_bin": ["Fav", "Balanced"],
                            "min_proba": 0.60,
                            "max_proba": 1.00,
                        }
                    ]

                tier_cfg = {
                    "tiers": tier_rules,
                    "default": {"code": "", "label": "Pass"}
                }

                TIER_CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
                TIER_CONFIG_OUT.write_text(
                    json.dumps(tier_cfg, indent=2),
                    encoding="utf-8"
                )
                print(f"🧩 Saved tier_config.json → {TIER_CONFIG_OUT}")
            except Exception as e:
                print(f"[warn] Could not auto-generate tier_config.json: {e}")
                    
            

            # odds_bin × market (test split)
            if "market" in X.columns:
                market_test = X.loc[X_test.index, "market"].astype(str)
                df_desc2_test = pd.DataFrame({"odds_bin": odds_bin_test, "market": market_test, "y": y_test_series})
                table_cross_test = (df_desc2_test.groupby(["odds_bin", "market"])
                                    .agg(n=("y", "size"), win_rate=("y", "mean"))
                                    .reset_index().sort_values(["odds_bin", "n"], ascending=[True, False]))
                table_cross_test.to_csv(MODEL_DIR / "descriptive_winrate_by_odds_bin_market_test.csv", index=False)
        # ============================================== #
    except Exception as e:
        print(f"[warn] Descriptive odds_bin tables skipped: {e}")


    # Save model + feature names
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], str(MODEL_PKL))
    joblib.dump(list(X.columns), str(FEATS_PKL))

    # Compatibility copies next to current working dir (optional)
    joblib.dump(best["model"], "model.pkl")
    joblib.dump(list(X.columns), "model_features.pkl")

    print(f"\n💾 Saved {MODEL_PKL} and {FEATS_PKL} (and compatibility copies in CWD)")
    if best["tuned"]:
        print(f"➡️  Suggested decision threshold (F1-opt): {best['tuned']['thr']:.2f}")
        
    # ---- Save per-row predictions for the whole dataset (easy import to Sheets) ----
    try:
        y_proba_all = best["model"].predict_proba(X)[:, 1]
        thr = float(best["tuned"]["thr"]) if (best.get("tuned") and best["tuned"] is not None) else 0.50
        y_hat_all = (y_proba_all >= thr).astype(int)

        df_pred_features = X.copy()
        df_pred_features.insert(0, "proba", y_proba_all)
        df_pred_features.insert(1, f"pred@{thr:.2f}", y_hat_all)

        y_aligned = y.loc[df_pred_features.index]
        df_pred_features.insert(2, "actual", y_aligned)

        csv_path = _resolve_data_file(csv_name)
        df_all = pd.read_csv(csv_path)
        meta_cols = [
            "Timestamp", "Date", "Sport", "League", "Market", "Direction",
            "Away Team", "Home Team", "Game Time", "Predicted", "Actual Winner"
        ]
        meta_cols = [c for c in meta_cols if c in df_all.columns]
        df_meta = df_all.loc[df_pred_features.index, meta_cols] if meta_cols else None

        if df_meta is not None:
            df_out = pd.concat([df_meta.reset_index(drop=True), df_pred_features.reset_index(drop=True)], axis=1)
        else:
            df_out = df_pred_features

        strong_thr = 0.65
        df_out["StrongBet"] = (df_out["proba"] >= strong_thr).astype(int)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        pred_path = MODEL_DIR / "predictions_latest.csv"
        df_out.to_csv(pred_path, index=False)
        print(f"📝 Saved per-row predictions with probabilities → {pred_path}")
        
    except Exception as e:
        print(f"[warn] Could not save predictions csv: {e}")
        
        
    import hashlib
    def _hash_file(p: Path):
        try:
            return hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            return None

    meta = {
        "model_name": best["name"],
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "dataset": csv_name,
        "dataset_hash": _hash_file(_resolve_data_file(csv_name)),
        "rows_total": int(len(X)),
        "class_balance": dict(Counter(y)),
        "auc": (best["auc"] if best["auc"] is not None else None),
        "accuracy": float(best["acc"]),
        "threshold": float(best["tuned"]["thr"]) if best.get("tuned") else 0.50,
        "features_used": list(X.columns),
        "learning_rate_hgb": (hgb.named_steps["hgb"].learning_rate if "hgb" in hgb.named_steps else None)
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "model_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"🗂️  Saved model_card.json → {MODEL_DIR}")        
        
        
        

        
        


        
        
# ============================
# PCA projection export (2D)
# ============================
def export_pca_projection(model, X_eval, y_eval, out_csv_path: Path):
    """
    Transforms X_eval using the model's fitted preprocessor, then applies PCA(2).
    Exports CSV with columns: pc1, pc2, actual, proba (if available), pred.
    """
    try:
        pre = model.named_steps.get("pre")
        if pre is None:
            print("[warn] PCA export: no preprocessor in pipeline")
            return
        Z = pre.transform(X_eval)  # dense due to OHE config
        pca = PCA(n_components=2, random_state=42)
        Z2 = pca.fit_transform(Z)

        proba = None
        try:
            proba = model.predict_proba(X_eval)[:, 1]
        except Exception:
            pass
        pred = model.predict(X_eval)

        df_pca = pd.DataFrame({
            "pc1": Z2[:, 0],
            "pc2": Z2[:, 1],
            "actual": y_eval.values.astype(int),
            "pred": pred.astype(int),
        })
        if proba is not None:
            df_pca["proba"] = proba

        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_pca.to_csv(out_csv_path, index=False)
        print(f"🗺️  Saved PCA projection → {out_csv_path}")
    except Exception as e:
        print(f"[warn] PCA export failed: {e}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="CSV filename in data/ (default: ConfirmedBets - AllObservations.csv)")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning (RandomizedSearchCV)")
    parser.add_argument("--pca",  action="store_true", help="Export a 2D PCA projection CSV for the test split")
    # NEW: chronological split flag
    parser.add_argument("--chronological", action="store_true", help="Use time-aware train/test split (train past, test future)")
    args = parser.parse_args()

    # pass toggles via env so train_models can pick them up without changing its signature
    if args.tune:
        os.environ["DO_TUNE"] = "1"
    if args.pca:
        os.environ["DO_PCA"] = "1"
    if args.chronological:
        os.environ["DO_CHRONO"] = "1"

    train_models(dataset_name=args.dataset)


