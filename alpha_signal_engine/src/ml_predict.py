# alpha_signal_engine/src/ml_predict.py
from __future__ import annotations

import os
import json
import pickle
import gzip
import bz2
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict



import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# --------------------------
# Environment + search paths
# --------------------------
# Primary env vars (old names kept for backward-compat)
MODEL_PATH_ENV: str = os.getenv("STRONG_MODEL_PATH") or os.getenv("MODEL_FILE", "")
FEATS_PATH_ENV: str  = os.getenv("STRONG_FEATURE_NAMES") or os.getenv("FEATURE_FILE", "")

_CANDIDATE_DIRS: List[Path] = [
    Path(r"C:\Users\Jcast\Documents\alpha_signal_engine\data\results\models"),
    Path(r"C:\Users\Jcast\Documents\alpha_signal_engine"),
    Path(r"C:\Users\Jcast\OneDrive\Documents\alpha_signal_engine\data\results\models"),
    Path(r"C:\Users\Jcast\OneDrive\Documents\alpha_signal_engine"),
    Path(r"C:\Users\John Castro\Documents\alpha_signal_engine\data\results\models"),
    Path(r"C:\Users\John Castro\Documents\alpha_signal_engine"),
    Path(r"C:\Users\John Castro\OneDrive\Documents\alpha_signal_engine\data\results\models"),
    Path(r"C:\Users\John Castro\OneDrive\Documents\alpha_signal_engine"),
    Path(__file__).resolve().parents[3] / "data" / "results" / "models",
    Path(__file__).resolve().parents[2],
]

# -----------------------
# Robust model/file I/O
# -----------------------
def _try_joblib(path: Path):
    import joblib
    return joblib.load(path)

def _try_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _try_gzip_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def _try_bz2_pickle(path: Path):
    with bz2.open(path, "rb") as f:
        return pickle.load(f)

_MODEL_READERS = (_try_joblib, _try_pickle, _try_gzip_pickle, _try_bz2_pickle)

_model: Any = None
_model_path: Optional[Path] = None
_feature_names: Optional[List[str]] = None  # pre-transform columns used in training

def _looks_like_feature_file(name_lower: str) -> bool:
    # Anything that hints it's a feature file
    return any(tok in name_lower for tok in ("features", "feature_names", "model_features", "feat"))

def _looks_like_model_file(name_lower: str) -> bool:
    # Prioritize obvious model names; avoid feature files
    if _looks_like_feature_file(name_lower):
        return False
    return any(k in name_lower for k in ("winloss", "strong", "final", "pipeline", "model", "clf", "estimator"))

def _auto_find_model() -> Optional[Path]:
    # Pick the newest plausible *model* file
    for root in _CANDIDATE_DIRS:
        if not root.exists():
            continue
        files = sorted((p for p in root.iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            n = p.name.lower()
            if _looks_like_model_file(n):
                return p
        # As last resort, pick newest file that is NOT an obvious feature file
        for p in files:
            if not _looks_like_feature_file(p.name.lower()):
                return p
    return None

def _auto_find_feature_file() -> Optional[Path]:
    # Try siblings of the chosen model first
    if _model_path:
        for cand in ("strong_feature_names.json", "feature_names.json",
                     "model_features.pkl", "model_features.joblib",
                     "model_features.pkl.gz", "model_features.pbz2"):
            t = _model_path.parent / cand
            if t.exists():
                return t
    # Then scan candidate dirs
    for root in _CANDIDATE_DIRS:
        if not root.exists():
            continue
        for cand in ("strong_feature_names.json", "feature_names.json",
                     "model_features.pkl", "model_features.joblib",
                     "model_features.pkl.gz", "model_features.pbz2"):
            t = root / cand
            if t.exists():
                return t
    return None

def _validate_model_object(obj: Any, src: Path) -> Any:
    # Accept (model, features) tuples too
    if isinstance(obj, tuple) and len(obj) == 2 and hasattr(obj[0], "predict_proba"):
        log.info("🧪 Model+features tuple detected at %s", src)
        return obj[0]
    if not hasattr(obj, "predict_proba"):
        raise TypeError(
            f"Loaded object from {src} is {type(obj)}; expected an estimator with predict_proba(). "
            "This often happens when a *feature* file is passed as the model."
        )
    return obj

def _load_model_once() -> Any:
    global _model, _model_path
    if _model is not None:
        return _model

    # Resolve path
    path = Path(MODEL_PATH_ENV) if MODEL_PATH_ENV else _auto_find_model()
    if not path or not path.exists():
        tried = [str(p) for p in _CANDIDATE_DIRS]
        raise FileNotFoundError(f"Model file not found. Env MODEL_FILE/STRONG_MODEL_PATH='{MODEL_PATH_ENV}'. Tried: {tried}")

    last_err: Optional[Exception] = None
    for reader in _MODEL_READERS:
        try:
            obj = reader(path)
            obj = _validate_model_object(obj, path)
            _model = obj
            _model_path = path
            log.info("🔮 Model loaded via %s: %s", reader.__name__, path)
            return _model
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load model at {path}. Last error: {last_err}")

def _load_feature_names_once() -> Optional[List[str]]:
    global _feature_names
    if _feature_names is not None:
        return _feature_names

    path: Optional[Path] = None
    if FEATS_PATH_ENV:
        cand = Path(FEATS_PATH_ENV)
        if cand.exists():
            path = cand

    if path is None:
        try:
            _load_model_once()  # ensures _model_path populated
        except Exception:
            pass
        path = _auto_find_feature_file()

    if path is None or not path.exists():
        return None

    try:
        # JSON is a clean list
        if path.suffix.lower() == ".json":
            _feature_names = list(json.loads(path.read_text(encoding="utf-8")))
        else:
            # joblib/pickle/gzip/bz2
            try:
                import joblib
                obj = joblib.load(path)
            except Exception:
                obj = _try_pickle(path)

            # If someone pointed us at a MODEL by mistake, ignore it
            if hasattr(obj, "predict_proba"):
                log.warning("Feature path %s appears to be a MODEL file; ignoring.", path)
                _feature_names = None
                return _feature_names

            if isinstance(obj, dict) and "feature_names" in obj:
                _feature_names = list(obj["feature_names"])
            else:
                _feature_names = list(obj)
        log.info("📐 Feature names loaded: %s", path)
    except Exception as e:
        log.warning("Feature names load failed from %s: %s", path, e)
        _feature_names = None
    return _feature_names

# ---------------------------------------
# Inference-time feature engineering
# Mirrors the training expectations in train_model.py
# ---------------------------------------

def _american_to_decimal(am: Optional[float]) -> float:
    try:
        am = float(am)
    except Exception:
        return np.nan
    if am > 0:
        return 1.0 + (am / 100.0)
    if am < 0:
        return 1.0 + (100.0 / abs(am))
    return np.nan

def _implied_prob_from_american(am: Optional[float]) -> float:
    try:
        am = float(am)
    except Exception:
        return np.nan
    if am > 0:
        return 100.0 / (am + 100.0)
    if am < 0:
        return abs(am) / (abs(am) + 100.0)
    return np.nan

def _clean_direction(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if t.startswith("home"): return "Home"
    if t.startswith("away"): return "Away"
    if "spread favorite" in t: return "Spread Favorite"
    if "spread underdog"  in t: return "Spread Underdog"
    if t == "underdog": return "Underdog"
    if t == "over":     return "Over"
    if t == "under":    return "Under"
    return "Other"

def _time_bin(mins: float) -> str:
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

def _odds_bin_from_imp(imp: float) -> str:
    if pd.isna(imp): return "unknown"
    if imp >= 0.70:  return "HeavyFav"
    if imp >= 0.55:  return "Fav"
    if imp >= 0.40:  return "Balanced"
    if imp >= 0.20:  return "Dog"
    return "Longshot"

# Payload keys we accept from the runner
BASE_NUMS = {
    "movement": np.nan, "minutes_to_start": np.nan,
    "open_am": np.nan, "open_dec": np.nan, "open_imp": np.nan,
    "curr_am": np.nan, "curr_dec": np.nan, "curr_imp": np.nan,
    "close_am": np.nan, "close_dec": np.nan, "close_imp": np.nan,
    "delta_am_close_open": np.nan, "delta_am_close_curr": np.nan,
    "delta_imp_close_open": np.nan, "delta_imp_close_curr": np.nan,
    "spread_home": np.nan, "spread_away": np.nan,
    "spread_home_abs": np.nan, "spread_away_abs": np.nan,
    "spread_abs_min": np.nan, "spread_abs_max": np.nan,
    "home_fav_flag": np.nan, "away_fav_flag": np.nan,
    "spread_dist_3": np.nan, "spread_dist_7": np.nan,
    "total_line": np.nan,
    "bet_limit_raw": np.nan, "bet_limit_log": np.nan, "bet_limit_z": np.nan,
    "consensus_close_imp": np.nan, "book_imp_range_close": np.nan, "book_outlier_flag": np.nan,
}
BASE_CATS = {
    "sport": "", "market": "", "direction_clean": "",
    "time_bin": "unknown", "limit_trend": "unknown", "odds_bin": "unknown"
}

def transform_for_inference(payload: Dict[str, Any], feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    p = dict(payload) if payload else {}

    # minutes_to_start
    minutes_to_start = None
    if "hours_to_start" in p and p["hours_to_start"] is not None:
        try:
            minutes_to_start = float(p["hours_to_start"]) * 60.0
        except Exception:
            minutes_to_start = None

    # odds + implied
    am  = p.get("american_odds", None)
    dec = p.get("decimal_odds", None)
    if dec is None or (isinstance(dec, (int,float)) and not np.isfinite(dec)):
        dec = _american_to_decimal(am)
    imp = _implied_prob_from_american(am) if am is not None else (1.0/dec if (isinstance(dec,(int,float)) and dec>1) else np.nan)

    direction_clean = _clean_direction(str(p.get("Direction", "")))
    sport   = str(p.get("sport", "")) or ""
    market  = str(p.get("market", "")) or ""
    limit_trend = str(p.get("limit_trend", p.get("Limit Trend", "unknown"))) or "unknown"
    tb = _time_bin(minutes_to_start if minutes_to_start is not None else np.nan)
    ob = _odds_bin_from_imp(imp)

    # spreads / totals
    def _to_spread(x):
        if isinstance(x, str) and x.strip().lower() in {"pk","pick","pickem","pick'em"}:
            return 0.0
        try: return float(x)
        except Exception: return np.nan
    sh = _to_spread(p.get("spread_line_home", np.nan))
    sa = _to_spread(p.get("spread_line_away", np.nan))
    sha = abs(sh) if pd.notna(sh) else np.nan
    saa = abs(sa) if pd.notna(sa) else np.nan
    smin = np.nanmin([sha, saa]) if (pd.notna(sha) or pd.notna(saa)) else np.nan
    smax = np.nanmax([sha, saa]) if (pd.notna(sha) or pd.notna(saa)) else np.nan
    home_fav_flag = float(sh < 0) if pd.notna(sh) else np.nan
    away_fav_flag = float(sa < 0) if pd.notna(sa) else np.nan

    total_line = None
    if "total_line" in p and p["total_line"] not in ("", None):
        try:
            total_line = float(p["total_line"])
        except Exception:
            total_line = np.nan

    # limits
    bet_limit_raw = p.get("bet_limit", p.get("Bet Limit", None))
    try:
        bet_limit_raw = float(bet_limit_raw) if bet_limit_raw is not None else np.nan
    except Exception:
        bet_limit_raw = np.nan
    bet_limit_log = np.log1p(bet_limit_raw) if pd.notna(bet_limit_raw) else np.nan
    bet_limit_z = np.nan

    nums = dict(BASE_NUMS)
    nums.update({
        "movement": float(p.get("Movement", np.nan)) if p.get("Movement", None) is not None else np.nan,
        "minutes_to_start": float(minutes_to_start) if minutes_to_start is not None else np.nan,
        "curr_am": float(am) if am is not None else np.nan,
        "curr_dec": float(dec) if dec is not None else np.nan,
        "curr_imp": float(imp) if imp is not None else np.nan,
        "spread_home": sh, "spread_away": sa,
        "spread_home_abs": sha, "spread_away_abs": saa,
        "spread_abs_min": smin, "spread_abs_max": smax,
        "home_fav_flag": home_fav_flag, "away_fav_flag": away_fav_flag,
        "spread_dist_3": (abs(smin - 3.0) if pd.notna(smin) else np.nan),
        "spread_dist_7": (abs(smin - 7.0) if pd.notna(smin) else np.nan),
        "total_line": total_line if total_line is not None else np.nan,
        "bet_limit_raw": bet_limit_raw, "bet_limit_log": bet_limit_log, "bet_limit_z": bet_limit_z,
    })
    cats = dict(BASE_CATS)
    cats.update({
        "sport": sport,
        "market": market,
        "direction_clean": direction_clean,
        "time_bin": tb,
        "limit_trend": limit_trend,
        "odds_bin": ob,
    })

    row = {**nums, **cats}
    df = pd.DataFrame([row])

    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan if col in BASE_NUMS else ""
        df = df[feature_names]

    nnz = df.notna().sum().sum()
    log.info(f"🧩 Inference row built | cols={len(df.columns)} | non-NA total={int(nnz)}")
    return df

def vectorize_for_model(df: pd.DataFrame, feature_names: Optional[List[str]], model: Any) -> pd.DataFrame:
    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan if col in BASE_NUMS else ""
        df = df[feature_names]
    return df

# ------------------------
# Public prediction helper
# ------------------------
def predict_win_prob(payload: dict) -> float:
    model = _load_model_once()
    feat_names = _load_feature_names_once()
    df = transform_for_inference(payload, feature_names=feat_names)
    X = vectorize_for_model(df, feature_names=feat_names, model=model)
    proba = float(model.predict_proba(X)[0][1])
    return proba

# ------------------------
# Optional convenience API
# ------------------------
def set_paths(model_path: Optional[str] = None, feature_names_path: Optional[str] = None) -> None:
    global MODEL_PATH_ENV, FEATS_PATH_ENV, _model, _model_path, _feature_names
    if model_path:
        MODEL_PATH_ENV = model_path
    if feature_names_path:
        FEATS_PATH_ENV = feature_names_path
    _model = None
    _model_path = None
    _feature_names = None
