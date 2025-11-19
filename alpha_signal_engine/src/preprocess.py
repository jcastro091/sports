# alpha_signal_engine/src/preprocess.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import math
import pandas as pd

_EXPECTED_COLS: List[str] = [
    "Market",
    "Direction",
    "Movement",
    "Kelly",
    "Confidence",
    "Odds Taken",
    "Tags",
]

_ALIAS_MAP: Dict[str, List[str]] = {
    "Market":       ["Market", "market", "market_name", "bet_market"],
    "Direction":    ["Direction", "direction", "side", "pick", "bet_side"],
    "Movement":     ["Movement", "movement", "odds_delta", "line_move", "delta"],
    "Kelly":        ["Kelly", "kelly", "kelly_stake", "kelly_pct", "kelly_fraction"],
    "Confidence":   ["Confidence", "confidence", "conf", "p_used", "prob", "p", "win_prob"],
    "Odds Taken":   ["Odds Taken", "american_odds", "odds_american", "odds", "price", "decimal_odds"],
    "Tags":         ["Tags", "tags", "tag", "reason", "reasons", "rationale"],
}

_DEFAULTS: Dict[str, Any] = {
    "Market": "",
    "Direction": "",
    "Movement": 0.0,
    "Kelly": 0.0,
    "Confidence": 0.50,
    "Odds Taken": 0,
    "Tags": "",
}

def _decimal_to_american(decimal_odds: float | int | str) -> int:
    try:
        d = float(decimal_odds)
    except Exception:
        return 0
    if not math.isfinite(d) or d <= 1.0:
        return 0
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100.0 / (d - 1.0)))

def _coerce_american_odds(series: pd.Series) -> pd.Series:
    def _one(x: Any) -> int:
        try:
            xi = int(x)
            if xi <= -100 or xi >= 100:
                return xi
        except Exception:
            pass
        try:
            xf = float(x)
            if 1.01 <= xf <= 50.0:
                return _decimal_to_american(xf)
        except Exception:
            return 0
        return 0
    return series.map(_one)

def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def transform_for_inference(payload: Dict[str, Any] | List[Dict[str, Any]], features: List[str] | None = None) -> pd.DataFrame:
    if isinstance(payload, dict):
        records = [payload]
    else:
        records = list(payload)
    df = pd.json_normalize(records)

    for target in _EXPECTED_COLS:
        if target in df.columns:
            continue
        alias = _pick_first_present(df, _ALIAS_MAP.get(target, []))
        if alias is not None:
            df[target] = df[alias]
        else:
            df[target] = _DEFAULTS[target]

    df["Odds Taken"] = _coerce_american_odds(df["Odds Taken"])

    for col in ("Market", "Direction", "Tags"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ("Movement", "Kelly", "Confidence"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(_DEFAULTS[col]).astype(float)

    df = df[_EXPECTED_COLS]
    return df

# --- NEW: vectorization & alignment ---

_NUMERIC_COLS = ["Movement", "Kelly", "Confidence", "Odds Taken"]
_CATEGORICAL_COLS = ["Market", "Direction"]  # match your training

def load_feature_names(path: str | Path) -> Optional[List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def vectorize_for_model(df: pd.DataFrame, feature_names: Optional[List[str]] = None, model: Any = None) -> pd.DataFrame:
    """
    Create the exact matrix your model expects:
      - one-hot on Market, Direction
      - keep numeric cols
      - align to feature_names or model.feature_names_in_
    """
    base = pd.DataFrame(index=df.index)
    # numeric
    for c in _NUMERIC_COLS:
        base[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # one-hot categoricals
    dummies = pd.get_dummies(df[_CATEGORICAL_COLS], prefix=_CATEGORICAL_COLS, dtype=float)
    X = pd.concat([base, dummies], axis=1)

    # resolve expected feature list
    expected: Optional[List[str]] = None
    if feature_names:
        expected = list(feature_names)
    elif hasattr(model, "feature_names_in_"):
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None

    if expected is None:
        # fallback: use current columns (works if model has no strict check)
        return X

    # Align: add missing as 0.0, drop extras
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    X = X.reindex(columns=expected, fill_value=0.0)
    return X
