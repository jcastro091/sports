# src/inference.py
import sys, json, os, joblib
import pandas as pd

from .preprocess_infer import transform_for_inference



# prefer package relative, then fallback to flat
try:
    from .preprocess import clean_tags, clean_direction, clean_movement, parse_odds_taken, load_allbets_clean  # type: ignore
    from .model_features import add_basic_features  # type: ignore
except Exception:
    from preprocess import clean_tags, clean_direction, clean_movement, parse_odds_taken  # type: ignore
    load_allbets_clean = None
    add_basic_features = None

MODEL_PRIMARY = os.path.join("data", "results", "models", "baseline_winloss.pkl")
MODEL_FALLBACK = "model.pkl"
FEATS_PRIMARY = os.path.join("data", "results", "models", "model_features.pkl")
FEATS_FALLBACK = "model_features.pkl"
OUT_PRED_CSV = os.path.join("data", "results", "ml_predictions.csv")

def map_confidence(val):
    m = {"High": 2, "Medium": 1, "Low": 0, 2: 2, 1: 1, 0: 0}
    return m.get(val, 1)

def transform_payload(payload: dict, feature_columns: list) -> pd.DataFrame:
    return transform_for_inference(payload, feature_columns)


def load_payload():
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch-duckdb":
        return "__BATCH__"
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            return json.load(f)
    elif len(sys.argv) >= 2:
        return json.loads(sys.argv[1])
    raise SystemExit("Pass a JSON payload, use --file payload.json, or --batch-duckdb")

def load_model_and_features():
    model_path = MODEL_PRIMARY if os.path.exists(MODEL_PRIMARY) else MODEL_FALLBACK
    feats_path = FEATS_PRIMARY if os.path.exists(FEATS_PRIMARY) else FEATS_FALLBACK
    model = joblib.load(model_path)
    features = joblib.load(feats_path)
    return model, features

def batch_from_duckdb():
    if load_allbets_clean is None or add_basic_features is None:
        raise SystemExit("DuckDB batch mode unavailable (load_allbets_clean/add_basic_features missing)")
    df = load_allbets_clean().reset_index(drop=True)
    df["row_idx"] = df.index
    df = add_basic_features(df)
    # Build feature frame from training features if available
    _, features = load_model_and_features()
    # If your training used dummy-encoded features, inference pipeline must mirror;
    # since we saved the pipeline object, we can pass raw columns and let the pipeline handle it.
    # Here, we align to saved feature columns if they exist (compatibility with older setup).
    if isinstance(features, list) and len(features) > 0:
        # Try to build a minimal X with matching columns
        X_cols = [c for c in features if c in df.columns]
        X = df[X_cols].copy() if X_cols else df.copy()
    else:
        X = df.copy()
    return df, X

if __name__ == "__main__":
    payload = load_payload()
    model, features = load_model_and_features()

    if payload == "__BATCH__":
        # Batch mode: write ml_predictions.csv with row_idx + ml_confidence
        # If model is a Pipeline with its own preprocessors (recommended), we can pass X as built in batch_from_duckdb.
        try:
            df, X = batch_from_duckdb()
        except SystemExit as e:
            print(str(e))
            sys.exit(1)

        # If model has predict_proba, use it; else use decision_function or predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            try:
                scores = model.decision_function(X)
                # min-max normalize to [0,1] as a rough proxy
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            except Exception:
                proba = model.predict(X).astype(float)

        out = pd.DataFrame({"row_idx": df["row_idx"], "ml_confidence": proba})
        os.makedirs(os.path.dirname(OUT_PRED_CSV), exist_ok=True)
        out.to_csv(OUT_PRED_CSV, index=False)
        print(f"[inference] wrote {OUT_PRED_CSV} ({len(out)})")
        sys.exit(0)

    # === Original single/array JSON payload behavior (no regression) ===
    if isinstance(payload, list):
        for i, p_item in enumerate(payload, start=1):
            X = transform_payload(p_item, features)
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(X)[0, 1])
                print(f"[{i}] win_prob={p:.3f}")
            else:
                print(f"[{i}] prediction={int(model.predict(X)[0])}")
    else:
        X = transform_payload(payload, features)
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X)[0, 1])
            print(f"win_prob={p:.3f}")
        else:
            print(f"prediction={int(model.predict(X)[0])}")
