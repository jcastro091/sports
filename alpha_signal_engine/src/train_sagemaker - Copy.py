import argparse
import os
import glob
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--target_col", type=str, default="Result")

    # SageMaker-provided paths
    parser.add_argument("--sm_model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--sm_output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--sm_channel_train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    return parser.parse_args()


def load_training_data(train_dir: str) -> pd.DataFrame:
    # train_dir will be /opt/ml/input/data/train
    pattern = os.path.join(train_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise RuntimeError(f"No CSV files found under {train_dir}")


    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def preprocess(df: pd.DataFrame, target_col: str = "Result"):
    """
    Build X, y from the raw dataframe.

    - If `target_col` is missing, derive it from "Prediction Result".
    - Supports values like "Win"/"Lose" OR "0"/"1" string/numeric.
    - Drops any rows with invalid labels (e.g. "P", "2", NaN).
    """
    # If explicit target column is missing, build it from Prediction Result
    if target_col not in df.columns:
        if "Prediction Result" in df.columns:
            raw = df["Prediction Result"]

            # Log raw values for debugging
            raw_str = raw.astype(str).str.strip()
            unique_vals = set(raw_str.dropna().unique())
            print("Raw 'Prediction Result' values (sample):", list(unique_vals)[:20])

            # Case 1: purely 0/1 encoded labels
            if unique_vals.issubset({"0", "1"}):
                df["Result"] = raw_str.astype(int)
                print(
                    "Created 'Result' from 0/1-encoded 'Prediction Result'. "
                    f"Unique values: {df['Result'].unique()}"
                )
            else:
                # Generic mapping that handles Win/Loss AND 0/1 strings
                mapping = {
                    "WIN": 1,
                    "W": 1,
                    "1": 1,
                    "LOSS": 0,
                    "LOSE": 0,
                    "L": 0,
                    "0": 0,
                }
                df["Result"] = raw_str.str.upper().map(mapping)
                print(
                    "Created 'Result' from string 'Prediction Result' with mapping. "
                    f"Unique values: {df['Result'].unique()}"
                )

            target_col = "Result"
        else:
            raise RuntimeError(
                f"Target column '{target_col}' not found and "
                f"'Prediction Result' not present. Columns: {df.columns.tolist()}"
            )

    # Build y as numeric labels from target_col (now guaranteed to exist)
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Drop rows where y is missing or invalid (e.g. P, 2, NaN)
    before = len(df)
    mask = y.isin([0, 1])
    dropped = before - mask.sum()
    if dropped > 0:
        print(f"🧹 Dropped {dropped} rows with invalid/missing target; remaining: {mask.sum()}")

    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    if len(y) == 0:
        raise RuntimeError(
            f"After cleaning, 0 rows left with non-null '{target_col}'. "
            "Check that 'Prediction Result' in S3 is populated correctly."
        )

    # Feature selection
    drop_cols = [
        target_col,
        "Prediction Result",
        "Actual Winner",
        "Timestamp",
        "Game Time",
        "Bet ID",
        "KalshiOrderId",
        "KalshiTs",
        "KalshiError",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Drop target + obvious non-feature columns first
    # Drop target + obvious non-feature columns first
    features_df = df.drop(columns=drop_cols)

    # Keep only numeric (and bool) feature columns
    numeric_cols = features_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    non_numeric = [c for c in features_df.columns if c not in numeric_cols]

    if non_numeric:
        print("🔻 Dropping non-numeric feature columns:", non_numeric)

    X = features_df[numeric_cols].astype(float)
    
    # Drop feature columns that are entirely NaN
    all_nan_cols = X.columns[X.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print("🪓 Dropping all-NaN feature columns:", list(all_nan_cols))
        X = X.drop(columns=all_nan_cols)



    # Handle NaNs in features: drop any rows with missing values
    # before = len(X)
    # mask = ~X.isna().any(axis=1)
    # dropped = before - mask.sum()
    # if dropped > 0:
        # print(f"🧽 Dropping {dropped} rows with NaN features; remaining: {mask.sum()}")
        # X = X.loc[mask]
        # y = y.loc[mask]
        
    # Handle NaNs in features: impute with column median instead of dropping all rows
    if X.isna().any().any():
        medians = X.median()
        X = X.fillna(medians)
        print("🧽 Imputed NaN features with column medians.")


    print(f"✅ Features shape: {X.shape}, Target positives: {y.sum()} / {len(y)}")
    return X, y



def train_and_evaluate(X, y, n_estimators, max_depth, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X, y)

    # simple in-sample AUC for now
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"🏁 Training complete. In-sample AUC = {auc:.4f}")

    return model, {"auc": float(auc)}


def main():
    args = parse_args()
    print("🔧 Args:", args)

    df = load_training_data(args.sm_channel_train)
    print("Loaded training data shape:", df.shape)

    # Let preprocess handle building Result from Prediction Result
    X, y = preprocess(df, target_col=args.target_col)

    model, metrics = train_and_evaluate(
        X,
        y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    os.makedirs(args.sm_model_dir, exist_ok=True)
    model_path = os.path.join(args.sm_model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path}")

    os.makedirs(args.sm_output_data_dir, exist_ok=True)
    metrics_path = os.path.join(args.sm_output_data_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"📊 Saved metrics to {metrics_path}")



if __name__ == "__main__":
    main()
