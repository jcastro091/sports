# alpha_signal_engine/src/train_model_mlflow_entry.py
"""
Entry point for AWS trainer container.

- Calls train_models(...) from train_model.py
- Reads model_card.json + model_health_latest.csv
- Logs metrics & artifacts to MLflow (if MLFLOW_TRACKING_URI is set)
- Uploads latest model artifacts to S3 for sports-runner to consume
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import mlflow

from train_model import train_models, MODEL_DIR
from model_registry_s3 import upload_latest_model_to_s3


def _load_model_card() -> dict:
    card_path = MODEL_DIR / "model_card.json"
    if not card_path.exists():
        return {}
    return json.loads(card_path.read_text(encoding="utf-8"))


def _load_health_metrics() -> dict:
    health_path = MODEL_DIR / "model_health_latest.csv"
    if not health_path.exists():
        return {}
    df = pd.read_csv(health_path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_NAME", "ConfirmedBets - AllObservations.csv"),
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable threshold tuning (sets DO_TUNE=1)",
    )
    parser.add_argument(
        "--pca", action="store_true",
        help="Enable PCA export (sets DO_PCA=1)",
    )
    parser.add_argument(
        "--chronological", action="store_true",
        help="Use time-aware split (sets DO_CHRONO=1)",
    )
    args = parser.parse_args()

    # Respect same env toggles train_model.py expects
    if args.tune:
        os.environ["DO_TUNE"] = "1"
    if args.pca:
        os.environ["DO_PCA"] = "1"
    if args.chronological:
        os.environ["DO_CHRONO"] = "1"

    # 1) Run the existing training pipeline
    train_models(dataset_name=args.dataset)

    # 2) Load metadata + health
    card = _load_model_card()
    health = _load_health_metrics()

    # 3) Log to MLflow (if configured)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "sharpsignal_winloss"))

        run_name = f"{card.get('model_name', 'model')}_{card.get('timestamp', '')}"

        with mlflow.start_run(run_name=run_name):
            # Core metrics from model card
            if "auc" in card:
                mlflow.log_metric("auc_test", float(card["auc"]))
            if "accuracy" in card:
                mlflow.log_metric("accuracy_test", float(card["accuracy"]))
            if "threshold" in card:
                mlflow.log_metric("threshold", float(card["threshold"]))
            if "rows_total" in card:
                mlflow.log_metric("rows_total", float(card["rows_total"]))

            # Health/drift metrics
            for k, v in health.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

            # Params / metadata
            mlflow.log_param("model_name", card.get("model_name"))
            mlflow.log_param("dataset", card.get("dataset"))
            mlflow.log_param("dataset_hash", card.get("dataset_hash"))
            mlflow.log_param("class_balance", json.dumps(card.get("class_balance", {})))

            # Log artifacts (models, reports, pca, etc.)
            if MODEL_DIR.exists():
                mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model_artifacts")

    # 4) Push latest model artifacts to S3 so sports-runner can pull them
    upload_latest_model_to_s3()


if __name__ == "__main__":
    main()
