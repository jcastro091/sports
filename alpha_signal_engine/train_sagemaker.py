import argparse
import os
import glob
import json
import shutil
from pathlib import Path
from typing import Optional

import mlflow
import train_model as tm  # this is fine if train_model.py is in the same source_dir

# ---------------------------------------------------------------------
# MLflow config: use your SageMaker MLflow tracking server directly
# ---------------------------------------------------------------------
MLFLOW_TRACKING_URI = (
    "arn:aws:sagemaker:us-east-2:159903201403:mlflow-tracking-server/sharpsignal-mlflow"
)
MLFLOW_EXPERIMENT_NAME = "sharpsignal-sagemaker"


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparams are accepted (for future tuning), but the real logic lives
    # in train_model.py. Keeping them avoids SageMaker complaining.
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=10)

    # Backward-compat: accepted but unused – train_model.py uses its own target.
    parser.add_argument(
        "--target_col",
        type=str,
        default="Result",
        help="Kept for backward compatibility; ignored by train_model.py.",
    )

    # Optional override for dataset name once copied into tm.DATA_DIR
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional filename to save the training CSV as inside data/.",
    )

    # Use chronological split if set (train past, test future)
    parser.add_argument(
        "--chronological",
        action="store_true",
        help="Use time-aware train/test split in train_model.py.",
    )

    # SageMaker-provided paths
    parser.add_argument("--sm_model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--sm_output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR"),
    )
    parser.add_argument(
        "--sm_channel_train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )

    return parser.parse_args()


def find_training_csv(train_dir: str) -> str:
    """
    Find the training CSV under the SageMaker `train` channel directory.
    Typically this is /opt/ml/input/data/train/....

    We prefer your daily all_observations_*.csv file.
    """
    if not train_dir or not os.path.isdir(train_dir):
        raise RuntimeError(f"Train directory does not exist: {train_dir}")

    pattern = os.path.join(train_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise RuntimeError(f"No CSV files found under {train_dir}")

    # Prefer all_observations_* if present
    preferred = [f for f in files if "all_observations_" in os.path.basename(f)]
    files_to_consider = preferred or files

    files_to_consider.sort()
    csv_path = files_to_consider[-1]  # "latest" by name
    print(f"📄 Using training CSV: {csv_path}")
    return csv_path


def copy_csv_into_engine(csv_src: str, dataset_name: Optional[str]) -> str:
    """
    Copy the SageMaker training CSV into the alpha_signal_engine data directory,
    so train_model.py can use its normal DATA_DIR / DATASET_NAME flow.

    Returns the dataset_name to pass into train_models().
    """
    data_dir: Path = tm.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name is None:
        dataset_name = os.path.basename(csv_src)

    dst_path = data_dir / dataset_name
    print(f"📥 Copying training CSV → {dst_path}")
    shutil.copy2(csv_src, dst_path)

    return dataset_name


def copy_artifacts_to_sagemaker_dirs(sm_model_dir: str, sm_output_dir: str):
    """
    After train_model.py finishes, take key artifacts from tm.MODEL_DIR
    and put them where SageMaker expects them.
    """
    model_dir: Path = tm.MODEL_DIR

    # === SM_MODEL_DIR: deployable model ===
    if sm_model_dir:
        sm_model_dir_path = Path(sm_model_dir)
        sm_model_dir_path.mkdir(parents=True, exist_ok=True)

        # Main model + feature list
        src_model = tm.MODEL_PKL
        if src_model.exists():
            dst_model = sm_model_dir_path / "model.pkl"
            print(f"💾 Copying model → {dst_model}")
            shutil.copy2(src_model, dst_model)

        src_feats = tm.FEATS_PKL
        if src_feats.exists():
            dst_feats = sm_model_dir_path / "model_features.pkl"
            print(f"💾 Copying feature list → {dst_feats}")
            shutil.copy2(src_feats, dst_feats)

        # Tier config is useful for inference containers
        if tm.TIER_CONFIG_OUT.exists():
            dst_tier = sm_model_dir_path / "tier_config.json"
            print(f"💾 Copying tier config → {dst_tier}")
            shutil.copy2(tm.TIER_CONFIG_OUT, dst_tier)

    # === SM_OUTPUT_DATA_DIR: diagnostics, reports, metrics.json ===
    if sm_output_dir:
        sm_output_dir_path = Path(sm_output_dir)
        sm_output_dir_path.mkdir(parents=True, exist_ok=True)

        candidates = [
            model_dir / "model_card.json",
            model_dir / "feature_importances.csv",
            model_dir / "predictions_latest.csv",
            model_dir / "weekly_metrics.csv",
            model_dir / "logreg_learning_curve.csv",
            model_dir / "rf_learning_curve.csv",
            tm.TIER_CONFIG_OUT,
        ]
        for src in candidates:
            if src.exists():
                dst = sm_output_dir_path / src.name
                print(f"📤 Copying artifact → {dst}")
                shutil.copy2(src, dst)

        # Also write a concise metrics.json from model_card.json for quick inspection
        card_path = model_dir / "model_card.json"
        if card_path.exists():
            try:
                with card_path.open("r", encoding="utf-8") as f:
                    card = json.load(f)

                metrics_out = {
                    "model_name": card.get("model_name"),
                    "auc": card.get("auc"),
                    "accuracy": card.get("accuracy"),
                    "threshold": card.get("threshold"),
                    "rows_total": card.get("rows_total"),
                    "dataset": card.get("dataset"),
                    "timestamp": card.get("timestamp"),
                }
                metrics_path = sm_output_dir_path / "metrics.json"
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics_out, f)
                print(f"📊 Wrote summary metrics → {metrics_path}")
            except Exception as e:
                print(f"[warn] Could not build metrics.json from model_card: {e}")


def main():
    args = parse_args()
    print("🔧 Args:", args)

    # -----------------------------------------------------------------
    # Configure MLflow to use your SageMaker MLflow Tracking Server
    # -----------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Allow overriding experiment name via env var if you want
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(exp_name)

    # Also expose to train_model.py if it reads env vars
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name

    # 1) Locate training CSV in SageMaker input channel
    csv_path = find_training_csv(args.sm_channel_train)

    # 2) Copy CSV into alpha_signal_engine/data so train_model.py can use it
    dataset_name = copy_csv_into_engine(csv_path, args.dataset_name)

    # 3) Wire env vars that train_model.py expects
    os.environ["DATASET_NAME"] = dataset_name
    os.environ["DO_CHRONO"] = "1" if args.chronological else os.environ.get("DO_CHRONO", "0")
    os.environ.setdefault("DO_TUNE", "0")
    os.environ.setdefault("DO_PCA", "0")

    # Use the SageMaker training job name as the parent run name if available
    run_name = os.getenv("TRAINING_JOB_NAME", "sagemaker-train")

    # 4) Run the SAME training pipeline as local train_model.py under an MLflow run
    with mlflow.start_run(run_name=run_name):
        tm.train_models(dataset_name=dataset_name)

        # Log key artifacts on the parent run for convenience
        artifact_files = [
            tm.MODEL_DIR / "model_card.json",
            tm.MODEL_DIR / "feature_importances.csv",
            tm.MODEL_DIR / "predictions_latest.csv",
            tm.MODEL_DIR / "weekly_metrics.csv",
            tm.MODEL_DIR / "logreg_learning_curve.csv",
            tm.MODEL_DIR / "rf_learning_curve.csv",
            tm.TIER_CONFIG_OUT,
        ]
        for f in artifact_files:
            if f.exists():
                mlflow.log_artifact(str(f))

    print("✅ SageMaker training (via train_model.py) complete.")

    # 5) Copy artifacts into SM_MODEL_DIR / SM_OUTPUT_DATA_DIR for SageMaker
    copy_artifacts_to_sagemaker_dirs(args.sm_model_dir, args.sm_output_data_dir)

    # 6) Also push artifacts to S3 + send health summary
    try:
        tm.upload_artifacts_to_s3()
    except Exception as e:
        print(f"[warn] upload_artifacts_to_s3 failed: {e}")

    try:
        tm.send_health_summary()
    except Exception as e:
        print(f"[warn] send_health_summary failed: {e}")


if __name__ == "__main__":
    main()
