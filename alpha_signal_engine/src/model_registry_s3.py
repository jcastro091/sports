# alpha_signal_engine/src/model_registry_s3.py
from __future__ import annotations

import os
from pathlib import Path

import boto3

# Recreate MODEL_DIR without importing train_model
# Engine root = parent directory of this src/ folder
ENGINE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ENGINE_ROOT / "data"
MODEL_DIR = DATA_DIR / "results" / "models"

# Env vars you set in ECS / Airflow
BUCKET = os.getenv("SHARPSIGNAL_MODEL_BUCKET", "")
PREFIX = os.getenv("SHARPSIGNAL_MODEL_PREFIX", "models/latest")

_S3 = boto3.client("s3")



def _upload_file(local: Path, key_suffix: str):
    if not BUCKET:
        print("[model_registry_s3] BUCKET not configured, skipping upload")
        return
    key = f"{PREFIX}/{key_suffix}"
    print(f"[model_registry_s3] Uploading {local} → s3://{BUCKET}/{key}")
    _S3.upload_file(str(local), BUCKET, key)


def _download_file(local: Path, key_suffix: str):
    if not BUCKET:
        print("[model_registry_s3] BUCKET not configured, skipping download")
        return
    key = f"{PREFIX}/{key_suffix}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model_registry_s3] Downloading s3://{BUCKET}/{key} → {local}")
    _S3.download_file(BUCKET, key, str(local))


def upload_latest_model_to_s3():
    """
    Called by trainer after successful training.
    Ships model.pkl, model_features.pkl, model_card.json, model_health_latest.csv.
    """
    files = [
        "model.pkl",
        "model_features.pkl",
        "model_card.json",
        "model_health_latest.csv",
    ]
    for name in files:
        local = MODEL_DIR / name
        if local.exists():
            _upload_file(local, name)
        else:
            print(f"[model_registry_s3] WARNING: {local} missing, skip")


def download_latest_model_from_s3(target_dir: Path | None = None):
    """
    Call this from sports-runner startup (or ml_predict) so inference uses the newest model.
    """
    if target_dir is None:
        target_dir = MODEL_DIR

    files = [
        "model.pkl",
        "model_features.pkl",
        "model_card.json",
        "model_health_latest.csv",
    ]
    for name in files:
        local = target_dir / name
        try:
            _download_file(local, name)
        except Exception as e:
            print(f"[model_registry_s3] ERROR downloading {name}: {e}")
