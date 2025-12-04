# launch_sagemaker_train.py

import os
from datetime import datetime, timezone

import boto3
from sagemaker import Session
from sagemaker.sklearn import SKLearn

# === CONFIG ===
AWS_REGION = "us-east-2"
BUCKET = "sharpsignal-ml-data"
TRAIN_PREFIX = "raw/all_observations/"
OUTPUT_PREFIX = "sagemaker-output/"
ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE",
    "arn:aws:iam::159903201403:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
)

# ---- MLflow (managed on SageMaker) ----
# Copy the ARN from the MLflow page in SageMaker; this is the one from your screenshot.
# MLflow managed tracking server (SageMaker)
MLFLOW_TRACKING_URI = (
    "arn:aws:sagemaker:us-east-2:159903201403:mlflow-tracking-server/sharpsignal-mlflow"
)
MLFLOW_EXPERIMENT_NAME = "sharpsignal-sagemaker"


def get_latest_all_observations_uri() -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    latest_key = None
    latest_modified = None

    for page in paginator.paginate(Bucket=BUCKET, Prefix=TRAIN_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Only consider your daily all_observations CSV files
            if not key.lower().endswith(".csv"):
                continue
            if "all_observations_" not in key:
                continue

            lm = obj["LastModified"]
            if latest_modified is None or lm > latest_modified:
                latest_modified = lm
                latest_key = key

    if not latest_key:
        raise RuntimeError(
            f"No CSV found under s3://{BUCKET}/{TRAIN_PREFIX}. "
            "Check that at least one all_observations_*.csv is uploaded there "
            "or adjust TRAIN_PREFIX."
        )

    return f"s3://{BUCKET}/{latest_key}"


def main():
    sess = Session(boto3.Session(region_name=AWS_REGION))

    # 1) Find latest CSV
    train_s3_uri = get_latest_all_observations_uri()
    print("Training on S3 URI:", train_s3_uri)

    # 2) Build job name + output path
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"sharpsignal-train-{ts}"
    output_path = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

    # 3) Define SKLearn estimator
    sk_estimator = SKLearn(
        entry_point="train_sagemaker.py",
        source_dir=".",  # same as before
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        hyperparameters={
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "target_col": "Result",
        },
        sagemaker_session=sess,
        output_path=output_path,        
        enable_sagemaker_metrics=True,
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        },
    )


    # 4) Kick off training
    print(f"Starting training job: {job_name}")
    sk_estimator.fit(
        inputs={"train": train_s3_uri},
        job_name=job_name,
        wait=True,
    )

    print("✅ SageMaker training job completed:", job_name)


if __name__ == "__main__":
    main()
