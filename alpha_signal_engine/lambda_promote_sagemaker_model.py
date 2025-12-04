# lambda_promote_sagemaker_model.py

import json
import os
import tarfile
import tempfile
from urllib.parse import urlparse

import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET = "sharpsignal-ml-data"

# Where the prod bundle lives (what ECS runner expects)
PROD_PREFIX = "models/prod"

sagemaker = boto3.client("sagemaker", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


def _parse_s3_uri(uri: str):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _read_auc_from_metrics(bucket: str, key: str) -> float | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        # adjust if your key name is different
        return float(data.get("auc") or data.get("AUC") or data.get("roc_auc"))
    except Exception as e:
        print(f"⚠️ Could not read AUC from s3://{bucket}/{key}: {e}")
        return None


def handler(event, context):
    print("Event:", json.dumps(event))

    detail = event.get("detail", {})
    training_job_name = detail.get("TrainingJobName")
    if not training_job_name:
        print("No TrainingJobName in event; skipping.")
        return

    print(f"⬆️ Promoting model from training job: {training_job_name}")

    # 1) Describe training job to find artifacts + output path
    resp = sagemaker.describe_training_job(TrainingJobName=training_job_name)
    model_artifacts_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    output_path = resp["OutputDataConfig"]["S3OutputPath"]

    # SageMaker stores output like:
    # s3://BUCKET/<OUTPUT_PREFIX>/<JOB_NAME>/output/model.tar.gz
    # and metrics in:
    # s3://BUCKET/<OUTPUT_PREFIX>/<JOB_NAME>/output/data/metrics.json
    out_bucket, out_prefix = _parse_s3_uri(output_path)
    metrics_key = f"{out_prefix.rstrip('/')}/{training_job_name}/output/data/metrics.json"

    new_auc = _read_auc_from_metrics(out_bucket, metrics_key)
    print(f"New model AUC: {new_auc}")

    # 2) Compare with current prod AUC (if exists)
    prod_metrics_key = f"{PROD_PREFIX}/metrics.json"
    prod_auc = _read_auc_from_metrics(BUCKET, prod_metrics_key)

    print(f"Current prod AUC: {prod_auc}")

    if prod_auc is not None and new_auc is not None and new_auc < prod_auc:
        print("❌ New model AUC worse than prod, skipping promotion.")
        return

    print("✅ Promoting model to prod…")

    # 3) Download and extract model.tar.gz
    art_bucket, art_key = _parse_s3_uri(model_artifacts_uri)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = os.path.join(tmpdir, "model.tar.gz")
        s3.download_file(art_bucket, art_key, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(tmpdir)

        # train_sagemaker.py currently saves model as model.pkl under /opt/ml/model/
        # When SageMaker packs model.tar.gz, it keeps that path.
        # Find model.pkl inside extracted tree:
        model_path = None
        for root, _, files in os.walk(tmpdir):
            if "model.pkl" in files:
                model_path = os.path.join(root, "model.pkl")
                break

        if not model_path:
            raise RuntimeError("model.pkl not found inside model.tar.gz")

        # 4) Upload model.pkl to prod location with the name the runner expects
        prod_model_key = f"{PROD_PREFIX}/baseline_winloss.pkl"
        s3.upload_file(model_path, BUCKET, prod_model_key)

        # 5) Also copy metrics.json into prod
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": out_bucket, "Key": metrics_key},
            Key=prod_metrics_key,
        )

        # (Optional) log which training job produced prod model
        info_key = f"{PROD_PREFIX}/model_info.json"
        info = {
            "training_job_name": training_job_name,
            "model_artifacts_uri": model_artifacts_uri,
            "output_path": output_path,
            "auc": new_auc,
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=info_key,
            Body=json.dumps(info, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    print(f"🎉 Promotion complete. Prod model at s3://{BUCKET}/{prod_model_key}")
