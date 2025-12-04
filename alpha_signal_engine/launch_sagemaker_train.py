# launch_sagemaker_train.py
from pathlib import Path
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import Session
import boto3
import datetime as dt



def main():
    session = Session()
    role = "arn:aws:iam::159903201403:role/service-role/AmazonSageMakerAdminIAMExecutionRole"  # you already have this set

    train_s3_uri = "s3://sharpsignal-ml-data/raw/all_observations/date=2025-11-30/all_observations_20251130.csv"

    script_dir = Path(__file__).resolve().parent  # folder with train_sagemaker.py AND train_model.py

    MLFLOW_TRACKING_ARN = (
        "arn:aws:sagemaker:us-east-2:159903201403:mlflow-tracking-server/sharpsignal-mlflow"
    )

    sk_estimator = SKLearn(
        entry_point="train_sagemaker.py",
        source_dir=str(script_dir),   # 🔴 IMPORTANT: include BOTH scripts
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.0-1",    # or whatever you set
        py_version="py3",
        hyperparameters={
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "target_col": "Result",
        },
        env={
            # This is what your training container will see
            "MLFLOW_TRACKING_ARN": MLFLOW_TRACKING_ARN,
            "MLFLOW_EXPERIMENT_NAME": "sharpsignal-sagemaker",
        },        
        
    )

    now = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"sharpsignal-train-{now}"
    print(f"Starting training job: {job_name}")

    sk_estimator.fit(
        inputs={"train": train_s3_uri},
        job_name=job_name,
        wait=True,
        logs=True,
    )

if __name__ == "__main__":
    main()
