# airflow/dags/sharpsignal_train_model_dag.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator

default_args = {
    "owner": "sharpsignal",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="sharpsignal_weekly_training",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 6 * * 1",  # Every Monday 6:00 UTC
    catchup=False,
    default_args=default_args,
) as dag:

    run_alpha_trainer = EcsRunTaskOperator(
        task_id="run_alpha_trainer",
        cluster="sharpsignal-ml-cluster",               # <-- your ECS cluster
        task_definition="alpha-trainer-task",           # <-- your task def
        launch_type="FARGATE",
        overrides={
            "containerOverrides": [
                {
                    "name": "trainer",
                    "environment": [
                        {"name": "DATASET_NAME", "value": "ConfirmedBets - AllObservations.csv"},
                        {"name": "DO_CHRONO", "value": "1"},
                        {
                            "name": "MLFLOW_TRACKING_URI",
                            "value": "http://mlflow-server.internal:5000",
                        },
                        {
                            "name": "MLFLOW_EXPERIMENT_NAME",
                            "value": "sharpsignal_winloss",
                        },
                        {
                            "name": "SHARPSIGNAL_MODEL_BUCKET",
                            "value": "sharpsignal-ml",  # your S3 bucket
                        },
                        {
                            "name": "SHARPSIGNAL_MODEL_PREFIX",
                            "value": "models/latest",
                        },
                        # Telegram / creds envs you already use in train_model for alerts
                        # {"name": "TELEGRAM_BOT_TOKEN", "value": "..."},
                        # {"name": "TELEGRAM_CHAT_ID", "value": "..."},
                    ],
                }
            ],
        },
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-xxxxxx"],
                "securityGroups": ["sg-xxxxxx"],
                "assignPublicIp": "ENABLED",
            }
        },
    )
