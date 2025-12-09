# SharpsSignal Sports Engine

SharpsSignal is an automated sports betting analytics and signal engine.

It ingests market data (odds + limits), logs predictions, and pushes curated bets to Telegram and dashboards — with a full ML pipeline (training, evaluation, and deployment) running on AWS.

> **Goal:** Turn raw odds + limits into high-quality, tiered betting signals (A/B/C) with clear performance metrics and minimal noise.

---

## High-Level Architecture

- **Data sources**
  - Sportsbook odds & limits (Pinnacle, others via The Odds API)
  - Google Sheets:
    - `ConfirmedBets` (tabs: `AllBets`, `ConfirmedBets`)
    - `ConfirmedTrades`
  - Final game scores (for model evaluation)

- **Core components**
  - **`sports.py`** – main real-time signal engine
    - Polls odds + limits on a schedule
    - Applies ML model + thresholds (`A/B/C/PASS`)
    - Sends Telegram alerts for actionable plays
    - Logs predictions to CSV + Google Sheets (`AllBets`)
  - **Telegram bot & logger (`telegram_confirm_logger.py`)**
    - Users confirm bets (and confidence) by replying in Telegram
      - ✅ High, 🟡 Medium, ⚪️ Low
    - Writes confirmed bets/trades into Google Sheets:
      - `ConfirmedBets` (sports bets)
      - `ConfirmedTrades` (stock/forex trades)
    - Auto-tags setups (H2H Underdog/Favorite, Spread, Totals, etc.)
  - **ML engine (`alpha_signal_engine/`)**
    - Fetches training data from S3 / Sheets
    - Feature engineering + model training (`train_model.py`)
    - Threshold tuning for A/B/C tiers
    - Model + metadata pushed to S3 & tracked with MLflow
  - **Pipelines & utilities**
    - `sheets_to_s3.py` – syncs Google Sheets → S3 (`raw/` partitions)
    - `scores_to_s3.py` / similar – syncs final scores → S3
    - `evaluation.py` – joins predictions + final scores, computes:
      - ROI, win rate
      - AUC/ROC
      - Calibration metrics
      - Market-based performance
    - Writes metrics to S3 (`metrics/daily/`) and logs to MLflow
  - **Dashboards**
    - Streamlit app powered by Google Sheets (and/or S3):
      - Filters by sport, market, setup, confidence
      - Visualizes win rate, ROI, top teams, tag-based insights
      - Used for internal QC + potential customer-facing views

- **Infrastructure**
  - **AWS S3** – central data lake for raw data, features, models, metrics  
    - Bucket: `sharpsignal-ml-data` (raw / features / models / metrics)
  - **AWS SageMaker** – batch training jobs
    - `train_sagemaker.py` wraps `train_model.py`
    - Outputs:
      - `baseline_winloss.pkl`
      - `model_features.pkl`
      - `model_card.json`
      - `thresholds.json` (auto-tuned A/B/C cutoffs)
  - **AWS ECS / ECR**
    - Docker images (e.g., `sharpsignal-sports`, `sharpsignal-trainer`)
    - **Sports Service**: long-running odds watcher (`sports.py`)
    - **Trainer / Eval Tasks**: scheduled one-shot ECS tasks for:
      - Model training
      - Daily evaluation
  - **MLflow**
    - Tracks experiments, runs, and metrics
    - Stores links to versioned models + thresholds

---

## Repo Layout (simplified)

> Names may vary slightly; this is the conceptual layout.

```text
.
├── sports.py                     # Main real-time signal engine (production)
├── out_sports*_EDGE.py           # ECS / Docker entrypoints
├── telegram_confirm_logger.py    # Telegram confirmation + Sheets logger
├── sheets_to_s3.py               # Google Sheets -> S3 sync
├── evaluation.py                 # Daily model evaluation
├── alpha_signal_engine/
│   ├── train_model.py            # Full training pipeline
│   ├── train_sagemaker.py        # SageMaker-compatible wrapper
│   └── data/results/models/
│       ├── baseline_winloss.pkl
│       ├── model_features.pkl
│       ├── thresholds.json
│       ├── model_card.json
│       └── weekly_metrics.csv
├── streamlit_app/                # Dashboard for bets/predictions
├── docker/                       # Dockerfiles, ECS task config
└── ...
