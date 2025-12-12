# test_thresholds_tuner.py  (example)

from pathlib import Path
import pandas as pd
from src.thresholds_tuner import tune_and_save_thresholds
import train_model as tm  # to get MODEL_DIR

preds_csv = tm.MODEL_DIR / "predictions_latest.csv"
df = pd.read_csv(preds_csv)

output_path = tm.MODEL_DIR / "thresholds.json"   # 🔴 changed from thresholds_test.json

thresholds, summary = tune_and_save_thresholds(
    df,
    output_path=output_path,
    proba_col="ModelProba",
    result_col="Result",
    odds_col="decimal_odds",
    stake_col="Stake",
)

print("Thresholds:", thresholds)
print(summary.tail())
