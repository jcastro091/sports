# src/backtest/engine.py
from ..preprocess import load_allbets_clean
from ..model_features import add_basic_features
from .metrics import summarize
import pandas as pd
import yaml, os

def load_settings(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_backtest(stake_rule="kelly_cap", kelly_cap=0.02):
    df = load_allbets_clean().reset_index(drop=True)
    df["row_idx"] = df.index  # used later to join ML preds
    df = add_basic_features(df)

    # --- stake sizing ---
    if "kelly" in df.columns:
        k = pd.to_numeric(df["kelly"], errors="coerce").astype("Float64")  # nullable float
        if k.dropna().gt(1).any():
            k = k / 100.0
        k = k.astype(float)  # regular float for math ops
        df["stake"] = k.clip(upper=kelly_cap).fillna(0.005).astype(float)
    else:
        df["stake"] = 0.01

    summary = summarize(df)
    return summary, df

def segment_summaries(df: pd.DataFrame):
    grp = df.groupby(["sport", "market"], dropna=False)
    try:
        seg = grp.apply(lambda g: pd.Series(summarize(g)), include_groups=False).reset_index()
    except TypeError:
        seg = grp.apply(lambda g: pd.Series(summarize(g))).reset_index()

    conf = None
    if "ml_confidence" in df.columns:
        bins = [0, 0.6, 0.75, 0.9, 1.01]
        labels = ["≤0.60", "0.60–0.75", "0.75–0.90", ">0.90"]
        df["conf_bucket"] = pd.cut(df["ml_confidence"], bins=bins, labels=labels, include_lowest=True)
        conf_grp = df.groupby("conf_bucket", dropna=False)
        try:
            conf = conf_grp.apply(lambda g: pd.Series(summarize(g)), include_groups=False).reset_index()
        except TypeError:
            conf = conf_grp.apply(lambda g: pd.Series(summarize(g))).reset_index()
    return seg, conf

if __name__ == "__main__":
    s = load_settings()
    os.makedirs(s["paths"]["results_dir"], exist_ok=True)

    summary, df = run_backtest()
    print(summary)

    # write main csv
    out_dir = s["paths"]["results_dir"]
    df.to_csv(os.path.join(out_dir, "backtest_latest.csv"), index=False)

    # base segments
    seg, conf = segment_summaries(df)
    seg.to_csv(os.path.join(out_dir, "segments_sport_market.csv"), index=False)
    if conf is not None:
        conf.to_csv(os.path.join(out_dir, "segments_confidence.csv"), index=False)

    # If ML preds exist, merge and write a combined file + confidence segments
    ml_path = os.path.join(out_dir, "ml_predictions.csv")
    if os.path.exists(ml_path):
        ml = pd.read_csv(ml_path)
        merged = df.merge(ml, on="row_idx", how="left")
        merged.to_csv(os.path.join(out_dir, "backtest_with_ml.csv"), index=False)

        seg2, conf2 = segment_summaries(merged)
        seg2.to_csv(os.path.join(out_dir, "segments_sport_market.csv"), index=False)
        if conf2 is not None:
            conf2.to_csv(os.path.join(out_dir, "segments_confidence.csv"), index=False)
