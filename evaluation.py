import argparse
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import boto3
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BUCKET_DEFAULT = os.environ.get("EVAL_S3_BUCKET", "sharpsignal-ml-data")

RAW_BASE_PREFIX = "raw/all_observations"
METRICS_BASE_PREFIX = "metrics/daily"


MLFLOW_TRACKING_URI = (
    os.getenv("MLFLOW_TRACKING_ARN")   # <-- ARN from SageMaker
    or os.getenv("MLFLOW_TRACKING_URI")
    or os.getenv("MLFLOW_TRACKING")
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EVAL_EXPERIMENT",
    "sharpsignal-eval",
)





TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_EVAL_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_EVAL_CHAT_ID")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily model evaluation for SharpSignal.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Evaluation date in YYYY-MM-DD (defaults to yesterday, UTC).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=BUCKET_DEFAULT,
        help=f"S3 bucket name (default: {BUCKET_DEFAULT})",
    )
    return parser.parse_args()


def resolve_eval_date(date_str: str | None) -> datetime:
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # default: yesterday UTC
    today_utc = datetime.now(timezone.utc).date()
    return datetime.combine(today_utc - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)


def build_s3_key_for_observations(eval_date: datetime) -> str:
    date_str = eval_date.strftime("%Y-%m-%d")
    yyyymmdd = eval_date.strftime("%Y%m%d")
    return f"{RAW_BASE_PREFIX}/date={date_str}/all_observations_{yyyymmdd}.csv"


def load_daily_observations(bucket: str, key: str) -> pd.DataFrame:
    logging.info("Loading observations from s3://%s/%s", bucket, key)
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        logging.error("File not found in S3: s3://%s/%s", bucket, key)
        raise
    buf = io.BytesIO(obj["Body"].read())
    df = pd.read_csv(buf)
    logging.info("Loaded %d rows, %d columns from observations CSV.", df.shape[0], df.shape[1])
    return df


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_core_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute ROI, win rate, AUC, calibration, and basic segment stats.

    We treat:
      - Prediction Result == 1 as win
      - Prediction Result == 0 as loss

    We include ALL bets (including PASS), but also provide subsets for:
      - ABC tiers only
      - Non-PASS/Unknown tiers
    """
    df = df.copy()

    df["Prediction Result"] = _clean_numeric(df.get("Prediction Result"))
    df["Stake Amount"] = _clean_numeric(df.get("Stake Amount"))
    df["Decimal Odds (Current)"] = _clean_numeric(df.get("Decimal Odds (Current)"))
    df["Model Probability"] = _clean_numeric(df.get("Model Probability"))

    # Valid bets for evaluation (ALL tiers)
    valid_mask = (
        df["Prediction Result"].isin([0, 1])
        & df["Stake Amount"].notna()
        & (df["Stake Amount"] > 0)
        & df["Decimal Odds (Current)"].notna()
        & df["Model Probability"].notna()
    )
    df_valid = df.loc[valid_mask].copy()
    logging.info("Found %d valid bets for evaluation.", len(df_valid))

    if df_valid.empty:
        raise ValueError("No valid bets found for evaluation (after filtering).")

    # Core fields
    y_true = df_valid["Prediction Result"].astype(int)
    stake = df_valid["Stake Amount"]
    odds = df_valid["Decimal Odds (Current)"]
    proba = df_valid["Model Probability"].clip(0.0, 1.0)

    # Profit per bet using closing decimal odds and stake
    profit = np.where(y_true == 1, stake * (odds - 1.0), -stake)
    df_valid["profit"] = profit

    total_profit = float(profit.sum())
    total_staked = float(stake.sum())
    roi_all = total_profit / total_staked if total_staked > 0 else 0.0
    win_rate_all = float((y_true == 1).mean())
    n_bets = int(len(df_valid))

    # Tier subsets (for extra ROI breakdowns)
    tier_code = df_valid.get("Tier Code", pd.Series(index=df_valid.index, dtype="object"))
    tier_label = df_valid.get("Tier Label", pd.Series(index=df_valid.index, dtype="object")).fillna("")

    is_abc = tier_code.isin(["A", "B", "C"])
    is_pass_like = (
        tier_label.str.contains("pass", case=False, na=False)
        | tier_code.eq("PASS")
        | tier_label.eq("Unknown")
    )

    subsets: Dict[str, Dict[str, Any]] = {}

    def _subset_metrics(mask: pd.Series) -> Dict[str, float]:
        df_sub = df_valid.loc[mask]
        if df_sub.empty:
            return {
                "n_bets": 0,
                "win_rate": None,
                "roi": None,
                "total_profit": 0.0,
                "total_staked": 0.0,
            }
        y = df_sub["Prediction Result"].astype(int)
        st = df_sub["Stake Amount"]
        od = df_sub["Decimal Odds (Current)"]
        pr = np.where(y == 1, st * (od - 1.0), -st)
        total_p = float(pr.sum())
        total_s = float(st.sum())
        roi = total_p / total_s if total_s > 0 else 0.0
        win_rate = float((y == 1).mean())
        return {
            "n_bets": int(len(df_sub)),
            "win_rate": win_rate,
            "roi": roi,
            "total_profit": total_p,
            "total_staked": total_s,
        }

    # ALL bets (including PASS)
    subsets["all"] = {
        "n_bets": n_bets,
        "win_rate": win_rate_all,
        "roi": roi_all,
        "total_profit": total_profit,
        "total_staked": total_staked,
    }
    # ABC tiers only
    subsets["tier_abc"] = _subset_metrics(is_abc)
    # Everything except obvious PASS/Unknown tiers
    subsets["no_pass_unknown"] = _subset_metrics(~is_pass_like)

    # AUC / ROC using all valid bets
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        logging.exception("Failed to compute ROC AUC.")
        auc = None

    # Calibration curve (10 bins)
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, proba, n_bins=10, strategy="uniform"
        )
        calibration_bins: List[Dict[str, Any]] = []
        for p_hat, p_true in zip(prob_pred, prob_true):
            calibration_bins.append(
                {
                    "p_hat": float(p_hat),
                    "empirical": float(p_true),
                }
            )
    except Exception:
        logging.exception("Failed to compute calibration curve.")
        calibration_bins = []

    # Segment summaries (all bets)
    segments: Dict[str, Any] = {}

    def _segment_by(col: str) -> List[Dict[str, Any]]:
        if col not in df_valid.columns:
            return []
        results: List[Dict[str, Any]] = []
        for value, group in df_valid.groupby(col):
            if group.empty:
                continue
            y = group["Prediction Result"].astype(int)
            st = group["Stake Amount"]
            od = group["Decimal Odds (Current)"]
            pr = np.where(y == 1, st * (od - 1.0), -st)
            total_p = float(pr.sum())
            total_s = float(st.sum())
            roi_g = total_p / total_s if total_s > 0 else 0.0
            win_rate_g = float((y == 1).mean())
            results.append(
                {
                    col: value if pd.notna(value) else "NaN",
                    "n_bets": int(len(group)),
                    "win_rate": win_rate_g,
                    "roi": roi_g,
                    "total_profit": total_p,
                    "total_staked": total_s,
                    "avg_stake": float(st.mean()),
                    "avg_model_prob": float(group["Model Probability"].mean()),
                    "avg_limit": float(group["Bet Limit"].mean())
                    if "Bet Limit" in group.columns
                    else None,
                }
            )
        # sort largest volume first
        results.sort(key=lambda r: r["n_bets"], reverse=True)
        return results

    segments["by_sport"] = _segment_by("Sport")
    segments["by_market"] = _segment_by("Market")
    segments["by_tier_label"] = _segment_by("Tier Label")
    segments["by_tier_code"] = _segment_by("Tier Code")

    metrics: Dict[str, Any] = {
        "totals": subsets["all"],   # ALL bets, including PASS
        "subsets": subsets,         # all / tier_abc / no_pass_unknown
        "auc": auc,
        "calibration": {"bins": calibration_bins},
        "segments": segments,
    }
    return metrics


def write_metrics_to_s3(bucket: str, eval_date: datetime, metrics: Dict[str, Any]) -> str:
    s3 = boto3.client("s3")
    date_str = eval_date.strftime("%Y-%m-%d")
    yyyymmdd = eval_date.strftime("%Y%m%d")
    key = f"{METRICS_BASE_PREFIX}/date={date_str}/evaluation_{yyyymmdd}.json"

    import json

    body = json.dumps(metrics, indent=2, default=str).encode("utf-8")

    logging.info("Writing metrics JSON to s3://%s/%s", bucket, key)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key


def log_metrics_to_mlflow(eval_date: datetime, metrics: Dict[str, Any]):
    # If no tracking URI, just skip MLflow logging
    if not MLFLOW_TRACKING_URI:
        logging.info("MLFLOW_TRACKING_URI not set; skipping MLflow logging.")
        return

    try:
        logging.info("Logging metrics to MLflow at %s", MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        date_str = eval_date.strftime("%Y-%m-%d")

        with mlflow.start_run(run_name=f"eval-{date_str}"):
            # Log top-level metrics (ALL bets)
            totals = metrics.get("totals", {})
            mlflow.log_metric("n_bets_all", totals.get("n_bets", 0))
            if totals.get("win_rate") is not None:
                mlflow.log_metric("win_rate_all", totals["win_rate"])
            if totals.get("roi") is not None:
                mlflow.log_metric("roi_all", totals["roi"])
            if totals.get("total_profit") is not None:
                mlflow.log_metric("total_profit_all", totals["total_profit"])
            if totals.get("total_staked") is not None:
                mlflow.log_metric("total_staked_all", totals["total_staked"])

            # Subset breakdowns
            subsets = metrics.get("subsets", {})
            for name, sub in subsets.items():
                prefix = f"{name}_"
                if sub.get("n_bets") is not None:
                    mlflow.log_metric(prefix + "n_bets", sub["n_bets"])
                if sub.get("win_rate") is not None:
                    mlflow.log_metric(prefix + "win_rate", sub["win_rate"])
                if sub.get("roi") is not None:
                    mlflow.log_metric(prefix + "roi", sub["roi"])

            auc = metrics.get("auc")
            if auc is not None:
                mlflow.log_metric("auc", auc)

            # Log date as a param
            mlflow.log_param("eval_date", date_str)

            # Log full metrics JSON as artifact
            import tempfile, json

            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, f"evaluation_{date_str}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, default=str)
                mlflow.log_artifact(json_path, artifact_path="evaluation")

    except Exception:
        logging.exception("MLflow logging failed; continuing without it.")


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials not set; skipping Telegram notification.")
        return
    import requests

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.error("Failed to send Telegram message: %s", resp.text)
    except Exception:
        logging.exception("Error sending Telegram message.")


def format_summary_message(eval_date: datetime, metrics: Dict[str, Any]) -> str:
    date_str = eval_date.strftime("%Y-%m-%d")
    totals = metrics.get("totals", {})
    roi = totals.get("roi")
    win_rate = totals.get("win_rate")
    n_bets = totals.get("n_bets", 0)
    profit = totals.get("total_profit")
    segments = metrics.get("segments", {})

    roi_str = f"{roi:.2%}" if roi is not None else "N/A"
    win_rate_str = f"{win_rate:.2%}" if win_rate is not None else "N/A"
    profit_str = f"{profit:+.2f}u" if profit is not None else "N/A"

    lines = [
        f"🧮 *SharpSignal Daily Eval — {date_str}*",
        "",
        f"*Bets:* {n_bets}",
        f"*Win rate:* {win_rate_str}",
        f"*ROI:* {roi_str}",
        f"*Profit:* {profit_str}",
    ]

    # sport + market top performers if available
    by_sport = segments.get("by_sport") or []
    if by_sport:
        top = by_sport[0]
        lines.append(
            f"*Top sport:* {top.get('Sport', 'N/A')} "
            f"({top['n_bets']} bets, ROI {top['roi']:.2%})"
        )

    by_market = segments.get("by_market") or []
    if by_market:
        top_m = by_market[0]
        lines.append(
            f"*Top market:* {top_m.get('Market', 'N/A')} "
            f"({top_m['n_bets']} bets, ROI {top_m['roi']:.2%})"
        )

    return "\n".join(lines)


def main():
    setup_logging()
    args = parse_args()

    eval_date = resolve_eval_date(args.date)
    logging.info("Running evaluation for date: %s", eval_date.strftime("%Y-%m-%d"))

    key = build_s3_key_for_observations(eval_date)
    try:
        df = load_daily_observations(args.bucket, key)
        metrics = compute_core_metrics(df)
        metrics["date"] = eval_date.strftime("%Y-%m-%d")
        metrics_s3_key = write_metrics_to_s3(args.bucket, eval_date, metrics)
        logging.info("Metrics written to s3://%s/%s", args.bucket, metrics_s3_key)

        log_metrics_to_mlflow(eval_date, metrics)

        # Success Telegram summary
        msg = format_summary_message(eval_date, metrics)
        send_telegram_message(msg)
        logging.info("Daily evaluation completed successfully.")
    except Exception as exc:
        logging.exception("Daily evaluation FAILED.")
        # Send failure alert
        fail_msg = (
            f"❌ *Daily evaluation FAILED* for {eval_date.strftime('%Y-%m-%d')}\n"
            f"Error: {exc}"
        )
        send_telegram_message(fail_msg)
        # Re-raise to make ECS / scheduler mark as failed
        raise


if __name__ == "__main__":
    main()
