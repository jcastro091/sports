#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime, timezone

import boto3
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


log = logging.getLogger("sheets_to_s3")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )


def get_gsheet_client(creds_path: str) -> gspread.Client:
    scope = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    return gspread.authorize(creds)


def load_worksheet_to_df(gc: gspread.Client, sheet_name: str, tab_name: str) -> pd.DataFrame:
    log.info(f"📄 Loading Google Sheet '{sheet_name}' tab '{tab_name}'...")
    sh = gc.open(sheet_name)
    ws = sh.worksheet(tab_name)
    data = ws.get_all_records()  # list[dict]
    df = pd.DataFrame(data)
    log.info(f"✅ Loaded {len(df):,} rows from {sheet_name}/{tab_name}")
    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            except Exception:
                pass

    return df


def write_df_to_s3_csv(df: pd.DataFrame, bucket: str, key: str):
    log.info(f"💾 Writing {len(df):,} rows to s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    log.info("✅ Upload complete")


def build_s3_key(table: str, run_date: datetime, base_prefix: str = "raw") -> str:
    date_str_partition = run_date.strftime("%Y-%m-%d")
    date_str_file = run_date.strftime("%Y%m%d")

    if table == "all_bets":
        subdir = "all_bets"
        filename = f"all_bets_{date_str_file}.csv"
    elif table == "all_observations":
        subdir = "all_observations"
        filename = f"all_observations_{date_str_file}.csv"
    else:
        raise ValueError(f"Unsupported table: {table}")

    key = f"{base_prefix}/{subdir}/date={date_str_partition}/{filename}"
    return key


def main():
    parser = argparse.ArgumentParser(description="Export Google Sheets to S3 for ML training.")
    parser.add_argument(
        "--table",
        choices=["all_bets", "all_observations", "all"],
        required=True,
        help="Which logical table to export.",
    )
    parser.add_argument(
        "--date",
        help="Run date in YYYY-MM-DD (defaults to today UTC).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.date:
        run_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        run_date = datetime.now(timezone.utc)

    creds_path = os.getenv("CREDS")
    if not creds_path:
        raise SystemExit("CREDS env var is not set (Google service account JSON).")

    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "ConfirmedBets")
    bucket = os.getenv("ML_DATA_BUCKET", "sharpsignal-ml-data")
    base_prefix = os.getenv("ML_DATA_PREFIX", "raw")

    log.info(f"Using Google Sheet: {sheet_name}")
    log.info(f"S3 bucket: {bucket}, base prefix: {base_prefix}")
    log.info(f"Run date (partition): {run_date.date().isoformat()}")

    gc = get_gsheet_client(creds_path)

    if args.table == "all":
        tables = ["all_bets", "all_observations"]
    else:
        tables = [args.table]

    for table in tables:
        if table == "all_bets":
            tab_name = "AllBets"
        elif table == "all_observations":
            tab_name = "AllObservations"
        else:
            raise ValueError(f"Unexpected table: {table}")

        df = load_worksheet_to_df(gc, sheet_name, tab_name)
        df = sanitize_df(df)
        df["ingest_date"] = run_date.date().isoformat()

        key = build_s3_key(table, run_date, base_prefix=base_prefix)
        write_df_to_s3_csv(df, bucket=bucket, key=key)

    log.info("🎉 Sheets → S3 export complete.")


if __name__ == "__main__":
    main()
