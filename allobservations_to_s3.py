import os
import datetime as dt
from io import StringIO

import boto3
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials


# ---------- CONFIG VIA ENV (with safe defaults) ----------

GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH", "creds/telegrambetlogger-35856685bc29w.json")

# ID of the Google Sheet (from URL)
OBS_SHEET_ID = os.getenv("OBS_SHEET_ID")  # required

# Tab name at the bottom of the workbook
OBS_TAB_NAME = os.getenv("OBS_TAB_NAME", "AllObservations")

S3_BUCKET = os.getenv("S3_DATA_BUCKET", "sharpsignal-ml-data")
S3_PREFIX_RAW = os.getenv("S3_PREFIX_RAW", "observations/raw")
S3_PREFIX_LATEST = os.getenv("S3_PREFIX_LATEST", "observations/latest")


def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(GOOGLE_CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)


def sheet_tab_to_df(gc, sheet_id: str, tab_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab_name)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    return df


def df_to_s3_csv(df: pd.DataFrame, bucket: str, key: str):
    s3 = boto3.client("s3")
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"✅ Uploaded to s3://{bucket}/{key} (rows={len(df)})")


def main():
    if not OBS_SHEET_ID:
        raise SystemExit("❌ OBS_SHEET_ID env var not set.")

    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    gc = get_gspread_client()

    print(f"📄 Reading sheet {OBS_SHEET_ID} tab '{OBS_TAB_NAME}'...")
    df_obs = sheet_tab_to_df(gc, OBS_SHEET_ID, OBS_TAB_NAME)
    print(f"   -> {len(df_obs)} rows")

    raw_key = f"{S3_PREFIX_RAW}/allobservations_snapshot_{today}.csv"
    latest_key = f"{S3_PREFIX_LATEST}/allobservations.csv"

    df_to_s3_csv(df_obs, S3_BUCKET, raw_key)
    df_to_s3_csv(df_obs, S3_BUCKET, latest_key)


if __name__ == "__main__":
    main()
