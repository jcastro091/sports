import duckdb
import os
from pathlib import Path

def get_con(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path)
