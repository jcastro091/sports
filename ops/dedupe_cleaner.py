#!/usr/bin/env python3
from __future__ import annotations
import os, time
from collections import defaultdict
from typing import Dict, List

# Reuse your existing helpers to read/write Google Sheets
# Assume you have a gspread wrapper like read_allbets(), delete_rows(ids), etc.
from your_module_for_sheets import read_allbets, delete_row_by_index  # <- adjust to your code

from ops.notify_tg import send as notify

TAB = os.getenv("ALLBETS_TAB", "AllBets")
TS_FIELD = os.getenv("TS_FIELD", "timestamp_iso")  # adjust to your schema
ID_FIELD = os.getenv("ID_FIELD", "bet_id")

def main() -> None:
    rows: List[Dict] = read_allbets(TAB)
    by_id: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        rid = str(r.get(ID_FIELD, "")).strip()
        if rid:
            by_id[rid].append(r)

    removed = 0
    for bid, lst in by_id.items():
        if len(lst) <= 1:
            continue
        keep = max(lst, key=lambda x: str(x.get(TS_FIELD, "")))
        for r in lst:
            if r is keep:
                continue
            # Your wrapper should let you delete by row index; adjust key name.
            idx = int(r.get("_row", 0)) or int(r.get("_row_number", 0))
            if idx:
                delete_row_by_index(TAB, idx)
                removed += 1

    if removed:
        notify(f"🧹 Dedupe removed *{removed}* rows in `{TAB}`")
    else:
        print("[dedupe] no duplicates found")

if __name__ == "__main__":
    main()
