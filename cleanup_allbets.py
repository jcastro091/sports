# cleanup_allbets.py

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ─── CONFIG ───────────────────────────────────────────────────────────
KEYFILE = "telegrambetlogger-35856685bc29.json"
SPREADSHEET_NAME = "ConfirmedBets"
WORKSHEET_NAME = "AllBets"
COLS = 17  # width of your table (A→Q)

# ─── AUTH & OPEN ───────────────────────────────────────────────────────
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name(KEYFILE, scope)
client = gspread.authorize(creds)
sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)

# ─── PULL EVERYTHING ────────────────────────────────────────────────────
all_vals = sheet.get_all_values()

if not all_vals:
    print("Sheet is empty, nothing to do.")
    exit()

# header is first row, only columns A→Q
header = all_vals[0][:COLS]

# collect cleaned rows here
cleaned = []

for row in all_vals[1:]:
    # 1) take the first block of length COLS
    block1 = row[:COLS]
    if any(cell.strip() for cell in block1):
        cleaned.append(block1)

    # 2) look for a second block further to the right
    #    (skip the first COLS cells)
    tail = row[COLS:]
    # find the first non-empty cell in tail
    for idx, cell in enumerate(tail):
        if cell.strip():
            start = COLS + idx
            block2 = row[start:start+COLS]
            if any(c.strip() for c in block2):
                cleaned.append(block2)
            break

# ─── WIPE & REWRITE ────────────────────────────────────────────────────
print(f"Found {len(cleaned)} total data-rows (plus header). Rewriting sheet…")
sheet.clear()

# write header + all rows in one call
sheet.update(
    [header] + cleaned,
    value_input_option='USER_ENTERED'
)

print("✅ Done! Everything is back in A→Q where it belongs.")
