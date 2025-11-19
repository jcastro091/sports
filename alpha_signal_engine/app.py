# app.py
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import os, time, csv

API_KEY = os.getenv("ML_API_KEY", "")  # set this
CSV_PATH = os.getenv("PRED_CSV", "data/results/models/predictions_latest.csv")
BET_COL_CANDIDATES = ["bet_id", "BetID", "id"]
PROB_COL_CANDIDATES = ["prob", "proba", "Pred_Prob", "ModelProb", "pred_proba"]

app = FastAPI(title="Alpha Signal ML API", version="1.0.0")

_cache = {"mtime": 0, "probs": {}}  # { bet_id: float }

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = urllib.parse.unquote(str(s)).strip()
    # normalize unicode and collapse all dash-like chars to ASCII '-'
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u2010-\u2015\u2212]+", "-", s)  # hyphen, en/em dashes, minus
    return s

def _load_probs() -> Dict[str, float]:
    if not os.path.exists(CSV_PATH):
        return {}
    mtime = os.path.getmtime(CSV_PATH)
    if mtime == _cache["mtime"]:
        return _cache["probs"]

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        probs = {}
        for row in reader:
            bet_key = next((k for k in BET_COL_CANDIDATES if k in row and row[k]), None)
            p_key   = next((k for k in PROB_COL_CANDIDATES if k in row and row[k] not in ("", None)), None)
            if not bet_key or not p_key:
                continue
            bid = str(row[bet_key]).strip()
            try:
                p = float(row[p_key])
                probs[bid] = p/100 if p > 1.0 else p
            except ValueError:
                continue
    _cache["mtime"] = mtime
    _cache["probs"] = probs
    return probs

@app.get("/api/health")
def health():
    return {"ok": True, "ts": time.time()}

@app.get("/api/probs")
def get_probs(bet_ids: str, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    ids = [s.strip() for s in bet_ids.split(",") if s.strip()]
    probs = _load_probs()
    return JSONResponse({"probs": {i: probs.get(i) for i in ids if i in probs}})
