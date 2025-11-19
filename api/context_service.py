"""
SharpsSignal Context Service (FastAPI)
-------------------------------------
Exposes endpoints to fetch context JSON/text (Free/Pro) and LLM-polished summaries.

Run locally:
    pip install fastapi uvicorn pydantic requests python-dotenv openai
    set ODDS_API_KEY=your_theodds_key        # Windows (use `export` on macOS/Linux)
    set OPENAI_API_KEY=sk-...                # only needed for /llm/* endpoints
    uvicorn api.context_service:app --reload --port 8099

Then test:
    curl -X POST http://localhost:8099/context/text -H 'Content-Type: application/json' -d '{
      "event_id":"123456",
      "sport":"baseball_mlb",
      "league":"MLB",
      "teams":{"home":"BOS","away":"NYY"},
      "start_time_utc":"2025-09-21T23:05:00Z",
      "pro": false
    }'

Folder expectations (inside sports-repo):
    src/context/context_builder.py
    src/context/adapters/*.py
    api/context_service.py   ← this file

You can deploy this as a service on a VPS and point your Next.js app to it.
"""
# api/context_service.py
from __future__ import annotations
from typing import Dict, Optional
import os, json, traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# replace your existing provider singletons with this:
from src.context.context_builder import (
    ContextBuilder, ProviderConfig, render_llm_prompt, NoopClient
)
from src.context.adapters import (
    TheOddsClient, OpenMeteoClient, MLBStatsInjuriesClient, BasicFormClient
)

# Optional OpenAI deps (some endpoints check this flag)
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False
    
    
import os

# ---- Provider singletons (only once) ----
USE_REAL_ODDS = bool(os.getenv("ODDS_API_KEY"))
odds = TheOddsClient(sport_key="baseball_mlb") if USE_REAL_ODDS else NoopClient()
weather = OpenMeteoClient()
injuries = MLBStatsInjuriesClient()
stats = BasicFormClient()

builder = ContextBuilder(ProviderConfig(
    odds_client=odds,
    matchup_client=stats,    # stub until real H2H client is wired
    injuries_client=injuries,
    weather_client=weather,
    stats_client=stats,
))

# -------------------------
# App & CORS
# -------------------------
# ---- FastAPI app ----
app = FastAPI(title="SharpsSignal Context Service", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://www.sharps-signal.com","https://sharps-signal.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- RAG router (NOW that app exists) ----
from api.rag_service import router as rag_router
app.include_router(rag_router)
# -------------------------
# Provider singletons
# -------------------------
# You can parameterize sport_key per-request later; for now MLB default
# Use real Odds API only if we have a key; else fall back to NoopClient
if os.getenv("ODDS_API_KEY"):
    odds = TheOddsClient(sport_key="baseball_mlb")
else:
    odds = NoopClient()

weather = OpenMeteoClient()
injuries = MLBStatsInjuriesClient()
stats = BasicFormClient()

builder = ContextBuilder(ProviderConfig(
    odds_client=odds,
    matchup_client=stats,   # stub for now
    injuries_client=injuries,
    weather_client=weather,
    stats_client=stats,
))

# -------------------------
# Request model
# -------------------------
class ContextRequest(BaseModel):
    event_id: str
    sport: str            # e.g., "baseball_mlb"
    league: str           # e.g., "MLB"
    teams: Dict[str, str] # {"home":"BOS","away":"NYY"}
    start_time_utc: str   # ISO 8601 (UTC)
    pro: bool = False
    # Optional: venue coordinates for weather (until we wire a venue DB)
    lat: Optional[float] = None
    lon: Optional[float] = None
    # Optional: your model explanation block for Pro
    model: Optional[Dict] = None

# -------------------------
# Health
# -------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# -------------------------
# Raw context endpoints
# -------------------------
@app.post("/context/json")
def context_json(req: ContextRequest):
    ctx = builder.build_context(
        event_id=req.event_id,
        sport=req.sport,
        league=req.league,
        teams=req.teams,
        start_time_utc=req.start_time_utc,
    )
    if req.pro and req.model:
        ctx["model"] = req.model
    return ctx if req.pro else builder.redact_for_free(ctx)

from fastapi import HTTPException
import traceback

@app.post("/context/text")
def context_text(req: ContextRequest):
    try:
        ctx = builder.build_context(
            event_id=req.event_id,
            sport=req.sport,
            league=req.league,
            teams=req.teams,
            start_time_utc=req.start_time_utc,
        )
        if req.pro and req.model:
            ctx["model"] = req.model
        payload = ctx if req.pro else builder.redact_for_free(ctx)
        return {"text": render_llm_prompt(payload, pro=req.pro)}
    except Exception as e:
        print("[context_text] Exception:\n", traceback.format_exc())  # shows full stack in server window
        raise HTTPException(status_code=400, detail=f"context_error: {type(e).__name__}: {e}")
        
        
from fastapi import HTTPException
import traceback

@app.post("/reason")
def reason(req: ContextRequest):
    """
    Rule-based, model-aware reason. NO OpenAI key needed.
    Returns: {"text": "..."} just like /context/text.
    """
    try:
        ctx = builder.build_context(
            event_id=req.event_id,
            sport=req.sport,
            league=req.league,
            teams=req.teams,
            start_time_utc=req.start_time_utc,
        )
        if req.model:
            ctx["model"] = req.model

        lm   = (ctx.get("market") or {}).get("line_movement") or {}
        pvs  = (ctx.get("market") or {}).get("public_vs_sharp") or {}
        lims = (ctx.get("market") or {}).get("limits") or {}

        # Movement summary
        dir_map = {"toward_home":"toward home","toward_away":"toward away","neutral":"flat"}
        direction = dir_map.get(lm.get("direction") or "neutral")
        delta = lm.get("open_to_now_cents")
        steam = lm.get("steam_moves")
        rev   = lm.get("reversals")
        vol   = lm.get("volatility")
        mv_bits = []
        if isinstance(delta,(int,float)): mv_bits.append(f"Δ {int(delta)}¢")
        if isinstance(steam,int) and steam>0: mv_bits.append(f"steam {steam}x")
        if isinstance(rev,int) and rev>0: mv_bits.append(f"reversals {rev}x")
        mv = f"Movement {direction}" + (f" ({', '.join(mv_bits)})" if mv_bits else "") + (f". Volatility {vol}." if vol else ".")

        # Limits
        lim_line = ""
        if isinstance(lims, dict) and (lims.get("open_limit") or lims.get("current_limit")):
            ol = lims.get("open_limit"); cl = lims.get("current_limit")
            if isinstance(ol,(int,float)) and isinstance(cl,(int,float)):
                trend = "rising" if cl>ol else ("falling" if cl<ol else "steady")
                lim_line = f" Limits {trend} into kickoff."
            else:
                lim_line = " Limits available."
        else:
            lim_line = " Limits n/a."

        # Splits
        sp = ""
        t = pvs.get("tickets_pct"); m = pvs.get("money_pct")
        if isinstance(t,(int,float)) and isinstance(m,(int,float)):
            gap = int(round(m - t))
            sp = f" Splits: tickets {int(t)}%, money {int(m)}% (gap {gap}pp)" + (" (RLM)." if pvs.get("rlm_flag") else ".")

        # Model (optional)
        mdl = ""
        if req.model:
            pick = req.model.get("pick"); market = req.model.get("market")
            conf = req.model.get("confidence"); reasons = req.model.get("reasons") or []
            head = f" Model favors {pick} ({market})"
            if conf: head += f" at {conf} confidence"
            head += "."
            tail = f" Reasons: {', '.join(reasons[:3])}." if reasons else ""
            mdl = " " + head + (" " + tail if tail else "")

        return {"text": (mv + lim_line + sp + mdl).strip()}
    except Exception as e:
        print("[/reason] Exception:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"reason_error: {type(e).__name__}: {e}")
    
        
        
        
@app.post("/echo")
def echo(req: dict):
    return {"ok": True, "you_sent": req}


# -------------------------
# LLM-polished endpoints (optional)
# -------------------------
SYSTEM_FREE = (
    "You are SharpsSignal Free Assistant. Write a concise, neutral pregame primer. "
    "Do NOT recommend bets. Explain factors (matchup, injuries/lineups, weather, "
    "consensus direction, volatility, recent form). Avoid book-level edges or limits."
)

SYSTEM_PRO = (
    "You are SharpsSignal Pro Assistant. Write a confident pregame report with model "
    "reasoning and market context (steam/reversals). Include pick, market, confidence, "
    "and 3–5 crisp reasons tied to data. Close with a short risk note."
)

@app.post("/llm/free")
def llm_free(req: ContextRequest):
    if not _OPENAI_OK:
        return {"error": "openai SDK not installed"}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    ctx = builder.build_context(
        event_id=req.event_id,
        sport=req.sport,
        league=req.league,
        teams=req.teams,
        start_time_utc=req.start_time_utc,
    )
    free_ctx = builder.redact_for_free(ctx)

    client = OpenAI(api_key=api_key)
    content = "Context JSON:\n" + json.dumps(free_ctx, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_FREE},
            {"role": "user", "content": content},
        ],
    )
    return {"text": resp.choices[0].message.content.strip()}

@app.post("/llm/pro")
def llm_pro(req: ContextRequest):
    if not _OPENAI_OK:
        return {"error": "openai SDK not installed"}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    ctx = builder.build_context(
        event_id=req.event_id,
        sport=req.sport,
        league=req.league,
        teams=req.teams,
        start_time_utc=req.start_time_utc,
    )
    if req.model:
        ctx["model"] = req.model

    client = OpenAI(api_key=api_key)
    content = "Context JSON:\n" + json.dumps(ctx, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PRO},
            {"role": "user", "content": content},
        ],
    )
    return {"text": resp.choices[0].message.content.strip()}
