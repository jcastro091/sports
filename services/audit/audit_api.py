import os, json
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from fastapi import FastAPI, Query
from pydantic import BaseModel
import asyncpg

app = FastAPI(title="Audit Log API", version="0.1.0")
DATABASE_URL = os.getenv("DATABASE_URL")

class AuditLog(BaseModel):
    user: str
    model: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    decision: str
    status: str
    meta: Optional[Dict[str, Any]] = None

async def _conn():
    return await asyncpg.connect(DATABASE_URL)

import traceback

@app.post("/api/audit_log")
async def add_log(entry: AuditLog):
    q = """INSERT INTO audit_log
           (ts_utc, user_id, model, input, output, decision, status, meta)
           VALUES($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7,$8::jsonb)
           RETURNING id"""
    conn = await _conn()
    try:
        rid = await conn.fetchval(
            q,
            datetime.now(timezone.utc),
            entry.user, entry.model,
            json.dumps(entry.input),
            json.dumps(entry.output),
            entry.decision, entry.status,
            json.dumps(entry.meta or {})
        )
        return {"ok": True, "id": rid}
    except Exception as e:
        print("❌ INSERT FAILED:", e)
        traceback.print_exc()
        # optional: return a clearer error to the client
        return {"ok": False, "error": str(e)}
    finally:
        await conn.close()


@app.get("/api/audit_log")
async def list_logs(user: Optional[str] = None,
                    decision: Optional[str] = None,
                    since: Optional[str] = Query(None, description="ISO8601"),
                    limit: int = 200):
    q = "SELECT id, ts_utc, user_id, model, input, output, decision, status, meta FROM audit_log WHERE 1=1"
    params: List[Any] = []
    if user:
        q += f" AND user_id = ${len(params)+1}"; params.append(user)
    if decision:
        q += f" AND decision = ${len(params)+1}"; params.append(decision)
    if since:
        q += f" AND ts_utc >= ${len(params)+1}"; params.append(datetime.fromisoformat(since))
    q += f" ORDER BY id DESC LIMIT ${len(params)+1}"; params.append(limit)

    conn = await _conn()
    try:
        rows = await conn.fetch(q, *params)
        return [dict(r) for r in rows]
    finally:
        await conn.close()
