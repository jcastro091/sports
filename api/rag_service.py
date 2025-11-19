# api/rag_service.py
from fastapi import APIRouter, HTTPException
from typing import Optional
import os

# Only import psycopg2 inside the handler (avoid module-import failures)
def _get_db_conn():
    import psycopg2
    url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL (or DATABASE_URL) is not set")
    # Supabase Postgres usually requires SSL
    return psycopg2.connect(url, sslmode="require")

def _embed_1536(text: str) -> list[float]:
    # TODO: replace with your real embedding call (OpenAI, etc.)
    # Placeholder zero-vector avoids import errors while wiring things up.
    return [0.0] * 1536

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/search")
def search_rag(
    q: str,
    sport: Optional[str] = None,
    market: Optional[str] = None,
    team_home: Optional[str] = None,
    team_away: Optional[str] = None,
    since: Optional[str] = None,
    k: int = 8,
):
    try:
        v = _embed_1536(q)
        filters, args = [], []
        if sport:     filters.append("sport = %s");      args.append(sport)
        if market:    filters.append("market = %s");     args.append(market)
        if team_home: filters.append("team_home ILIKE %s"); args.append(team_home)
        if team_away: filters.append("team_away ILIKE %s"); args.append(team_away)
        if since:     filters.append("created_at >= %s");   args.append(since)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        sql = f"""
            SELECT id, source_id, doc_id, text, metadata, sport, market, team_home, team_away, game_date,
                   1 - (embedding <=> %s::vector) AS cos_sim
            FROM rag_chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        with _get_db_conn() as conn:
            import psycopg2.extras as pe
            with conn.cursor(cursor_factory=pe.RealDictCursor) as cur:
                cur.execute(sql, [v] + args + [v, k])
                rows = cur.fetchall()
        return {"results": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag_search_error: {type(e).__name__}: {e}")
