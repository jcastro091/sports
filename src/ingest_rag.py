# ingest_rag.py
import os, json, math
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# 1) fetch source rows (from Sheets API or from your Supabase views)
# rows = [...]

def summarize(row):
  parts = [
    f"{row['sport']} | {row['away']} @ {row['home']} | {row['market']}",
    f"Model {row.get('proba',0):.0%}, Kelly {row.get('kelly',0):.2f}",
  ]
  if row.get('reason'):
    parts.append(f"Reason: {row['reason']}")
  return " • ".join(parts)

def embed_many(texts):
  # call your embedding model here (OpenAI text-embedding-3-small => 1536 dims)
  # return list[list[float]]
  ...

def upsert_chunks(chunks):
  conn = psycopg2.connect(os.getenv("SUPABASE_DB_URL"))  # or pgbouncer URL
  with conn, conn.cursor() as cur:
    sql = """
    insert into rag_chunks
      (source_id, doc_id, chunk_idx, text, metadata,
       sport, market, team_home, team_away, game_date, embedding)
    values %s
    on conflict (id) do nothing
    """
    execute_values(cur, sql, chunks)
  conn.close()

# Build payloads
texts, vals = [], []
for r in rows:
  t = summarize(r)
  texts.append(t)
embs = embed_many(texts)

for r, e, t in zip(rows, embs, texts):
  vals.append((
    r['source_id'], r['doc_id'], 0, t, json.dumps(r['metadata']),
    r['sport'], r['market'], r['home'], r['away'], r['game_date'], e
  ))

upsert_chunks(vals)
