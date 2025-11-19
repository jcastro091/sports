import asyncpg, asyncio, os
os.environ["DATABASE_URL"] = "postgresql://postgres:Yankees091!@localhost:5432/postgres"

async def test():
    try:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        print("✅ Connected to Postgres successfully!")
        await conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

asyncio.run(test())
