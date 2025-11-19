import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

app = FastAPI(title="LLM Stream API", version="0.1.0")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def _gen(prompt: str):
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

@app.get("/api/stream")
async def stream(prompt: str):
    return StreamingResponse(_gen(prompt), media_type="text/plain")
