# execution/resolver/kalshi_resolver.py
from __future__ import annotations
import requests

class KalshiResolver:
    def __init__(self, session: requests.Session, base_url: str):
        self.session = session
        self.base = base_url.rstrip("/")

    def get_market(self, ticker: str) -> dict:
        r = self.session.get(f"{self.base}/markets/{ticker}", timeout=10)
        if r.status_code == 404:
            raise FileNotFoundError(f"Market not found for ticker '{ticker}'")
        r.raise_for_status()
        return r.json()

    def search_markets(self, q: str, limit: int = 10) -> list[dict]:
        # Depending on API, this could be ?search= or ?ticker= prefix; keep it simple:
        r = self.session.get(f"{self.base}/markets", params={"search": q, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("markets", [])
