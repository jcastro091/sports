from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, time, requests

BASE = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("ODDS_API_KEY", "")

class TheOddsClient:
    def __init__(self, sport_key: str, market: str = "h2h", regions: str = "us", odds_format: str = "american"):
        self.sport_key = sport_key
        self.market = market
        self.regions = regions
        self.odds_format = odds_format
        self.session = requests.Session()

    def get_consensus(self, event_id: str) -> Dict[str, Any]:
        url = f"{BASE}/sports/{self.sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": API_KEY,
            "regions": self.regions,
            "markets": ",".join(["h2h", "spreads", "totals"]),
            "oddsFormat": self.odds_format,
        }
        data = self._get(url, params)
        prices_h2h, prices_spread, prices_total = [], [], []
        for bk in data.get("bookmakers", []):
            for m in bk.get("markets", []):
                key = m.get("key")
                for o in m.get("outcomes", []):
                    if key == "h2h" and o.get("price") is not None:
                        prices_h2h.append(o["price"])
                    if key == "spreads" and o.get("price") is not None:
                        prices_spread.append(o["price"])
                    if key == "totals" and o.get("price") is not None:
                        prices_total.append(o["price"])
        def _median(xs):
            xs = sorted(x for x in xs if isinstance(x, (int, float)))
            if not xs: return None
            n = len(xs)
            return xs[n//2] if n % 2 == 1 else (xs[n//2-1] + xs[n//2]) / 2
        market_summary = {
            "h2h_median_price": _median(prices_h2h),
            "spread_median_price": _median(prices_spread),
            "total_median_price": _median(prices_total),
        }
        return {"market_summary": market_summary, "direction": None, "volatility": "low"}

    def get_line_history(self, event_id: str) -> List[Dict[str, Any]]:
        snapshots = []
        now = int(time.time())
        for offset in (0, 2*3600, 6*3600):
            ts = now - offset
            url = f"{BASE}/sports/{self.sport_key}/events/{event_id}/odds"
            params = {"apiKey": API_KEY, "regions": self.regions, "markets": "h2h", "oddsFormat": self.odds_format}
            try:
                data = self._get(url, params)
            except Exception:
                continue
            prices = []
            for bk in data.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m.get("key") == "h2h":
                        for o in m.get("outcomes", []):
                            if isinstance(o.get("price"), (int, float)):
                                prices.append(o["price"])
            if prices:
                mean_price = sum(prices) / len(prices)
                snapshots.append({"ts": ts, "book": "consensus", "market": "h2h", "price": mean_price})
        return sorted(snapshots, key=lambda r: r.get("ts", 0))

    def get_limits(self, event_id: str) -> Optional[Dict[str, Any]]:
        return None

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not API_KEY:
            raise RuntimeError("ODDS_API_KEY is not set")
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
