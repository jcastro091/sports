from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests

MLB_BASE = "https://statsapi.mlb.com/api/v1"

class MLBStatsInjuriesClient:
    def __init__(self):
        self.session = requests.Session()

    def get_injuries(self, event_id: str) -> List[Dict[str, Any]]:
        """Free Stats API doesn’t expose structured daily injury reports.
        Start with empty list or extend with team transaction endpoints.
        """
        return []

    def get_projected_lineup(self, event_id: str) -> Optional[Dict[str, Any]]:
        """If event_id is a numeric MLB gamePk, hydrate probable pitchers."""
        try:
            game_pk = int(event_id)
        except Exception:
            return None

        url = f"{MLB_BASE}/game/{game_pk}/boxscore"
        try:
            data = self._get(url)
        except Exception:
            return None

        def _pitchers(side: str):
            t = data.get("teams", {}).get(side, {})
            out = []
            for _, info in (t.get("players", {}) or {}).items():
                pos = (info.get("position") or {}).get("abbreviation")
                if pos == "P":
                    p = info.get("person", {})
                    if p:
                        out.append(p.get("fullName"))
            return out

        return {"home": _pitchers("home"), "away": _pitchers("away")}

    def _get(self, url: str) -> Dict[str, Any]:
        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
