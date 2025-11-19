from __future__ import annotations
from typing import Any, Dict, Optional
import datetime as dt
import requests

OM_BASE = "https://api.open-meteo.com/v1/forecast"

class OpenMeteoClient:
    def __init__(self):
        self.session = requests.Session()

    def get_weather(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        kickoff_iso: Optional[str] = None,
        **_
    ) -> Optional[Dict[str, Any]]:
        if lat is None or lon is None or not kickoff_iso:
            return None

        kickoff = dt.datetime.fromisoformat(
            kickoff_iso.replace("Z", "+00:00")
        ).replace(tzinfo=dt.timezone.utc)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC",
        }
        data = self._get(OM_BASE, params)
        hours = data.get("hourly", {})
        times = hours.get("time", [])

        if not times:
            return None

        # Find nearest hour to kickoff
        idx = min(
            range(len(times)),
            key=lambda i: abs(
                (dt.datetime.fromisoformat(times[i].replace("Z", "+00:00")).replace(tzinfo=dt.timezone.utc) - kickoff
                ).total_seconds()
            ),
        )

        def _get(arr, i):
            try:
                return arr[i]
            except Exception:
                return None

        return {
            "temp_f": _c_to_f(_get(hours.get("temperature_2m", []), idx)),
            "wind_mph": _ms_to_mph(_get(hours.get("wind_speed_10m", []), idx)),
            "wind_dir": _get(hours.get("wind_direction_10m", []), idx),
            "precip_prob": _get(hours.get("precipitation_probability", []), idx),
        }

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

def _c_to_f(c):
    return round((c * 9 / 5) + 32) if c is not None else None

def _ms_to_mph(ms):
    return round(ms * 2.23694) if ms is not None else None
