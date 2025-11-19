import requests
from typing import Dict, List, Optional, Tuple
from config.settings import Settings

BASE = "https://api.the-odds-api.com/v4"

def _get(path: str, params: Dict) -> any:
    key = Settings.ODDS_API_KEY
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY")
    params = dict(params)
    params["apiKey"] = key
    r = requests.get(f"{BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def list_events(sport_key: str, hours: int = 12) -> List[dict]:
    """Upcoming events within hours for given sport."""
    return _get(
        f"/sports/{sport_key}/events",
        {"daysFrom": max(1, hours // 24), "regions": Settings.ODDS_API_REGION},
    )

def event_odds(
    sport_key: str,
    event_id: str,
    markets: List[str] = ["h2h", "spreads", "totals"],
    bookmakers: Optional[List[str]] = None,
) -> dict:
    """Get odds for a specific event_id."""
    params = {
        "regions": Settings.ODDS_API_REGION,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    return _get(f"/sports/{sport_key}/events/{event_id}/odds", params)

def american_to_decimal(american_odds: float) -> float:
    o = float(american_odds)
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))

def _normalize_team(s: str) -> str:
    return s.lower().strip()

def best_quote_h2h(
    event_data: dict,
    side: str,  # "home" or "away"
    preferred: Optional[List[str]] = None,
    blocked: Optional[List[str]] = None,
) -> Optional[Tuple[str, float]]:
    """
    Return (bookmaker_name, american_odds) with the BEST price for the chosen side.
    The Odds API uses team names in H2H outcomes; we resolve your side to the event's team name.
    """
    preferred = [b.lower() for b in (preferred or [])]
    blocked = [b.lower() for b in (blocked or [])]

    home_team = event_data.get("home_team") or ""
    away_team = event_data.get("away_team") or ""
    target_team = home_team if side.lower() == "home" else away_team
    target_norm = _normalize_team(target_team)

    offers = []
    for bm in event_data.get("bookmakers", []):
        name = bm["title"]
        lname = name.lower()
        if lname in blocked:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                # Outcome name is the TEAM NAME (e.g., "New York Yankees")
                if _normalize_team(outcome.get("name","")) == target_norm:
                    price = outcome["price"]
                    offers.append((name, price, american_to_decimal(price)))

    if not offers:
        return None

    preferred_offers = [o for o in offers if o[0].lower() in preferred] if preferred else []
    pool = preferred_offers or offers
    pool.sort(key=lambda x: x[2], reverse=True)  # highest decimal odds
    best = pool[0]
    return best[0], best[1]
