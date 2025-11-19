"""
Context Builder for SharpsSignal
--------------------------------

Goal
====
A single importable module that gathers rich, bettor-ready context per event
(matchup history, injuries/lineups, weather, line movement/consensus, recent form,
market/limits), then returns a normalized JSON payload that your LLM (free or
paid) can use to generate explanations.

Design
======
- Keep `sports21.py` lean. This module exposes a small API and swappable fetch
  adapters so you can integrate any provider (The Odds API, Pinnacle, LowVig,
  Sportradar, StatsBomb, Rotowire, FantasyPros, Weather APIs, etc.).
- Includes: retry/backoff, timeouts, and basic in-memory caching to reduce
  rate-limit issues.
- Includes a `redact_for_free` function to convert full context → free payload.
- Pure-Python, no deps required; you can inject real HTTP clients at runtime.

Usage
=====
from context_builder import ContextBuilder, ProviderConfig

cfg = ProviderConfig(
    odds_client=your_odds_client,          # must implement .get_consensus(), .get_line_history(), .get_limits()
    matchup_client=your_matchup_client,    # must implement .get_h2h(teamA, teamB, lookback_games)
    injuries_client=your_injuries_client,  # must implement .get_injuries(event_id) and .get_projected_lineup(event_id)
    weather_client=your_weather_client,    # must implement .get_weather(event_id) or (lat, lon, time)
    stats_client=your_stats_client,        # must implement .get_form(team_id, n), .get_advanced(team_id)
)

builder = ContextBuilder(cfg)
full_ctx = builder.build_context(event_id="abc123", sport="baseball_mlb", league="MLB", teams={"home":"BOS","away":"NYY"}, start_time_utc="2025-09-21T23:05:00Z")
free_ctx = builder.redact_for_free(full_ctx)

# Feed full_ctx (Pro) or free_ctx (Free) into your LLM prompt/template.

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import math

# -------------------------
# Provider Contracts
# -------------------------

class OddsClient:
    """Contract for odds/line data providers.
    Implement with The Odds API + book-specific lookups (Pinnacle, etc.).
    """
    def get_consensus(self, event_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_line_history(self, event_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_limits(self, event_id: str) -> Optional[Dict[str, Any]]:
        return None

class MatchupClient:
    def get_h2h(self, team_a: str, team_b: str, lookback_games: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

class InjuriesClient:
    def get_injuries(self, event_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_projected_lineup(self, event_id: str) -> Optional[Dict[str, Any]]:
        return None

class WeatherClient:
    def get_weather(self, *, event_id: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None, 
                    kickoff_iso: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return None

class StatsClient:
    def get_form(self, team: str, n: int = 5) -> Dict[str, Any]:
        raise NotImplementedError

    def get_advanced(self, team: str) -> Dict[str, Any]:
        return {}

# -------------------------
# Config & Simple Cache
# -------------------------

@dataclass
class ProviderConfig:
    odds_client: OddsClient
    matchup_client: MatchupClient
    injuries_client: InjuriesClient
    weather_client: WeatherClient
    stats_client: StatsClient
    lookback_games: int = 10
    recent_n: int = 5
    enable_limits: bool = True

class _Cache:
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        v = self._store.get(key)
        if not v:
            return None
        ts, data = v
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return data

    def set(self, key: str, value: Any):
        self._store[key] = (time.time(), value)

# -------------------------
# Context Builder
# -------------------------

class ContextBuilder:
    def __init__(self, config: ProviderConfig, cache_ttl: int = 30):
        self.cfg = config
        self.cache = _Cache(ttl_seconds=cache_ttl)

    # Public API
    def build_context(self, *, event_id: str, sport: str, league: str, teams: Dict[str, str], start_time_utc: str) -> Dict[str, Any]:
        """Gather everything needed for a rich, single-event report.
        Returns a normalized dict that both Free and Pro prompts can consume.
        """
        key = f"ctx:{event_id}"
        cached = self.cache.get(key)
        if cached:
            return cached

        home = teams.get("home")
        away = teams.get("away")

        # Retry wrapper to be resilient to transient 429/5xx
        def _retry(fn, *args, tries=3, base=0.4, **kwargs):
            for i in range(tries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # replace with narrower exceptions in real impl
                    if i == tries - 1:
                        raise
                    time.sleep(base * (2 ** i))

        consensus = _retry(self.cfg.odds_client.get_consensus, event_id)
        line_history = _retry(self.cfg.odds_client.get_line_history, event_id)
        limits = _retry(self.cfg.odds_client.get_limits, event_id) if self.cfg.enable_limits else None

        h2h = _retry(self.cfg.matchup_client.get_h2h, away, home, self.cfg.lookback_games)
        injuries = _retry(self.cfg.injuries_client.get_injuries, event_id)
        lineup = _retry(self.cfg.injuries_client.get_projected_lineup, event_id)
        weather = _retry(self.cfg.weather_client.get_weather, event_id=event_id, kickoff_iso=start_time_utc)
        form = {
            "home": _retry(self.cfg.stats_client.get_form, home, self.cfg.recent_n),
            "away": _retry(self.cfg.stats_client.get_form, away, self.cfg.recent_n),
        }
        advanced = {
            "home": _retry(self.cfg.stats_client.get_advanced, home),
            "away": _retry(self.cfg.stats_client.get_advanced, away),
        }

        # Compute derived insights
        lm = _analyze_line_movement(line_history)
        public_vs_sharp = _estimate_public_vs_sharp(consensus)

        ctx = {
            "event": {
                "id": event_id,
                "sport": sport,
                "league": league,
                "teams": teams,
                "start_time_utc": start_time_utc,
            },
            "matchup": {
                "h2h_recent": h2h,  # list of {date, home, away, score_home, score_away, ats_winner?, total_over?}
            },
            "availability": {
                "injuries": injuries,  # list of {player, team, status, note}
                "projected_lineup": lineup,  # {home:[players], away:[players]}
            },
            "environment": {
                "weather": weather,  # {temp_f, wind_mph, wind_dir, precip_prob, narrative}
            },
            "market": {
                "consensus": consensus,  # snapshot of current market odds across books
                "line_history": line_history,  # [{ts, book, market, price, spread/total}]
                "line_movement": lm,  # derived metrics: open→now deltas, RLM flags
                "public_vs_sharp": public_vs_sharp,  # {tickets_pct, money_pct, rlm_flag}
                "limits": limits,  # optional {open_limit, current_limit}
            },
            "form": form,  # recent form (last N): records, ats, o/u, bullpen ERA, pace, etc.
            "advanced": advanced,  # team-level advanced metrics (pace, efficiency, etc.)
            # Slot for your model to attach its explanation before LLM render (Pro only)
            "model": None,  # e.g., {pick: ..., market: ..., edge_bps: ..., confidence: ..., reasons: [...]}
            "disclaimer": "Not financial advice. Bet responsibly.",
        }

        self.cache.set(key, ctx)
        return ctx

    def redact_for_free(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return a version safe for the Free LLM (no picks, no precise book edges).
        Keeps educational value, removes actionable arbitrage or best-price signals.
        """
        # Deep copy without import to keep deps minimal
        import copy
        free = copy.deepcopy(ctx)

        # Remove model picks/explanations
        free["model"] = None

        # Trim market data: keep consensus direction + movement summary, drop per-book best prices/limits
        market = free.get("market", {})
        market.pop("limits", None)

        # Collapse consensus to an aggregate view (hide book-level edges)
        consensus = market.get("consensus") or {}
        market["consensus"] = {
            "market_summary": consensus.get("market_summary"),  # e.g., median/avg line & total
            "direction": consensus.get("direction"),  # e.g., "favoring away spread", "total drifting up"
            "volatility": consensus.get("volatility"),  # e.g., low/med/high
        }

        # Keep line_movement derived metrics but drop book-by-book history
        market.pop("line_history", None)

        # Keep public vs sharp percentages but round/coarsen
        pvs = market.get("public_vs_sharp") or {}
        def _bucket(x):
            if x is None:
                return None
            # round to nearest 5% to avoid being too revealing
            return int(round(x / 5.0) * 5)
        market["public_vs_sharp"] = {
            "tickets_pct": _bucket(pvs.get("tickets_pct")),
            "money_pct": _bucket(pvs.get("money_pct")),
            "rlm_flag": pvs.get("rlm_flag"),
        }

        # Keep injuries, lineups, weather, form, advanced unchanged (these are helpful context)
        return free

# -------------------------
# Analytics Helpers
# -------------------------

def _analyze_line_movement(line_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce concise metrics from raw line history across books.
    Output example:
    {
      'open_to_now_cents': 12,
      'direction': 'toward_away',  # toward_home/toward_away/neutral
      'steam_moves': 2,
      'reversals': 1,
      'volatility': 'medium'
    }
    """
    if not line_history:
        return {"open_to_now_cents": 0, "direction": "neutral", "steam_moves": 0, "reversals": 0, "volatility": "low"}

    # Very rough heuristic on aggregated market price deltas
    sorted_by_time = sorted(line_history, key=lambda r: r.get("ts", 0))
    open_price = sorted_by_time[0].get("price")
    last_price = sorted_by_time[-1].get("price")

    open_to_now = None
    if isinstance(open_price, (int, float)) and isinstance(last_price, (int, float)):
        open_to_now = int(round((last_price - open_price)))

    # Count steam moves as sharp jumps >= 8 cents within short time windows (placeholder)
    steam_moves = 0
    reversals = 0
    prev = None
    for row in sorted_by_time:
        price = row.get("price")
        if prev is not None and isinstance(price, (int, float)) and isinstance(prev, (int, float)):
            delta = price - prev
            if abs(delta) >= 8:
                steam_moves += 1
            # reversal when direction flips beyond small threshold
            if (prev - open_price) * (price - prev) < -16:  # crude sign flip heuristic
                reversals += 1
        prev = price

    direction = "neutral"
    if open_to_now is not None:
        if open_to_now > 0:
            direction = "toward_away"
        elif open_to_now < 0:
            direction = "toward_home"

    volatility = "low"
    if line_history and len(line_history) >= 6:
        volatility = "medium"
    if steam_moves >= 3:
        volatility = "high"

    return {
        "open_to_now_cents": open_to_now or 0,
        "direction": direction,
        "steam_moves": steam_moves,
        "reversals": reversals,
        "volatility": volatility,
    }


def _estimate_public_vs_sharp(consensus: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a simple Public vs Sharp view from available consensus/splits.
    Prefer real splits if provided by your data source; this function only
    harmonizes the fields and guards for missing data.
    """
    if not consensus:
        return {"tickets_pct": None, "money_pct": None, "rlm_flag": False}

    tickets = consensus.get("tickets_pct")  # % on away or home depending on your schema
    money = consensus.get("money_pct")

    rlm_flag = False
    if isinstance(tickets, (int, float)) and isinstance(money, (int, float)):
        # Reverse Line Movement flag: money >> tickets (gap >= 10pp)
        rlm_flag = (money - tickets) >= 10

    return {"tickets_pct": tickets, "money_pct": money, "rlm_flag": rlm_flag}

# -------------------------
# Example adapter stubs (optional)
# -------------------------

class NoopClient(OddsClient, MatchupClient, InjuriesClient, WeatherClient, StatsClient):
    """Minimal client providing empty responses. Useful for dry runs/tests."""
    def get_consensus(self, event_id: str) -> Dict[str, Any]:
        return {"market_summary": None, "direction": None, "volatility": None}

    def get_line_history(self, event_id: str) -> List[Dict[str, Any]]:
        return []

    def get_limits(self, event_id: str) -> Optional[Dict[str, Any]]:
        return None

    def get_h2h(self, team_a: str, team_b: str, lookback_games: int = 10) -> List[Dict[str, Any]]:
        return []

    def get_injuries(self, event_id: str) -> List[Dict[str, Any]]:
        return []

    def get_projected_lineup(self, event_id: str) -> Optional[Dict[str, Any]]:
        return None

    def get_weather(self, *, event_id: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None, kickoff_iso: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return None

    def get_form(self, team: str, n: int = 5) -> Dict[str, Any]:
        return {"last_n_record": None, "ats": None, "ou": None}

    def get_advanced(self, team: str) -> Dict[str, Any]:
        return {}

# -------------------------
# Pro/free rendering helper (optional)
# -------------------------

def render_llm_prompt(ctx: Dict[str, Any], *, pro: bool) -> str:
    """Returns a concise text prompt for the LLM combining context + (optional) model.
    You likely already have a better prompt, but this gives you a clean starting point.
    """
    ev = ctx["event"]
    teams = ev["teams"]
    parts = [
        f"Game: {teams['away']} @ {teams['home']} ({ev['league']})",
        f"Start: {ev['start_time_utc']}",
        "\nMatchup/Trends:",
        f"- Recent H2H (last {len(ctx['matchup']['h2h_recent'])}): summarize wins/ATS...",
        "\nAvailability:",
        f"- Injuries: {len(ctx['availability']['injuries'])} listed",
        f"- Projected lineups: {'yes' if ctx['availability']['projected_lineup'] else 'no'}",
        "\nEnvironment:",
        f"- Weather: {ctx['environment']['weather']}",
        "\nMarket:",
        f"- Direction: {ctx['market']['line_movement']['direction']}; Δ={ctx['market']['line_movement']['open_to_now_cents']}¢; steam={ctx['market']['line_movement']['steam_moves']}; rev={ctx['market']['line_movement']['reversals']}; vol={ctx['market']['line_movement']['volatility']}",
        f"- Public vs Money: tix={ctx['market']['public_vs_sharp']['tickets_pct']}%, $={ctx['market']['public_vs_sharp']['money_pct']}%, RLM={ctx['market']['public_vs_sharp']['rlm_flag']}",
    ]
    if pro and ctx.get("model"):
        parts += [
            "\nSharpsSignal Model:",
            f"- Pick: {ctx['model'].get('pick')} | Market: {ctx['model'].get('market')} | Confidence: {ctx['model'].get('confidence')}",
            f"- Reasons: {', '.join(ctx['model'].get('reasons', []))}",
        ]
    parts.append("\nDisclaimer: Not financial advice. Bet responsibly.")
    return "\n".join(parts)
