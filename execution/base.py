from dataclasses import dataclass
from typing import Optional

@dataclass
class BetOrder:
    sport_key: str
    event_id: str
    market: str  # 'h2h'
    side: str    # 'home' or 'away'
    bookmaker: str
    american_odds: float
    stake: float
    reason: str  # short

class Executor:
    """Interface for execution adapters."""
    def execute(self, order: BetOrder) -> str:
        """Return an execution id or path (eg CSV row id)."""
        raise NotImplementedError
