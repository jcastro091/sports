from dataclasses import dataclass
from typing import List, Optional
from config.settings import Settings
from providers.odds_api import american_to_decimal
from .kelly import kelly_fraction

@dataclass
class UserProfile:
    bankroll: float = Settings.DEFAULT_BANKROLL
    kelly_scale: float = Settings.DEFAULT_KELLY_FRACTION
    max_stake_pct: float = Settings.MAX_STAKE_PCT
    preferred_books: List[str] = None
    blocked_books: List[str] = None

    def __post_init__(self):
        self.preferred_books = self.preferred_books or Settings.PREFERRED_BOOKS
        self.blocked_books = self.blocked_books or Settings.BLOCKED_BOOKS

    def stake_for(self, win_prob: float, american_odds: float) -> float:
        dec = american_to_decimal(american_odds)
        k_full = kelly_fraction(win_prob, dec)
        k_scaled = k_full * self.kelly_scale
        stake = self.bankroll * min(self.max_stake_pct, max(0.0, k_scaled))
        return round(stake, 2)
