from typing import Optional
from providers.odds_api import event_odds, best_quote_h2h, american_to_decimal
from personalization.profile import UserProfile
from execution.base import BetOrder
from compliance.checks import basic_time_check

class QuoteAgent:
    def __init__(self, user: Optional[UserProfile] = None, min_minutes_to_start: int = 5):
        self.user = user or UserProfile()
        self.min_minutes_to_start = min_minutes_to_start

    def best_quote_and_stake(
        self,
        sport_key: str,
        event_id: str,
        side: str,
        win_prob: float,
        bookmakers: Optional[list] = None,
    ):
        data = event_odds(sport_key, event_id, markets=["h2h"], bookmakers=bookmakers)
        if not basic_time_check(data.get("commence_time", ""), self.min_minutes_to_start):
            raise RuntimeError(f"Event is too close to start time (<= {self.min_minutes_to_start}m). Skipping.")
        best = best_quote_h2h(
            data,
            side=side,
            preferred=self.user.preferred_books,
            blocked=self.user.blocked_books,
        )
        if not best:
            raise RuntimeError("No H2H quotes found for requested side.")
        bm, american = best
        stake = self.user.stake_for(win_prob, american)
        reason = f"Kelly-scaled stake for p={win_prob:.2f}; best price across allowed books."
        order = BetOrder(
            sport_key=sport_key,
            event_id=event_id,
            market="h2h",
            side=side,
            bookmaker=bm,
            american_odds=american,
            stake=stake,
            reason=reason,
        )
        return order, data
