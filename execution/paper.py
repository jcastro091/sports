import csv, os, time
from typing import Optional
from .base import BetOrder, Executor

class PaperExecutor(Executor):
    def __init__(self, path: str = "data/executions.csv"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","sport_key","event_id","market","side","bookmaker","american_odds","stake","reason"])

    def execute(self, order: BetOrder) -> str:
        row_id = str(int(time.time()*1000))
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row_id, order.sport_key, order.event_id, order.market, order.side,
                        order.bookmaker, order.american_odds, order.stake, order.reason])
        return row_id
