# execution/risk.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskConfig:
    max_order_usd: float = 150.0
    daily_exposure_usd: float = 1000.0
    min_price: float = 0.05
    max_price: float = 0.95

class RiskManager:
    """
    Minimal guardrails; expand with bankroll tracking if desired.
    """
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._today_exposure = 0.0

    def check(self, price: float, size: float):
        if size > self.cfg.max_order_usd:
            raise ValueError(f"Order size {size} exceeds max_order_usd {self.cfg.max_order_usd}")
        if not (self.cfg.min_price <= price <= self.cfg.max_price):
            raise ValueError(f"Price {price} outside [{self.cfg.min_price},{self.cfg.max_price}]")
        if self._today_exposure + size > self.cfg.daily_exposure_usd:
            raise ValueError("Daily exposure cap reached")

    def book(self, size: float):
        self._today_exposure += size
