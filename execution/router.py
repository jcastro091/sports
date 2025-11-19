# execution/router.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

@dataclass
class Order:
    row_id: str           # sheet row key or unique id
    ticker: str           # Kalshi ticker, e.g., "CPI.Y25"
    side: str             # "YES" or "NO"
    price: float          # 0.01–0.99
    size: float           # USD size
    price_floor: float    # min acceptable price
    tif: str              # "IOC"|"FOK"|"GTD"
    notes: str = ""

@dataclass
class Fill:
    status: str           # "FILLED"|"PARTIAL"|"OPEN"|"REJECTED"|"CANCELED"
    avg_price: float
    filled_size: float
    venue_ref: str        # kalshi order id or reason
    ts_utc: str

class BrokerAdapter(Protocol):
    def place(self, o: Order) -> Fill: ...

class ExecutionRouter:
    """Thin router that enforces idempotency and basic parameter checks."""
    def __init__(self, adapter: BrokerAdapter):
        self.adapter = adapter
        self.seen: set[str] = set()

    def route(self, o: Order) -> Fill:
        if o.row_id in self.seen:
            return Fill("REJECTED", 0.0, 0.0, "DUPLICATE", datetime.now(timezone.utc).isoformat())
        self._pre_checks(o)
        fill = self.adapter.place(o)
        self.seen.add(o.row_id)
        return fill

    def _pre_checks(self, o: Order):
        if not (0.01 <= o.price <= 0.99):
            raise ValueError(f"Invalid price: {o.price}")
        if not (0.01 <= o.price_floor <= 0.99):
            raise ValueError(f"Invalid price_floor: {o.price_floor}")
        if o.size <= 0:
            raise ValueError("Size must be > 0")
        if o.side not in ("YES", "NO"):
            raise ValueError(f"Invalid side: {o.side}")
        if o.tif not in ("IOC", "FOK", "GTD"):
            raise ValueError(f"Invalid tif: {o.tif}")
