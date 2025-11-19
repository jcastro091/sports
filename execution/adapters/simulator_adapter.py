# execution/adapters/simulator_adapter.py
from __future__ import annotations
from datetime import datetime, timezone
from ..router import Order, Fill

class SimulatorAdapter:
    def __init__(self, slip_bps: int = 0):
        self.slip_bps = slip_bps

    def place(self, o: Order) -> Fill:
        px = round(o.price * (1 - self.slip_bps/10000), 4)
        if px < o.price_floor:
            return Fill("REJECTED", 0.0, 0.0, "SIM_PRICE_FLOOR", datetime.now(timezone.utc).isoformat())
        return Fill("FILLED", px, o.size, f"SIM-{o.row_id}", datetime.now(timezone.utc).isoformat())
