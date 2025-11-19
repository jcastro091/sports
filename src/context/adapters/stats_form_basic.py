from __future__ import annotations
from typing import Any, Dict

class BasicFormClient:
    """
    Minimal recent-form/advanced-stats adapter.

    Plug in your real backend later (Google Sheets, DuckDB, public box-score APIs).
    If a backend with matching methods is provided, we delegate; otherwise return
    placeholders so the rest of the pipeline keeps working.
    """

    def __init__(self, backend=None):
        self.backend = backend  # expected optional methods: get_form(team, n), get_advanced(team)

    def get_form(self, team: str, n: int = 5) -> Dict[str, Any]:
        if self.backend and hasattr(self.backend, "get_form"):
            return self.backend.get_form(team, n)
        # Fallback placeholder
        return {
            "team": team,
            "last_n_record": None,  # e.g., "3-2"
            "ats": None,            # e.g., "3-2 ATS"
            "ou": None,             # e.g., "2 Over / 3 Under"
            "notes": "Form backend not configured",
        }

    def get_advanced(self, team: str) -> Dict[str, Any]:
        if self.backend and hasattr(self.backend, "get_advanced"):
            return self.backend.get_advanced(team)
        # Fallback placeholder
        return {}
