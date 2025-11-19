from datetime import datetime, timezone

def basic_time_check(start_time_iso: str, min_minutes: int = 5) -> bool:
    """
    Return True if we are at least `min_minutes` before event start.
    """
    try:
        # e.g. '2025-09-18T19:05:00Z'
        if start_time_iso.endswith("Z"):
            cutoff = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        else:
            cutoff = datetime.fromisoformat(start_time_iso)
        now = datetime.now(timezone.utc)
        return (cutoff - now).total_seconds() > min_minutes * 60
    except Exception:
        # If unknown, allow upstream logic to decide
        return True
