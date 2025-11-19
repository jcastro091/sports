def kelly_fraction(p: float, decimal_odds: float) -> float:
    """
    Kelly in decimal odds:
    b = decimal_odds - 1
    f* = (b*p - (1-p)) / b
    Returns full Kelly fraction (0..1). Caller may scale by user fraction.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1 - p)) / b)
