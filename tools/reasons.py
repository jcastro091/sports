# tools/reasons.py
from typing import Optional

def _fmt_am(dec: Optional[float]) -> str:
    if dec is None: return "n/a"
    try:
        d = float(dec)
    except:
        return "n/a"
    if d <= 1.0: return "n/a"
    if d >= 2.0: return f"+{int(round((d-1)*100))}"
    return f"-{int(round(100/(d-1)))}"

def build_reason(
    *,
    sport_key: str,
    market_label: str,          # e.g., "H2H Away", "Spread Home", "Total Over"
    pick_label: str,            # e.g., "NYM Mets", "Over"
    decimal_odds: float,        # current price we’d take
    open_dec: Optional[float],  # opening price
    peak_dec: Optional[float],  # prior peak before reversal
    p_ml: float,                # model prob 0..1
    min_proba: float,           # from strong_config
    odds_bin: str,              # live bin "Fav/Dog/Balanced/Longshot"
    bin_allowed: bool,          # does strong_config allow this bin?
    limit_val: Optional[float], # current bet limit (if present)
    limit_trend: str,           # "rising"/"flat"
    kelly_quarter: float,       # 0..1
    setup_tag: str              # derived tag(s)
) -> str:
    am = _fmt_am(decimal_odds)
    open_am = _fmt_am(open_dec)
    peak_am = _fmt_am(peak_dec)

    why = []
    # 1) Model edge
    edge = round(100*(p_ml - 1/decimal_odds), 1)
    why.append(f"Model p={p_ml:.0%} (edge {edge:+.1f}pts vs {am}).")

    # 2) Reversal/value narrative
    if open_dec and peak_dec:
        why.append(f"Reversal after move {open_am} → {peak_am} → {am} near start.")

    # 3) Risk guardrails from strong_config
    gates = []
    if p_ml >= min_proba: gates.append(f"p≥{min_proba:.0%}")
    if bin_allowed: gates.append(f"bin={odds_bin}")
    if gates:
        why.append("Passes gates: " + ", ".join(gates) + ".")

    # 4) Liquidity / confidence via bet limits
    if isinstance(limit_val, (int, float)):
        why.append(f"Limits ~${int(limit_val):,} ({limit_trend}).")

    # 5) Bankroll sizing
    why.append(f"Stake = ¼ Kelly ({kelly_quarter:.2%}).")

    # 6) Setup tag
    if setup_tag:
        why.append(f"Setup: {setup_tag}.")

    return f"{market_label} • {pick_label}: " + " ".join(why)
