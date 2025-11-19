# scripts/context_demo.py
import json
from src.context.context_builder import ContextBuilder, ProviderConfig, render_llm_prompt
from src.context.adapters import TheOddsClient, OpenMeteoClient, MLBStatsInjuriesClient, BasicFormClient

# OPTIONAL: quick venue lat/lon hardcode to test weather
VENUE = {"lat": 42.3467, "lon": -71.0972}  # Fenway Park

def main():
    odds = TheOddsClient(sport_key="baseball_mlb")
    weather = OpenMeteoClient()
    injuries = MLBStatsInjuriesClient()
    stats = BasicFormClient()

    # Temporary: reuse stats as matchup_client until you plug a real one
    builder = ContextBuilder(ProviderConfig(
        odds_client=odds,
        matchup_client=stats,
        injuries_client=injuries,
        weather_client=weather,
        stats_client=stats,
    ))

    ctx = builder.build_context(
        event_id="123456",  # MLB gamePk if you have it; otherwise some features may be None
        sport="baseball_mlb",
        league="MLB",
        teams={"home": "BOS", "away": "NYY"},
        start_time_utc="2025-09-21T23:05:00Z",
    )

    # Attach a dummy model block (Pro only) so you can see formatting
    ctx["model"] = {
        "pick": "NYY -1.5",
        "market": "spread",
        "confidence": "High",
        "reasons": [
            "Reverse line move with rising liquidity",
            "Bullpen form edge (last 7d)",
            "Key bat returning",
            "Wind out to LF supports power",
        ],
    }

    print("\n=== PRO (text render) ===")
    print(render_llm_prompt(ctx, pro=True))

    free_ctx = builder.redact_for_free(ctx)
    print("\n=== FREE (text render) ===")
    print(render_llm_prompt(free_ctx, pro=False))

if __name__ == "__main__":
    main()
