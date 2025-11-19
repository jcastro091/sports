import argparse, sys
from agents.quote_agent import QuoteAgent
from execution.paper import PaperExecutor
from execution.telegram_confirm import TelegramNotifyExecutor
from providers.odds_api import list_events

def main():
    p = argparse.ArgumentParser(description="Odds shopping + personalized stake + paper/telegram execution")
    p = argparse.ArgumentParser(description="Odds shopping + personalized stake + paper/telegram execution")
    p.add_argument("--sport", required=True, help="e.g., baseball_mlb, basketball_nba")
    p.add_argument("--hours", type=int, default=24, help="Look ahead this many hours for events (default 24).")
    p.add_argument("--list", type=int, default=0, help="List the next N events and exit.")
    p.add_argument("--event-id", help="Odds API event id. If omitted, --event-index is used.")
    p.add_argument("--event-index", type=int, default=0, help="Use Nth upcoming event if no event-id provided.")
    p.add_argument("--side", choices=["home","away"], help="Which side to back for H2H.")
    p.add_argument("--p", type=float, default=0.55, help="Your model-estimated win probability for this pick (0..1).")
    p.add_argument("--books", default="", help="Comma separated bookmakers to query (optional).")
    p.add_argument("--min-mins", type=int, default=5, help="Minimum minutes before start (default 5).")
    p.add_argument("--paper", action="store_true", help="Write paper trade to CSV.")
    p.add_argument("--telegram", action="store_true", help="Send Telegram proposal message.")
    args = p.parse_args()
    
    
    events = list_events(args.sport, hours=args.hours)

    if args.list:
        # Print availability so you can pick an index that actually has H2H odds
        from providers.odds_api import event_odds
        print(f"Found {len(events)} upcoming events in next {args.hours}h\n")
        for i, ev in enumerate(events[:args.list]):
            ev_id = ev["id"]
            home = ev.get("home_team")
            away = ev.get("away_team")
            ts = ev.get("commence_time")
            try:
                data = event_odds(args.sport, ev_id, markets=["h2h"])
                has_h2h = any(m.get("key")=="h2h" for b in data.get("bookmakers",[]) for m in b.get("markets",[]))
            except Exception:
                has_h2h = False
            print(f"[{i}] {home} vs {away} | {ts} | event_id={ev_id} | H2H={'YES' if has_h2h else 'NO'}")
        return

    sport_key = args.sport
    event_id = args.event_id

    if not event_id:
        events = list_events(sport_key, hours=args.hours)
        if not events:
            print("No upcoming events found.")
            sys.exit(0)
        idx = min(max(args.event_index, 0), len(events)-1)
        event = events[idx]
        event_id = event["id"]
        print(f"Using event_index={idx} → event_id={event_id} | {event.get('home_team')} vs {event.get('away_team')} at {event.get('commence_time')}")

    books = [b.strip() for b in args.books.split(",") if b.strip()] or None

    qa = QuoteAgent(min_minutes_to_start=args.min_mins)
    try:
        order, event_data = qa.best_quote_and_stake(
            sport_key=sport_key,
            event_id=event_id,
            side=args.side,
            win_prob=args.p,
            bookmakers=books
        )
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    print("=== Proposed Order ===")
    print(order)

    exec_ids = []
    if args.paper:
        paper = PaperExecutor()
        exec_ids.append(("paper", paper.execute(order)))
        print("[paper] Logged to data/executions.csv.")
    if args.telegram:
        tg = TelegramNotifyExecutor()
        exec_ids.append(("telegram", tg.execute(order)))
        print("[telegram] Proposal sent.")

    if not exec_ids:
        print("No executor selected. Use --paper and/or --telegram.")

if __name__ == "__main__":
    main()
