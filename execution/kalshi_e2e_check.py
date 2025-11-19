import argparse, logging, os, json
from watcher_allcontracts_to_kalshi import load_cfg, _startup_probe, _resolve_tradeable_ticker, _place_raw_limit
from adapters.kalshi_adapter import KalshiAdapter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Desired ticker or search text")
    ap.add_argument("--side", choices=["YES","NO"], default="YES")
    ap.add_argument("--price-cents", type=int, default=1, help="limit price in cents")
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--tif", default="GTC")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    logging.basicConfig(level=logging.DEBUG)
    kcfg = cfg["kalshi"]
    kalshi = KalshiAdapter(base_url=kcfg["base_url"], key_id=kcfg["key_id"], private_key_file=kcfg["private_key_file"])

    if not _startup_probe(kalshi):
        raise SystemExit("Auth probe failed (check PROD keys / perms).")

    # resolve market
    rt = _resolve_tradeable_ticker(kalshi, args.ticker)
    logging.info("resolved ticker => %s", rt)
    if not rt:
        raise SystemExit("Could not resolve a tradeable ticker from: " + args.ticker)

    if args.dry_run:
        logging.info("DRY RUN: would place %s %s @ %sc x %s", args.side, rt, args.price_cents, args.qty)
        return

    # place a tiny order (count = qty); price in cents
    resp = _place_raw_limit(kalshi, rt, args.side, args.price_cents, args.qty, args.tif)
    logging.info("✅ order placed: %s", json.dumps(resp)[:800])

if __name__ == "__main__":
    main()
