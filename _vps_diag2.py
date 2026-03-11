import asyncio, json, os, time, sys
sys.path.insert(0, ".")

async def check():
    import redis.asyncio as rm
    from datetime import datetime
    from microservices.autonomous_trader.entry_scanner import EntryScanner
    from microservices.autonomous_trader.funding_rate_filter import get_filtered_symbols

    r = rm.from_url("redis://localhost:6379", decode_responses=True)

    # Simulate exactly what autonomous_trader does
    candidate_symbols = ["ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","BNBUSDT",
                         "ADAUSDT","SUIUSDT","LINKUSDT","AVAXUSDT","LTCUSDT","DOTUSDT","NEARUSDT"]
    safe_symbols = await get_filtered_symbols(candidate_symbols)
    print("Safe symbols after funding filter: {}".format(safe_symbols))

    scanner = EntryScanner(r, min_confidence=0.65, max_positions=10, symbols=safe_symbols)
    print("Scanner symbols ({}): {}".format(len(scanner.symbols), scanner.symbols))

    # Run scan
    opps = await scanner.scan_for_entries(current_position_count=0)
    print("Opportunities found: {}".format(len(opps)))
    for o in opps:
        print("  {} {} conf={}".format(o.symbol, o.side, o.confidence))

    await r.aclose()

asyncio.run(check())
