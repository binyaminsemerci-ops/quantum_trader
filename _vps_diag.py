import asyncio,json,os,time,sys
sys.path.insert(0,".")
async def check():
    import redis.asyncio as rm
    r = rm.from_url("redis://localhost:6379",decode_responses=True)
    msgs = await r.xrevrange("quantum:stream:ai.signal_generated",count=50)
    for mid,data in msgs:
        p = json.loads(data.get("payload","{}"))
        if p.get("symbol") == "BTCUSDT":
            from datetime import datetime
            ts_str = data.get("timestamp","")
            ts = int(datetime.fromisoformat(ts_str.replace("Z","+00:00")).timestamp())
            age = int(time.time()) - ts
            print("BTCUSDT age={} conf={} action={}".format(age,p.get("confidence"),p.get("action")))
            break
    from microservices.autonomous_trader.funding_rate_filter import get_filtered_symbols
    syms = ["BTCUSDT","ETHUSDT","XRPUSDT","LTCUSDT","DOTUSDT"]
    safe = await get_filtered_symbols(syms)
    print("Safe {}/{}: {}".format(len(safe),len(syms),safe))
    await r.aclose()
asyncio.run(check())