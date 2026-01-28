#!/usr/bin/env python3
"""P3.1 Test Data Injector - Execution Results"""
import time
import json
import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
STREAM = os.getenv("EXECUTION_STREAM", "quantum:stream:execution.result")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def xadd(symbol, pnl, qty):
    """Add execution result to stream"""
    now = int(time.time())
    msg = {
        "symbol": symbol,
        "pnl_usd": str(pnl),
        "filled_qty": str(qty),
        "ts": str(now),
        "timestamp": str(now),
        "source": "proof_p31",
    }
    r.xadd(STREAM, msg, maxlen=50000, approximate=True)
    print(f"  {symbol}: pnl=${pnl:+.2f} qty={qty}")

if __name__ == "__main__":
    print("Injecting execution result samples...")
    
    # BTC: mixed results (net positive)
    xadd("BTCUSDT", +12.5, 0.002)
    xadd("BTCUSDT", -3.1, 0.002)
    xadd("BTCUSDT", +8.7, 0.002)
    
    # ETH: volatile with drawdown
    xadd("ETHUSDT", -9.7, 0.05)
    xadd("ETHUSDT", +2.2, 0.05)
    xadd("ETHUSDT", -1.4, 0.05)
    xadd("ETHUSDT", +5.3, 0.05)
    
    # SOL: consistent positive
    xadd("SOLUSDT", +4.2, 0.5)
    xadd("SOLUSDT", +3.8, 0.5)
    xadd("SOLUSDT", +2.1, 0.5)
    
    print("OK: injected execution samples")
