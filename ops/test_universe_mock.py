#!/usr/bin/env python3
"""
Mock test of Universe Service (for demonstration)
Shows expected Redis key structure without actually calling Binance API
"""

import json
import time

# Mock data structure that would be published to Redis
mock_universe = {
    "asof_epoch": int(time.time()),
    "source": "binance_futures_exchangeInfo",
    "mode": "testnet",
    "symbols": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "SOLUSDT", "MATICUSDT",
        "LINKUSDT", "LTCUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
        "VETUSDT", "TRXUSDT", "FILUSDT", "AAVEUSDT", "THETAUSDT"
        # ... (would be 400-800 symbols in real data)
    ],
    "filters": {
        "contractType": "PERPETUAL",
        "status": "TRADING"
    }
}

mock_meta = {
    "asof_epoch": str(int(time.time())),
    "last_ok_epoch": str(int(time.time())),
    "count": str(len(mock_universe["symbols"])),
    "stale": "0",
    "error": ""
}

print("╔════════════════════════════════════════════════╗")
print("║   UNIVERSE SERVICE - MOCK OUTPUT               ║")
print("╚════════════════════════════════════════════════╝")
print()

print("Example Redis Key: quantum:cfg:universe:active")
print("Value (JSON):")
print(json.dumps(mock_universe, indent=2))
print()

print("Example Redis Key: quantum:cfg:universe:meta (hash)")
print("Fields:")
for key, value in mock_meta.items():
    print(f"  {key}: {value}")
print()

print("Example Proof Script Output:")
print("-" * 50)
print(f"Mode:           {mock_universe['mode']}")
print(f"Active Symbols: {len(mock_universe['symbols'])}")
print(f"Last Update:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Age:            0 minutes (just updated)")
print()
print("✅ STATUS: FRESH (recently updated)")
print()
print("Sample Symbols (first 20):")
for i, symbol in enumerate(mock_universe['symbols'][:20], 1):
    print(f" {i:2d}. {symbol}")
print()

print("Usage from other services:")
print("-" * 50)
print("""
import redis
import json

r = redis.Redis(decode_responses=True)
universe_json = r.get('quantum:cfg:universe:active')
universe = json.loads(universe_json)

symbols = universe['symbols']  # List[str]
mode = universe['mode']        # 'testnet' or 'mainnet'
asof = universe['asof_epoch']  # int (Unix timestamp)

print(f"Trading universe has {len(symbols)} symbols")
print(f"Mode: {mode}")
print(f"First 5: {symbols[:5]}")
""")
