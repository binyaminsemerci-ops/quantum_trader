#!/usr/bin/env python3
"""
Inject test market ticks to trigger AI Engine signal generation
and verify dedup fix is working
"""
import asyncio
import redis
import json
from datetime import datetime

async def inject_test_ticks():
    """Inject 10 identical ETHUSDT ticks within 1 second to test dedup"""
    r = redis.Redis(host='46.224.116.254', port=6379, decode_responses=True)
    
    test_symbols = [
        {"symbol": "ETHUSDT", "price": 3330.51},
        {"symbol": "BTCUSDT", "price": 42500.00},
        {"symbol": "BNBUSDT", "price": 610.25}
    ]
    
    print("📤 Injecting test market ticks to trigger AI Engine signals...")
    print("=" * 70)
    
    for test_data in test_symbols:
        symbol = test_data["symbol"]
        price = test_data["price"]
        
        print(f"\n🔄 Injecting 10 identical ticks for {symbol} @ ${price}")
        
        # Inject 10 identical ticks very quickly (should all trigger same signal)
        for i in range(10):
            tick_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "price": price,
                "volume": 100.0,
                "bid": price - 0.01,
                "ask": price + 0.01,
                "exchange": "binance",
                "source": "test_inject"
            }
            
            # Add to market.tick stream
            r.xadd(
                "quantum:stream:market.tick",
                tick_payload,
                maxlen=10000,
                approximate=True
            )
            
            if (i + 1) % 2 == 0:
                print(f"  ✓ Injected tick {i+1}/10")
            
            await asyncio.sleep(0.05)  # 50ms between ticks
        
        print(f"  ✅ Completed {symbol}")
        await asyncio.sleep(1)
    
    print("\n" + "=" * 70)
    print("⏳ Waiting 5 seconds for AI Engine to process...")
    await asyncio.sleep(5)
    
    # Check how many signals were generated
    stream_len = r.xlen("quantum:stream:trade.intent")
    print(f"\n📊 Trade intent stream now has: {stream_len} total entries")
    
    # Analyze the last signals for duplicates
    print("\n🔍 Analyzing for duplicate signals...")
    print("=" * 70)
    
    # Get last 50 signals
    signals = r.xrevrange("quantum:stream:trade.intent", count=50)
    
    signal_map = {}
    for entry_id, data in signals:
        symbol = data.get("symbol", "UNKNOWN")
        action = data.get("side", "UNKNOWN")
        key = f"{symbol}_{action}"
        
        if key not in signal_map:
            signal_map[key] = []
        signal_map[key].append(entry_id)
    
    print("\n📌 Signal summary (last 50 entries):")
    total_signals = 0
    for key, ids in sorted(signal_map.items()):
        print(f"  {key}: {len(ids)} signals")
        total_signals += len(ids)
    
    print(f"\nTotal signals analyzed: {total_signals}")
    
    # Expected vs Actual
    print("\n✨ DEDUP VERIFICATION:")
    print("-" * 70)
    for test_data in test_symbols:
        symbol = test_data["symbol"]
        # Expected: ~1-2 signals per symbol (due to dedup working)
        # Without dedup: 10+ signals per symbol
        
        for action in ["BUY", "SELL"]:
            key = f"{symbol}_{action}"
            count = len(signal_map.get(key, []))
            if count > 0:
                status = "✅ DEDUP OK" if count <= 2 else "❌ DUPLICATES DETECTED"
                print(f"  {key}: {count} signals {status}")

if __name__ == "__main__":
    asyncio.run(inject_test_ticks())
