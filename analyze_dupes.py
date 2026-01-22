#!/usr/bin/env python3
"""Analyze trade.intent duplicates"""
import redis
import json
from collections import Counter

r = redis.Redis(decode_responses=False)

# Get last 100 messages
msgs = r.xrevrange('quantum:stream:trade.intent', count=100)

symbols = []
timestamps = []

for msg_id, data in msgs:
    try:
        payload_bytes = data[b'data']
        payload = json.loads(payload_bytes.decode('utf-8'))
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = payload.get('timestamp', 'UNKNOWN')
        action = payload.get('action', payload.get('side', 'UNKNOWN'))
        
        symbols.append(f"{symbol}_{action}")
        timestamps.append(timestamp)
    except Exception as e:
        print(f"Error parsing message: {e}")

print("\n📊 Last 100 trade.intent signals (symbol_action):")
print("=" * 60)
for symbol_action, count in Counter(symbols).most_common(20):
    symbol, action = symbol_action.split('_')
    print(f"  {symbol:12} {action:4} - {count:3} signals")

print(f"\n📈 Total unique symbol_action pairs: {len(set(symbols))}")
print(f"📈 Total signals analyzed: {len(symbols)}")

# Check for rapid duplicates (same symbol+action within 5 seconds)
rapid_dupes = 0
prev_sig = {}
for i, (sym_act, ts) in enumerate(zip(symbols, timestamps)):
    if sym_act in prev_sig:
        # Check if within 5 seconds
        try:
            from datetime import datetime
            dt1 = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(prev_sig[sym_act].replace('Z', '+00:00'))
            diff = abs((dt1 - dt2).total_seconds())
            if diff < 5:
                rapid_dupes += 1
                if rapid_dupes <= 5:  # Show first 5
                    print(f"\n⚠️  Rapid duplicate: {sym_act} - {diff:.1f}s apart")
        except:
            pass
    prev_sig[sym_act] = ts

print(f"\n🔥 Total rapid duplicates (< 5s apart): {rapid_dupes}")
