#!/usr/bin/env python3
"""Test REDUCE_ONLY intent parsing in intent_bridge"""
import redis
import json
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Test intent with reduceOnly=true (camelCase - NEW format)
test_intent = {
    "symbol": "HYPEUSDT",
    "side": "BUY",
    "qty": 100.0,
    "intent_type": "REDUCE_ONLY",
    "reason": "[TEST] Manual test of reduceOnly parsing",
    "reduceOnly": True,  # CAMELCASE
    "source": "test_script",
    "correlation_id": f"test:manual:{int(time.time())}",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
}

entry_id = r.xadd(
    'quantum:stream:trade.intent',
    {'payload': json.dumps(test_intent)}
)

print(f"✅ Published test REDUCE_ONLY intent to quantum:stream:trade.intent")
print(f"   Entry ID: {entry_id}")
print(f"   Symbol: {test_intent['symbol']}")
print(f"   Side: {test_intent['side']}")
print(f"   Qty: {test_intent['qty']}")
print(f"   reduceOnly: {test_intent['reduceOnly']}")
print()
print("Now check intent_bridge logs:")
print("  journalctl -u quantum-intent-bridge -f | grep -i reduce")
