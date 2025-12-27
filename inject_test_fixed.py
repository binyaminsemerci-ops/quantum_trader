#!/usr/bin/env python3
"""
Test event injection with CORRECT format for new trade_intent_subscriber.py
Tests both OLD format (direction) and NEW format (side)
"""
import redis
import json
from datetime import datetime

r = redis.Redis(host='quantum_redis', port=6379, decode_responses=False)

# TEST 1: Old AI Engine format (direction=LONG)
old_format_payload = {
    'event_type': 'trade.intent',
    'symbol': 'BTCUSDT',
    'direction': 'LONG',  # OLD format
    'leverage': 10,
    'entry_price': 95000,
    'confidence': 0.85,
    'position_size_usd': 50,
    'atr_value': 0.03,
    'volatility_factor': 1.5,
    'timestamp': int(datetime.utcnow().timestamp() * 1000),
    'source': 'test_old_format'
}

msg_id_1 = r.xadd(
    b'quantum:stream:trade.intent',
    old_format_payload,
    maxlen=10000
)
print(f'✅ OLD FORMAT published: {msg_id_1.decode()} (direction=LONG)')

# TEST 2: New Trading Bot format (side=BUY with payload wrapper)
new_format = {
    'event_type': 'trade.intent',
    'payload': json.dumps({
        'symbol': 'ETHUSDT',
        'side': 'BUY',  # NEW format
        'leverage': 8,
        'entry_price': 3500,
        'confidence': 0.78,
        'position_size_usd': 40,
        'atr_value': 0.025,
        'volatility_factor': 1.2,
        'timestamp': datetime.utcnow().isoformat(),
        'model': 'test_new_format'
    }),
    'timestamp': datetime.utcnow().isoformat(),
    'source': 'test_bot'
}

msg_id_2 = r.xadd(
    b'quantum:stream:trade.intent',
    new_format,
    maxlen=10000
)
print(f'✅ NEW FORMAT published: {msg_id_2.decode()} (side=BUY)')

# TEST 3: SHORT position (old format)
short_payload = {
    'event_type': 'trade.intent',
    'symbol': 'BNBUSDT',
    'direction': 'SHORT',
    'leverage': 5,
    'entry_price': 650,
    'confidence': 0.72,
    'position_size_usd': 30,
    'timestamp': int(datetime.utcnow().timestamp() * 1000),
    'source': 'test_short'
}

msg_id_3 = r.xadd(
    b'quantum:stream:trade.intent',
    short_payload,
    maxlen=10000
)
print(f'✅ SHORT FORMAT published: {msg_id_3.decode()} (direction=SHORT → side=SELL)')

print(f'\n📊 Alle 3 test events injected!')
print(f'   - Old format (direction=LONG): BTCUSDT $50')
print(f'   - New format (side=BUY): ETHUSDT $40')
print(f'   - Short format (direction=SHORT): BNBUSDT $30')
