#!/usr/bin/env python3
import redis
import json
from datetime import datetime

r = redis.Redis(host='redis', port=6379, decode_responses=False)

# Create test trade intent with all ExitBrain v3.5 fields
payload = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'position_size_usd': 50,
    'leverage': 15,
    'confidence': 0.85,
    'volatility_factor': 1.2,
    'atr_value': 0.03,
    'source': 'exitbrain_v35_final_test',
    'timestamp': int(datetime.utcnow().timestamp() * 1000)
}

# Publish using EventBus format
message = {
    'event_type': 'trade.intent',
    'payload': json.dumps(payload),
    'trace_id': 'test_exitbrain_v35_final',
    'timestamp': datetime.utcnow().isoformat(),
    'source': 'manual_test'
}

msg_id = r.xadd('quantum:stream:trade.intent', message, maxlen=10000)
print(f'✅ Published test message: {msg_id.decode()}')
print(f'📊 Payload: leverage={payload["leverage"]}x, volatility={payload["volatility_factor"]}')
