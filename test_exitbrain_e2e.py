#!/usr/bin/env python3
"""
End-to-End Test: ExitBrain v3.5 Adaptive Levels
Injects a fresh trade intent and monitors for adaptive levels computation
"""
import sys
sys.path.insert(0, '/app/backend')

import redis
import json
import time
from datetime import datetime

def inject_test_message():
    """Inject a fresh trade intent message"""
    r = redis.Redis(host='redis', port=6379, decode_responses=False)
    
    # Create test trade intent with ExitBrain v3.5 fields
    payload = {
        'symbol': 'BTCUSDT',
        'side': 'LONG',  # Must be LONG or SHORT (not BUY/SELL)
        'position_size_usd': 50,
        'leverage': 15,  # Will trigger adaptive calculations
        'confidence': 0.85,
        'volatility_factor': 1.2,  # High volatility
        'atr_value': 0.03,
        'source': 'exitbrain_v35_e2e_test',
        'timestamp': int(datetime.utcnow().timestamp() * 1000)  # Current timestamp
    }
    
    # Publish using EventBus format
    message = {
        b'event_type': b'trade.intent',
        b'payload': json.dumps(payload).encode('utf-8'),
        b'trace_id': b'test_exitbrain_v35_e2e',
        b'timestamp': datetime.utcnow().isoformat().encode('utf-8'),
        b'source': b'e2e_test_script'
    }
    
    msg_id = r.xadd('quantum:stream:trade.intent', message, maxlen=10000)
    
    print('=' * 70)
    print('✅ TEST MESSAGE INJECTED')
    print('=' * 70)
    print(f'Message ID: {msg_id.decode()}')
    print(f'Symbol: {payload["symbol"]}')
    print(f'Side: {payload["side"]}')
    print(f'Leverage: {payload["leverage"]}x')
    print(f'Volatility: {payload["volatility_factor"]}')
    print(f'Position Size: ${payload["position_size_usd"]}')
    print(f'Timestamp: {payload["timestamp"]} ({datetime.fromtimestamp(payload["timestamp"]/1000).isoformat()})')
    print('=' * 70)
    
    return msg_id.decode(), payload

def check_adaptive_levels_stream():
    """Check if adaptive levels were written to the stream"""
    r = redis.Redis(host='redis', port=6379, decode_responses=False)
    
    # Get stream length
    stream_len = r.xlen('quantum:stream:exitbrain.adaptive_levels')
    print(f'\n📊 Adaptive Levels Stream: {stream_len} messages')
    
    if stream_len > 0:
        # Read last 3 messages
        messages = r.xrevrange('quantum:stream:exitbrain.adaptive_levels', count=3)
        
        print('\n' + '=' * 70)
        print('🎯 RECENT ADAPTIVE LEVELS OUTPUT')
        print('=' * 70)
        
        for msg_id, data in messages:
            print(f'\nMessage ID: {msg_id.decode()}')
            for key, value in data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                val_str = value.decode() if isinstance(value, bytes) else value
                
                # Try to parse payload as JSON
                if key_str == 'payload':
                    try:
                        payload = json.loads(val_str)
                        print(f'  Symbol: {payload.get("symbol")}')
                        print(f'  Leverage: {payload.get("leverage")}x')
                        print(f'  TP1: {payload.get("tp1", "N/A")}%')
                        print(f'  TP2: {payload.get("tp2", "N/A")}%')
                        print(f'  TP3: {payload.get("tp3", "N/A")}%')
                        print(f'  SL: {payload.get("sl", "N/A")}%')
                        print(f'  LSF: {payload.get("LSF", "N/A")}')
                        print(f'  Harvest: {payload.get("harvest_scheme", "N/A")}')
                    except:
                        print(f'  {key_str}: {val_str}')
                else:
                    print(f'  {key_str}: {val_str}')
            print('-' * 70)
    
    return stream_len

def main():
    print('\n' + '=' * 70)
    print('🧪 EXITBRAIN V3.5 END-TO-END TEST')
    print('=' * 70)
    print(f'Timestamp: {datetime.utcnow().isoformat()}Z')
    print('=' * 70)
    
    # Step 1: Inject test message
    print('\n[STEP 1] Injecting fresh test trade intent...')
    msg_id, payload = inject_test_message()
    
    # Step 2: Wait for processing
    print('\n[STEP 2] Waiting 5 seconds for consumer to process...')
    time.sleep(5)
    
    # Step 3: Check adaptive levels stream
    print('\n[STEP 3] Checking adaptive levels stream...')
    count = check_adaptive_levels_stream()
    
    # Step 4: Summary
    print('\n' + '=' * 70)
    print('📋 TEST SUMMARY')
    print('=' * 70)
    print(f'✅ Test message injected: {msg_id}')
    print(f'📊 Adaptive levels in stream: {count}')
    
    if count > 0:
        print('🎉 SUCCESS: ExitBrain v3.5 is computing adaptive levels!')
    else:
        print('⚠️  No adaptive levels found - check consumer logs')
        print('Tip: docker logs quantum_trade_intent_consumer --tail 50')
    
    print('=' * 70)

if __name__ == '__main__':
    main()
