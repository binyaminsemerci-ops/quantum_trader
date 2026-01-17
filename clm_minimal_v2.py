#!/usr/bin/env python3
"""
CLM Minimal v2 - Continuous Learning Manager
Reads UTF stream → Computes drift/health → Emits CLM intents
Uses Python redis library for proper Stream support
"""
import os
import sys
import json
import time
import socket
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import redis
except ImportError:
    print("ERROR: python3-redis not installed. Run: apt-get install python3-redis", file=sys.stderr)
    sys.exit(1)

# Configuration
UTF_STREAM = os.getenv('UTF_STREAM', 'quantum:stream:utf')
CLM_GROUP = os.getenv('CLM_GROUP', 'clm')
CLM_CONSUMER = os.getenv('CLM_CONSUMER', socket.gethostname())
CLM_INTENT_STREAM = os.getenv('CLM_INTENT_STREAM', 'quantum:stream:clm.intent')
CLM_ERROR_THRESHOLD_PER_HOUR = int(os.getenv('CLM_ERROR_THRESHOLD_PER_HOUR', '20'))
LOG_FILE = '/var/log/quantum/clm_minimal.log'

# Redis client
redis_client = None

# Simple logger
def log(msg: str, level: str = 'INFO'):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} [{level}] {msg}\n"
    print(line, end='', flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write log: {e}", file=sys.stderr)

# Emit CLM intent
def emit_intent(reason: str, unit: str, symbol: Optional[str] = None, 
                severity: str = 'medium', suggested_action: str = 'investigate'):
    intent = {
        'ts': int(time.time()),
        'reason': reason,
        'unit': unit,
        'symbol': symbol,
        'severity': severity,
        'suggested_action': suggested_action,
        'consumer': CLM_CONSUMER
    }
    
    try:
        redis_client.xadd(CLM_INTENT_STREAM, {'intent': json.dumps(intent)})
        log(f"CLM Intent: {reason} (unit={unit}, symbol={symbol}, action={suggested_action})")
    except Exception as e:
        log(f"Failed to emit intent: {e}", 'ERROR')

# Process UTF event
def process_event(event_id: str, event_data: Dict[str, Any]):
    try:
        # Decode event JSON
        event_json = event_data.get(b'event', event_data.get('event', '{}'))
        if isinstance(event_json, bytes):
            event_json = event_json.decode('utf-8')
        
        event = json.loads(event_json)
        
        # Extract fields
        unit = event.get('unit', 'unknown')
        symbol = event.get('symbol')
        level = int(event.get('level', 6))
        message = event.get('message', '')
        ts = event.get('ts', 0)
        
        # Determine hour bucket
        if isinstance(ts, str):
            ts = int(ts) // 1000000  # Convert microseconds to seconds
        hour = datetime.fromtimestamp(ts).strftime('%Y%m%d%H')
        
        # Update counters with pipeline for efficiency
        pipe = redis_client.pipeline()
        
        count_key = f'quantum:clm:count:{unit}:{hour}'
        pipe.incr(count_key)
        pipe.expire(count_key, 86400)  # 24 hours TTL
        
        if symbol:
            symbol_key = f'quantum:clm:symbol:{symbol}:{hour}'
            pipe.incr(symbol_key)
            pipe.expire(symbol_key, 86400)
        
        # Track errors (PRIORITY: 3=error, 4=warning)
        if level <= 4 or 'ERROR' in message.upper() or 'EXCEPTION' in message.upper():
            error_key = f'quantum:clm:errors:{unit}:{hour}'
            pipe.incr(error_key)
            pipe.expire(error_key, 86400)
            
            # Execute pipeline
            results = pipe.execute()
            
            # Check threshold (error count is in results)
            error_count = int(results[-2]) if len(results) >= 2 else 0
            
            if error_count > CLM_ERROR_THRESHOLD_PER_HOUR:
                emit_intent(
                    reason=f"High error rate: {error_count} errors/hour",
                    unit=unit,
                    symbol=symbol,
                    severity='high',
                    suggested_action='investigate'
                )
        else:
            # Execute pipeline without checking
            pipe.execute()
        
    except Exception as e:
        log(f"Error processing event {event_id}: {e}", 'ERROR')

# Main consumer loop
def consume_loop():
    global redis_client
    
    log("CLM Consumer v2 starting...")
    log(f"Reading from: {UTF_STREAM}, Group: {CLM_GROUP}, Consumer: {CLM_CONSUMER}")
    log(f"Error threshold: {CLM_ERROR_THRESHOLD_PER_HOUR}/hour")
    
    # Connect to Redis
    try:
        redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=False)
        redis_client.ping()
        log("Redis connection established")
    except Exception as e:
        log(f"Failed to connect to Redis: {e}", 'ERROR')
        sys.exit(1)
    
    # Ensure consumer group exists
    try:
        redis_client.xgroup_create(UTF_STREAM, CLM_GROUP, id='0', mkstream=True)
        log(f"Consumer group created: {CLM_GROUP}")
    except redis.exceptions.ResponseError as e:
        if 'BUSYGROUP' in str(e):
            log(f"Consumer group already exists: {CLM_GROUP}")
        else:
            log(f"Error creating consumer group: {e}", 'ERROR')
            sys.exit(1)
    
    processed_count = 0
    last_heartbeat = time.time()
    
    while True:
        try:
            # Read from stream
            messages = redis_client.xreadgroup(
                groupname=CLM_GROUP,
                consumername=CLM_CONSUMER,
                streams={UTF_STREAM: '>'},
                count=200,
                block=2000  # 2 second timeout
            )
            
            if messages:
                for stream_name, events in messages:
                    for event_id, event_data in events:
                        try:
                            # Process event
                            process_event(event_id, event_data)
                            processed_count += 1
                            
                            # ACK immediately
                            redis_client.xack(UTF_STREAM, CLM_GROUP, event_id)
                            
                            if processed_count % 100 == 0:
                                log(f"Processed {processed_count} total events")
                        
                        except Exception as e:
                            log(f"Error processing event {event_id}: {e}", 'ERROR')
            
            # Heartbeat
            if time.time() - last_heartbeat > 60:
                log(f"CLM Consumer heartbeat: {processed_count} events processed")
                last_heartbeat = time.time()
        
        except KeyboardInterrupt:
            log("CLM Consumer shutting down...")
            break
        except Exception as e:
            log(f"Consume loop error: {e}", 'ERROR')
            time.sleep(5)

def main():
    consume_loop()

if __name__ == '__main__':
    main()
