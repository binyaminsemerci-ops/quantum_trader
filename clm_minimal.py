#!/usr/bin/env python3
"""
CLM Minimal - Continuous Learning Manager
Reads UTF stream → Computes drift/health → Emits CLM intents
"""
import os
import sys
import json
import time
import subprocess
import socket
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configuration
UTF_STREAM = os.getenv('UTF_STREAM', 'quantum:stream:utf')
CLM_GROUP = os.getenv('CLM_GROUP', 'clm')
CLM_CONSUMER = os.getenv('CLM_CONSUMER', socket.gethostname())
CLM_INTENT_STREAM = os.getenv('CLM_INTENT_STREAM', 'quantum:stream:clm.intent')
CLM_ERROR_THRESHOLD_PER_HOUR = int(os.getenv('CLM_ERROR_THRESHOLD_PER_HOUR', '20'))
LOG_FILE = '/var/log/quantum/clm_minimal.log'

# Simple logger
def log(msg: str, level: str = 'INFO'):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} [{level}] {msg}\n"
    print(line, end='', flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write log: {e}", file=sys.stderr)

# Redis command wrapper
def redis_cmd(args: List[str], timeout: int = 5) -> Optional[str]:
    try:
        result = subprocess.run(['redis-cli'] + args, 
                              capture_output=True, timeout=timeout, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            log(f"Redis command failed: {' '.join(args)} - {result.stderr}", 'ERROR')
            return None
    except Exception as e:
        log(f"Redis command exception: {e}", 'ERROR')
        return None

# Ensure consumer group exists
def ensure_consumer_group():
    log(f"Ensuring consumer group {CLM_GROUP} exists on {UTF_STREAM}")
    result = redis_cmd(['XGROUP', 'CREATE', UTF_STREAM, CLM_GROUP, '0', 'MKSTREAM'])
    if result and 'BUSYGROUP' not in result:
        log(f"Consumer group created: {CLM_GROUP}")
    else:
        log(f"Consumer group already exists or created: {CLM_GROUP}")

# Increment Redis counter with TTL
def incr_counter(key: str, ttl_hours: int = 24):
    redis_cmd(['INCR', key])
    redis_cmd(['EXPIRE', key, str(ttl_hours * 3600)])

# Get counter value
def get_counter(key: str) -> int:
    result = redis_cmd(['GET', key])
    try:
        return int(result) if result and result != '(nil)' else 0
    except:
        return 0

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
    
    intent_json = json.dumps(intent)
    result = redis_cmd(['XADD', CLM_INTENT_STREAM, '*', 'intent', intent_json])
    
    if result:
        log(f"CLM Intent: {reason} (unit={unit}, symbol={symbol}, action={suggested_action})")
    else:
        log(f"Failed to emit intent: {reason}", 'ERROR')

# Process UTF event
def process_event(event_id: str, event_data: Dict[str, Any]):
    try:
        # Decode event JSON
        event_json = event_data.get('event', '{}')
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
            ts = int(ts) // 1000000  # Convert microseconds
        hour = datetime.fromtimestamp(ts).strftime('%Y%m%d%H')
        
        # Update counters
        incr_counter(f'quantum:clm:count:{unit}:{hour}')
        
        if symbol:
            incr_counter(f'quantum:clm:symbol:{symbol}:{hour}')
        
        # Track errors (PRIORITY: 3=error, 4=warning)
        if level <= 4 or 'ERROR' in message.upper() or 'EXCEPTION' in message.upper():
            error_key = f'quantum:clm:errors:{unit}:{hour}'
            incr_counter(error_key)
            
            # Check threshold
            error_count = get_counter(error_key)
            if error_count > CLM_ERROR_THRESHOLD_PER_HOUR:
                emit_intent(
                    reason=f"High error rate: {error_count} errors/hour",
                    unit=unit,
                    symbol=symbol,
                    severity='high',
                    suggested_action='investigate'
                )
        
        # Additional drift heuristics can be added here
        # e.g., symbol activity drop detection, performance degradation, etc.
        
    except Exception as e:
        log(f"Error processing event {event_id}: {e}", 'ERROR')

# Main consumer loop
def consume_loop():
    log("CLM Consumer starting...")
    log(f"Reading from: {UTF_STREAM}, Group: {CLM_GROUP}, Consumer: {CLM_CONSUMER}")
    log(f"Error threshold: {CLM_ERROR_THRESHOLD_PER_HOUR}/hour")
    
    ensure_consumer_group()
    
    processed_count = 0
    last_heartbeat = time.time()
    
    while True:
        try:
            # Read from stream (XREADGROUP)
            # Format: XREADGROUP GROUP group consumer COUNT count BLOCK ms STREAMS stream >
            result = subprocess.run([
                'redis-cli', 'XREADGROUP', 'GROUP', CLM_GROUP, CLM_CONSUMER,
                'COUNT', '200', 'BLOCK', '2000', 'STREAMS', UTF_STREAM, '>'
            ], capture_output=True, timeout=10, text=True)
            
            if result.returncode != 0:
                log(f"XREADGROUP failed: {result.stderr}", 'ERROR')
                time.sleep(5)
                continue
            
            output = result.stdout.strip()
            
            # Parse response (simple parsing for redis-cli output)
            if output and output != '(nil)' and not output.startswith('(empty'):
                lines = output.split('\n')
                
                # Very simple parser: look for event IDs and event data
                event_ids = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    # Event ID format: 1) "1737084000000-0"
                    if line and line[0].isdigit() and ')' in line:
                        # Extract ID
                        parts = line.split('"')
                        if len(parts) >= 2:
                            event_id = parts[1]
                            
                            # Next few lines contain field-value pairs
                            # Look for "event" field
                            event_json = None
                            for j in range(i+1, min(i+10, len(lines))):
                                if '"event"' in lines[j]:
                                    # Next line should have the value
                                    if j+1 < len(lines):
                                        event_json = lines[j+1].strip().strip('"')
                                    break
                            
                            if event_json:
                                try:
                                    process_event(event_id, {'event': event_json})
                                    event_ids.append(event_id)
                                    processed_count += 1
                                except Exception as e:
                                    log(f"Failed to process event {event_id}: {e}", 'ERROR')
                    
                    i += 1
                
                # ACK processed events
                if event_ids:
                    for event_id in event_ids:
                        redis_cmd(['XACK', UTF_STREAM, CLM_GROUP, event_id])
                    
                    if len(event_ids) % 50 == 0:
                        log(f"Processed {processed_count} total events")
            
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
