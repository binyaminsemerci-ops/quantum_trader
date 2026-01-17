#!/usr/bin/env python3
"""
UTF Publisher - Unified Training Feed
Tails journald logs from quantum services → Redis Stream
"""
import os
import sys
import json
import time
import re
import subprocess
import socket
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Configuration from environment
UTF_REDIS_URL = os.getenv('UTF_REDIS_URL', 'redis://127.0.0.1:6379/0')
UTF_STREAM = os.getenv('UTF_STREAM', 'quantum:stream:utf')
UTF_MAXLEN = int(os.getenv('UTF_MAXLEN', '200000'))
UTF_UNITS = os.getenv('UTF_UNITS', '').split()
UTF_POLL_SEC = int(os.getenv('UTF_POLL_SEC', '2'))
CURSOR_DIR = Path('/var/lib/quantum/utf/cursors')
LOG_FILE = '/var/log/quantum/utf_publisher.log'

HOSTNAME = socket.gethostname()

# Unit to source mapping
UNIT_SOURCE_MAP = {
    'quantum-ai-engine.service': 'ai_engine',
    'quantum-strategy-brain.service': 'strategy',
    'quantum-risk-brain.service': 'risk',
    'quantum-execution-engine.service': 'execution',
    'quantum-execution.service': 'execution',
}

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

# Extract symbol from message (best effort)
def extract_symbol(msg: str) -> Optional[str]:
    match = re.search(r'\b([A-Z]{3,10}USDT)\b', msg)
    return match.group(1) if match else None

# Extract correlation_id (best effort)
def extract_correlation_id(msg: str, json_data: Dict) -> Optional[str]:
    # Try from JSON structure first
    if 'correlation_id' in json_data:
        return json_data['correlation_id']
    
    # Try from message
    match = re.search(r'correlation_id[=:]?\s*["\']?([a-f0-9-]{36})["\']?', msg, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try UUID pattern
    match = re.search(r'\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b', msg)
    return match.group(1) if match else None

# Extract confidence (best effort)
def extract_confidence(msg: str) -> Optional[float]:
    match = re.search(r'confidence[=:]?\s*(\d+\.?\d*)', msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    return None

# Extract decision (BUY/SELL/HOLD)
def extract_decision(msg: str) -> Optional[str]:
    for decision in ['BUY', 'SELL', 'HOLD']:
        if decision in msg.upper():
            return decision
    return None

# Parse journald JSON to UTF event
def parse_journal_entry(entry: Dict[str, Any], unit: str) -> Dict[str, Any]:
    message = entry.get('MESSAGE', '')
    
    # Try to parse embedded JSON in message
    extra_data = {}
    try:
        # Some logs have JSON structure
        if isinstance(message, str) and message.strip().startswith('{'):
            extra_data = json.loads(message)
            message = extra_data.get('msg', message)
    except:
        pass
    
    # Build UTF event
    utf_event = {
        'ts': entry.get('__REALTIME_TIMESTAMP', str(int(time.time() * 1000000))),
        'unit': unit,
        'level': entry.get('PRIORITY', '6'),  # 6 = INFO
        'message': message[:500],  # Truncate long messages
        'symbol': extract_symbol(message),
        'correlation_id': extract_correlation_id(message, extra_data),
        'source': UNIT_SOURCE_MAP.get(unit, 'unknown'),
        'host': HOSTNAME,
        'tags': []
    }
    
    # Optional fields
    confidence = extract_confidence(message)
    if confidence is not None:
        utf_event['confidence'] = confidence
    
    decision = extract_decision(message)
    if decision:
        utf_event['decision'] = decision
    
    # Add error tag if needed
    if int(utf_event['level']) <= 4:  # 3=ERROR, 4=WARNING
        utf_event['tags'].append('error')
    
    return utf_event

# Load cursor for unit
def load_cursor(unit: str) -> Optional[str]:
    cursor_file = CURSOR_DIR / f"{unit.replace('.service', '')}.cursor"
    try:
        if cursor_file.exists():
            return cursor_file.read_text().strip()
    except Exception as e:
        log(f"Failed to load cursor for {unit}: {e}", 'WARN')
    return None

# Save cursor for unit
def save_cursor(unit: str, cursor: str):
    cursor_file = CURSOR_DIR / f"{unit.replace('.service', '')}.cursor"
    try:
        CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        cursor_file.write_text(cursor)
    except Exception as e:
        log(f"Failed to save cursor for {unit}: {e}", 'ERROR')

# Publish to Redis Stream (using redis-cli)
def publish_to_redis(event: Dict[str, Any]) -> bool:
    try:
        event_json = json.dumps(event)
        cmd = [
            'redis-cli',
            'XADD', UTF_STREAM,
            'MAXLEN', '~', str(UTF_MAXLEN),
            '*',  # Auto-generate ID
            'event', event_json
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
        if result.returncode != 0:
            log(f"Redis publish failed: {result.stderr}", 'ERROR')
            return False
        return True
    except Exception as e:
        log(f"Redis publish exception: {e}", 'ERROR')
        return False

# Tail journald for a unit
def tail_unit(unit: str):
    cursor = load_cursor(unit)
    
    # Build journalctl command
    cmd = ['journalctl', '-u', unit, '-o', 'json', '-n', '0', '--follow']
    if cursor:
        cmd.extend(['--after-cursor', cursor])
    
    log(f"Starting tail for {unit} (cursor: {cursor[:20] if cursor else 'none'}...)")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        event_count = 0
        last_cursor = cursor
        
        while True:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            try:
                entry = json.loads(line.strip())
                
                # Extract cursor
                if '__CURSOR' in entry:
                    last_cursor = entry['__CURSOR']
                
                # Parse to UTF event
                utf_event = parse_journal_entry(entry, unit)
                
                # Publish to Redis
                if publish_to_redis(utf_event):
                    event_count += 1
                    if event_count % 100 == 0:
                        log(f"{unit}: Published {event_count} events")
                        save_cursor(unit, last_cursor)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                log(f"Error processing entry from {unit}: {e}", 'ERROR')
                continue
    
    except Exception as e:
        log(f"Fatal error tailing {unit}: {e}", 'ERROR')
        raise

def main():
    log("UTF Publisher starting...")
    log(f"Configuration: stream={UTF_STREAM}, maxlen={UTF_MAXLEN}, poll={UTF_POLL_SEC}s")
    log(f"Units: {UTF_UNITS}")
    
    # Create cursor directory
    CURSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Redis stream if needed
    try:
        subprocess.run(['redis-cli', 'XADD', UTF_STREAM, '*', 'init', '1'], 
                      capture_output=True, timeout=5)
        subprocess.run(['redis-cli', 'XTRIM', UTF_STREAM, 'MAXLEN', '0'], 
                      capture_output=True, timeout=5)
        log("Redis stream initialized")
    except Exception as e:
        log(f"Failed to initialize Redis stream: {e}", 'ERROR')
        sys.exit(1)
    
    # For simplicity, tail all units in parallel using subprocesses
    # In production, use threading or asyncio
    import threading
    
    threads = []
    for unit in UTF_UNITS:
        if unit:
            thread = threading.Thread(target=tail_unit, args=(unit,), daemon=True)
            thread.start()
            threads.append(thread)
    
    log(f"Started {len(threads)} tailers")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            log(f"UTF Publisher heartbeat: {len(threads)} active tailers")
    except KeyboardInterrupt:
        log("UTF Publisher shutting down...")
        sys.exit(0)

if __name__ == '__main__':
    main()
