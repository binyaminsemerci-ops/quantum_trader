#!/usr/bin/env python3
"""
RL Policy Publisher v0 (Shadow Mode)
Continuously publishes fresh RL policies to Redis to prevent stale policy gates.
"""
import os
import sys
import time
import json
import redis
from datetime import datetime, timezone


def load_env():
    """Load configuration from environment variables."""
    return {
        'redis_host': os.getenv('REDIS_HOST', '127.0.0.1'),
        'redis_port': int(os.getenv('REDIS_PORT', '6379')),
        'interval': int(os.getenv('PUBLISH_INTERVAL_SEC', '30')),
        'symbols': os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,SOLUSDT').split(','),
        'mode': os.getenv('MODE', 'shadow'),
        'default_action': os.getenv('DEFAULT_ACTION', 'HOLD'),
        'default_conf': float(os.getenv('DEFAULT_CONF', '0.80')),
        'action_map': parse_map(os.getenv('ACTION_MAP', '')),
        'conf_map': parse_conf_map(os.getenv('CONF_MAP', '')),
        'kill_switch': os.getenv('KILL_SWITCH', 'false').lower() == 'true',
        'prefix': os.getenv('PREFIX', 'quantum:rl:policy:'),
        'version': os.getenv('VERSION', 'v2.0'),
        'reason': os.getenv('REASON', 'publisher_v0')
    }


def parse_map(value: str) -> dict:
    """Parse ACTION_MAP: BTCUSDT:BUY,ETHUSDT:SELL"""
    if not value:
        return {}
    result = {}
    for pair in value.split(','):
        if ':' in pair:
            k, v = pair.split(':', 1)
            result[k.strip()] = v.strip()
    return result


def parse_conf_map(value: str) -> dict:
    """Parse CONF_MAP: BTCUSDT:0.85,ETHUSDT:0.78"""
    if not value:
        return {}
    result = {}
    for pair in value.split(','):
        if ':' in pair:
            k, v = pair.split(':', 1)
            result[k.strip()] = float(v.strip())
    return result


def publish_policies(redis_client, config):
    """Publish fresh policies for all symbols."""
    now = int(time.time())
    count = 0
    
    for symbol in config['symbols']:
        symbol = symbol.strip()
        if not symbol:
            continue
        
        action = config['action_map'].get(symbol, config['default_action'])
        confidence = config['conf_map'].get(symbol, config['default_conf'])
        
        policy = {
            'action': action,
            'confidence': confidence,
            'version': config['version'],
            'timestamp': now,
            'reason': config['reason']
        }
        
        key = f"{config['prefix']}{symbol}"
        redis_client.set(key, json.dumps(policy))
        count += 1
    
    return count


def main():
    """Main loop."""
    config = load_env()
    
    # Connect to Redis
    try:
        r = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        r.ping()
        print(f"[RL-POLICY-PUB] ✅ Connected to Redis {config['redis_host']}:{config['redis_port']}")
    except Exception as e:
        print(f"[RL-POLICY-PUB] ❌ Redis connection failed: {e}")
        sys.exit(1)
    
    print(f"[RL-POLICY-PUB] 🚀 Starting publisher: mode={config['mode']}, interval={config['interval']}s, symbols={config['symbols']}")
    
    iteration = 0
    while True:
        iteration += 1
        start = time.time()
        
        if config['kill_switch']:
            print(f"[RL-POLICY-PUB] ⛔ KILL_SWITCH active - skipping publish (iteration {iteration})")
        else:
            try:
                count = publish_policies(r, config)
                elapsed = time.time() - start
                timestamp = datetime.now(timezone.utc).isoformat()
                print(f"[RL-POLICY-PUB] 📢 Published {count} policies in {elapsed:.3f}s | interval={config['interval']}s | kill=false | iteration={iteration} | ts={timestamp}")
            except Exception as e:
                print(f"[RL-POLICY-PUB] ❌ Error publishing policies: {e}")
        
        # Sleep until next interval
        time.sleep(config['interval'])


if __name__ == '__main__':
    main()
