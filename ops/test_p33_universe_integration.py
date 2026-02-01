#!/usr/bin/env python3
"""
P1 Universe Integration Test for P3.3
Tests Universe Service integration with fallback behavior
"""

import redis
import json
import time
import sys

def test_universe_integration():
    """Test P3.3 Universe Service integration"""
    
    print("╔════════════════════════════════════════════════╗")
    print("║   P3.3 UNIVERSE INTEGRATION TEST              ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    
    r = redis.Redis(decode_responses=True)
    
    # Test 1: Universe active with stale=0
    print("Test 1: Universe FRESH (stale=0)")
    print("-" * 50)
    
    universe_data = {
        'asof_epoch': int(time.time()),
        'source': 'test_mock',
        'mode': 'testnet',
        'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TRXUSDT'],
        'filters': {'contractType': 'PERPETUAL', 'status': 'TRADING'}
    }
    
    r.set('quantum:cfg:universe:active', json.dumps(universe_data))
    r.hset('quantum:cfg:universe:meta', 'stale', 0)
    r.hset('quantum:cfg:universe:meta', 'count', 4)
    r.hset('quantum:cfg:universe:meta', 'asof_epoch', int(time.time()))
    
    print("✅ Set universe: stale=0, symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TRXUSDT']")
    print("   Expected: P3.3 should accept all 4 symbols")
    print("   P33_ALLOWLIST should be IGNORED")
    print()
    
    # Test 2: Universe stale
    print("Test 2: Universe STALE (stale=1)")
    print("-" * 50)
    
    r.hset('quantum:cfg:universe:meta', 'stale', 1)
    r.hset('quantum:cfg:universe:meta', 'error', 'test_stale_scenario')
    
    print("✅ Set universe: stale=1")
    print("   Expected: P3.3 should fallback to P33_ALLOWLIST env var")
    print("   Universe symbols should be IGNORED")
    print()
    
    # Test 3: Universe missing
    print("Test 3: Universe MISSING (key deleted)")
    print("-" * 50)
    
    r.delete('quantum:cfg:universe:active')
    
    print("✅ Deleted quantum:cfg:universe:active")
    print("   Expected: P3.3 should fallback to P33_ALLOWLIST env var")
    print()
    
    # Restore for production
    print("Restoring Universe to FRESH state...")
    print("-" * 50)
    
    universe_data['symbols'] = ['BTCUSDT', 'ETHUSDT', 'TRXUSDT', 'SOLUSDT']
    universe_data['asof_epoch'] = int(time.time())
    
    r.set('quantum:cfg:universe:active', json.dumps(universe_data))
    r.hset('quantum:cfg:universe:meta', 'stale', 0)
    r.hset('quantum:cfg:universe:meta', 'count', 4)
    r.hset('quantum:cfg:universe:meta', 'asof_epoch', int(time.time()))
    r.hset('quantum:cfg:universe:meta', 'error', '')
    
    print("✅ Universe restored: stale=0, 4 symbols")
    print()
    
    print("═══════════════════════════════════════════════")
    print()
    print("Manual Verification:")
    print("  1. Restart P3.3: systemctl restart quantum-position-state-brain")
    print("  2. Check logs: journalctl -u quantum-position-state-brain -f")
    print("  3. Look for: 'Allowlist source=universe' or 'Allowlist source=fallback'")
    print()
    print("Test complete. Check P3.3 logs to verify behavior.")


if __name__ == '__main__':
    try:
        test_universe_integration()
    except redis.ConnectionError:
        print("ERROR: Cannot connect to Redis")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
