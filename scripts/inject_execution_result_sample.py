#!/usr/bin/env python3
"""
Execution Result Injector for P3.0 Testing

Publishes sample execution results to quantum:stream:execution.result
for testing P3.0 Performance Attribution Brain attribution computation.

Usage:
    python3 inject_execution_result_sample.py [--count N] [--symbol SYMBOL]
"""

import sys
import time
import argparse
import redis

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# Stream key (matches P3.0 consumption)
STREAM_KEY = 'quantum:stream:execution.result'


def inject_execution_results(count: int = 3, symbol: str = 'BTCUSDT'):
    """Inject sample execution results to Redis stream."""
    
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    
    print(f"=== Execution Result Injector ===")
    print(f"Target: {STREAM_KEY}")
    print(f"Symbol: {symbol}")
    print(f"Count: {count}")
    print()
    
    # Sample execution results with varying P&L
    sample_executions = [
        {
            'symbol': symbol,
            'realized_pnl': '125.50',
            'timestamp': str(int(time.time())),
            'regime': 'BULLISH',
            'cluster': 'MOMENTUM',
            'signal': 'LONG_ENTRY',
        },
        {
            'symbol': symbol,
            'realized_pnl': '-45.20',
            'timestamp': str(int(time.time()) + 1),
            'regime': 'BEARISH',
            'cluster': 'REVERSAL',
            'signal': 'SHORT_ENTRY',
        },
        {
            'symbol': symbol,
            'realized_pnl': '78.90',
            'timestamp': str(int(time.time()) + 2),
            'regime': 'BULLISH',
            'cluster': 'MOMENTUM',
            'signal': 'LONG_EXIT',
        },
        {
            'symbol': symbol,
            'realized_pnl': '210.00',
            'timestamp': str(int(time.time()) + 3),
            'regime': 'TREND',
            'cluster': 'BREAKOUT',
            'signal': 'LONG_ENTRY',
        },
        {
            'symbol': symbol,
            'realized_pnl': '-12.50',
            'timestamp': str(int(time.time()) + 4),
            'regime': 'CHOP',
            'cluster': 'NOISE',
            'signal': 'SHORT_EXIT',
        },
    ]
    
    injected = 0
    for i in range(min(count, len(sample_executions))):
        event = sample_executions[i]
        
        # Add to stream
        entry_id = client.xadd(STREAM_KEY, event)
        
        print(f"✓ Injected execution {i+1}: {symbol} P&L=${event['realized_pnl']} (ID: {entry_id})")
        injected += 1
        
        time.sleep(0.1)  # Small delay between injections
    
    print()
    print(f"=== Injection Complete: {injected} events ===")
    print(f"Stream length: {client.xlen(STREAM_KEY)}")
    print()
    print("P3.0 should process these events in next loop cycle (5s interval)")
    
    return injected


def main():
    parser = argparse.ArgumentParser(description='Inject execution results for P3.0 testing')
    parser.add_argument('--count', type=int, default=3, help='Number of events to inject')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol for execution results')
    
    args = parser.parse_args()
    
    try:
        inject_execution_results(count=args.count, symbol=args.symbol)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
