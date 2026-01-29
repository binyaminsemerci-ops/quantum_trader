#!/usr/bin/env python3
"""
Proof Helper: Inject P3.1 Efficiency Data for Testing

Writes quantum:capital:efficiency:{symbol} hash with specified fields.
Used by proof_p31_step1_allocation_shadow.sh.

Usage:
  python3 inject_efficiency.py SYMBOL SCORE CONFIDENCE [stale]

Examples:
  python3 inject_efficiency.py BTCUSDT 0.9 0.9
  python3 inject_efficiency.py ETHUSDT 0.3 0.5 stale
"""

import sys
import time
import redis

def inject_efficiency(symbol: str, score: float, confidence: float, stale: bool = False):
    """Inject efficiency data into Redis"""
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    now_ts = int(time.time())
    
    # If stale requested, set timestamp 700s ago (beyond 600s threshold)
    if stale:
        now_ts -= 700
    
    key = f"quantum:capital:efficiency:{symbol}"
    
    data = {
        "efficiency_score": str(score),
        "confidence": str(confidence),
        "ts": str(now_ts),
        "mode": "enforce",
        "version": "p31_v1"
    }
    
    r.hset(key, mapping=data)
    r.expire(key, 600)
    
    print(f"✓ Injected efficiency for {symbol}: score={score:.2f} conf={confidence:.2f} ts={now_ts} stale={stale}")

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    
    symbol = sys.argv[1]
    score = float(sys.argv[2])
    confidence = float(sys.argv[3])
    stale = len(sys.argv) > 4 and sys.argv[4] == 'stale'
    
    inject_efficiency(symbol, score, confidence, stale)

if __name__ == "__main__":
    main()
