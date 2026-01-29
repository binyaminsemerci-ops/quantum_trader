#!/usr/bin/env python3
"""
Proof Helper: Inject Apply Plan for Testing

Writes quantum:stream:apply.plan entry for governor to process.
Used by proof_p31_step2_governor_downsize.sh.

Usage:
  python3 inject_plan.py PLAN_ID SYMBOL ACTION DECISION

Examples:
  python3 inject_plan.py abc123def456 BTCUSDT OPEN_PROPOSED EXECUTE
  python3 inject_plan.py xyz789uvw012 ETHUSDT FULL_CLOSE_PROPOSED EXECUTE
"""

import sys
import time
import redis
import hashlib

def inject_plan(plan_id: str, symbol: str, action: str, decision: str):
    """Inject plan into Redis stream"""
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    now_ts = int(time.time())
    
    # Plan entry
    plan_entry = {
        "plan_id": plan_id,
        "symbol": symbol,
        "action": action,
        "decision": decision,
        "kill_score": "0.1",
        "side": "BUY",
        "qty": "0.01",
        "timestamp": str(now_ts)
    }
    
    # Add to stream
    stream_id = r.xadd("quantum:stream:apply.plan", plan_entry, maxlen=10000)
    
    print(f"✓ Injected plan {plan_id[:8]}: symbol={symbol} action={action} decision={decision} stream_id={stream_id}")

def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    
    plan_id = sys.argv[1]
    symbol = sys.argv[2]
    action = sys.argv[3]
    decision = sys.argv[4]
    
    inject_plan(plan_id, symbol, action, decision)

if __name__ == "__main__":
    main()
