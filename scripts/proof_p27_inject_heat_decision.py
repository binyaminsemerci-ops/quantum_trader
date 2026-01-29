#!/usr/bin/env python3
"""
P2.7 HeatBridge - Proof Inject Helper
======================================
Injects heat decision messages into quantum:stream:harvest.heat.decision for testing.
"""

import sys
import time
import redis
import hashlib
import argparse


def inject_heat_decision(
    r: redis.Redis,
    symbol: str = "BTCUSDT",
    plan_id: str = None,
    heat_level: str = "cold",
    heat_action: str = "NONE",
    out_action: str = "FULL_CLOSE_PROPOSED",
    score: float = 0.2,
    partial: float = None
):
    """Inject a heat decision message to the stream."""
    ts_now = int(time.time())
    
    # Generate plan_id if not provided
    if plan_id is None:
        plan_id = hashlib.md5(f"plan_{symbol}_{ts_now}".encode()).hexdigest()[:12]
    
    decision_data = {
        "ts_epoch": ts_now,
        "symbol": symbol,
        "plan_id": plan_id,
        "in_action": "FULL_CLOSE_PROPOSED",
        "out_action": out_action,
        "heat_level": heat_level,
        "heat_score": score,
        "heat_action": heat_action,
        "recommended_partial": partial if partial else "",
        "reason": "ok",
        "inputs_age_sec": "",
        "mode": "shadow",
        "debug_json": "{}"
    }
    
    r.xadd("quantum:stream:harvest.heat.decision", decision_data)
    print(f"✓ Injected: {symbol}/{plan_id} heat_level={heat_level} heat_action={heat_action}")
    print(f"  plan_id={plan_id}")
    
    return plan_id


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inject heat decision into stream for testing"
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--plan_id", default=None, help="Plan ID (auto-generated if not provided)")
    parser.add_argument("--heat_level", default="cold", choices=["cold", "warm", "hot", "unknown"])
    parser.add_argument("--heat_action", default="NONE", choices=["NONE", "DOWNGRADE_FULL_TO_PARTIAL", "HOLD_CLOSE"])
    parser.add_argument("--out_action", default="FULL_CLOSE_PROPOSED")
    parser.add_argument("--score", type=float, default=0.2, help="Heat score 0-1")
    parser.add_argument("--partial", type=float, default=None, help="Recommended partial (0.25/0.50/0.75)")
    
    args = parser.parse_args()
    
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    
    try:
        r.ping()
    except Exception as e:
        print(f"ERROR: Redis not available: {e}")
        sys.exit(1)
    
    plan_id = inject_heat_decision(
        r,
        symbol=args.symbol,
        plan_id=args.plan_id,
        heat_level=args.heat_level,
        heat_action=args.heat_action,
        out_action=args.out_action,
        score=args.score,
        partial=args.partial
    )
    
    print(f"Waiting 2s for HeatBridge to process...")
    time.sleep(2)
    
    # Verify lookup keys were created
    by_plan_key = f"quantum:harvest:heat:by_plan:{plan_id}"
    if r.exists(by_plan_key):
        print(f"✓ Lookup key created: {by_plan_key}")
    else:
        print(f"✗ Lookup key NOT found: {by_plan_key}")
        sys.exit(1)


if __name__ == "__main__":
    main()
