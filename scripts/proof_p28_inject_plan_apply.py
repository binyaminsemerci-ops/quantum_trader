#!/usr/bin/env python3
"""
P2.8A Apply Heat Observer - Proof Injection Script

Injects synthetic harvest proposal + HeatBridge heat key for testing.
Redis-only, no exchange credentials required.
"""

import sys
import time
import json
import hashlib
import argparse
import redis


def inject_plan_with_heat(
    symbol: str = "BTCUSDT",
    action: str = "FULL_CLOSE_PROPOSED",
    kill_score: float = 0.25,
    heat_level: str = "warm",
    heat_action: str = "DOWNGRADE_FULL_TO_PARTIAL",
    heat_out_action: str = "PARTIAL_50_PROPOSED",
    redis_host: str = "localhost",
    redis_port: int = 6379
):
    """
    Inject a synthetic harvest proposal and corresponding HeatBridge key.
    
    Returns:
        plan_id (str): Generated plan ID for verification
    """
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Generate stable plan_id (matches Apply's create_plan_id logic)
    ts_now = int(time.time())
    computed_at_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_now))
    fingerprint = f"{symbol}:{action}:{kill_score:.6f}:none:{computed_at_utc}"
    plan_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    print(f"Injecting plan: {plan_id[:16]}")
    print(f"  Symbol: {symbol}")
    print(f"  Action: {action}")
    print(f"  Kill score: {kill_score}")
    print(f"  Heat level: {heat_level}")
    print(f"  Heat action: {heat_action}")
    print()
    
    # 1. Inject harvest proposal to Redis hash
    proposal_key = f"quantum:harvest:{symbol}:proposal"
    proposal_data = {
        "symbol": symbol,
        "harvest_action": action,
        "kill_score": str(kill_score),
        "k_regime_flip": "0.05",
        "k_sigma_spike": "0.10",
        "k_ts_drop": "0.05",
        "k_age_penalty": "0.05",
        "new_sl_proposed": "",
        "R_net": "2.5",
        "last_update_epoch": str(ts_now),
        "computed_at_utc": computed_at_utc,
        "reason_codes": "test_injection"
    }
    
    # Clear existing proposal
    r.delete(proposal_key)
    
    # Write new proposal
    r.hset(proposal_key, mapping=proposal_data)
    print(f"✓ Harvest proposal written: {proposal_key}")
    
    # 2. Inject HeatBridge by_plan key
    heat_key = f"quantum:harvest:heat:by_plan:{plan_id}"
    heat_data = {
        "ts_epoch": str(ts_now),
        "symbol": symbol,
        "plan_id": plan_id,
        "in_action": action,
        "out_action": heat_out_action,
        "heat_level": heat_level,
        "heat_score": str(kill_score),
        "heat_action": heat_action,
        "recommended_partial": "0.5" if "PARTIAL_50" in heat_out_action else "",
        "reason": "ok",
        "inputs_age_sec": "",
        "mode": "shadow",
        "debug_json": json.dumps({"test": "p28_injection"})
    }
    
    # Clear existing heat key
    r.delete(heat_key)
    
    # Write new heat key with TTL
    r.hset(heat_key, mapping=heat_data)
    r.expire(heat_key, 1800)  # 30 min TTL
    print(f"✓ HeatBridge key written: {heat_key}")
    print(f"  TTL: 1800s (30 min)")
    print()
    
    print(f"Injection complete. Plan ID: {plan_id}")
    print()
    print("Apply will process this proposal on next cycle (typically 5s).")
    print()
    print("Verify with:")
    print(f"  redis-cli HGETALL {proposal_key}")
    print(f"  redis-cli HGETALL {heat_key}")
    print(f"  redis-cli XREVRANGE quantum:stream:apply.heat.observed + - COUNT 1")
    print()
    
    return plan_id


def inject_plan_without_heat(
    symbol: str = "ETHUSDT",
    action: str = "FULL_CLOSE_PROPOSED",
    kill_score: float = 0.15,
    redis_host: str = "localhost",
    redis_port: int = 6379
):
    """
    Inject a synthetic harvest proposal WITHOUT heat key (for testing heat_found=0).
    
    Returns:
        plan_id (str): Generated plan ID for verification
    """
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Generate stable plan_id
    ts_now = int(time.time())
    computed_at_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_now))
    fingerprint = f"{symbol}:{action}:{kill_score:.6f}:none:{computed_at_utc}"
    plan_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    print(f"Injecting plan WITHOUT heat: {plan_id[:16]}")
    print(f"  Symbol: {symbol}")
    print(f"  Action: {action}")
    print(f"  Kill score: {kill_score}")
    print(f"  Heat key: NOT CREATED (testing heat_found=0)")
    print()
    
    # Inject only harvest proposal (no heat key)
    proposal_key = f"quantum:harvest:{symbol}:proposal"
    proposal_data = {
        "symbol": symbol,
        "harvest_action": action,
        "kill_score": str(kill_score),
        "k_regime_flip": "0.03",
        "k_sigma_spike": "0.05",
        "k_ts_drop": "0.03",
        "k_age_penalty": "0.04",
        "new_sl_proposed": "",
        "R_net": "1.8",
        "last_update_epoch": str(ts_now),
        "computed_at_utc": computed_at_utc,
        "reason_codes": "test_no_heat"
    }
    
    r.delete(proposal_key)
    r.hset(proposal_key, mapping=proposal_data)
    print(f"✓ Harvest proposal written: {proposal_key}")
    print()
    
    print(f"Injection complete. Plan ID: {plan_id}")
    print()
    print("Apply should emit observed event with heat_found=0 and heat_reason=missing.")
    print()
    
    return plan_id


def main():
    parser = argparse.ArgumentParser(description="Inject test plan + heat for P2.8A proof")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--action", default="FULL_CLOSE_PROPOSED", help="Harvest action")
    parser.add_argument("--kill_score", type=float, default=0.25, help="Kill score")
    parser.add_argument("--heat_level", default="warm", choices=["cold", "warm", "hot"], help="Heat level")
    parser.add_argument("--heat_action", default="DOWNGRADE_FULL_TO_PARTIAL", help="Heat action")
    parser.add_argument("--no_heat", action="store_true", help="Inject without heat key (test heat_found=0)")
    parser.add_argument("--redis_host", default="localhost", help="Redis host")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis port")
    
    args = parser.parse_args()
    
    if args.no_heat:
        plan_id = inject_plan_without_heat(
            symbol=args.symbol,
            action=args.action,
            kill_score=args.kill_score,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
    else:
        heat_out_action = "PARTIAL_50_PROPOSED" if args.heat_action == "DOWNGRADE_FULL_TO_PARTIAL" else args.action
        
        plan_id = inject_plan_with_heat(
            symbol=args.symbol,
            action=args.action,
            kill_score=args.kill_score,
            heat_level=args.heat_level,
            heat_action=args.heat_action,
            heat_out_action=heat_out_action,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
