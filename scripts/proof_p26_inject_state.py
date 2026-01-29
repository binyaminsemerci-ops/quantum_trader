#!/usr/bin/env python3
"""
P2.6 Heat Gate - Proof Inject Helper
=====================================
Injects test portfolio state and harvest proposals for proof script.
"""

import sys
import time
import redis
import hashlib


def inject_portfolio_state(
    r: redis.Redis,
    gross_exposure: float,
    dd_ewma: float,
    burst_ewma: float,
    fee_ewma: float,
    churn_ewma: float
):
    """Inject portfolio state to Redis."""
    ts_now = int(time.time())
    
    state_data = {
        "ts_epoch": ts_now,
        "net_exposure_usd": gross_exposure * 0.8,  # Estimate
        "gross_exposure_usd": gross_exposure,
        "dd_ewma_usd": dd_ewma,
        "lossburst_ewma_usd": burst_ewma,
        "fee_burden_ewma_usd": fee_ewma,
        "churn_ewma": churn_ewma
    }
    
    r.hset("quantum:portfolio:state", mapping=state_data)
    print(f"✓ Injected portfolio state: gross={gross_exposure}, dd={dd_ewma}, burst={burst_ewma}")


def inject_harvest_proposal(
    r: redis.Redis,
    symbol: str,
    action: str,
    kill_score: float,
    decision: str = "EXECUTE"
) -> str:
    """Inject harvest proposal to stream. Returns plan_id."""
    ts_now = int(time.time())
    
    # Generate plan_id
    plan_id = hashlib.md5(f"plan_{symbol}_{ts_now}".encode()).hexdigest()[:16]
    
    proposal_data = {
        "ts_epoch": ts_now,
        "symbol": symbol,
        "plan_id": plan_id,
        "action": action,
        "decision": decision,
        "kill_score": kill_score
    }
    
    r.xadd("quantum:stream:harvest.proposal", proposal_data)
    print(f"✓ Injected harvest proposal: {symbol} {action} (plan_id={plan_id}, kill_score={kill_score})")
    
    return plan_id


def clear_state(r: redis.Redis):
    """Clear portfolio state for clean test."""
    r.delete("quantum:portfolio:state")
    print("✓ Cleared portfolio state")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 proof_p26_inject_state.py clear")
        print("  python3 proof_p26_inject_state.py state <gross> <dd> <burst> <fee> <churn>")
        print("  python3 proof_p26_inject_state.py proposal <symbol> <action> <kill_score>")
        sys.exit(1)
    
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    
    cmd = sys.argv[1]
    
    if cmd == "clear":
        clear_state(r)
    
    elif cmd == "state":
        if len(sys.argv) < 7:
            print("ERROR: state requires 5 args: gross dd burst fee churn")
            sys.exit(1)
        
        gross = float(sys.argv[2])
        dd = float(sys.argv[3])
        burst = float(sys.argv[4])
        fee = float(sys.argv[5])
        churn = float(sys.argv[6])
        
        inject_portfolio_state(r, gross, dd, burst, fee, churn)
    
    elif cmd == "proposal":
        if len(sys.argv) < 5:
            print("ERROR: proposal requires 3 args: symbol action kill_score")
            sys.exit(1)
        
        symbol = sys.argv[2]
        action = sys.argv[3]
        kill_score = float(sys.argv[4])
        
        plan_id = inject_harvest_proposal(r, symbol, action, kill_score)
        print(f"Plan ID: {plan_id}")
    
    else:
        print(f"ERROR: Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
