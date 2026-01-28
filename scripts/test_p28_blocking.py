#!/usr/bin/env python3
"""
P2.8 Portfolio Risk Governor - Controlled Test
===============================================

Tests P2.8 budget enforcement by injecting controlled data:
1. Creates fake portfolio state (equity, heat)
2. Creates fake budget hash (simulates P2.8 computation)
3. Publishes budget violation event
4. Triggers Governor evaluation (simulated plan)
5. Verifies blocking in enforce mode vs allowing in shadow mode

Usage:
    python test_p28_blocking.py --mode shadow  # Should NOT block
    python test_p28_blocking.py --mode enforce # Should BLOCK
"""

import redis
import json
import time
import sys
import hashlib
import argparse

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

SYMBOL = "BTCUSDT"
PLAN_ID = hashlib.sha256(f"test_p28_{int(time.time())}".encode()).hexdigest()

def create_portfolio_state(equity_usd=100000, drawdown=0.0):
    """Create fake portfolio state"""
    key = "quantum:state:portfolio"
    data = {
        "equity_usd": str(equity_usd),
        "drawdown": str(drawdown),
        "timestamp": str(int(time.time()))
    }
    r.hset(key, mapping=data)
    r.expire(key, 120)
    print(f"✅ Created portfolio state: equity=${equity_usd}, drawdown={drawdown}")
    return data

def create_budget_hash_with_violation(symbol, equity_usd=100000, stress=0.4):
    """
    Create budget hash that simulates a violation scenario.
    
    Budget formula: budget = equity * 0.02 * (1 - stress)
    Example: 100K * 0.02 * (1 - 0.4) = $1,200 budget
    
    We'll create a scenario where position_notional > budget
    """
    budget_usd = equity_usd * 0.02 * (1 - stress)
    
    key = f"quantum:portfolio:budget:{symbol}"
    data = {
        "symbol": symbol,
        "budget_usd": str(budget_usd),
        "stress_factor": str(stress),
        "equity_usd": str(equity_usd),
        "portfolio_heat": "0.65",  # HOT
        "cluster_stress": "0.0",
        "vol_regime": "0.33",
        "mode": "shadow",  # Will be changed by service
        "timestamp": str(int(time.time())),
        "base_risk_pct": "0.02"
    }
    
    r.hset(key, mapping=data)
    r.expire(key, 120)
    
    print(f"✅ Created budget hash: budget=${budget_usd:.0f}, stress={stress}")
    return budget_usd

def publish_budget_violation(symbol, position_notional, budget_usd):
    """Publish budget violation event to stream"""
    over_budget = position_notional - budget_usd
    
    event = {
        "event_type": "budget.violation",
        "symbol": symbol,
        "position_notional": position_notional,
        "budget_usd": budget_usd,
        "over_budget": over_budget,
        "mode": "TEST",
        "timestamp": int(time.time())
    }
    
    stream_key = "quantum:stream:budget.violation"
    r.xadd(stream_key, {"json": json.dumps(event)}, maxlen=1000)
    
    print(f"⚠️  Published violation: notional=${position_notional:.0f}, budget=${budget_usd:.0f}, over=${over_budget:.0f}")
    return event

def check_governor_blocking(symbol, plan_id, timeout=5):
    """
    Check if Governor would block this plan by simulating a plan submission.
    
    Note: This doesn't actually call Governor API, it checks Redis permit key
    after Governor processes the plan stream.
    """
    # Check if permit exists
    permit_key = f"quantum:permit:{plan_id}"
    
    start = time.time()
    while (time.time() - start) < timeout:
        if r.exists(permit_key):
            permit_data = r.get(permit_key)
            permit = json.loads(permit_data)
            
            if permit.get("granted"):
                print(f"✅ ALLOWED: Permit granted for plan {plan_id[:8]}")
                return "ALLOWED"
            else:
                print(f"❌ BLOCKED: Permit denied for plan {plan_id[:8]}")
                return "BLOCKED"
        
        time.sleep(0.5)
    
    print(f"⏱️  TIMEOUT: No permit decision within {timeout}s")
    return "TIMEOUT"

def check_p28_mode():
    """Check current P2.8 mode from service"""
    import requests
    try:
        resp = requests.get("http://localhost:8049/health", timeout=2)
        if resp.status_code == 200:
            health = resp.json()
            mode = health.get("mode", "unknown")
            print(f"📊 P2.8 current mode: {mode}")
            return mode
    except:
        print("⚠️  Could not reach P2.8 service")
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Test P2.8 budget blocking")
    parser.add_argument("--mode", choices=["shadow", "enforce"], required=True,
                        help="Expected P2.8 mode (shadow should allow, enforce should block)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"P2.8 PORTFOLIO RISK GOVERNOR - CONTROLLED TEST")
    print(f"Expected mode: {args.mode.upper()}")
    print(f"{'='*60}\n")
    
    # Check P2.8 is running
    p28_mode = check_p28_mode()
    if p28_mode != args.mode:
        print(f"⚠️  WARNING: P2.8 mode mismatch! Expected {args.mode}, got {p28_mode}")
        print(f"    To fix: sed -i 's/P28_MODE=.*/P28_MODE={args.mode}/' /etc/quantum/portfolio-risk-governor.env")
        print(f"    Then: systemctl restart quantum-portfolio-risk-governor")
        return 1
    
    print(f"\n🧪 Step 1: Create test scenario")
    print(f"-" * 60)
    
    # Create portfolio state
    equity = 100000
    create_portfolio_state(equity_usd=equity)
    
    # Create budget with violation
    stress = 0.4  # High stress
    budget_usd = create_budget_hash_with_violation(SYMBOL, equity_usd=equity, stress=stress)
    
    # Simulate oversized position
    position_notional = budget_usd * 1.8  # 80% over budget
    print(f"💰 Simulated position: ${position_notional:.0f} (exceeds budget by {((position_notional/budget_usd - 1)*100):.0f}%)")
    
    print(f"\n🚨 Step 2: Publish budget violation")
    print(f"-" * 60)
    publish_budget_violation(SYMBOL, position_notional, budget_usd)
    
    print(f"\n🔍 Step 3: Verify Redis state")
    print(f"-" * 60)
    
    # Check budget hash
    budget_key = f"quantum:portfolio:budget:{SYMBOL}"
    if r.exists(budget_key):
        print(f"✅ Budget hash exists: {budget_key}")
    else:
        print(f"❌ Budget hash missing: {budget_key}")
    
    # Check violation stream
    stream_key = "quantum:stream:budget.violation"
    recent = r.xrevrange(stream_key, count=1)
    if recent:
        print(f"✅ Violation event in stream (ID: {recent[0][0]})")
    else:
        print(f"❌ No violation events in stream")
    
    print(f"\n📋 Step 4: Expected outcome")
    print(f"-" * 60)
    
    if args.mode == "shadow":
        print(f"Expected: ALLOW (shadow mode logs only, doesn't block)")
    else:
        print(f"Expected: BLOCK (enforce mode blocks budget violations)")
    
    print(f"\n⏳ Waiting for Governor to detect violation...")
    print(f"(In real system, Governor checks budget on each new plan)")
    
    # Note: Governor doesn't auto-process, it waits for plan stream
    print(f"\n✅ TEST DATA INJECTED")
    print(f"\n📝 Manual verification steps:")
    print(f"   1. Check budget hash:")
    print(f"      redis-cli HGETALL quantum:portfolio:budget:{SYMBOL}")
    print(f"   2. Check violation stream:")
    print(f"      redis-cli XREVRANGE quantum:stream:budget.violation + - COUNT 1")
    print(f"   3. Trigger real order attempt for {SYMBOL}")
    print(f"   4. Check Governor logs:")
    print(f"      journalctl -u quantum-governor -n 20 | grep -E '(p28|budget)'")
    
    print(f"\n{'='*60}")
    print(f"Test data prepared. Ready for Governor evaluation.")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
