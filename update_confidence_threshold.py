#!/usr/bin/env python3
"""
Update global_min_confidence to 0.68 in policy store
"""

import sys
sys.path.insert(0, '/app/backend')

import asyncio
import redis.asyncio as redis
from services.policy_store.store import RedisPolicyStore
from services.policy_store.models import GlobalPolicy, RiskMode

async def update_confidence():
    print("\n" + "="*80)
    print("🔧 UPDATING GLOBAL CONFIDENCE THRESHOLD")
    print("="*80 + "\n")
    
    # Connect to Redis
    redis_client = redis.Redis(
        host='redis',
        port=6379,
        db=0,
        decode_responses=True
    )
    
    store = RedisPolicyStore(redis_client)
    
    # Get current policy
    current = await store.get_policy()
    
    print("📊 CURRENT POLICY:")
    print(f"   Risk Mode: {current.risk_mode}")
    print(f"   Global Min Confidence: {current.global_min_confidence}")
    print(f"   Max Risk per Trade: {current.max_risk_per_trade}")
    print(f"   Max Positions: {current.max_positions}")
    print(f"   Max Daily Trades: {current.max_daily_trades}")
    print()
    
    old_confidence = current.global_min_confidence
    
    # Update confidence to 0.68
    new_policy = GlobalPolicy(
        risk_mode=current.risk_mode,
        allowed_strategies=current.allowed_strategies,
        max_risk_per_trade=current.max_risk_per_trade,
        max_positions=current.max_positions,
        global_min_confidence=0.68,  # NEW VALUE
        max_daily_trades=current.max_daily_trades,
        updated_by="admin",
    )
    
    await store.set_policy(new_policy)
    
    print("✅ POLICY UPDATED!")
    print(f"   Confidence: {old_confidence:.2f} → 0.68")
    print()
    
    # Verify update
    verified = await store.get_policy()
    print("🔍 VERIFICATION:")
    print(f"   New Global Min Confidence: {verified.global_min_confidence}")
    print(f"   Status: {'✅ SUCCESS' if verified.global_min_confidence == 0.68 else '❌ FAILED'}")
    print()
    
    print("="*80)
    print("💡 IMPACT:")
    print("="*80)
    print()
    print(f"OLD: Confidence ≥ {old_confidence:.0%} required")
    print(f"     → Very selective, only highest quality signals")
    print()
    print(f"NEW: Confidence ≥ 68% required")
    print(f"     → Slightly more permissive, allows good quality trades")
    print()
    print("EXPECTED RESULT:")
    print("  • More signals will pass confidence filter")
    print("  • Still maintain high quality (68%+ is solid)")
    print("  • Should increase trade frequency slightly")
    print("  • Win rate should remain strong (68%+ confidence)")
    print()
    print("="*80)
    print()
    
    await redis_client.aclose()

if __name__ == "__main__":
    asyncio.run(update_confidence())
