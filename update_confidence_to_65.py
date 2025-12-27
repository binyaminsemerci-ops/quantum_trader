#!/usr/bin/env python3
"""
Update confidence threshold to 0.65 (more permissive)
"""

import asyncio
import sys
sys.path.insert(0, 'backend')

async def update_confidence():
    from backend.services.policy_store.store import RedisPolicyStore
    from backend.services.policy_store.models import GlobalPolicy
    import redis.asyncio as redis
    
    print("📊 Updating confidence threshold to 0.65...")
    
    # Connect to Redis
    redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=False)
    store = RedisPolicyStore(redis_client)
    
    # Get current policy
    policy = await store.get_policy()
    
    print(f"\n📊 CURRENT POLICY:")
    print(f"   Risk Mode: {policy.risk_mode}")
    print(f"   Global Min Confidence: {policy.global_min_confidence}")
    print(f"   Max Risk per Trade: {policy.max_risk_per_trade}")
    print(f"   Max Positions: {policy.max_positions}")
    print(f"   Max Daily Trades: {policy.max_daily_trades}")
    
    # Update confidence
    old_confidence = policy.global_min_confidence
    policy.global_min_confidence = 0.65
    
    # Save
    await store.set_policy(policy)
    
    print(f"\n✅ POLICY UPDATED!")
    print(f"   Confidence: {old_confidence} → 0.65")
    
    # Verify
    updated_policy = await store.get_policy()
    print(f"\n🔍 VERIFICATION:")
    print(f"   New Global Min Confidence: {updated_policy.global_min_confidence}")
    print(f"   Status: {'✅ SUCCESS' if updated_policy.global_min_confidence == 0.65 else '❌ FAILED'}")
    
    print(f"\n💡 IMPACT:")
    print(f"OLD: Confidence ≥ {old_confidence*100:.0f}% required")
    print(f"NEW: Confidence ≥ 65% required")
    print(f"     → More permissive - allows more quality trades")
    
    print(f"\nEXPECTED RESULT:")
    print(f"  • More signals will pass confidence filter")
    print(f"  • Higher trade frequency")
    print(f"  • Still maintain good quality (65%+ is solid)")
    print(f"  • Combined with entry price fix = accurate risk management")

if __name__ == "__main__":
    asyncio.run(update_confidence())
