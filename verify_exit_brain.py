#!/usr/bin/env python3
"""
Quick verification script for Exit Brain v3 deployment.
Tests that Exit Brain v3 is properly enabled and functional on VPS.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/qt/quantum_trader')

def check_exit_brain_availability():
    """Check if Exit Brain v3 modules can be imported"""
    print("1️⃣  Checking Exit Brain v3 availability...")
    
    try:
        from backend.domains.exits.exit_brain_v3.router import ExitRouter
        from backend.domains.exits.exit_brain_v3.models import ExitPlan
        print("   ✅ Exit Brain v3 modules available")
        return True
    except ImportError as e:
        print(f"   ❌ Exit Brain v3 not available: {e}")
        return False


def check_exit_brain_enabled():
    """Check if EXIT_BRAIN_V3_ENABLED is set"""
    print("\n2️⃣  Checking EXIT_BRAIN_V3_ENABLED environment variable...")
    
    enabled = os.getenv("EXIT_BRAIN_V3_ENABLED", "false").lower() == "true"
    
    if enabled:
        print("   ✅ EXIT_BRAIN_V3_ENABLED=true")
    else:
        print("   ❌ EXIT_BRAIN_V3_ENABLED=false or not set")
    
    return enabled


async def test_exit_router_initialization():
    """Test that ExitRouter can be initialized"""
    print("\n3️⃣  Testing ExitRouter initialization...")
    
    try:
        from backend.domains.exits.exit_brain_v3.router import ExitRouter
        
        router = ExitRouter()
        print("   ✅ ExitRouter initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ ExitRouter initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_create_simple_plan():
    """Test creating a simple exit plan"""
    print("\n4️⃣  Testing exit plan creation...")
    
    try:
        from backend.domains.exits.exit_brain_v3.router import ExitRouter
        
        router = ExitRouter()
        
        # Simple test position
        position = {
            "symbol": "XRPUSDT",
            "positionAmt": "100.0",
            "entryPrice": "0.50",
            "markPrice": "0.50",
            "leverage": "10",
            "unrealizedProfit": "0.0"
        }
        
        # Create plan
        plan = await router.get_or_create_plan(
            position=position,
            rl_hints={"tp_target_pct": 0.03, "confidence": 0.75},
            risk_context={"risk_mode": "NORMAL"},
            market_data={"current_price": 0.50}
        )
        
        if plan:
            print(f"   ✅ Plan created: {plan.strategy_id} with {len(plan.legs)} legs")
            return True
        else:
            print("   ❌ Plan creation returned None")
            return False
            
    except Exception as e:
        print(f"   ❌ Plan creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks"""
    print("="*60)
    print("EXIT BRAIN V3 VERIFICATION")
    print("="*60)
    
    # Check 1: Availability
    available = check_exit_brain_availability()
    if not available:
        print("\n❌ VERIFICATION FAILED: Exit Brain v3 not available")
        return False
    
    # Check 2: Enabled
    enabled = check_exit_brain_enabled()
    if not enabled:
        print("\n⚠️  WARNING: Exit Brain v3 available but not enabled")
        print("   Set EXIT_BRAIN_V3_ENABLED=true in environment")
    
    # Check 3: Router initialization
    import asyncio
    router_ok = asyncio.run(test_exit_router_initialization())
    if not router_ok:
        print("\n❌ VERIFICATION FAILED: ExitRouter initialization failed")
        return False
    
    # Check 4: Plan creation
    plan_ok = asyncio.run(test_create_simple_plan())
    if not plan_ok:
        print("\n❌ VERIFICATION FAILED: Plan creation failed")
        return False
    
    # All checks passed
    print("\n" + "="*60)
    print("✅ ALL VERIFICATION CHECKS PASSED")
    print("="*60)
    print("\nExit Brain v3 Status:")
    print(f"  • Available: ✅")
    print(f"  • Enabled: {'✅' if enabled else '⚠️  Not enabled (set EXIT_BRAIN_V3_ENABLED=true)'}")
    print(f"  • Router: ✅")
    print(f"  • Plan Creation: ✅")
    print("\nExit Brain v3 is ready to use!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
