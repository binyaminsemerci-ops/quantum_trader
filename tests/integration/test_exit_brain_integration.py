"""
Integration tests for Exit Brain v3 integration with Event-Driven Executor and Position Monitor.

Tests verify that:
1. Event-Driven Executor creates Exit Brain plans when placing orders
2. Position Monitor creates retroactive plans for positions without plans
3. Trailing Stop Manager can read Exit Brain plans correctly
"""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Test imports
try:
    from backend.domains.exits.exit_brain_v3.router import ExitRouter
    from backend.domains.exits.exit_brain_v3.models import ExitPlan, ExitLeg, ExitKind
    from backend.domains.exits.exit_brain_v3.integration import to_trailing_config
    EXIT_BRAIN_AVAILABLE = True
except ImportError as e:
    EXIT_BRAIN_AVAILABLE = False
    print(f"⚠️  Exit Brain v3 not available: {e}")


@pytest.mark.skipif(not EXIT_BRAIN_AVAILABLE, reason="Exit Brain v3 not installed")
class TestExitBrainIntegration:
    """Integration tests for Exit Brain v3"""
    
    @pytest.mark.asyncio
    async def test_exit_router_initialization(self):
        """Test that ExitRouter can be initialized"""
        router = ExitRouter()
        assert router is not None
        print("✅ ExitRouter initialized successfully")
    
    @pytest.mark.asyncio
    async def test_create_exit_plan(self):
        """Test creating an exit plan for a position"""
        router = ExitRouter()
        
        # Mock position (simulating a LONG position on XRPUSDT)
        position = {
            "symbol": "XRPUSDT",
            "positionAmt": "100.0",  # Long 100 XRP
            "entryPrice": "0.50",
            "markPrice": "0.50",
            "leverage": "10",
            "unrealizedProfit": "0.0"
        }
        
        # RL hints (from AI decision)
        rl_hints = {
            "tp_target_pct": 0.03,  # 3% TP
            "sl_target_pct": 0.02,  # 2% SL
            "trail_callback_pct": 0.015,  # 1.5% trail
            "confidence": 0.75,
        }
        
        # Risk context
        risk_context = {
            "risk_mode": "NORMAL",
            "daily_pnl_pct": 0.0,
            "position_count": 1,
            "max_positions": 10,
        }
        
        # Market data
        market_data = {
            "current_price": 0.50,
            "volatility": 0.02,
            "atr": 0.01,
            "regime": "NORMAL",
        }
        
        # Create plan
        plan = await router.get_or_create_plan(
            position=position,
            rl_hints=rl_hints,
            risk_context=risk_context,
            market_data=market_data
        )
        
        # Verify plan created
        assert plan is not None, "Plan should not be None"
        assert plan.symbol == "XRPUSDT"
        assert len(plan.legs) > 0, "Plan should have at least one leg"
        
        print(f"✅ Exit plan created: {plan.strategy_id} with {len(plan.legs)} legs")
        
        # Verify plan structure
        for i, leg in enumerate(plan.legs):
            print(f"   Leg {i+1}: {leg.kind.name} @ {leg.trigger_pct*100:+.2f}% ({leg.size_pct*100:.0f}%)")
        
        return plan
    
    @pytest.mark.asyncio
    async def test_plan_caching(self):
        """Test that plans are cached by symbol"""
        router = ExitRouter()
        
        position = {
            "symbol": "BTCUSDT",
            "positionAmt": "0.01",
            "entryPrice": "95000",
            "markPrice": "95000",
            "leverage": "10",
            "unrealizedProfit": "0.0"
        }
        
        # Create plan
        plan1 = await router.get_or_create_plan(
            position=position,
            rl_hints={"tp_target_pct": 0.03, "confidence": 0.70},
            risk_context={"risk_mode": "NORMAL"},
            market_data={"current_price": 95000}
        )
        
        # Get same plan (should be cached)
        plan2 = router.get_active_plan("BTCUSDT")
        
        assert plan1 is not None
        assert plan2 is not None
        assert plan1.symbol == plan2.symbol
        assert plan1.strategy_id == plan2.strategy_id
        
        print(f"✅ Plan caching works: {plan1.strategy_id}")
        
        return plan1
    
    @pytest.mark.asyncio
    async def test_trailing_config_conversion(self):
        """Test converting Exit Brain plan to trailing config"""
        router = ExitRouter()
        
        position = {
            "symbol": "ETHUSDT",
            "positionAmt": "1.0",
            "entryPrice": "3500",
            "markPrice": "3500",
            "leverage": "5",
            "unrealizedProfit": "0.0"
        }
        
        # Create plan
        plan = await router.get_or_create_plan(
            position=position,
            rl_hints={"trail_callback_pct": 0.02, "confidence": 0.80},
            risk_context={"risk_mode": "NORMAL"},
            market_data={"current_price": 3500}
        )
        
        # Convert to trailing config
        trail_config = to_trailing_config(plan, None)
        
        assert trail_config is not None, "Trailing config should not be None"
        assert trail_config.get("enabled"), "Trailing should be enabled"
        assert trail_config.get("callback_pct") > 0, "Callback percentage should be set"
        
        print(f"✅ Trailing config: {trail_config}")
        
        return trail_config
    
    @pytest.mark.asyncio
    async def test_retroactive_plan_creation(self):
        """Test Position Monitor creating plans retroactively"""
        router = ExitRouter()
        
        # Simulate position WITHOUT existing plan
        symbol = "SOLUSDT"
        
        # Verify no plan exists
        existing_plan = router.get_active_plan(symbol)
        assert existing_plan is None, "Should start with no plan"
        
        # Position Monitor would call this
        position = {
            "symbol": symbol,
            "positionAmt": "10.0",
            "entryPrice": "140",
            "markPrice": "145",  # +3.6% profit
            "leverage": "10",
            "unrealizedProfit": "50.0"
        }
        
        # Create retroactive plan
        plan = await router.get_or_create_plan(
            position=position,
            rl_hints={"tp_target_pct": 0.03, "trail_callback_pct": 0.015},
            risk_context={"risk_mode": "NORMAL"},
            market_data={"current_price": 145}
        )
        
        assert plan is not None
        assert plan.symbol == symbol
        
        # Verify plan now cached
        cached_plan = router.get_active_plan(symbol)
        assert cached_plan is not None
        assert cached_plan.symbol == symbol
        
        print(f"✅ Retroactive plan created: {plan.strategy_id}")
        
        return plan


@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow: create plan → cache → retrieve → convert"""
    if not EXIT_BRAIN_AVAILABLE:
        pytest.skip("Exit Brain v3 not available")
    
    print("\n" + "="*60)
    print("FULL WORKFLOW TEST")
    print("="*60)
    
    router = ExitRouter()
    
    # 1. Event-Driven Executor creates plan on order fill
    print("\n1️⃣  Event-Driven Executor: Creating plan on order fill...")
    position = {
        "symbol": "DOTUSDT",
        "positionAmt": "100.0",
        "entryPrice": "2.30",
        "markPrice": "2.30",
        "leverage": "10",
        "unrealizedProfit": "0.0"
    }
    
    plan = await router.get_or_create_plan(
        position=position,
        rl_hints={"tp_target_pct": 0.04, "trail_callback_pct": 0.02, "confidence": 0.85},
        risk_context={"risk_mode": "NORMAL", "position_count": 1},
        market_data={"current_price": 2.30, "volatility": 0.025}
    )
    
    print(f"   ✅ Plan created: {plan.strategy_id} with {len(plan.legs)} legs")
    
    # 2. Trailing Stop Manager retrieves plan
    print("\n2️⃣  Trailing Stop Manager: Retrieving plan...")
    cached_plan = router.get_active_plan("DOTUSDT")
    assert cached_plan is not None
    print(f"   ✅ Plan retrieved from cache")
    
    # 3. Convert to trailing config
    print("\n3️⃣  Converting to trailing config...")
    trail_config = to_trailing_config(cached_plan, None)
    assert trail_config is not None
    assert trail_config.get("enabled")
    print(f"   ✅ Trailing config: callback={trail_config.get('callback_pct')*100:.2f}%")
    
    # 4. Position Monitor checks for plan (finds it)
    print("\n4️⃣  Position Monitor: Checking for plan...")
    existing = router.get_active_plan("DOTUSDT")
    assert existing is not None
    print(f"   ✅ Plan exists - no retroactive creation needed")
    
    # 5. Simulate position without plan (retroactive creation)
    print("\n5️⃣  Simulating position without plan...")
    no_plan = router.get_active_plan("ADAUSDT")
    assert no_plan is None
    print(f"   ⚠️  No plan for ADAUSDT")
    
    print("\n6️⃣  Position Monitor: Creating retroactive plan...")
    retroactive_position = {
        "symbol": "ADAUSDT",
        "positionAmt": "200.0",
        "entryPrice": "0.47",
        "markPrice": "0.50",
        "leverage": "10",
        "unrealizedProfit": "6.0"
    }
    
    retroactive_plan = await router.get_or_create_plan(
        position=retroactive_position,
        rl_hints={"tp_target_pct": 0.03, "trail_callback_pct": 0.015},
        risk_context={"risk_mode": "NORMAL"},
        market_data={"current_price": 0.50}
    )
    
    assert retroactive_plan is not None
    print(f"   ✅ Retroactive plan created: {retroactive_plan.strategy_id}")
    
    print("\n" + "="*60)
    print("✅ FULL WORKFLOW TEST PASSED")
    print("="*60)


if __name__ == "__main__":
    """Run tests directly without pytest"""
    print("🧪 Running Exit Brain v3 Integration Tests\n")
    
    if not EXIT_BRAIN_AVAILABLE:
        print("❌ Exit Brain v3 not available - tests skipped")
        exit(1)
    
    # Run tests
    test_suite = TestExitBrainIntegration()
    
    try:
        print("Test 1: ExitRouter Initialization")
        asyncio.run(test_suite.test_exit_router_initialization())
        print()
        
        print("Test 2: Create Exit Plan")
        asyncio.run(test_suite.test_create_exit_plan())
        print()
        
        print("Test 3: Plan Caching")
        asyncio.run(test_suite.test_plan_caching())
        print()
        
        print("Test 4: Trailing Config Conversion")
        asyncio.run(test_suite.test_trailing_config_conversion())
        print()
        
        print("Test 5: Retroactive Plan Creation")
        asyncio.run(test_suite.test_retroactive_plan_creation())
        print()
        
        print("Test 6: Full Workflow")
        asyncio.run(test_full_workflow())
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
