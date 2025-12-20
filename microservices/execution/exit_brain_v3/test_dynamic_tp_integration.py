"""
Test Dynamic TP Integration with Exit Brain V3
"""
import asyncio
import logging
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.models import ExitContext

logging.basicConfig(level=logging.INFO)


async def test_dynamic_tp():
    """Test dynamic TP with XRPUSDT-like position"""
    
    # Create planner with dynamic TP enabled
    config = {
        "use_profiles": True,
        "use_dynamic_tp": True,
        "strategy_id": "RL_V3"
    }
    planner = ExitBrainV3(config=config)
    
    # Test case 1: HIGH LEVERAGE position (like current XRPUSDT)
    print("\n" + "="*80)
    print("TEST 1: HIGH LEVERAGE POSITION (20x)")
    print("="*80)
    
    ctx = ExitContext(
        symbol="XRPUSDT",
        side="LONG",
        entry_price=1.9983,
        current_price=2.0081,
        size=743.0,  # Number of coins
        unrealized_pnl_usd=7.28,
        unrealized_pnl_pct=0.49,
        leverage=20.0,  # HIGH LEVERAGE
        market_regime="NORMAL",
        volatility=0.025,  # 2.5% volatility
        signal_confidence=0.85,  # High confidence
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan = await planner.build_exit_plan(ctx)
    
    print(f"\n📊 Position Info:")
    print(f"   Symbol: {ctx.symbol}")
    print(f"   Entry: ${ctx.entry_price:.4f}")
    print(f"   Current: ${ctx.current_price:.4f}")
    print(f"   Size: {ctx.size:.0f} coins (${ctx.size * ctx.entry_price:.2f} USD)")
    print(f"   Leverage: {ctx.leverage}x")
    print(f"   Volatility: {ctx.volatility:.2%}")
    print(f"   Confidence: {ctx.signal_confidence:.1%}")
    
    print(f"\n🎯 Exit Plan:")
    print(f"   Strategy: {plan.strategy_id}")
    print(f"   Total Legs: {len(plan.legs)}")
    
    tp_legs = [leg for leg in plan.legs if leg.kind.value == "TP"]
    sl_legs = [leg for leg in plan.legs if leg.kind.value == "SL"]
    
    print(f"\n📈 TP Levels ({len(tp_legs)}):")
    for leg in tp_legs:
        tp_price = ctx.entry_price * (1 + leg.trigger_pct)
        is_dynamic = leg.metadata.get("dynamic", False)
        marker = "🎯 DYNAMIC" if is_dynamic else "📊 STATIC"
        print(f"   {marker} {leg.reason}")
        print(f"      Trigger: {leg.trigger_pct:.2%} (${tp_price:.4f})")
        print(f"      Size: {leg.size_pct:.1%}")
        if is_dynamic and "reasoning" in leg.metadata:
            print(f"      Reasoning: {leg.metadata['reasoning']}")
    
    print(f"\n🛑 SL Levels ({len(sl_legs)}):")
    for leg in sl_legs:
        sl_price = ctx.entry_price * (1 + leg.trigger_pct)
        print(f"   {leg.reason}")
        print(f"      Trigger: {leg.trigger_pct:.2%} (${sl_price:.4f})")
    
    # Test case 2: LARGE POSITION
    print("\n" + "="*80)
    print("TEST 2: LARGE POSITION ($5000 USD)")
    print("="*80)
    
    ctx2 = ExitContext(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=95000.0,
        current_price=95500.0,
        size=0.0526,  # ~$5000 position
        unrealized_pnl_usd=26.30,
        unrealized_pnl_pct=0.53,
        leverage=10.0,
        market_regime="NORMAL",
        volatility=0.030,  # 3% volatility
        signal_confidence=0.90,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan2 = await planner.build_exit_plan(ctx2)
    
    print(f"\n📊 Position Info:")
    print(f"   Symbol: {ctx2.symbol}")
    print(f"   Size: ${ctx2.size * ctx2.entry_price:.2f} USD")
    print(f"   Leverage: {ctx2.leverage}x")
    
    tp_legs2 = [leg for leg in plan2.legs if leg.kind.value == "TP"]
    print(f"\n📈 TP Levels ({len(tp_legs2)}):")
    for leg in tp_legs2:
        is_dynamic = leg.metadata.get("dynamic", False)
        marker = "🎯 DYNAMIC" if is_dynamic else "📊 STATIC"
        print(f"   {marker} {leg.reason}")
        if is_dynamic and "reasoning" in leg.metadata:
            print(f"      Reasoning: {leg.metadata['reasoning']}")
    
    # Test case 3: HIGH VOLATILITY + TRENDING
    print("\n" + "="*80)
    print("TEST 3: HIGH VOLATILITY + TRENDING REGIME")
    print("="*80)
    
    ctx3 = ExitContext(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=180.0,
        current_price=182.0,
        size=10.0,  # $1800 position
        unrealized_pnl_usd=20.0,
        unrealized_pnl_pct=1.11,
        leverage=15.0,
        market_regime="TRENDING",  # TRENDING regime
        volatility=0.045,  # 4.5% volatility (HIGH)
        signal_confidence=0.88,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan3 = await planner.build_exit_plan(ctx3)
    
    print(f"\n📊 Position Info:")
    print(f"   Symbol: {ctx3.symbol}")
    print(f"   Regime: {ctx3.market_regime}")
    print(f"   Volatility: {ctx3.volatility:.2%}")
    
    tp_legs3 = [leg for leg in plan3.legs if leg.kind.value == "TP"]
    print(f"\n📈 TP Levels ({len(tp_legs3)}):")
    for leg in tp_legs3:
        is_dynamic = leg.metadata.get("dynamic", False)
        marker = "🎯 DYNAMIC" if is_dynamic else "📊 STATIC"
        print(f"   {marker} {leg.reason}")
        if is_dynamic and "reasoning" in leg.metadata:
            print(f"      Reasoning: {leg.metadata['reasoning']}")
    
    # Test case 4: Compare STATIC vs DYNAMIC
    print("\n" + "="*80)
    print("TEST 4: STATIC vs DYNAMIC COMPARISON")
    print("="*80)
    
    # Static TP
    config_static = {
        "use_profiles": True,
        "use_dynamic_tp": False,  # DISABLE dynamic TP
        "strategy_id": "RL_V3"
    }
    planner_static = ExitBrainV3(config=config_static)
    
    ctx4 = ExitContext(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=3500.0,
        current_price=3520.0,
        size=0.5,  # $1750 position
        unrealized_pnl_usd=10.0,
        unrealized_pnl_pct=0.57,
        leverage=20.0,
        market_regime="NORMAL",
        volatility=0.028,
        signal_confidence=0.82,
        risk_mode="NORMAL",
        trail_enabled=True
    )
    
    plan_static = await planner_static.build_exit_plan(ctx4)
    plan_dynamic = await planner.build_exit_plan(ctx4)
    
    tp_static = [leg for leg in plan_static.legs if leg.kind.value == "TP"]
    tp_dynamic = [leg for leg in plan_dynamic.legs if leg.kind.value == "TP"]
    
    print(f"\n📊 STATIC TP (Profile-based):")
    for i, leg in enumerate(tp_static):
        print(f"   TP{i+1}: {leg.trigger_pct:.2%} @ {leg.size_pct:.0%}")
    
    print(f"\n🎯 DYNAMIC TP (AI-adjusted):")
    for i, leg in enumerate(tp_dynamic):
        print(f"   TP{i+1}: {leg.trigger_pct:.2%} @ {leg.size_pct:.0%}")
    
    print(f"\n💡 Difference:")
    for i in range(len(tp_static)):
        static_pct = tp_static[i].trigger_pct
        dynamic_pct = tp_dynamic[i].trigger_pct
        diff_pct = ((dynamic_pct / static_pct) - 1) * 100 if static_pct > 0 else 0
        print(f"   TP{i+1}: {diff_pct:+.1f}% change ({static_pct:.2%} → {dynamic_pct:.2%})")


if __name__ == "__main__":
    asyncio.run(test_dynamic_tp())
