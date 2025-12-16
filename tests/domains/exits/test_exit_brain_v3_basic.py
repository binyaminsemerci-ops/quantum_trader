"""
Tests for Exit Brain v3 - Basic functionality
"""

import pytest
from datetime import datetime, timezone

from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitKind
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3


@pytest.mark.asyncio
async def test_build_plan_normal_regime():
    """Test standard 3-leg plan in normal market regime (legacy mode)"""
    brain = ExitBrainV3(config={"use_profiles": False})  # Use legacy mode
    
    ctx = ExitContext(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        size=1.0,
        leverage=10.0,
        current_price=50100.0,
        unrealized_pnl_pct=0.2,
        market_regime="NORMAL",
        risk_mode="NORMAL"
    )
    
    plan = await brain.build_exit_plan(ctx)
    
    # Should have 3 TP legs + 1 SL
    assert len(plan.legs) == 4
    assert plan.strategy_id == "STANDARD_LADDER"
    
    # Check leg structure
    tp_legs = plan.get_legs_by_kind(ExitKind.TP)
    sl_legs = plan.get_legs_by_kind(ExitKind.SL)
    
    assert len(tp_legs) == 2  # TP1 and TP2
    assert len(sl_legs) == 1
    
    # Check trailing
    assert plan.has_trailing is True
    trail_legs = plan.get_legs_by_kind(ExitKind.TRAIL)
    assert len(trail_legs) == 1
    assert trail_legs[0].size_pct == 0.50


@pytest.mark.asyncio
async def test_build_plan_with_rl_hints():
    """Test plan incorporates RL hints (legacy mode)"""
    brain = ExitBrainV3(config={"use_profiles": False})  # Use legacy mode
    
    ctx = ExitContext(
        symbol="ETHUSDT",
        side="SHORT",
        entry_price=3000.0,
        size=10.0,
        leverage=5.0,
        current_price=2950.0,
        unrealized_pnl_pct=3.33,
        rl_tp_hint=0.05,  # RL suggests 5% TP
        rl_sl_hint=0.02,  # RL suggests 2% SL
        rl_confidence=0.85,
        market_regime="NORMAL",
        risk_mode="NORMAL"
    )
    
    plan = await brain.build_exit_plan(ctx)
    
    # Should use RL hints as base
    primary_tp = plan.get_primary_tp()
    primary_sl = plan.get_primary_sl()
    
    assert primary_tp is not None
    assert primary_sl is not None
    
    # TP should be influenced by RL hint (5%)
    # With default template: TP1=50%*5%=2.5%, TP2=100%*5%=5.0%
    assert 0.02 <= primary_tp.trigger_pct <= 0.06


@pytest.mark.asyncio
async def test_build_plan_critical_risk_mode():
    """Test plan tightens in CRITICAL risk mode"""
    brain = ExitBrainV3()
    
    ctx_normal = ExitContext(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        size=100.0,
        leverage=20.0,
        current_price=102.0,
        unrealized_pnl_pct=0.4,
        market_regime="NORMAL",
        risk_mode="NORMAL"
    )
    
    ctx_critical = ExitContext(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        size=100.0,
        leverage=20.0,
        current_price=102.0,
        unrealized_pnl_pct=0.4,
        market_regime="NORMAL",
        risk_mode="CRITICAL"
    )
    
    plan_normal = await brain.build_exit_plan(ctx_normal)
    plan_critical = await brain.build_exit_plan(ctx_critical)
    
    # Critical mode should have tighter TP (lower percentage)
    tp_normal = plan_normal.get_primary_tp().trigger_pct
    tp_critical = plan_critical.get_primary_tp().trigger_pct
    
    assert tp_critical < tp_normal
    
    # Critical mode should have tighter SL (smaller loss allowed)
    sl_normal = abs(plan_normal.get_primary_sl().trigger_pct)
    sl_critical = abs(plan_critical.get_primary_sl().trigger_pct)
    
    assert sl_critical < sl_normal


@pytest.mark.asyncio
async def test_build_plan_ess_active():
    """Test emergency exit in ESS mode"""
    brain = ExitBrainV3()
    
    ctx = ExitContext(
        symbol="BNBUSDT",
        side="LONG",
        entry_price=400.0,
        size=50.0,
        leverage=10.0,
        current_price=398.0,
        unrealized_pnl_pct=-0.5,
        market_regime="VOLATILE",
        risk_mode="ESS_ACTIVE"  # Emergency!
    )
    
    plan = await brain.build_exit_plan(ctx)
    
    # Should have emergency leg
    assert plan.has_emergency is True
    assert plan.strategy_id == "EMERGENCY_EXIT"
    
    emergency_legs = plan.get_legs_by_kind(ExitKind.EMERGENCY)
    assert len(emergency_legs) == 1
    assert emergency_legs[0].size_pct == 1.0  # Close entire position


@pytest.mark.asyncio
async def test_build_plan_profit_lock():
    """Test profit locking when PnL > 10%"""
    brain = ExitBrainV3()
    
    ctx = ExitContext(
        symbol="ADAUSDT",
        side="LONG",
        entry_price=0.50,
        size=1000.0,
        leverage=15.0,
        current_price=0.56,
        unrealized_pnl_pct=18.0,  # High profit!
        market_regime="TRENDING",
        risk_mode="NORMAL"
    )
    
    plan = await brain.build_exit_plan(ctx)
    
    # Should use profit lock strategy
    assert plan.strategy_id == "PROFIT_LOCK"
    
    # SL should be at breakeven or small profit
    sl_leg = plan.get_primary_sl()
    assert sl_leg.trigger_pct >= 0.0  # Above entry (breakeven+)
    assert sl_leg.trigger_pct < 0.01  # But not too far
