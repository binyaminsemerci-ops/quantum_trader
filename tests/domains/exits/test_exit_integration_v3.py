"""
Tests for Exit Brain v3 Integration - Compatibility with existing systems
"""

import pytest
from unittest.mock import Mock

from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitKind, ExitLeg, ExitPlan
from backend.domains.exits.exit_brain_v3.integration import (
    to_dynamic_tpsl,
    to_trailing_config,
    to_partial_exit_config,
    build_context_from_position
)


def test_to_dynamic_tpsl():
    """Test conversion to dynamic_tpsl format"""
    plan = ExitPlan(
        symbol="BTCUSDT",
        legs=[
            ExitLeg(kind=ExitKind.TP, size_pct=0.25, trigger_pct=0.015, priority=0, reason="TP1"),
            ExitLeg(kind=ExitKind.TP, size_pct=0.25, trigger_pct=0.03, priority=1, reason="TP2"),
            ExitLeg(kind=ExitKind.TRAIL, size_pct=0.50, trail_callback=0.02, priority=2, reason="TP3"),
            ExitLeg(kind=ExitKind.SL, size_pct=1.0, trigger_pct=-0.025, priority=0, reason="SL"),
        ],
        strategy_id="STANDARD",
        source="EXIT_BRAIN_V3",
        reason="Test"
    )
    
    ctx = ExitContext(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        size=1.0
    )
    
    result = to_dynamic_tpsl(plan, ctx)
    
    assert "tp_percent" in result
    assert "sl_percent" in result
    assert "trail_percent" in result
    assert "partial_tp" in result
    
    assert result["tp_percent"] == 0.015  # Primary TP (TP1)
    assert result["sl_percent"] == 0.025  # Absolute value
    assert result["trail_percent"] == 0.02
    assert result["partial_tp"] is True  # Multiple TP legs


def test_to_trailing_config():
    """Test conversion to trailing stop configuration"""
    plan = ExitPlan(
        symbol="ETHUSDT",
        legs=[
            ExitLeg(kind=ExitKind.TP, size_pct=0.50, trigger_pct=0.02, priority=0, reason="TP1"),
            ExitLeg(kind=ExitKind.TRAIL, size_pct=0.50, trail_callback=0.015, priority=1, reason="Trail"),
        ],
        strategy_id="TRAIL",
        source="EXIT_BRAIN_V3",
        reason="Test"
    )
    
    ctx = ExitContext(
        symbol="ETHUSDT",
        side="SHORT",
        entry_price=3000.0,
        size=10.0
    )
    
    result = to_trailing_config(plan, ctx)
    
    assert result is not None
    assert result["enabled"] is True
    assert result["callback_pct"] == 0.015
    assert result["size_pct"] == 0.50
    assert "activation_pct" in result


def test_to_trailing_config_no_trail():
    """Test returns None when no trailing configured"""
    plan = ExitPlan(
        symbol="SOLUSDT",
        legs=[
            ExitLeg(kind=ExitKind.TP, size_pct=1.0, trigger_pct=0.03, priority=0, reason="TP"),
            ExitLeg(kind=ExitKind.SL, size_pct=1.0, trigger_pct=-0.025, priority=0, reason="SL"),
        ],
        strategy_id="SIMPLE",
        source="EXIT_BRAIN_V3",
        reason="Test"
    )
    
    ctx = ExitContext(symbol="SOLUSDT", side="LONG", entry_price=100.0, size=100.0)
    
    result = to_trailing_config(plan, ctx)
    assert result is None


def test_to_partial_exit_config():
    """Test conversion to partial exit ladder"""
    plan = ExitPlan(
        symbol="ADAUSDT",
        legs=[
            ExitLeg(kind=ExitKind.TP, size_pct=0.25, trigger_pct=0.01, priority=0, reason="TP1"),
            ExitLeg(kind=ExitKind.TP, size_pct=0.25, trigger_pct=0.025, priority=1, reason="TP2"),
            ExitLeg(kind=ExitKind.TP, size_pct=0.50, trigger_pct=0.05, priority=2, reason="TP3"),
            ExitLeg(kind=ExitKind.SL, size_pct=1.0, trigger_pct=-0.02, priority=0, reason="SL"),
        ],
        strategy_id="LADDER",
        source="EXIT_BRAIN_V3",
        reason="Test"
    )
    
    ctx = ExitContext(symbol="ADAUSDT", side="LONG", entry_price=0.50, size=1000.0)
    
    result = to_partial_exit_config(plan, ctx)
    
    assert len(result) == 3  # 3 TP legs (SL not included)
    assert result[0]["trigger_pct"] == 0.01  # Sorted ascending
    assert result[1]["trigger_pct"] == 0.025
    assert result[2]["trigger_pct"] == 0.05
    assert all(item["kind"] == "TP" for item in result)


def test_build_context_from_position():
    """Test building ExitContext from Binance position"""
    position = {
        "symbol": "BTCUSDT",
        "positionAmt": "1.5",  # LONG
        "entryPrice": "50000.0",
        "markPrice": "51000.0",
        "leverage": "10"
    }
    
    rl_hints = {
        "tp_pct": 0.04,
        "sl_pct": 0.02,
        "confidence": 0.75
    }
    
    risk_context = {
        "mode": "CONSERVATIVE",
        "max_loss": 0.02
    }
    
    ctx = build_context_from_position(position, rl_hints, risk_context)
    
    assert ctx.symbol == "BTCUSDT"
    assert ctx.side == "LONG"
    assert ctx.entry_price == 50000.0
    assert ctx.current_price == 51000.0
    assert ctx.size == 1.5
    assert ctx.leverage == 10.0
    assert ctx.rl_tp_hint == 0.04
    assert ctx.rl_sl_hint == 0.02
    assert ctx.rl_confidence == 0.75
    assert ctx.risk_mode == "CONSERVATIVE"
    assert ctx.unrealized_pnl_pct > 0  # In profit


def test_build_context_short_position():
    """Test building context for SHORT position"""
    position = {
        "symbol": "ETHUSDT",
        "positionAmt": "-10.0",  # SHORT
        "entryPrice": "3000.0",
        "markPrice": "2900.0",  # Price dropped → profit
        "leverage": "5"
    }
    
    ctx = build_context_from_position(position)
    
    assert ctx.side == "SHORT"
    assert ctx.size == 10.0  # Absolute value
    assert ctx.unrealized_pnl_pct > 0  # Profitable short
