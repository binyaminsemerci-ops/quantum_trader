"""
Unit tests for Position Invariant Enforcer.

Tests the critical bug fix: preventing simultaneous long/short positions on same symbol.
"""

import pytest
from datetime import datetime, timezone
from backend.services.execution.position_invariant import (
    PositionInvariantEnforcer,
    PositionInvariantViolation,
    BlockedOrderEvent,
    get_position_invariant_enforcer,
    reset_enforcer,
)


class TestPositionInvariantEnforcer:
    """Test position invariant enforcement logic."""
    
    def setup_method(self):
        """Reset global enforcer before each test."""
        reset_enforcer()
    
    def test_initialization_default(self):
        """Test enforcer initializes with hedging disabled by default."""
        enforcer = PositionInvariantEnforcer()
        assert enforcer.allow_hedging is False
        assert len(enforcer.blocked_orders) == 0
    
    def test_initialization_with_hedging(self):
        """Test enforcer can be initialized with hedging enabled."""
        enforcer = PositionInvariantEnforcer(allow_hedging=True)
        assert enforcer.allow_hedging is True
    
    def test_allow_long_when_flat(self):
        """Should allow opening LONG when no position exists."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is True
        assert reason is None
    
    def test_allow_short_when_flat(self):
        """Should allow opening SHORT when no position exists."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="ETHUSDT",
            side="sell",
            quantity=0.1,
            current_positions=current_positions
        )
        
        assert can_open is True
        assert reason is None
    
    def test_block_long_when_short_exists(self):
        """Should BLOCK opening LONG when SHORT position exists - CRITICAL BUG FIX."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": -0.001}  # Existing SHORT
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="buy",  # Try to open LONG
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is False
        assert reason is not None
        assert "Cannot open LONG" in reason
        assert "existing SHORT" in reason
        assert "BTCUSDT" in reason
    
    def test_block_short_when_long_exists(self):
        """Should BLOCK opening SHORT when LONG position exists - CRITICAL BUG FIX."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"ETHUSDT": 0.1}  # Existing LONG
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="ETHUSDT",
            side="sell",  # Try to open SHORT
            quantity=0.1,
            current_positions=current_positions
        )
        
        assert can_open is False
        assert reason is not None
        assert "Cannot open SHORT" in reason
        assert "existing LONG" in reason
        assert "ETHUSDT" in reason
    
    def test_allow_adding_to_long(self):
        """Should allow adding to existing LONG position."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.001}  # Existing LONG
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="buy",  # Add more to LONG
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is True
        assert reason is None
    
    def test_allow_adding_to_short(self):
        """Should allow adding to existing SHORT position."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"ETHUSDT": -0.1}  # Existing SHORT
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="ETHUSDT",
            side="sell",  # Add more to SHORT
            quantity=0.1,
            current_positions=current_positions
        )
        
        assert can_open is True
        assert reason is None
    
    def test_multiple_symbols_independent(self):
        """Positions on different symbols should not interfere."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {
            "BTCUSDT": 0.001,   # LONG on BTC
            "ETHUSDT": -0.1     # SHORT on ETH
        }
        
        # Should allow SHORT on different symbol
        can_open_btc_short, _ = enforcer.check_can_open_position(
            symbol="SOLUSDT",  # Different symbol
            side="sell",
            quantity=1.0,
            current_positions=current_positions
        )
        assert can_open_btc_short is True
    
    def test_side_normalization_long(self):
        """Should recognize 'long' as equivalent to 'buy'."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        can_open, _ = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="long",  # Alternative notation
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is True
    
    def test_side_normalization_short(self):
        """Should recognize 'short' as equivalent to 'sell'."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        can_open, _ = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="short",  # Alternative notation
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is True
    
    def test_hedging_mode_allows_opposite_long_to_short(self):
        """When hedging mode enabled, should allow opposite positions."""
        enforcer = PositionInvariantEnforcer(allow_hedging=True)
        current_positions = {"BTCUSDT": 0.001}  # Existing LONG
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="sell",  # Open SHORT (opposite)
            quantity=0.001,
            current_positions=current_positions
        )
        
        assert can_open is True  # Hedging allowed
        assert reason is None
    
    def test_hedging_mode_allows_opposite_short_to_long(self):
        """When hedging mode enabled, should allow opposite positions."""
        enforcer = PositionInvariantEnforcer(allow_hedging=True)
        current_positions = {"ETHUSDT": -0.1}  # Existing SHORT
        
        can_open, reason = enforcer.check_can_open_position(
            symbol="ETHUSDT",
            side="buy",  # Open LONG (opposite)
            quantity=0.1,
            current_positions=current_positions
        )
        
        assert can_open is True  # Hedging allowed
        assert reason is None
    
    def test_enforce_raises_exception_on_conflict(self):
        """enforce_before_order should raise exception when check fails."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": -0.001}  # Existing SHORT
        
        with pytest.raises(PositionInvariantViolation) as exc_info:
            enforcer.enforce_before_order(
                symbol="BTCUSDT",
                side="buy",  # Try to open LONG
                quantity=0.001,
                current_positions=current_positions
            )
        
        assert "Cannot open LONG" in str(exc_info.value)
        assert "existing SHORT" in str(exc_info.value)
    
    def test_enforce_succeeds_when_valid(self):
        """enforce_before_order should not raise when check passes."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        # Should not raise
        enforcer.enforce_before_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.001,
            current_positions=current_positions
        )
    
    def test_blocked_order_recorded(self):
        """Blocked orders should be recorded for metrics."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.001}  # Existing LONG
        
        initial_count = len(enforcer.blocked_orders)
        
        # Attempt to open SHORT (will be blocked)
        can_open, _ = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="sell",
            quantity=0.001,
            current_positions=current_positions,
            account="test_account",
            exchange="binance"
        )
        
        assert can_open is False
        assert len(enforcer.blocked_orders) == initial_count + 1
        
        # Verify recorded event
        event = enforcer.blocked_orders[-1]
        assert event.symbol == "BTCUSDT"
        assert event.attempted_side == "SHORT"
        assert event.attempted_quantity == 0.001
        assert event.existing_quantity == 0.001
        assert event.account == "test_account"
        assert event.exchange == "binance"
        assert isinstance(event.timestamp, datetime)
    
    def test_get_blocked_orders_count(self):
        """Should return count of blocked orders."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.001}
        
        # Block some orders
        for _ in range(3):
            enforcer.check_can_open_position(
                "BTCUSDT", "sell", 0.001, current_positions
            )
        
        count = enforcer.get_blocked_orders_count()
        assert count == 3
    
    def test_get_blocked_orders_summary(self):
        """Should return summary of recent blocked orders."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.001}
        
        # Block an order
        enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, current_positions
        )
        
        summary = enforcer.get_blocked_orders_summary(limit=5)
        
        assert len(summary) > 0
        assert isinstance(summary[0], dict)
        assert "symbol" in summary[0]
        assert "attempted_side" in summary[0]
        assert "timestamp" in summary[0]
    
    def test_account_and_exchange_parameters(self):
        """Should handle account and exchange parameters."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.001}
        
        can_open, _ = enforcer.check_can_open_position(
            symbol="BTCUSDT",
            side="sell",
            quantity=0.001,
            current_positions=current_positions,
            account="main_account",
            exchange="binance_futures"
        )
        
        assert can_open is False
        
        # Verify account/exchange recorded
        event = enforcer.blocked_orders[-1]
        assert event.account == "main_account"
        assert event.exchange == "binance_futures"


class TestGlobalEnforcerSingleton:
    """Test global singleton enforcer access."""
    
    def setup_method(self):
        """Reset global enforcer before each test."""
        reset_enforcer()
    
    def test_get_enforcer_creates_instance(self):
        """get_position_invariant_enforcer should create singleton."""
        enforcer1 = get_position_invariant_enforcer()
        enforcer2 = get_position_invariant_enforcer()
        
        assert enforcer1 is enforcer2  # Same instance
    
    def test_get_enforcer_with_explicit_hedging(self):
        """Can initialize with explicit hedging setting."""
        enforcer = get_position_invariant_enforcer(allow_hedging=True)
        assert enforcer.allow_hedging is True
    
    def test_reset_enforcer_clears_singleton(self):
        """reset_enforcer should clear global instance."""
        enforcer1 = get_position_invariant_enforcer()
        reset_enforcer()
        enforcer2 = get_position_invariant_enforcer()
        
        assert enforcer1 is not enforcer2  # Different instances


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Reset global enforcer before each test."""
        reset_enforcer()
    
    def test_zero_position_treated_as_flat(self):
        """Zero quantity should be treated as flat position."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.0}  # Explicitly zero
        
        # Should allow both directions
        can_buy, _ = enforcer.check_can_open_position(
            "BTCUSDT", "buy", 0.001, current_positions
        )
        can_sell, _ = enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, current_positions
        )
        
        assert can_buy is True
        assert can_sell is True
    
    def test_very_small_position_detected(self):
        """Even very small positions should be detected."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"BTCUSDT": 0.00000001}  # Tiny LONG
        
        can_open, reason = enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, current_positions
        )
        
        assert can_open is False
        assert "Cannot open SHORT" in reason
    
    def test_case_insensitive_side(self):
        """Side parameter should be case-insensitive."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {}
        
        # All variations should work
        for side in ["BUY", "Buy", "buy", "LONG", "Long"]:
            can_open, _ = enforcer.check_can_open_position(
                "BTCUSDT", side, 0.001, current_positions
            )
            assert can_open is True
        
        for side in ["SELL", "Sell", "sell", "SHORT", "Short"]:
            can_open, _ = enforcer.check_can_open_position(
                "ETHUSDT", side, 0.1, current_positions
            )
            assert can_open is True
    
    def test_symbol_not_in_positions_treated_as_flat(self):
        """Symbol not in positions dict should be treated as flat."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        current_positions = {"ETHUSDT": 0.1}  # Only ETH
        
        # BTCUSDT not in dict, should be treated as flat
        can_open, _ = enforcer.check_can_open_position(
            "BTCUSDT", "buy", 0.001, current_positions
        )
        
        assert can_open is True


class TestIntegrationScenario:
    """Test realistic trading scenarios."""
    
    def setup_method(self):
        """Reset global enforcer before each test."""
        reset_enforcer()
    
    def test_normal_trading_flow(self):
        """Test normal flow: flat -> long -> close -> short."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        
        # Start flat
        positions = {}
        
        # Open LONG - should succeed
        can_open, _ = enforcer.check_can_open_position(
            "BTCUSDT", "buy", 0.001, positions
        )
        assert can_open is True
        
        # Update positions (simulating order fill)
        positions["BTCUSDT"] = 0.001
        
        # Try to open SHORT while LONG exists - should FAIL
        can_open, reason = enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, positions
        )
        assert can_open is False
        assert "Cannot open SHORT" in reason
        
        # Close LONG (position goes to zero)
        positions["BTCUSDT"] = 0.0
        
        # Now can open SHORT
        can_open, _ = enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, positions
        )
        assert can_open is True
    
    def test_scaling_into_position(self):
        """Test scaling into existing position."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        positions = {"BTCUSDT": 0.001}  # Initial LONG
        
        # Scale in - add more LONG
        for i in range(3):
            can_open, _ = enforcer.check_can_open_position(
                "BTCUSDT", "buy", 0.001, positions
            )
            assert can_open is True
            positions["BTCUSDT"] += 0.001
        
        # Final position should be larger
        assert positions["BTCUSDT"] == 0.004
    
    def test_multi_symbol_portfolio(self):
        """Test managing multiple symbols independently."""
        enforcer = PositionInvariantEnforcer(allow_hedging=False)
        positions = {
            "BTCUSDT": 0.001,   # LONG
            "ETHUSDT": -0.1,    # SHORT
            "SOLUSDT": 0.0      # FLAT
        }
        
        # Can add to BTC LONG
        can_open, _ = enforcer.check_can_open_position(
            "BTCUSDT", "buy", 0.001, positions
        )
        assert can_open is True
        
        # Can add to ETH SHORT
        can_open, _ = enforcer.check_can_open_position(
            "ETHUSDT", "sell", 0.1, positions
        )
        assert can_open is True
        
        # Can open either direction on SOL (flat)
        can_buy_sol, _ = enforcer.check_can_open_position(
            "SOLUSDT", "buy", 1.0, positions
        )
        can_sell_sol, _ = enforcer.check_can_open_position(
            "SOLUSDT", "sell", 1.0, positions
        )
        assert can_buy_sol is True
        assert can_sell_sol is True
        
        # Cannot reverse BTC (long to short)
        can_open, _ = enforcer.check_can_open_position(
            "BTCUSDT", "sell", 0.001, positions
        )
        assert can_open is False
        
        # Cannot reverse ETH (short to long)
        can_open, _ = enforcer.check_can_open_position(
            "ETHUSDT", "buy", 0.1, positions
        )
        assert can_open is False
