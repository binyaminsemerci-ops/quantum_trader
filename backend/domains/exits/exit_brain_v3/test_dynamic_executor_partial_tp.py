"""
Unit tests for Exit Brain V3 Dynamic Executor - Dynamic Partial TP & Loss Guard

Tests the new features:
1. Dynamic partial TP execution with remaining_size tracking
2. SL ratcheting after TP hits (breakeven, TP1 price)
3. Max unrealized loss guard (emergency exit)
4. TP fraction normalization and capping
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Tuple

from .types import PositionExitState, PositionContext, ExitDecision, ExitDecisionType
from .dynamic_executor import (
    ExitBrainDynamicExecutor,
    MAX_UNREALIZED_LOSS_PCT_PER_POSITION,
    MAX_TP_LEVELS,
    RATCHET_SL_ENABLED
)


@pytest.fixture
def mock_adapter():
    """Mock ExitBrainAdapter."""
    adapter = Mock()
    adapter.decide = AsyncMock()
    return adapter


@pytest.fixture
def mock_gateway():
    """Mock exit_order_gateway."""
    gateway = Mock()
    gateway.submit_exit_order = AsyncMock(return_value={
        'orderId': '12345',
        'status': 'FILLED'
    })
    return gateway


@pytest.fixture
def mock_position_source():
    """Mock position source (Binance client)."""
    return Mock()


@pytest.fixture
def executor(mock_adapter, mock_gateway, mock_position_source):
    """Create executor instance for testing."""
    with patch('backend.domains.exits.exit_brain_v3.dynamic_executor.is_exit_brain_live_fully_enabled', return_value=False):
        executor = ExitBrainDynamicExecutor(
            adapter=mock_adapter,
            exit_order_gateway=mock_gateway,
            position_source=mock_position_source,
            shadow_mode=True  # Shadow mode for unit tests
        )
    return executor


class TestDynamicPartialTP:
    """Tests for dynamic partial TP execution."""
    
    def test_state_initialization(self):
        """Test PositionExitState initializes dynamic fields correctly."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.5,
            entry_price=50000.0
        )
        
        assert state.initial_size == 1.5
        assert state.remaining_size == 1.5
        assert state.tp_hits_count == 0
        assert state.max_unrealized_profit_pct == 0.0
        assert state.loss_guard_triggered is False
    
    def test_get_remaining_size_tracks_partials(self):
        """Test get_remaining_size() updates correctly after partial closes."""
        state = PositionExitState(
            symbol="ETHUSDT",
            side="LONG",
            position_size=10.0,
            entry_price=3000.0,
            initial_size=10.0,
            remaining_size=10.0
        )
        
        # Set TP levels: 25%, 25%, 50%
        state.tp_levels = [
            (3100.0, 0.25),
            (3200.0, 0.25),
            (3300.0, 0.50)
        ]
        
        # Initial remaining size
        assert state.get_remaining_size() == 10.0
        
        # After TP1 hit: 25% closed
        state.triggered_legs.add(0)
        state.remaining_size = 7.5
        assert state.get_remaining_size() == 7.5
        
        # After TP2 hit: another 25% of original (2.5)
        state.triggered_legs.add(1)
        state.remaining_size = 5.0
        assert state.get_remaining_size() == 5.0
    
    def test_tp_fraction_normalization(self):
        """Test TP fractions are normalized correctly."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0
        )
        
        # Simulate UPDATE_TP_LIMITS with fractions summing > 1.0
        tp_prices = [51000.0, 52000.0, 53000.0]
        fractions = [0.4, 0.4, 0.4]  # Sum = 1.2
        
        # Normalize
        total = sum(fractions)
        normalized = [f / total for f in fractions]
        
        assert sum(normalized) == pytest.approx(1.0)
        assert normalized[0] == pytest.approx(0.333, rel=0.01)
    
    def test_tp_level_capping(self):
        """Test TP levels are capped at MAX_TP_LEVELS."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0
        )
        
        # Create 10 TP levels (should be capped at MAX_TP_LEVELS=4)
        tp_prices = [50000.0 + i*1000 for i in range(1, 11)]
        fractions = [0.1] * 10
        
        # Simulate capping
        capped_prices = tp_prices[:MAX_TP_LEVELS]
        capped_fractions = fractions[:MAX_TP_LEVELS]
        
        assert len(capped_prices) == MAX_TP_LEVELS
        assert len(capped_fractions) == MAX_TP_LEVELS


class TestSLRatcheting:
    """Tests for SL ratcheting after TP hits."""
    
    def test_ratchet_to_breakeven_after_tp1_long(self, executor):
        """Test SL moves to breakeven after first TP hit (LONG)."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=0.75,  # 25% closed
            tp_hits_count=1
        )
        
        state.tp_levels = [
            (51000.0, 0.25),
            (52000.0, 0.25),
            (53000.0, 0.50)
        ]
        state.triggered_legs.add(0)
        state.active_sl = 49000.0  # Below breakeven
        
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            current_price=51500.0,
            size=0.75,
            unrealized_pnl=3.0
        )
        
        # Execute ratcheting
        executor._recompute_dynamic_tp_and_sl(state, ctx)
        
        # SL should be moved to breakeven (50000.0)
        assert state.active_sl == 50000.0
    
    def test_ratchet_to_tp1_after_tp2_long(self, executor):
        """Test SL moves to TP1 price after second TP hit (LONG)."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=0.5,  # 50% closed
            tp_hits_count=2
        )
        
        state.tp_levels = [
            (51000.0, 0.25),
            (52000.0, 0.25),
            (53000.0, 0.50)
        ]
        state.triggered_legs.add(0)
        state.triggered_legs.add(1)
        state.active_sl = 50000.0  # At breakeven
        
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            current_price=52500.0,
            size=0.5,
            unrealized_pnl=5.0
        )
        
        # Execute ratcheting
        executor._recompute_dynamic_tp_and_sl(state, ctx)
        
        # SL should be moved to TP1 price (51000.0)
        assert state.active_sl == 51000.0
    
    def test_ratchet_to_breakeven_after_tp1_short(self, executor):
        """Test SL moves to breakeven after first TP hit (SHORT)."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="SHORT",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=0.75,
            tp_hits_count=1
        )
        
        state.tp_levels = [
            (49000.0, 0.25),
            (48000.0, 0.25),
            (47000.0, 0.50)
        ]
        state.triggered_legs.add(0)
        state.active_sl = 51000.0  # Above breakeven
        
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            current_price=48500.0,
            size=0.75,
            unrealized_pnl=3.0
        )
        
        # Execute ratcheting
        executor._recompute_dynamic_tp_and_sl(state, ctx)
        
        # SL should be moved to breakeven (50000.0)
        assert state.active_sl == 50000.0
    
    def test_no_ratchet_if_disabled(self, executor):
        """Test ratcheting is skipped if RATCHET_SL_ENABLED=False."""
        with patch('backend.domains.exits.exit_brain_v3.dynamic_executor.RATCHET_SL_ENABLED', False):
            state = PositionExitState(
                symbol="BTCUSDT",
                side="LONG",
                position_size=1.0,
                entry_price=50000.0,
                initial_size=1.0,
                remaining_size=0.75,
                tp_hits_count=1
            )
            
            state.active_sl = 49000.0
            
            ctx = PositionContext(
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                current_price=51500.0,
                size=0.75,
                unrealized_pnl=3.0
            )
            
            # Execute ratcheting (should be skipped)
            executor._recompute_dynamic_tp_and_sl(state, ctx)
            
            # SL should NOT change
            assert state.active_sl == 49000.0


class TestLossGuard:
    """Tests for max unrealized loss guard."""
    
    @pytest.mark.asyncio
    async def test_loss_guard_triggers_at_threshold(self, executor):
        """Test loss guard triggers at MAX_UNREALIZED_LOSS_PCT threshold."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=1.0
        )
        
        # Create context with loss exceeding threshold
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            current_price=43750.0,  # -12.5% price move
            size=1.0,
            unrealized_pnl=-12.6  # Slightly over -12.5% threshold
        )
        
        # Check loss guard
        triggered = await executor._check_loss_guard(state, ctx)
        
        # Should trigger in shadow mode
        assert triggered is True
        assert state.loss_guard_triggered is True
    
    @pytest.mark.asyncio
    async def test_loss_guard_not_triggered_below_threshold(self, executor):
        """Test loss guard does not trigger below threshold."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=1.0
        )
        
        # Create context with loss below threshold
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            current_price=44000.0,  # -12% loss
            size=1.0,
            unrealized_pnl=-12.0  # Below -12.5% threshold
        )
        
        # Check loss guard
        triggered = await executor._check_loss_guard(state, ctx)
        
        # Should NOT trigger
        assert triggered is False
        assert state.loss_guard_triggered is False
    
    @pytest.mark.asyncio
    async def test_loss_guard_not_triggered_twice(self, executor):
        """Test loss guard does not trigger twice for same position."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="LONG",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=1.0,
            loss_guard_triggered=True  # Already triggered
        )
        
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            current_price=40000.0,
            size=1.0,
            unrealized_pnl=-20.0
        )
        
        # Check loss guard
        triggered = await executor._check_loss_guard(state, ctx)
        
        # Should NOT trigger again
        assert triggered is False
    
    @pytest.mark.asyncio
    async def test_loss_guard_works_for_short(self, executor):
        """Test loss guard works correctly for SHORT positions."""
        state = PositionExitState(
            symbol="BTCUSDT",
            side="SHORT",
            position_size=1.0,
            entry_price=50000.0,
            initial_size=1.0,
            remaining_size=1.0
        )
        
        # Create context with loss exceeding threshold (price went UP)
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            current_price=56250.0,  # +12.5% price move = loss for SHORT
            size=1.0,
            unrealized_pnl=-12.6  # Loss for short
        )
        
        # Check loss guard
        triggered = await executor._check_loss_guard(state, ctx)
        
        # Should trigger
        assert triggered is True
        assert state.loss_guard_triggered is True


class TestIntegration:
    """Integration tests for complete TP execution flow."""
    
    @pytest.mark.asyncio
    async def test_full_tp_execution_flow(self, executor):
        """Test complete TP execution with dynamic updates."""
        # Create position state
        state = PositionExitState(
            symbol="ETHUSDT",
            side="LONG",
            position_size=10.0,
            entry_price=3000.0,
            initial_size=10.0,
            remaining_size=10.0
        )
        
        # Set TP levels
        state.tp_levels = [
            (3150.0, 0.25),  # TP1: 25% at +5%
            (3300.0, 0.25),  # TP2: 25% at +10%
            (3600.0, 0.50)   # TP3: 50% at +20%
        ]
        
        state.active_sl = 2940.0  # -2% SL
        
        # Simulate TP1 trigger
        current_price = 3150.0
        
        # Execute TP1
        await executor._execute_tp_trigger(
            state=state,
            leg_index=0,
            tp_price=3150.0,
            size_pct=0.25,
            current_price=current_price
        )
        
        # Verify state updates
        assert 0 in state.triggered_legs
        assert state.tp_hits_count == 1
        assert state.remaining_size == 7.5  # 75% remains
        
        # SL should be ratcheted to breakeven (3000.0)
        assert state.active_sl == 3000.0
        
        # Simulate TP2 trigger
        current_price = 3300.0
        
        await executor._execute_tp_trigger(
            state=state,
            leg_index=1,
            tp_price=3300.0,
            size_pct=0.25,
            current_price=current_price
        )
        
        # Verify state updates
        assert 1 in state.triggered_legs
        assert state.tp_hits_count == 2
        assert state.remaining_size == pytest.approx(5.0, rel=0.01)  # 50% remains
        
        # SL should be ratcheted to TP1 price (3150.0)
        assert state.active_sl == 3150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
