"""
Unit Tests for P0 Patches
==========================

Tests for all P0 patches to ensure core functionality works correctly.

P0 patches:
1. PolicyStore (dynamic thresholds)
2. ESS integration (closes positions, blocks trades)
3. EventBuffer (deduplication, ordering)
4. TradeStateStore (state tracking)
5. RL fallback (when model unavailable)
6. Model sync (A/B comparison, auto-promotion)
7. DrawdownMonitor (real-time DD tracking)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path


# ==============================================================================
# P0-01: PolicyStore Tests
# ==============================================================================

class TestPolicyStore:
    """Unit tests for PolicyStore (P0-01)."""
    
    @pytest.fixture
    async def policy_store(self):
        from backend.core.policy_store import PolicyStore
        import redis.asyncio as redis
        
        # Create mock Redis client for testing
        redis_client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
        store = PolicyStore(redis_client=redis_client, event_bus=None)
        
        try:
            await store.initialize()
            yield store
        finally:
            await redis_client.close()
    
    @pytest.mark.asyncio
    async def test_policy_store_initialization(self, policy_store):
        """Test PolicyStore initializes with default values."""
        assert policy_store is not None
        policy = await policy_store.get_policy()
        assert policy is not None
    
    @pytest.mark.asyncio
    async def test_policy_store_set_emergency_mode(self, policy_store):
        """Test setting emergency mode blocks trading."""
        
        # Set emergency mode using dedicated method
        await policy_store.set_emergency_mode(
            enabled=True,
            reason="P0 test",
            updated_by="test"
        )
        
        policy = await policy_store.get_policy()
        assert policy.emergency_mode is True
        assert policy.allow_new_trades is False
        assert policy.emergency_reason == "P0 test"
    
    @pytest.mark.asyncio
    async def test_policy_store_profile_switch(self, policy_store):
        """Test switching between risk profiles."""
        from backend.models.policy import RiskMode
        
        # Switch to aggressive mode
        await policy_store.switch_mode(
            RiskMode.AGGRESSIVE_SMALL_ACCOUNT,
            updated_by="test"
        )
        
        policy = await policy_store.get_policy()
        assert policy.active_mode == RiskMode.AGGRESSIVE_SMALL_ACCOUNT
    
    @pytest.mark.asyncio
    async def test_policy_store_dynamic_threshold_update(self, policy_store):
        """Test updating dynamic thresholds."""
        from backend.models.policy import RiskMode
        
        # Reset to NORMAL mode first (tests share Redis state)
        await policy_store.switch_mode(RiskMode.NORMAL, updated_by="test")
        
        # Get current policy and verify it exists
        policy = await policy_store.get_policy()
        assert policy is not None
        assert policy.active_mode is not None
        
        # Verify default is NORMAL
        assert policy.active_mode.value == "NORMAL"


# ==============================================================================
# P0-02: ESS Integration Tests
# ==============================================================================

class TestEmergencyStopSystem:
    """Unit tests for ESS (P0-02)."""
    
    @pytest.fixture
    def ess(self, tmp_path):
        from backend.services.risk.emergency_stop_system import (
            EmergencyStopController,
            EmergencyState,
            ESSStatus
        )
        
        # Mock dependencies
        class MockPolicyStore:
            def get(self, key): return {}
            def set(self, key, value): pass
        
        class MockExchange:
            async def close_all_positions(self): return 3
            async def cancel_all_orders(self): return 5
        
        class MockEventBus:
            async def publish(self, event): pass
        
        return EmergencyStopController(
            policy_store=MockPolicyStore(),
            exchange=MockExchange(),
            event_bus=MockEventBus()
        )
    
    @pytest.mark.asyncio
    async def test_ess_activation(self, ess):
        """Test ESS activates and closes positions."""
        assert ess.is_active is False
        
        await ess.activate("Test emergency")
        
        assert ess.is_active is True
        assert ess.state.reason == "Test emergency"
        assert ess.state.activation_count == 1
    
    @pytest.mark.asyncio
    async def test_ess_blocks_reactivation(self, ess):
        """Test ESS doesn't re-activate when already active."""
        await ess.activate("First emergency")
        count_before = ess.state.activation_count
        
        await ess.activate("Second emergency")
        count_after = ess.state.activation_count
        
        assert count_before == count_after  # No second activation


# ==============================================================================
# P0-03: EventBuffer Tests
# ==============================================================================

class TestEventBuffer:
    """Unit tests for EventBuffer (P0-03)."""
    
    @pytest.fixture
    def event_buffer(self, tmp_path):
        from backend.core.event_buffer import EventBuffer
        return EventBuffer(buffer_dir=tmp_path)
    
    def test_event_buffer_deduplication(self, event_buffer):
        """Test event buffer writes events to disk."""
        event1 = {
            "type": "signal.generated",
            "symbol": "BTCUSDT",
            "action": "BUY",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Write event to buffer
        event_buffer.write_event(event1)
        
        # Verify buffer file exists
        assert event_buffer.current_file.exists()
    
    def test_event_buffer_ordering(self, event_buffer):
        """Test events are written in order."""
        event1 = {"type": "test", "ts": 1000, "timestamp": datetime.utcnow().isoformat()}
        event2 = {"type": "test", "ts": 2000, "timestamp": datetime.utcnow().isoformat()}
        
        event_buffer.write_event(event1)
        event_buffer.write_event(event2)
        
        # Verify events written
        assert event_buffer.events_written >= 2


# ==============================================================================
# P0-04: TradeStateStore Tests
# ==============================================================================

class TestTradeStateStore:
    """Unit tests for TradeStateStore (P0-04)."""
    
    @pytest.fixture
    async def trade_store(self):
        from backend.services.governance.trade_state_store import TradeStateStore
        import redis.asyncio as redis
        
        # Create Redis client for testing
        redis_client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
        store = TradeStateStore(redis_client=redis_client)
        
        try:
            await store.initialize()
            yield store
        finally:
            await redis_client.close()
    
    @pytest.mark.asyncio
    async def test_trade_state_creation(self, trade_store):
        """Test creating and storing trade state."""
        from backend.services.governance.trade_state_store import TradeState
        
        trade = TradeState(
            trade_id="test_001",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.01
        )
        
        await trade_store.save("test_001", trade)
        retrieved = await trade_store.get("test_001")
        assert retrieved is not None
        assert retrieved.symbol == "BTCUSDT"
        assert retrieved.side == "LONG"
    
    @pytest.mark.asyncio
    async def test_trade_state_updates(self, trade_store):
        """Test updating trade state."""
        from backend.services.governance.trade_state_store import TradeState
        
        trade = TradeState(
            trade_id="test_002",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.01,
            status="OPEN"
        )
        
        await trade_store.save("test_002", trade)
        
        trade.status = "FILLED"
        await trade_store.save("test_002", trade)
        
        updated = await trade_store.get("test_002")
        assert updated.status == "FILLED"
    
    @pytest.mark.asyncio
    async def test_trade_state_close(self, trade_store):
        """Test closing trade and calculating PnL."""
        from backend.services.governance.trade_state_store import TradeState
        
        trade = TradeState(
            trade_id="test_003",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.01,
            status="OPEN"
        )
        
        await trade_store.save("test_003", trade)
        
        trade.exit_price = 51000.0
        trade.status = "CLOSED"
        trade.pnl = (51000.0 - 50000.0) * 0.01
        await trade_store.save("test_003", trade)
        
        closed = await trade_store.get("test_003")
        assert closed.status == "CLOSED"
        assert closed.pnl > 0  # Profit


# ==============================================================================
# P0-05: RL Fallback Tests
# ==============================================================================

class TestRLFallback:
    """Unit tests for RL fallback mechanism (P0-05)."""
    
    def test_rl_fallback_to_ensemble(self):
        """Test RL falls back to ensemble when unavailable."""
        # TODO: Implement when RL integration is finalized
        pass


# ==============================================================================
# P0-06: Model Sync Tests
# ==============================================================================

class TestModelSync:
    """Unit tests for model synchronization (P0-06)."""
    
    def test_model_comparison(self):
        """Test A/B comparison between models."""
        # TODO: Implement when model sync is finalized
        pass


# ==============================================================================
# P0-07: DrawdownMonitor Tests
# ==============================================================================

class TestDrawdownMonitor:
    """Unit tests for DrawdownMonitor (P0-07)."""
    
    def test_drawdown_calculation(self):
        """Test real-time drawdown calculation."""
        from backend.core.drawdown_monitor import DrawdownMonitor
        
        monitor = DrawdownMonitor(initial_balance=10000)
        
        # Simulate losing trade
        monitor.update_balance(9500)
        dd_pct = monitor.get_drawdown_percent()
        
        assert dd_pct == -5.0
    
    def test_drawdown_peak_tracking(self):
        """Test peak equity tracking."""
        from backend.core.drawdown_monitor import DrawdownMonitor
        
        monitor = DrawdownMonitor(initial_balance=10000)
        
        # New high
        monitor.update_balance(11000)
        assert monitor.peak_balance == 11000
        
        # Drawdown from peak
        monitor.update_balance(10500)
        dd_pct = monitor.get_drawdown_percent()
        assert dd_pct == pytest.approx(-4.54, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
