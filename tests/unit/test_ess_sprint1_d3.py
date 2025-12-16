"""
SPRINT 1 - D3: Emergency Stop System (ESS) Tests

Tests for ESS core functionality:
- State machine transitions
- Threshold-based tripping
- Policy integration
- Manual reset
- Order blocking
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from backend.core.safety.ess import EmergencyStopSystem, ESSState, ESSMetrics


# ==============================================================================
# Test ESS Core Functionality
# ==============================================================================

class TestESSCore:
    """Test Emergency Stop System core functionality."""
    
    @pytest.fixture
    def mock_policy_store(self):
        """Create mock PolicyStore."""
        store = Mock()
        store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 5.0,
            "ess.max_open_loss_pct": 10.0,
            "ess.max_execution_errors": 5,
            "ess.cooldown_minutes": 15,
            "ess.allow_manual_reset": True
        }.get(key, default))
        return store
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock EventBus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def mock_clock(self):
        """Create mock clock."""
        base_time = datetime(2025, 12, 4, 12, 0, 0)
        return Mock(return_value=base_time)
    
    @pytest.fixture
    def ess(self, mock_policy_store, mock_event_bus, mock_clock):
        """Create ESS instance."""
        return EmergencyStopSystem(
            mock_policy_store,
            mock_event_bus,
            clock=mock_clock
        )
    
    def test_ess_initialization_armed(self, ess):
        """Test ESS initializes in ARMED state by default."""
        assert ess.state == ESSState.ARMED
        assert ess.trip_count == 0
        assert ess.reset_count == 0
    
    def test_ess_initialization_disabled(self, mock_event_bus, mock_clock):
        """Test ESS initializes in DISABLED state when policy disables it."""
        policy_store = Mock()
        policy_store.get = Mock(return_value=False)  # ess.enabled=False
        
        ess = EmergencyStopSystem(policy_store, mock_event_bus, clock=mock_clock)
        assert ess.state == ESSState.DISABLED
    
    @pytest.mark.asyncio
    async def test_can_execute_orders_when_armed(self, ess):
        """Test can_execute_orders returns True when ARMED."""
        assert ess.state == ESSState.ARMED
        assert await ess.can_execute_orders() is True
    
    @pytest.mark.asyncio
    async def test_can_execute_orders_when_disabled(self, mock_event_bus, mock_clock):
        """Test can_execute_orders returns True when DISABLED."""
        policy_store = Mock()
        policy_store.get = Mock(return_value=False)  # ess.enabled=False
        
        ess = EmergencyStopSystem(policy_store, mock_event_bus, clock=mock_clock)
        assert await ess.can_execute_orders() is True


# ==============================================================================
# Test ESS Tripping
# ==============================================================================

class TestESSTripping:
    """Test ESS tripping behavior."""
    
    @pytest.fixture
    def mock_policy_store(self):
        """Create mock PolicyStore with thresholds."""
        store = Mock()
        store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 5.0,
            "ess.max_open_loss_pct": 10.0,
            "ess.max_execution_errors": 5,
        }.get(key, default))
        return store
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock EventBus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def ess(self, mock_policy_store, mock_event_bus):
        """Create ESS instance."""
        return EmergencyStopSystem(mock_policy_store, mock_event_bus)
    
    @pytest.mark.asyncio
    async def test_trip_on_daily_drawdown(self, ess, mock_event_bus):
        """Test ESS trips when daily drawdown exceeds threshold."""
        # Update with drawdown above threshold (5.0%)
        await ess.update_metrics(daily_drawdown_pct=6.0)
        
        # Verify tripped
        assert ess.state == ESSState.TRIPPED
        assert ess.trip_reason is not None
        assert "drawdown" in ess.trip_reason.lower()
        assert ess.trip_count == 1
        
        # Verify event published
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "ess.tripped"
    
    @pytest.mark.asyncio
    async def test_trip_on_open_loss(self, ess, mock_event_bus):
        """Test ESS trips when open loss exceeds threshold."""
        # Update with open loss above threshold (10.0%)
        await ess.update_metrics(open_loss_pct=12.0)
        
        # Verify tripped
        assert ess.state == ESSState.TRIPPED
        assert "open loss" in ess.trip_reason.lower()
        assert ess.trip_count == 1
    
    @pytest.mark.asyncio
    async def test_trip_on_execution_errors(self, ess, mock_event_bus):
        """Test ESS trips when execution errors exceed threshold."""
        # Update with errors above threshold (5)
        await ess.update_metrics(execution_errors=6)
        
        # Verify tripped
        assert ess.state == ESSState.TRIPPED
        assert "error" in ess.trip_reason.lower()
        assert ess.trip_count == 1
    
    @pytest.mark.asyncio
    async def test_no_trip_below_threshold(self, ess):
        """Test ESS does not trip when metrics are below thresholds."""
        await ess.update_metrics(
            daily_drawdown_pct=3.0,  # Below 5.0%
            open_loss_pct=8.0,  # Below 10.0%
            execution_errors=3  # Below 5
        )
        
        # Verify still ARMED
        assert ess.state == ESSState.ARMED
        assert ess.trip_count == 0
    
    @pytest.mark.asyncio
    async def test_cannot_execute_when_tripped(self, ess):
        """Test can_execute_orders returns False when TRIPPED."""
        await ess.update_metrics(daily_drawdown_pct=6.0)
        
        assert ess.state == ESSState.TRIPPED
        assert await ess.can_execute_orders() is False


# ==============================================================================
# Test ESS Policy Integration
# ==============================================================================

class TestESSPolicyIntegration:
    """Test ESS integration with PolicyStore."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock EventBus."""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_event_bus):
        """Test ESS uses custom thresholds from PolicyStore."""
        # Create policy store with lower thresholds
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 2.0,  # Lower threshold
            "ess.max_open_loss_pct": 5.0,  # Lower threshold
            "ess.max_execution_errors": 3,  # Lower threshold
        }.get(key, default))
        
        ess = EmergencyStopSystem(policy_store, mock_event_bus)
        
        # Update with values that would be OK with default thresholds
        # but trigger with custom lower thresholds
        await ess.update_metrics(daily_drawdown_pct=3.0)
        
        # Should trip with lower threshold
        assert ess.state == ESSState.TRIPPED
    
    @pytest.mark.asyncio
    async def test_higher_thresholds(self, mock_event_bus):
        """Test ESS works with higher thresholds."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 15.0,  # Higher threshold
            "ess.max_open_loss_pct": 20.0,
            "ess.max_execution_errors": 10,
        }.get(key, default))
        
        ess = EmergencyStopSystem(policy_store, mock_event_bus)
        
        # Update with values that would trip default thresholds
        await ess.update_metrics(daily_drawdown_pct=8.0)
        
        # Should NOT trip with higher threshold
        assert ess.state == ESSState.ARMED


# ==============================================================================
# Test ESS Manual Reset
# ==============================================================================

class TestESSManualReset:
    """Test ESS manual reset functionality."""
    
    @pytest.fixture
    def mock_policy_store(self):
        """Create mock PolicyStore."""
        store = Mock()
        store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 5.0,
            "ess.allow_manual_reset": True
        }.get(key, default))
        return store
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock EventBus."""
        return AsyncMock()
    
    @pytest.fixture
    async def tripped_ess(self, mock_policy_store, mock_event_bus):
        """Create ESS instance that is tripped."""
        ess = EmergencyStopSystem(mock_policy_store, mock_event_bus)
        await ess.update_metrics(daily_drawdown_pct=6.0)
        assert ess.state == ESSState.TRIPPED
        return ess
    
    @pytest.mark.asyncio
    async def test_manual_reset_success(self, tripped_ess, mock_event_bus):
        """Test successful manual reset."""
        result = await tripped_ess.manual_reset(user="admin", reason="Testing")
        
        assert result is True
        assert tripped_ess.state == ESSState.ARMED
        assert tripped_ess.trip_reason is None
        assert tripped_ess.reset_count == 1
        
        # Verify event published
        assert mock_event_bus.publish.call_count == 2  # trip + reset
        last_call = mock_event_bus.publish.call_args_list[-1]
        assert last_call[0][0] == "ess.manual_reset"
    
    @pytest.mark.asyncio
    async def test_manual_reset_disabled_by_policy(self, mock_event_bus):
        """Test manual reset fails when disabled by policy."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "ess.enabled": True,
            "ess.max_daily_drawdown_pct": 5.0,
            "ess.allow_manual_reset": False  # Disabled
        }.get(key, default))
        
        ess = EmergencyStopSystem(policy_store, mock_event_bus)
        await ess.update_metrics(daily_drawdown_pct=6.0)
        
        result = await ess.manual_reset(user="admin")
        
        assert result is False
        assert ess.state == ESSState.TRIPPED  # Still tripped
    
    @pytest.mark.asyncio
    async def test_manual_reset_when_armed(self, mock_policy_store, mock_event_bus):
        """Test manual reset fails when already ARMED."""
        ess = EmergencyStopSystem(mock_policy_store, mock_event_bus)
        
        result = await ess.manual_reset(user="admin")
        
        assert result is False
        assert ess.state == ESSState.ARMED
    
    @pytest.mark.asyncio
    async def test_can_execute_after_reset(self, tripped_ess):
        """Test can_execute_orders returns True after reset."""
        await tripped_ess.manual_reset(user="admin")
        
        assert await tripped_ess.can_execute_orders() is True


# ==============================================================================
# Test ESS Status Reporting
# ==============================================================================

class TestESSStatus:
    """Test ESS status reporting."""
    
    @pytest.fixture
    def ess(self):
        """Create ESS instance."""
        policy_store = Mock()
        policy_store.get = Mock(return_value=True)
        event_bus = AsyncMock()
        return EmergencyStopSystem(policy_store, event_bus)
    
    def test_get_status_armed(self, ess):
        """Test get_status returns correct info when ARMED."""
        status = ess.get_status()
        
        assert status["state"] == "ARMED"
        assert status["can_execute"] is True
        assert status["trip_reason"] is None
        assert "metrics" in status
        assert "statistics" in status
    
    @pytest.mark.asyncio
    async def test_get_status_tripped(self, ess):
        """Test get_status returns correct info when TRIPPED."""
        await ess.update_metrics(daily_drawdown_pct=10.0)
        
        status = ess.get_status()
        
        assert status["state"] == "TRIPPED"
        assert status["can_execute"] is False
        assert status["trip_reason"] is not None
        assert status["metrics"]["daily_drawdown_pct"] == 10.0
        assert status["statistics"]["trip_count"] == 1


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
