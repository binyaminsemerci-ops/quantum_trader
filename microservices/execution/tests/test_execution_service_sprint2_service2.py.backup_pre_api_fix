"""
Test: Execution Service - Sprint 2 Service #2

Tests:
- Service startup/shutdown
- EventBus event handling (ai.decision.made)
- ESS integration with risk-safety-service
- Order execution flow
- TradeStore persistence
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from microservices.execution.service import ExecutionService
from microservices.execution.models import OrderRequest, OrderStatus


@pytest.fixture
async def service():
    """Create execution service with mocked dependencies."""
    with patch('microservices.execution.service.TradeStore') as mock_trade_store, \
         patch('microservices.execution.service.GlobalRateLimiter') as mock_rate_limiter, \
         patch('microservices.execution.service.create_binance_wrapper') as mock_binance, \
         patch('microservices.execution.service.ExecutionSafetyGuard') as mock_safety_guard, \
         patch('microservices.execution.service.SafeOrderExecutor') as mock_safe_executor, \
         patch('microservices.execution.service.EventBus') as mock_event_bus, \
         patch('microservices.execution.service.DiskBuffer') as mock_disk_buffer, \
         patch('microservices.execution.service.httpx.AsyncClient') as mock_http_client:
        
        # Configure mocks
        mock_trade_store.return_value.initialize = AsyncMock()
        mock_event_bus.return_value.subscribe = MagicMock()
        mock_event_bus.return_value.publish = AsyncMock()
        
        service = ExecutionService()
        
        # Manually set mocks
        service.trade_store = mock_trade_store.return_value
        service.rate_limiter = mock_rate_limiter.return_value
        service.binance_client = mock_binance.return_value
        service.safety_guard = mock_safety_guard.return_value
        service.safe_executor = mock_safe_executor.return_value
        service.event_bus = mock_event_bus.return_value
        service.disk_buffer = mock_disk_buffer.return_value
        service.http_client = mock_http_client.return_value
        
        service._running = True
        
        yield service
        
        service._running = False


@pytest.mark.asyncio
async def test_service_health_all_components_healthy(service):
    """Test health check when all components are healthy."""
    # Mock Binance ping
    service.binance_client._signed_request = AsyncMock(return_value={})
    
    # Mock ESS check
    service.http_client.get = AsyncMock(return_value=MagicMock(
        json=lambda: {"state": "NORMAL", "can_execute": True}
    ))
    
    # Mock rate limiter
    service.rate_limiter.get_tokens_available = MagicMock(return_value=1000)
    
    health = await service.get_health()
    
    assert health["healthy"] is True
    assert health["service"] == "execution"
    assert health["running"] is True
    assert "binance" in health["components"]
    assert "event_bus" in health["components"]
    assert "trade_store" in health["components"]
    assert "rate_limiter" in health["components"]


@pytest.mark.asyncio
async def test_handle_ai_decision_ess_allows_execution(service):
    """Test ai.decision.made event when ESS allows execution."""
    # Mock ESS: NORMAL state, can execute
    service.http_client.get = AsyncMock(return_value=MagicMock(
        json=lambda: {"state": "NORMAL", "can_execute": True}
    ))
    
    # Mock safety guard: Valid order
    service.safety_guard.validate_and_adjust_order = AsyncMock(return_value=MagicMock(
        is_valid=True,
        reason=None,
        adjusted_sl=None,
        adjusted_tp=None
    ))
    
    # Mock Binance ticker
    service.binance_client._signed_request = AsyncMock(return_value={"price": "50000.00"})
    
    # Mock TradeStore
    service.trade_store.save_new_trade = AsyncMock()
    
    event_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "confidence": 0.85,
        "quantity": 0.001,
        "entry_price": 50000,
        "stop_loss": 49000,
        "take_profit": 52000,
        "leverage": 10,
        "model": "ensemble"
    }
    
    await service._handle_ai_decision(event_data)
    
    # Verify order.placed event published
    service.event_bus.publish.assert_any_call("order.placed", pytest.approx({}, rel=1.0))
    
    # Verify trade.opened event published
    service.event_bus.publish.assert_any_call("trade.opened", pytest.approx({}, rel=1.0))
    
    # Verify TradeStore saved trade
    service.trade_store.save_new_trade.assert_called_once()


@pytest.mark.asyncio
async def test_handle_ai_decision_ess_blocks_execution(service):
    """Test ai.decision.made event when ESS blocks execution."""
    # Mock ESS: CRITICAL state, cannot execute
    service.http_client.get = AsyncMock(return_value=MagicMock(
        json=lambda: {
            "state": "CRITICAL",
            "can_execute": False,
            "trip_reason": "Drawdown exceeded 10%"
        }
    ))
    
    event_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "confidence": 0.85,
        "quantity": 0.001,
        "entry_price": 50000,
        "stop_loss": 49000,
        "take_profit": 52000,
        "leverage": 10
    }
    
    await service._handle_ai_decision(event_data)
    
    # Verify order.failed event published
    service.event_bus.publish.assert_called_once()
    call_args = service.event_bus.publish.call_args
    assert call_args[0][0] == "order.failed"
    assert "ESS blocked" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_execute_order_safety_validation_fails(service):
    """Test execute_order when ExecutionSafetyGuard fails validation."""
    # Mock Binance ticker
    service.binance_client._signed_request = AsyncMock(return_value={"price": "50000.00"})
    
    # Mock safety guard: Invalid order (excessive slippage)
    service.safety_guard.validate_and_adjust_order = AsyncMock(return_value=MagicMock(
        is_valid=False,
        reason="Slippage exceeds 0.5%",
        adjusted_sl=None,
        adjusted_tp=None
    ))
    
    request = OrderRequest(
        symbol="BTCUSDT",
        side="long",
        quantity=0.001,
        price=50000,
        leverage=10,
        stop_loss=49000,
        take_profit=52000
    )
    
    response = await service.execute_order(request)
    
    assert response.success is False
    assert response.status == OrderStatus.FAILED
    assert "Validation failed" in response.message


@pytest.mark.asyncio
async def test_execute_order_success(service):
    """Test successful order execution."""
    # Mock Binance ticker
    service.binance_client._signed_request = AsyncMock(return_value={"price": "50000.00"})
    
    # Mock safety guard: Valid order
    service.safety_guard.validate_and_adjust_order = AsyncMock(return_value=MagicMock(
        is_valid=True,
        reason=None,
        adjusted_sl=None,
        adjusted_tp=None
    ))
    
    # Mock TradeStore
    service.trade_store.save_new_trade = AsyncMock()
    
    request = OrderRequest(
        symbol="BTCUSDT",
        side="long",
        quantity=0.001,
        price=50000,
        leverage=10,
        stop_loss=49000,
        take_profit=52000,
        metadata={"source": "manual", "confidence": 0.80}
    )
    
    response = await service.execute_order(request)
    
    assert response.success is True
    assert response.status == OrderStatus.FILLED
    assert response.symbol == "BTCUSDT"
    assert response.side == "long"
    
    # Verify order.placed event
    service.event_bus.publish.assert_any_call("order.placed", pytest.approx({}, rel=1.0))
    
    # Verify trade.opened event
    service.event_bus.publish.assert_any_call("trade.opened", pytest.approx({}, rel=1.0))
    
    # Verify TradeStore save
    service.trade_store.save_new_trade.assert_called_once()


@pytest.mark.asyncio
async def test_handle_ess_tripped_logs_error(service):
    """Test ess.tripped event handler logs critical error."""
    event_data = {
        "reason": "Drawdown exceeded 10%",
        "trip_level": "CRITICAL",
        "current_drawdown": 12.5
    }
    
    # Should log error but not raise exception
    await service._handle_ess_tripped(event_data)


@pytest.mark.asyncio
async def test_handle_policy_updated_logs_change(service):
    """Test policy.updated event handler logs change."""
    event_data = {
        "key": "max_leverage",
        "old_value": 20,
        "new_value": 10
    }
    
    # Should log info but not raise exception
    await service._handle_policy_updated(event_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
