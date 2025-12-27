"""
SPRINT 1 - D7: Execution Safety Guard + Safe Order Executor - Test Suite

Tests slippage validation, retry logic, -2021 error handling, and PolicyStore integration.

Test Categories:
1. ExecutionSafetyGuard - Slippage validation
2. ExecutionSafetyGuard - SL/TP validation
3. SafeOrderExecutor - Retry logic
4. SafeOrderExecutor - -2021 error handling
5. Integration tests

Run: pytest tests/unit/test_execution_safety_sprint1_d7.py -v
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import D7 modules
from backend.services.execution.execution_safety import ExecutionSafetyGuard, ValidationResult
from backend.services.execution.safe_order_executor import SafeOrderExecutor, OrderResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_policy_store():
    """Mock PolicyStore with default execution policies."""
    store = Mock()
    store.get_policy = Mock(side_effect=lambda key: {
        "execution.max_slippage_pct": 0.005,  # 0.5%
        "execution.max_order_retries": 5,
        "execution.retry_backoff_base_sec": 0.5,
        "execution.max_retry_backoff_sec": 10.0,
        "execution.sl_adjustment_buffer_pct": 0.001,  # 0.1%
        "execution.max_sl_distance_pct": 0.10,  # 10%
        "execution.max_tp_distance_pct": 0.50,  # 50%
    }.get(key))
    return store


@pytest.fixture
def safety_guard(mock_policy_store):
    """ExecutionSafetyGuard with mocked PolicyStore."""
    return ExecutionSafetyGuard(policy_store=mock_policy_store)


@pytest.fixture
def safe_executor(mock_policy_store, safety_guard):
    """SafeOrderExecutor with mocked PolicyStore and SafetyGuard."""
    return SafeOrderExecutor(
        policy_store=mock_policy_store,
        safety_guard=safety_guard
    )


@pytest.fixture
def mock_client():
    """Mock Binance client."""
    client = Mock()
    client._signed_request = AsyncMock()
    return client


# ============================================================================
# TEST 1: ExecutionSafetyGuard - Slippage Validation
# ============================================================================

@pytest.mark.asyncio
async def test_safety_guard_accepts_low_slippage(safety_guard):
    """Test that orders with acceptable slippage pass validation."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50100.0,  # 0.2% slippage
        sl_price=49000.0,
        tp_price=52000.0
    )
    
    assert result.is_valid, f"Expected valid order, got: {result.reason}"
    assert result.adjusted_entry == 50100.0  # Uses current market price
    assert "successfully" in result.reason.lower()


@pytest.mark.asyncio
async def test_safety_guard_rejects_high_slippage(safety_guard):
    """Test that orders with excessive slippage are rejected."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50400.0,  # 0.794% slippage (exceeds 0.5% limit)
        sl_price=49000.0,
        tp_price=52000.0
    )
    
    assert not result.is_valid, "Expected rejection for high slippage"
    assert "slippage" in result.reason.lower()
    assert "0.79" in result.reason or "0.794" in result.reason  # Percentage in error message


@pytest.mark.asyncio
async def test_safety_guard_stricter_for_high_leverage(safety_guard):
    """Test that slippage limits are tighter for high leverage trades."""
    # Normal leverage - 0.4% slippage (should pass with 0.5% limit)
    result_normal = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50200.0,  # 0.4% slippage
        leverage=5
    )
    assert result_normal.is_valid, "Expected valid for normal leverage"
    
    # High leverage - 0.4% slippage (should fail with 0.25% adjusted limit)
    result_high = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50200.0,  # 0.4% slippage
        leverage=20  # High leverage halves limit to 0.25%
    )
    assert not result_high.is_valid, "Expected rejection for high leverage + high slippage"


# ============================================================================
# TEST 2: ExecutionSafetyGuard - SL/TP Validation & Adjustment
# ============================================================================

@pytest.mark.asyncio
async def test_safety_guard_adjusts_invalid_long_sl(safety_guard):
    """Test that LONG SL above entry is adjusted down."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50000.0,
        sl_price=50500.0,  # INVALID: LONG SL above entry
        tp_price=52000.0
    )
    
    assert result.is_valid, f"Expected valid after adjustment, got: {result.reason}"
    assert result.adjusted_sl is not None, "Expected SL adjustment"
    assert result.adjusted_sl < 50000.0, f"Expected SL < entry, got {result.adjusted_sl}"
    assert "adjusted" in result.reason.lower()


@pytest.mark.asyncio
async def test_safety_guard_adjusts_invalid_short_sl(safety_guard):
    """Test that SHORT SL below entry is adjusted up."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="sell",
        planned_entry_price=50000.0,
        current_market_price=50000.0,
        sl_price=49500.0,  # INVALID: SHORT SL below entry
        tp_price=48000.0
    )
    
    assert result.is_valid, f"Expected valid after adjustment, got: {result.reason}"
    assert result.adjusted_sl is not None, "Expected SL adjustment"
    assert result.adjusted_sl > 50000.0, f"Expected SL > entry, got {result.adjusted_sl}"


@pytest.mark.asyncio
async def test_safety_guard_rejects_sl_too_far(safety_guard):
    """Test that SL more than 10% away from entry is rejected."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50000.0,
        sl_price=44000.0,  # 12% away (exceeds 10% limit)
        tp_price=52000.0
    )
    
    assert not result.is_valid, "Expected rejection for SL too far"
    assert "distance" in result.reason.lower()


@pytest.mark.asyncio
async def test_safety_guard_adjusts_invalid_long_tp(safety_guard):
    """Test that LONG TP below entry is adjusted up."""
    result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50000.0,
        sl_price=49000.0,
        tp_price=49500.0  # INVALID: LONG TP below entry
    )
    
    assert result.is_valid, f"Expected valid after adjustment, got: {result.reason}"
    assert result.adjusted_tp is not None, "Expected TP adjustment"
    assert result.adjusted_tp > 50000.0, f"Expected TP > entry, got {result.adjusted_tp}"


# ============================================================================
# TEST 3: SafeOrderExecutor - Retry Logic
# ============================================================================

@pytest.mark.asyncio
async def test_safe_executor_success_first_attempt(safe_executor):
    """Test successful order placement on first attempt."""
    mock_submit = AsyncMock(return_value={"orderId": "12345"})
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={"symbol": "BTCUSDT", "side": "BUY"},
        symbol="BTCUSDT",
        side="buy"
    )
    
    assert result.success, f"Expected success, got: {result.error_message}"
    assert result.order_id == "12345"
    assert result.attempts == 1
    mock_submit.assert_called_once()


@pytest.mark.asyncio
async def test_safe_executor_retry_on_transient_error(safe_executor, mock_policy_store):
    """Test retry on transient errors (-1001, -1021, etc.)."""
    # Mock: First 3 attempts fail with -1001, 4th succeeds
    call_count = 0
    async def mock_submit(**params):
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            error = Exception("Internal error")
            error.code = -1001  # Retryable error
            raise error
        return {"orderId": "12345"}
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={"symbol": "BTCUSDT", "side": "BUY"},
        symbol="BTCUSDT",
        side="buy"
    )
    
    assert result.success, f"Expected success after retries, got: {result.error_message}"
    assert result.order_id == "12345"
    assert result.attempts == 4, f"Expected 4 attempts, got {result.attempts}"


@pytest.mark.asyncio
async def test_safe_executor_fails_after_max_retries(safe_executor):
    """Test that executor gives up after max retries."""
    # Mock: Always fail with retryable error
    async def mock_submit(**params):
        error = Exception("Persistent error")
        error.code = -1001  # Retryable
        raise error
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={"symbol": "BTCUSDT", "side": "BUY"},
        symbol="BTCUSDT",
        side="buy"
    )
    
    assert not result.success, "Expected failure after max retries"
    assert result.error_code == -1001
    assert result.attempts == 5, f"Expected 5 attempts, got {result.attempts}"


@pytest.mark.asyncio
async def test_safe_executor_no_retry_on_non_retryable_error(safe_executor):
    """Test that non-retryable errors fail immediately."""
    # Mock: Fail with non-retryable error
    async def mock_submit(**params):
        error = Exception("Invalid symbol")
        error.code = -1121  # Invalid symbol (not in retryable list)
        raise error
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={"symbol": "INVALID", "side": "BUY"},
        symbol="INVALID",
        side="buy"
    )
    
    assert not result.success, "Expected immediate failure"
    assert result.attempts == 1, f"Expected 1 attempt (no retry), got {result.attempts}"


# ============================================================================
# TEST 4: SafeOrderExecutor - -2021 Error Handling (Order Would Immediately Trigger)
# ============================================================================

@pytest.mark.asyncio
async def test_safe_executor_adjusts_sl_on_2021_error(safe_executor, mock_client):
    """Test that -2021 error triggers SL price adjustment."""
    # Mock client to return current price
    mock_client._signed_request = AsyncMock(return_value={"markPrice": "50000.0"})
    
    # Mock submit: First attempt -2021, second succeeds
    call_count = 0
    async def mock_submit(**params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            error = Exception("Order would immediately trigger")
            error.code = -2021
            raise error
        return {"orderId": "12345"}
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": 50100.0  # LONG SL above current price (will trigger -2021)
        },
        symbol="BTCUSDT",
        side="buy",
        sl_price=50100.0,
        client=mock_client,
        order_type="sl"
    )
    
    assert result.success, f"Expected success after adjustment, got: {result.error_message}"
    assert result.attempts == 2, f"Expected 2 attempts (1 fail, 1 success), got {result.attempts}"


@pytest.mark.asyncio
async def test_safe_executor_adjusts_long_sl_below_price(safe_executor, mock_client):
    """Test that LONG SL is adjusted BELOW current price on -2021."""
    # Mock current mark price
    mock_client._signed_request = AsyncMock(return_value={"markPrice": "50000.0"})
    
    # Capture adjusted SL price
    adjusted_sl = None
    call_count = 0
    async def mock_submit(**params):
        nonlocal call_count, adjusted_sl
        call_count += 1
        if call_count == 1:
            error = Exception("Order would immediately trigger")
            error.code = -2021
            raise error
        # Capture adjusted price on second attempt
        adjusted_sl = float(params.get("stopPrice", 0))
        return {"orderId": "12345"}
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": 50100.0  # Above current price (invalid for LONG)
        },
        symbol="BTCUSDT",
        side="buy",  # LONG
        sl_price=50100.0,
        client=mock_client,
        order_type="sl"
    )
    
    assert result.success, "Expected success after adjustment"
    assert adjusted_sl is not None, "Expected SL adjustment"
    assert adjusted_sl < 50000.0, f"Expected LONG SL < 50000, got {adjusted_sl}"


@pytest.mark.asyncio
async def test_safe_executor_adjusts_short_sl_above_price(safe_executor, mock_client):
    """Test that SHORT SL is adjusted ABOVE current price on -2021."""
    # Mock current mark price
    mock_client._signed_request = AsyncMock(return_value={"markPrice": "50000.0"})
    
    # Capture adjusted SL price
    adjusted_sl = None
    call_count = 0
    async def mock_submit(**params):
        nonlocal call_count, adjusted_sl
        call_count += 1
        if call_count == 1:
            error = Exception("Order would immediately trigger")
            error.code = -2021
            raise error
        # Capture adjusted price on second attempt
        adjusted_sl = float(params.get("stopPrice", 0))
        return {"orderId": "12345"}
    
    result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "STOP_MARKET",
            "stopPrice": 49900.0  # Below current price (invalid for SHORT)
        },
        symbol="BTCUSDT",
        side="sell",  # SHORT
        sl_price=49900.0,
        client=mock_client,
        order_type="sl"
    )
    
    assert result.success, "Expected success after adjustment"
    assert adjusted_sl is not None, "Expected SL adjustment"
    assert adjusted_sl > 50000.0, f"Expected SHORT SL > 50000, got {adjusted_sl}"


# ============================================================================
# TEST 5: Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_integration_slippage_and_retry(safety_guard, safe_executor, mock_client):
    """Test full flow: slippage validation + retry on failure."""
    # Step 1: Validate with safety guard
    validation_result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50100.0,  # 0.2% slippage (acceptable)
        sl_price=49000.0,
        tp_price=52000.0
    )
    
    assert validation_result.is_valid, "Expected valid order"
    
    # Step 2: Submit with safe executor (with retry)
    call_count = 0
    async def mock_submit(**params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            error = Exception("Temporary network error")
            error.code = -1001
            raise error
        return {"orderId": "12345"}
    
    order_result = await safe_executor.place_order_with_safety(
        submit_func=mock_submit,
        order_params={"symbol": "BTCUSDT", "side": "BUY"},
        symbol="BTCUSDT",
        side="buy"
    )
    
    assert order_result.success, "Expected successful order after retry"
    assert order_result.attempts == 2, "Expected 2 attempts (1 fail, 1 success)"


@pytest.mark.asyncio
async def test_integration_excessive_slippage_blocks_order(safety_guard):
    """Test that excessive slippage prevents order from being submitted."""
    validation_result = await safety_guard.validate_and_adjust_order(
        symbol="BTCUSDT",
        side="buy",
        planned_entry_price=50000.0,
        current_market_price=50600.0,  # 1.2% slippage (exceeds 0.5% limit)
        sl_price=49000.0,
        tp_price=52000.0
    )
    
    assert not validation_result.is_valid, "Expected rejection for excessive slippage"
    # In production, this would prevent safe_executor from being called


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
