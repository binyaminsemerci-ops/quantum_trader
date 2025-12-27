"""
Exchange Failover Tests

EPIC-EXCH-FAIL-001: Test multi-exchange failover router.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.policies.exchange_failover_policy import (
    get_failover_chain,
    set_failover_chain,
    is_healthy,
    choose_exchange_with_failover,
)


# ============================================================================
# UNIT TESTS: Failover Policy
# ============================================================================

def test_get_failover_chain_configured():
    """Configured exchange returns full failover chain."""
    chain = get_failover_chain("binance")
    
    assert len(chain) > 1
    assert chain[0] == "binance"  # Primary always first
    assert "bybit" in chain
    assert "okx" in chain


def test_get_failover_chain_unconfigured():
    """Unconfigured exchange returns single-item chain."""
    chain = get_failover_chain("unknown_exchange")
    
    assert chain == ["unknown_exchange"]


def test_set_failover_chain():
    """set_failover_chain updates chain dynamically."""
    original = get_failover_chain("binance")
    
    # Set custom chain
    set_failover_chain("binance", ["binance", "bybit"])
    updated = get_failover_chain("binance")
    
    assert updated == ["binance", "bybit"]
    
    # Restore original
    set_failover_chain("binance", original)


def test_is_healthy_ok():
    """Status 'ok' is healthy."""
    health = {"status": "ok", "latency_ms": 50, "last_error": None}
    assert is_healthy(health) is True


def test_is_healthy_degraded():
    """Status 'degraded' is not healthy."""
    health = {"status": "degraded", "latency_ms": 500, "last_error": "Timeout"}
    assert is_healthy(health) is False


def test_is_healthy_down():
    """Status 'down' is not healthy."""
    health = {"status": "down", "latency_ms": 0, "last_error": "Connection refused"}
    assert is_healthy(health) is False


# ============================================================================
# INTEGRATION TESTS: Failover Selection
# ============================================================================

@pytest.mark.asyncio
async def test_choose_exchange_primary_healthy():
    """Primary exchange healthy → use primary."""
    
    # Mock health checks
    async def mock_health(exchange):
        return {"status": "ok", "latency_ms": 45, "last_error": None}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await choose_exchange_with_failover("binance", "binance")
        
        assert exchange == "binance"


@pytest.mark.asyncio
async def test_choose_exchange_primary_down_secondary_ok():
    """Primary down, secondary healthy → use secondary."""
    
    # Mock health checks: binance down, bybit ok
    async def mock_health(exchange):
        if exchange == "binance":
            return {"status": "down", "latency_ms": 0, "last_error": "Connection refused"}
        elif exchange == "bybit":
            return {"status": "ok", "latency_ms": 50, "last_error": None}
        else:
            return {"status": "down", "latency_ms": 0, "last_error": "Unavailable"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await choose_exchange_with_failover("binance", "binance")
        
        # Should failover to bybit (first healthy in chain)
        assert exchange == "bybit"


@pytest.mark.asyncio
async def test_choose_exchange_all_down():
    """All exchanges down → return default_exchange anyway."""
    
    # Mock all exchanges down
    async def mock_health(exchange):
        return {"status": "down", "latency_ms": 0, "last_error": "All exchanges offline"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await choose_exchange_with_failover("binance", "binance")
        
        # Should return default_exchange (let execution handle error)
        assert exchange == "binance"


@pytest.mark.asyncio
async def test_choose_exchange_health_check_exception():
    """Health check exception → try next exchange."""
    
    call_count = {"count": 0}
    
    # Mock: binance throws exception, bybit ok
    async def mock_health(exchange):
        call_count["count"] += 1
        if exchange == "binance":
            raise Exception("Network error")
        elif exchange == "bybit":
            return {"status": "ok", "latency_ms": 60, "last_error": None}
        else:
            return {"status": "down", "latency_ms": 0, "last_error": "Offline"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await choose_exchange_with_failover("binance", "binance")
        
        # Should skip binance (exception) and use bybit
        assert exchange == "bybit"
        assert call_count["count"] >= 2  # Tried binance + bybit


@pytest.mark.asyncio
async def test_choose_exchange_degraded_skipped():
    """Degraded exchange skipped, next healthy used."""
    
    # Mock: binance degraded, bybit degraded, okx ok
    async def mock_health(exchange):
        if exchange in ("binance", "bybit"):
            return {"status": "degraded", "latency_ms": 800, "last_error": "High latency"}
        elif exchange == "okx":
            return {"status": "ok", "latency_ms": 70, "last_error": None}
        else:
            return {"status": "down", "latency_ms": 0, "last_error": "Offline"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await choose_exchange_with_failover("binance", "binance")
        
        # Should skip binance (degraded), skip bybit (degraded), use okx (healthy)
        assert exchange == "okx"


# ============================================================================
# EXECUTION INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_resolve_exchange_with_failover():
    """resolve_exchange_with_failover wrapper works correctly."""
    from backend.services.execution.execution import resolve_exchange_with_failover
    
    # Mock: firi down, binance ok
    async def mock_health(exchange):
        if exchange == "firi":
            return {"status": "down", "latency_ms": 0, "last_error": "API offline"}
        elif exchange == "binance":
            return {"status": "ok", "latency_ms": 40, "last_error": None}
        else:
            return {"status": "down", "latency_ms": 0, "last_error": "Offline"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        exchange = await resolve_exchange_with_failover("firi", "binance")
        
        # Should failover from firi → binance
        assert exchange == "binance"


@pytest.mark.asyncio
async def test_failover_with_strategy_routing():
    """Failover works with strategy-based routing."""
    from backend.services.execution.execution import resolve_exchange_for_signal, resolve_exchange_with_failover
    from backend.policies.exchange_policy import set_strategy_exchange_mapping
    
    # Setup: Strategy maps to okx
    set_strategy_exchange_mapping({"scalper_test": "okx"})
    
    # Get primary exchange from routing
    primary = resolve_exchange_for_signal(signal_exchange=None, strategy_id="scalper_test")
    assert primary == "okx"
    
    # Mock: okx down, bybit ok
    async def mock_health(exchange):
        if exchange == "okx":
            return {"status": "down", "latency_ms": 0, "last_error": "Maintenance"}
        elif exchange == "bybit":
            return {"status": "ok", "latency_ms": 55, "last_error": None}
        else:
            return {"status": "down", "latency_ms": 0, "last_error": "Offline"}
    
    with patch(
        "backend.policies.exchange_failover_policy.get_exchange_health",
        side_effect=mock_health
    ):
        # Apply failover
        final = await resolve_exchange_with_failover(primary, "binance")
        
        # Should failover from okx → bybit
        assert final == "bybit"
