"""
Exchange Routing Tests

EPIC-EXCH-ROUTING-001: Test strategy → exchange mapping logic.
"""

import pytest
from backend.policies.exchange_policy import (
    get_exchange_for_strategy,
    validate_exchange_name,
    set_strategy_exchange_mapping,
    ALLOWED_EXCHANGES,
    DEFAULT_EXCHANGE,
)
from backend.services.execution.execution import resolve_exchange_for_signal


def test_get_exchange_for_strategy_with_mapping():
    """Strategy with explicit mapping returns mapped exchange."""
    # Setup: Add test mapping
    set_strategy_exchange_mapping({"test_scalper": "bybit"})
    
    exchange = get_exchange_for_strategy("test_scalper")
    
    assert exchange == "bybit"


def test_get_exchange_for_strategy_without_mapping():
    """Strategy without mapping returns default exchange."""
    exchange = get_exchange_for_strategy("unmapped_strategy")
    
    assert exchange == DEFAULT_EXCHANGE


def test_get_exchange_for_strategy_none():
    """None strategy_id returns default exchange."""
    exchange = get_exchange_for_strategy(None)
    
    assert exchange == DEFAULT_EXCHANGE


def test_validate_exchange_name_valid():
    """Valid exchange name passes validation."""
    for exchange in ALLOWED_EXCHANGES:
        validated = validate_exchange_name(exchange)
        assert validated == exchange


def test_validate_exchange_name_invalid():
    """Invalid exchange name falls back to default."""
    validated = validate_exchange_name("unknown_exchange")
    
    assert validated == DEFAULT_EXCHANGE


def test_resolve_exchange_for_signal_explicit_override():
    """Explicit signal.exchange takes priority."""
    exchange = resolve_exchange_for_signal(
        signal_exchange="okx",
        strategy_id="test_strategy"
    )
    
    assert exchange == "okx"


def test_resolve_exchange_for_signal_strategy_mapping():
    """Strategy mapping used when no explicit exchange."""
    set_strategy_exchange_mapping({"test_swing": "kraken"})
    
    exchange = resolve_exchange_for_signal(
        signal_exchange=None,
        strategy_id="test_swing"
    )
    
    assert exchange == "kraken"


def test_resolve_exchange_for_signal_fallback():
    """Default exchange used when no explicit exchange or strategy."""
    exchange = resolve_exchange_for_signal(
        signal_exchange=None,
        strategy_id=None
    )
    
    assert exchange == DEFAULT_EXCHANGE


def test_resolve_exchange_for_signal_invalid_exchange():
    """Invalid exchange validated and falls back."""
    exchange = resolve_exchange_for_signal(
        signal_exchange="fake_exchange",
        strategy_id=None
    )
    
    assert exchange == DEFAULT_EXCHANGE


def test_build_execution_adapter_with_override():
    """build_execution_adapter respects exchange_override."""
    from backend.services.execution.execution import build_execution_adapter
    from backend.config import ExecutionConfig
    
    config = ExecutionConfig(exchange="binance")
    
    # Override to paper mode (always available)
    adapter = build_execution_adapter(config, exchange_override="paper")
    
    # PaperExchangeAdapter should be returned
    assert adapter.__class__.__name__ == "PaperExchangeAdapter"


def test_set_strategy_exchange_mapping():
    """set_strategy_exchange_mapping updates mapping."""
    original_count = len(get_exchange_for_strategy.__globals__['STRATEGY_EXCHANGE_MAP'])
    
    set_strategy_exchange_mapping({
        "new_strategy_1": "firi",
        "new_strategy_2": "kucoin",
    })
    
    from backend.policies.exchange_policy import STRATEGY_EXCHANGE_MAP
    assert "new_strategy_1" in STRATEGY_EXCHANGE_MAP
    assert STRATEGY_EXCHANGE_MAP["new_strategy_1"] == "firi"
