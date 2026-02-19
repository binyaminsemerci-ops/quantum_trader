import os

import pytest

from backend.config import (
    ExecutionConfig,
    LiquidityConfig,
    RiskConfig,
    load_execution_config,
    load_liquidity_config,
    load_risk_config,
)


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch):
    """Ensure each test starts with a clean configuration environment."""

    keys = [
        # Liquidity
        "QT_LIQUIDITY_UNIVERSE_MAX",
        "QT_LIQUIDITY_SELECTION_MIN",
        "QT_LIQUIDITY_SELECTION_MAX",
        "QT_LIQUIDITY_MIN_QUOTE_VOLUME",
        "QT_LIQUIDITY_MOMENTUM_WEIGHT",
        "QT_LIQUIDITY_BINANCE_WEIGHT",
        "QT_LIQUIDITY_COINGECKO_WEIGHT",
        "QT_LIQUIDITY_LIQ_WEIGHT",
        "QT_LIQUIDITY_MODEL_WEIGHT",
        "QT_LIQUIDITY_MODEL_SELL_THRESHOLD",
        "QT_LIQUIDITY_STABLE_QUOTES",
        "QT_LIQUIDITY_BLACKLIST_SUFFIXES",
        "QT_LIQUIDITY_MAX_PER_BASE",
        # Execution
        "QT_EXECUTION_MIN_NOTIONAL",
        "QT_EXECUTION_MAX_ORDERS",
        "QT_EXECUTION_CASH_BUFFER",
        "QT_EXECUTION_ALLOW_PARTIAL",
        "QT_EXECUTION_EXCHANGE",
        "QT_EXECUTION_QUOTE_ASSET",
        "QT_EXECUTION_BINANCE_TESTNET",
        "DEFAULT_QUOTE",
        # Risk
        "STAGING_MODE",
        "QT_KILL_SWITCH",
        "QT_MAX_NOTIONAL_PER_TRADE",
        "QT_MAX_DAILY_LOSS",
        "QT_ALLOWED_SYMBOLS",
        "QT_FAILSAFE_RESET_MINUTES",
        "QT_RISK_STATE_DB",
        "QT_ADMIN_TOKEN",
        "QT_MAX_POSITION_PER_SYMBOL",
        "QT_MAX_GROSS_EXPOSURE",
        "QT_MIN_UNIT_PRICE",
        "QT_MAX_UNIT_PRICE",
        "QT_MAX_PRICE_STALENESS_SECONDS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_load_liquidity_config_defaults():
    cfg = load_liquidity_config()
    assert isinstance(cfg, LiquidityConfig)
    assert cfg.universe_max == 100
    assert cfg.selection_min == 3
    assert cfg.selection_max == 10
    assert cfg.liquidity_weight + cfg.model_weight == pytest.approx(1.0)
    assert cfg.max_per_base == 1
    assert cfg.blacklist_suffixes == ("UP", "DOWN", "BULL", "BEAR", "PERP")


def test_load_liquidity_config_env_overrides(monkeypatch):
    monkeypatch.setenv("QT_LIQUIDITY_LIQ_WEIGHT", "0.2")
    monkeypatch.setenv("QT_LIQUIDITY_MODEL_WEIGHT", "0.8")
    monkeypatch.setenv("QT_LIQUIDITY_SELECTION_MIN", "5")
    monkeypatch.setenv("QT_LIQUIDITY_SELECTION_MAX", "3")  # min should win
    monkeypatch.setenv("QT_LIQUIDITY_STABLE_QUOTES", "usdt, eur")
    monkeypatch.setenv("QT_LIQUIDITY_MAX_PER_BASE", "-2")

    cfg = load_liquidity_config()
    assert cfg.selection_min == 5
    assert cfg.selection_max == 5  # corrected to min when max < min
    assert cfg.liquidity_weight == pytest.approx(0.2 / (0.2 + 0.8))
    assert cfg.model_weight == pytest.approx(0.8 / (0.2 + 0.8))
    assert cfg.stable_quote_assets == ("USDT", "EUR")
    assert cfg.max_per_base == 0  # negative coerced to zero


def test_load_execution_config_defaults():
    cfg = load_execution_config()
    assert isinstance(cfg, ExecutionConfig)
    assert cfg.min_notional == 50.0
    assert cfg.max_orders == 10
    assert cfg.allow_partial is True
    assert cfg.exchange == "paper"
    assert cfg.quote_asset == "USDT"


def test_load_execution_config_env_overrides(monkeypatch):
    monkeypatch.setenv("QT_EXECUTION_MIN_NOTIONAL", "75")
    monkeypatch.setenv("QT_EXECUTION_MAX_ORDERS", "-3")  # coerced to >= 0
    monkeypatch.setenv("QT_EXECUTION_ALLOW_PARTIAL", "false")
    monkeypatch.setenv("QT_EXECUTION_EXCHANGE", "BINANCE")
    monkeypatch.setenv("QT_EXECUTION_QUOTE_ASSET", "eth")
    monkeypatch.setenv("QT_EXECUTION_BINANCE_TESTNET", "1")
    monkeypatch.setenv("QT_EXECUTION_CASH_BUFFER", "-10")

    cfg = load_execution_config()
    assert cfg.min_notional == 75.0
    assert cfg.max_orders == 0
    assert cfg.allow_partial is False
    assert cfg.exchange == "binance"
    assert cfg.quote_asset == "ETH"
    assert cfg.binance_testnet is True
    assert cfg.cash_buffer == 0.0


def test_load_risk_config_defaults():
    cfg = load_risk_config()
    assert isinstance(cfg, RiskConfig)
    assert cfg.kill_switch is False
    assert cfg.allowed_symbols == ["BTCUSDT", "ETHUSDT"]
    assert cfg.max_notional_per_trade == 1000.0
    assert cfg.max_daily_loss == 500.0
    assert cfg.max_position_per_symbol is None


def test_load_risk_config_env_overrides(monkeypatch):
    monkeypatch.setenv("QT_KILL_SWITCH", "1")
    monkeypatch.setenv("QT_ALLOWED_SYMBOLS", "btceth, ada usdt ,")
    monkeypatch.setenv("QT_MAX_POSITION_PER_SYMBOL", "-100")  # invalid -> None
    monkeypatch.setenv("QT_MAX_GROSS_EXPOSURE", "2000")
    monkeypatch.setenv("QT_MIN_UNIT_PRICE", "10")
    monkeypatch.setenv("QT_MAX_UNIT_PRICE", " 100 ")
    monkeypatch.setenv("QT_MAX_PRICE_STALENESS_SECONDS", "900")

    cfg = load_risk_config()
    assert cfg.kill_switch is True
    assert cfg.allowed_symbols == ["BTCETH", "ADAUSDT"]
    assert cfg.max_position_per_symbol is None
    assert cfg.max_gross_exposure == 2000.0
    assert cfg.min_unit_price == 10.0
    assert cfg.max_unit_price == 100.0
    assert cfg.max_price_staleness_seconds == 900
