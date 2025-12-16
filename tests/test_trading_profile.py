"""
Unit tests for Trading Profile System.

Tests all 6 subsystems:
1. Liquidity filtering
2. Position sizing
3. Dynamic TP/SL
4. Funding protection
5. Universe classification
6. Trade validation

Author: Quantum Trader Team
Date: 2025-11-26
"""

import pytest
from datetime import datetime, timezone, timedelta
from backend.services.ai.trading_profile import (
    SymbolMetrics,
    UniverseTier,
    LiquidityConfig,
    RiskConfig,
    TpslConfig,
    FundingConfig,
    DynamicTpslLevels,
    compute_spread,
    compute_spread_bps,
    compute_liquidity_score,
    is_symbol_tradeable,
    filter_and_rank_universe,
    compute_position_margin,
    compute_effective_leverage,
    compute_position_size,
    compute_dynamic_tpsl_long,
    compute_dynamic_tpsl_short,
    is_funding_window_blocked,
    is_funding_rate_unfavourable,
    check_funding_protection,
    validate_trade,
    classify_symbol_tier,
    calculate_risk_reward_ratio,
)


# ==================== Fixtures ====================

@pytest.fixture
def btc_metrics():
    """High-quality BTC metrics."""
    return SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,  # $1B volume
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,  # $5M depth
        funding_rate=0.0001,  # 0.01%
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )


@pytest.fixture
def tao_metrics():
    """Low-quality TAO metrics (excluded symbol)."""
    return SymbolMetrics(
        symbol="TAOUSDT",
        quote_volume_24h=1_000_000,  # Only $1M volume
        bid=500.0,
        ask=505.0,  # 1% spread!
        depth_notional_5bps=50_000,  # $50k depth (low)
        funding_rate=-0.001,  # -0.1% (high)
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=502.5,
        index_price=502.0,
        open_interest=10_000_000,
        universe_tier=UniverseTier.EXCLUDED
    )


@pytest.fixture
def liquidity_config():
    """Default liquidity configuration."""
    return LiquidityConfig(
        min_quote_volume_24h=5_000_000,
        max_spread_bps=3.0,
        min_depth_notional=200_000,
        w_volume=0.5,
        w_spread=0.3,
        w_depth=0.2,
        min_liquidity_score=0.0,
        max_universe_size=20,
        allowed_tiers=[UniverseTier.MAIN, UniverseTier.L1, UniverseTier.L2]
    )


@pytest.fixture
def risk_config():
    """Default risk configuration."""
    return RiskConfig(
        base_risk_frac=0.01,
        max_risk_frac=0.03,
        min_margin=10.0,
        max_margin=1000.0,
        max_total_risk=0.15,
        max_positions=8,
        min_ai_risk_factor=0.5,
        max_ai_risk_factor=1.5,
        default_leverage=30,
        effective_leverage_main=15.0,
        effective_leverage_l1=12.0,
        effective_leverage_l2=10.0,
        effective_leverage_min=8.0
    )


@pytest.fixture
def tpsl_config():
    """Default TP/SL configuration."""
    return TpslConfig(
        atr_period=14,
        atr_timeframe="15m",
        atr_mult_base=1.0,
        atr_mult_sl=1.0,
        atr_mult_tp1=1.5,
        atr_mult_tp2=2.5,
        atr_mult_tp3=4.0,
        atr_mult_be=1.0,
        be_buffer_bps=5.0,
        trail_dist_mult=0.8,
        trail_activation_mult=2.5,
        partial_close_frac_tp1=0.5,
        partial_close_frac_tp2=0.3
    )


@pytest.fixture
def funding_config():
    """Default funding configuration."""
    return FundingConfig(
        pre_window_minutes=40,
        post_window_minutes=20,
        min_long_funding=-0.0003,
        max_short_funding=0.0003,
        extreme_funding_threshold=0.001,
        high_funding_threshold=0.0005
    )


# ==================== Liquidity Tests ====================

def test_spread_calculation(btc_metrics, tao_metrics):
    """Test spread calculation."""
    btc_spread = compute_spread(btc_metrics)
    tao_spread = compute_spread(tao_metrics)
    
    assert btc_spread == pytest.approx(5.0, rel=0.01)  # $5 spread
    assert tao_spread == pytest.approx(5.0, rel=0.01)  # $5 spread
    
    # BPS calculation
    btc_spread_bps = compute_spread_bps(btc_metrics)
    tao_spread_bps = compute_spread_bps(tao_metrics)
    
    assert btc_spread_bps < 2.0  # ~1.15 bps (good)
    assert tao_spread_bps > 50.0  # ~99 bps (bad)


def test_liquidity_score(btc_metrics, tao_metrics, liquidity_config):
    """Test liquidity scoring."""
    btc_score = compute_liquidity_score(btc_metrics, liquidity_config)
    tao_score = compute_liquidity_score(tao_metrics, liquidity_config)
    
    # BTC should have much higher score
    assert btc_score > tao_score
    assert btc_score > 10.0  # High score
    assert tao_score < 5.0   # Low score


def test_symbol_tradeability(btc_metrics, tao_metrics, liquidity_config):
    """Test symbol tradeable check."""
    btc_ok, btc_reason = is_symbol_tradeable(btc_metrics, liquidity_config)
    tao_ok, tao_reason = is_symbol_tradeable(tao_metrics, liquidity_config)
    
    assert btc_ok is True
    assert "tradeable" in btc_reason.lower()
    
    assert tao_ok is False
    assert "volume" in tao_reason.lower() or "spread" in tao_reason.lower()


def test_universe_filtering(btc_metrics, tao_metrics, liquidity_config):
    """Test universe filtering and ranking."""
    all_symbols = [btc_metrics, tao_metrics]
    
    filtered = filter_and_rank_universe(all_symbols, liquidity_config)
    
    # Only BTC should pass
    assert len(filtered) == 1
    assert filtered[0].symbol == "BTCUSDT"


def test_universe_tier_classification():
    """Test symbol tier classification."""
    assert classify_symbol_tier("BTCUSDT") == UniverseTier.MAIN
    assert classify_symbol_tier("ETHUSDT") == UniverseTier.MAIN
    assert classify_symbol_tier("SOLUSDT") == UniverseTier.L1
    assert classify_symbol_tier("ARBUSDT") == UniverseTier.L2
    assert classify_symbol_tier("TAOUSDT") == UniverseTier.EXCLUDED
    assert classify_symbol_tier("DOGEUSDT") == UniverseTier.MEME
    assert classify_symbol_tier("UNKNOWN") == UniverseTier.L2  # Default


# ==================== Position Sizing Tests ====================

def test_position_margin_base_case(risk_config):
    """Test basic position margin calculation."""
    equity = 10000.0
    ai_risk_factor = 1.0  # Neutral
    
    margin = compute_position_margin(equity, ai_risk_factor, risk_config)
    
    # Should be 1% of equity
    assert margin == pytest.approx(100.0, rel=0.01)


def test_position_margin_conservative_ai(risk_config):
    """Test margin with conservative AI (low conviction)."""
    equity = 10000.0
    ai_risk_factor = 0.5  # Conservative
    
    margin = compute_position_margin(equity, ai_risk_factor, risk_config)
    
    # Should be 0.5% of equity
    assert margin == pytest.approx(50.0, rel=0.01)


def test_position_margin_aggressive_ai(risk_config):
    """Test margin with aggressive AI (high conviction)."""
    equity = 10000.0
    ai_risk_factor = 1.5  # Aggressive
    
    margin = compute_position_margin(equity, ai_risk_factor, risk_config)
    
    # Should be 1.5% of equity
    assert margin == pytest.approx(150.0, rel=0.01)


def test_position_margin_caps(risk_config):
    """Test margin respects min/max caps."""
    # Test min cap
    equity = 100.0  # Small account
    margin_min = compute_position_margin(equity, 1.0, risk_config)
    assert margin_min == risk_config.min_margin  # $10
    
    # Test max cap
    equity = 200_000.0  # Large account
    margin_max = compute_position_margin(equity, 1.5, risk_config)
    assert margin_max == risk_config.max_margin  # $1000


def test_effective_leverage_by_tier(btc_metrics, risk_config):
    """Test effective leverage calculation by universe tier."""
    # BTC (MAIN tier)
    btc_lev = compute_effective_leverage(btc_metrics, risk_config)
    assert btc_lev == 15.0  # Main tier: 15x
    
    # Change tier to L1
    btc_metrics.universe_tier = UniverseTier.L1
    l1_lev = compute_effective_leverage(btc_metrics, risk_config)
    assert l1_lev == 12.0  # L1 tier: 12x
    
    # Change tier to L2
    btc_metrics.universe_tier = UniverseTier.L2
    l2_lev = compute_effective_leverage(btc_metrics, risk_config)
    assert l2_lev == 10.0  # L2 tier: 10x


def test_position_size_calculation(risk_config):
    """Test position size calculation."""
    margin = 100.0
    leverage = 15.0
    entry_price = 43500.0
    
    quantity = compute_position_size(margin, leverage, entry_price)
    notional = quantity * entry_price
    
    # Notional should be margin * leverage
    assert notional == pytest.approx(1500.0, rel=0.01)
    # Quantity should be notional / price
    assert quantity == pytest.approx(0.0345, rel=0.01)


# ==================== TP/SL Tests ====================

def test_tpsl_long_calculation(tpsl_config):
    """Test dynamic TP/SL for LONG positions."""
    entry_price = 43500.0
    atr = 650.0  # $650 ATR
    
    levels = compute_dynamic_tpsl_long(entry_price, atr, tpsl_config)
    
    # SL should be 1R below entry
    expected_sl = entry_price - (atr * 1.0)
    assert levels.sl_init == pytest.approx(expected_sl, rel=0.01)
    assert levels.sl_init == pytest.approx(42850.0, rel=0.01)
    
    # TP1 should be 1.5R above entry
    expected_tp1 = entry_price + (atr * 1.5)
    assert levels.tp1 == pytest.approx(expected_tp1, rel=0.01)
    assert levels.tp1 == pytest.approx(44475.0, rel=0.01)
    
    # TP2 should be 2.5R above entry
    expected_tp2 = entry_price + (atr * 2.5)
    assert levels.tp2 == pytest.approx(expected_tp2, rel=0.01)
    assert levels.tp2 == pytest.approx(45125.0, rel=0.01)
    
    # Break-even trigger should be 1R above entry
    expected_be_trigger = entry_price + (atr * 1.0)
    assert levels.be_trigger == pytest.approx(expected_be_trigger, rel=0.01)
    
    # Break-even price should be entry + buffer
    assert levels.be_price >= entry_price


def test_tpsl_short_calculation(tpsl_config):
    """Test dynamic TP/SL for SHORT positions."""
    entry_price = 43500.0
    atr = 650.0
    
    levels = compute_dynamic_tpsl_short(entry_price, atr, tpsl_config)
    
    # SL should be 1R above entry
    expected_sl = entry_price + (atr * 1.0)
    assert levels.sl_init == pytest.approx(expected_sl, rel=0.01)
    assert levels.sl_init == pytest.approx(44150.0, rel=0.01)
    
    # TP1 should be 1.5R below entry
    expected_tp1 = entry_price - (atr * 1.5)
    assert levels.tp1 == pytest.approx(expected_tp1, rel=0.01)
    assert levels.tp1 == pytest.approx(42525.0, rel=0.01)
    
    # TP2 should be 2.5R below entry
    expected_tp2 = entry_price - (atr * 2.5)
    assert levels.tp2 == pytest.approx(expected_tp2, rel=0.01)
    assert levels.tp2 == pytest.approx(41875.0, rel=0.01)


def test_risk_reward_ratio():
    """Test risk/reward ratio calculation."""
    entry = 43500.0
    sl = 42850.0
    tp = 44475.0
    
    rr_ratio = calculate_risk_reward_ratio(entry, sl, tp)
    
    # Risk: 650, Reward: 975 → RR = 1.5
    assert rr_ratio == pytest.approx(1.5, rel=0.01)


# ==================== Funding Protection Tests ====================

def test_funding_window_pre_blocked(funding_config):
    """Test funding window blocks entries before funding."""
    now = datetime.now(timezone.utc)
    
    # Funding in 30 minutes (within 40min window)
    metrics = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=0.0001,
        next_funding_time=now + timedelta(minutes=30),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    blocked, reason = is_funding_window_blocked(now, metrics, funding_config)
    
    assert blocked is True
    assert "before" in reason.lower() or "pre" in reason.lower()


def test_funding_window_post_blocked(funding_config):
    """Test funding window blocks entries after funding."""
    now = datetime.now(timezone.utc)
    
    # Funding 10 minutes ago (within 20min window)
    metrics = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=0.0001,
        next_funding_time=now - timedelta(minutes=10),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    blocked, reason = is_funding_window_blocked(now, metrics, funding_config)
    
    assert blocked is True
    assert "after" in reason.lower() or "post" in reason.lower()


def test_funding_window_allowed(funding_config):
    """Test funding window allows entries when safe."""
    now = datetime.now(timezone.utc)
    
    # Funding in 4 hours (safe)
    metrics = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=0.0001,
        next_funding_time=now + timedelta(hours=4),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    blocked, reason = is_funding_window_blocked(now, metrics, funding_config)
    
    assert blocked is False


def test_funding_rate_long_unfavourable(funding_config):
    """Test unfavourable funding rate for LONG."""
    # Negative funding (longs pay)
    metrics = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=-0.0005,  # -0.05% (high negative)
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    unfav, reason = is_funding_rate_unfavourable("LONG", metrics, funding_config)
    
    assert unfav is True
    assert "long" in reason.lower() and "negative" in reason.lower()


def test_funding_rate_short_unfavourable(funding_config):
    """Test unfavourable funding rate for SHORT."""
    # Positive funding (shorts pay)
    metrics = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=0.0005,  # +0.05% (high positive)
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    unfav, reason = is_funding_rate_unfavourable("SHORT", metrics, funding_config)
    
    assert unfav is True
    assert "short" in reason.lower() and "positive" in reason.lower()


# ==================== Integration Tests ====================

def test_validate_trade_btc_pass(btc_metrics, liquidity_config, funding_config):
    """Test complete trade validation for BTC (should pass)."""
    valid, reason = validate_trade(
        btc_metrics,
        "LONG",
        liquidity_config,
        funding_config
    )
    
    assert valid is True
    assert "validation passed" in reason.lower() or "tradeable" in reason.lower()


def test_validate_trade_tao_fail(tao_metrics, liquidity_config, funding_config):
    """Test complete trade validation for TAO (should fail)."""
    valid, reason = validate_trade(
        tao_metrics,
        "LONG",
        liquidity_config,
        funding_config
    )
    
    assert valid is False
    # Should fail on liquidity (volume or spread)
    assert "volume" in reason.lower() or "spread" in reason.lower() or "tier" in reason.lower()


def test_validate_trade_funding_window_blocked(btc_metrics, liquidity_config, funding_config):
    """Test trade validation fails during funding window."""
    # Set funding time to 30 minutes from now
    btc_metrics.next_funding_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    
    valid, reason = validate_trade(
        btc_metrics,
        "LONG",
        liquidity_config,
        funding_config
    )
    
    assert valid is False
    assert "funding" in reason.lower()


def test_complete_workflow():
    """Test complete workflow from validation to TP/SL calculation."""
    # 1. Create symbol metrics
    symbol = SymbolMetrics(
        symbol="BTCUSDT",
        quote_volume_24h=1_000_000_000,
        bid=43500.0,
        ask=43505.0,
        depth_notional_5bps=5_000_000,
        funding_rate=0.0001,
        next_funding_time=datetime.now(timezone.utc) + timedelta(hours=4),
        mark_price=43502.5,
        index_price=43502.0,
        open_interest=10_000_000_000,
        universe_tier=UniverseTier.MAIN
    )
    
    # 2. Validate trade
    liq_cfg = LiquidityConfig()
    fund_cfg = FundingConfig()
    valid, reason = validate_trade(symbol, "LONG", liq_cfg, fund_cfg)
    assert valid is True
    
    # 3. Calculate position size
    risk_cfg = RiskConfig()
    equity = 10000.0
    ai_risk_factor = 1.2
    
    margin = compute_position_margin(equity, ai_risk_factor, risk_cfg)
    leverage = compute_effective_leverage(symbol, risk_cfg)
    quantity = compute_position_size(margin, leverage, 43500.0)
    
    assert margin > 0
    assert leverage == 15.0  # BTC = MAIN tier
    assert quantity > 0
    
    # 4. Calculate TP/SL
    tpsl_cfg = TpslConfig()
    atr = 650.0
    levels = compute_dynamic_tpsl_long(43500.0, atr, tpsl_cfg)
    
    assert levels.sl_init < 43500.0
    assert levels.tp1 > 43500.0
    assert levels.tp2 > levels.tp1
    
    # Verify risk/reward ratios
    assert levels.risk_r == pytest.approx(1.0, rel=0.01)
    assert levels.reward_r_tp1 == pytest.approx(1.5, rel=0.01)
    assert levels.reward_r_tp2 == pytest.approx(2.5, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
