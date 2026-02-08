"""
Unit tests for Arbiter Agent #5 (Market Understanding)

Tests the Arbiter agent's ability to make trading decisions based on
technical indicators when Meta-Agent V2 escalates cases.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.agents.arbiter_agent import ArbiterAgent


@pytest.fixture
def arbiter():
    """Create default Arbiter agent."""
    return ArbiterAgent(
        confidence_threshold=0.70,
        enable_conservative_mode=True
    )


@pytest.fixture
def market_data_oversold():
    """Market data with oversold signals (should suggest BUY)."""
    return {
        'indicators': {
            'rsi': 25,  # Oversold
            'macd': 0.05,
            'macd_signal': -0.02,  # Bullish crossover
            'bb_position': 0.15,  # Near lower band
            'volume_trend': 1.8,  # High volume
            'momentum_1h': 0.025  # +2.5% momentum
        },
        'regime': {
            'volatility': 0.4,
            'trend_strength': 0.2  # Weak uptrend
        }
    }


@pytest.fixture
def market_data_overbought():
    """Market data with overbought signals (should suggest SELL)."""
    return {
        'indicators': {
            'rsi': 78,  # Overbought
            'macd': -0.03,
            'macd_signal': 0.02,  # Bearish crossunder
            'bb_position': 0.88,  # Near upper band
            'volume_trend': 1.6,
            'momentum_1h': -0.022  # -2.2% momentum
        },
        'regime': {
            'volatility': 0.5,
            'trend_strength': -0.3  # Downtrend
        }
    }


@pytest.fixture
def market_data_neutral():
    """Neutral market data (should suggest HOLD)."""
    return {
        'indicators': {
            'rsi': 52,
            'macd': 0.001,
            'macd_signal': 0.0005,
            'bb_position': 0.50,
            'volume_trend': 1.0,
            'momentum_1h': 0.005
        },
        'regime': {
            'volatility': 0.3,
            'trend_strength': 0.05
        }
    }


def test_arbiter_initialization():
    """Test Arbiter agent initialization."""
    arbiter = ArbiterAgent(
        confidence_threshold=0.75,
        enable_conservative_mode=False
    )
    
    assert arbiter.confidence_threshold == 0.75
    assert arbiter.enable_conservative_mode == False
    assert arbiter.total_calls == 0
    assert arbiter.decisions_made == 0


def test_arbiter_predict_oversold_buy_signal(arbiter, market_data_oversold):
    """Test Arbiter recognizes oversold conditions and suggests BUY."""
    result = arbiter.predict(
        market_data=market_data_oversold,
        symbol="BTCUSDT"
    )
    
    assert result['action'] == 'BUY', f"Expected BUY for oversold market, got {result['action']}"
    assert result['confidence'] >= 0.70, f"Expected high confidence, got {result['confidence']}"
    assert 'arbiter_invoked' in result
    assert result['arbiter_invoked'] == True
    assert 'oversold' in result['reason'].lower() or 'buy' in result['reason'].lower()
    assert arbiter.total_calls == 1


def test_arbiter_predict_overbought_sell_signal(arbiter, market_data_overbought):
    """Test Arbiter recognizes overbought conditions and suggests SELL."""
    result = arbiter.predict(
        market_data=market_data_overbought,
        symbol="ETHUSDT"
    )
    
    assert result['action'] == 'SELL', f"Expected SELL for overbought market, got {result['action']}"
    assert result['confidence'] >= 0.70
    assert 'overbought' in result['reason'].lower() or 'sell' in result['reason'].lower()
    assert arbiter.total_calls == 1


def test_arbiter_predict_neutral_hold(arbiter, market_data_neutral):
    """Test Arbiter returns HOLD for neutral market."""
    result = arbiter.predict(
        market_data=market_data_neutral,
        symbol="BTCUSDT"
    )
    
    # Neutral market may return HOLD or low confidence BUY/SELL
    assert result['action'] in ['BUY', 'SELL', 'HOLD']
    
     # If not HOLD, confidence should be relatively low
    if result['action'] != 'HOLD':
        assert result['confidence'] < 0.75, "Neutral market should have low confidence"
    
    assert arbiter.total_calls == 1


def test_arbiter_gating_low_confidence(arbiter):
    """Test Arbiter gating rejects low confidence decisions."""
    # Low confidence should NOT override
    assert arbiter.should_override_ensemble('BUY', 0.65) == False
    assert arbiter.should_override_ensemble('SELL', 0.68) == False
    
    # High confidence should override
    assert arbiter.should_override_ensemble('BUY', 0.75) == True
    assert arbiter.should_override_ensemble('SELL', 0.82) == True


def test_arbiter_gating_hold_rejection(arbiter):
    """Test Arbiter gating rejects HOLD actions."""
    # HOLD should never override, regardless of confidence
    assert arbiter.should_override_ensemble('HOLD', 0.95) == False
    assert arbiter.should_override_ensemble('HOLD', 0.70) == False
    
    # Active actions should override with high confidence
    assert arbiter.should_override_ensemble('BUY', 0.70) == True
    assert arbiter.should_override_ensemble('SELL', 0.70) == True


def test_arbiter_statistics_tracking(arbiter, market_data_oversold, market_data_neutral):
    """Test Arbiter tracks statistics correctly."""
    # Make several predictions
    result1 = arbiter.predict(market_data_oversold)
    result2 = arbiter.predict(market_data_neutral)
    result3 = arbiter.predict(market_data_oversold)
    
    stats = arbiter.get_statistics()
    
    assert stats['total_calls'] == 3
    assert stats['decisions_made'] >= 0  # BUY/SELL count
    assert stats['holds_returned'] >= 0
    assert 'decision_rate' in stats
    
    # Reset statistics
    arbiter.reset_statistics()
    stats_after = arbiter.get_statistics()
    
    assert stats_after['total_calls'] == 0
    assert stats_after['decisions_made'] == 0


def test_arbiter_no_market_data(arbiter):
    """Test Arbiter handles missing market data gracefully."""
    result = arbiter.predict(
        market_data=None,
        symbol="BTCUSDT"
    )
    
    assert result['action'] == 'HOLD'
    assert result['confidence'] == 0.5
    assert result['reason'] == 'no_market_data'


def test_arbiter_empty_indicators(arbiter):
    """Test Arbiter handles empty indicators."""
    result = arbiter.predict(
        market_data={'indicators': {}, 'regime': {}},
        symbol="BTCUSDT"
    )
    
    # Should still return a decision (likely HOLD with low confidence)
    assert result['action'] in ['BUY', 'SELL', 'HOLD']
    assert 0.0 <= result['confidence'] <= 1.0


def test_arbiter_with_base_signals_context(arbiter, market_data_oversold):
    """Test Arbiter uses base signals as context (not for voting)."""
    base_signals = {
        'xgb': {'action': 'SELL', 'confidence': 0.70},
        'lgbm': {'action': 'BUY', 'confidence': 0.65}
    }
    
    result = arbiter.predict(
        market_data=market_data_oversold,
        base_signals=base_signals,
        symbol="BTCUSDT"
    )
    
    # Arbiter should make decision based on technicals, not base signals
    # (base signals are context only)
    assert result['action'] == 'BUY'  # Based on oversold conditions
    assert result['confidence'] >= 0.70


def test_arbiter_conservative_mode_vs_normal(market_data_oversold):
    """Test conservative mode requires stronger signals."""
    arbiter_conservative = ArbiterAgent(
        confidence_threshold=0.70,
        enable_conservative_mode=True
    )
    
    arbiter_aggressive = ArbiterAgent(
        confidence_threshold=0.70,
        enable_conservative_mode=False
    )
    
    result_conservative = arbiter_conservative.predict(market_data_oversold)
    result_aggressive = arbiter_aggressive.predict(market_data_oversold)
    
    # Both should recognize BUY signal, but thresholds differ internally
    assert result_conservative['action'] == 'BUY'
    assert result_aggressive['action'] == 'BUY'


def test_arbiter_rsi_analysis(arbiter):
    """Test RSI-based decision making."""
    # Extreme oversold (RSI 20)
    data_oversold = {
        'indicators': {'rsi': 20, 'macd': 0, 'macd_signal': 0, 
                      'bb_position': 0.5, 'volume_trend': 1.0, 'momentum_1h': 0},
        'regime': {'volatility': 0.3, 'trend_strength': 0}
    }
    
    result = arbiter.predict(data_oversold)
    assert result['action'] == 'BUY', "RSI 20 should trigger BUY"
    
    # Extreme overbought (RSI 80)
    data_overbought = {
        'indicators': {'rsi': 80, 'macd': 0, 'macd_signal': 0,
                      'bb_position': 0.5, 'volume_trend': 1.0, 'momentum_1h': 0},
        'regime': {'volatility': 0.3, 'trend_strength': 0}
    }
    
    result = arbiter.predict(data_overbought)
    assert result['action'] == 'SELL', "RSI 80 should trigger SELL"


def test_arbiter_macd_momentum(arbiter):
    """Test MACD-based momentum detection."""
    # Strong bullish MACD
    data_bullish = {
        'indicators': {'rsi': 50, 'macd': 0.08, 'macd_signal': -0.02,
                      'bb_position': 0.5, 'volume_trend': 1.5, 'momentum_1h': 0.03},
        'regime': {'volatility': 0.4, 'trend_strength': 0.3}
    }
    
    result = arbiter.predict(data_bullish)
    assert result['action'] == 'BUY', "Strong bullish MACD should trigger BUY"


def test_arbiter_bollinger_bands(arbiter):
    """Test Bollinger Bands positioning."""
    # Price at lower band
    data_lower_band = {
        'indicators': {'rsi': 45, 'macd': 0, 'macd_signal': 0,
                      'bb_position': 0.05, 'volume_trend': 1.0, 'momentum_1h': 0},
        'regime': {'volatility': 0.3, 'trend_strength': 0}
    }
    
    result = arbiter.predict(data_lower_band)
    # Lower band suggests potential bounce → BUY
    # (may not trigger if other indicators are weak)
    assert result['action'] in ['BUY', 'HOLD']


def test_arbiter_high_volume_confirmation(arbiter):
    """Test volume trend amplifies signals."""
    # High volume + positive momentum
    data_high_vol = {
        'indicators': {'rsi': 40, 'macd': 0.03, 'macd_signal': 0,
                      'bb_position': 0.3, 'volume_trend': 2.5, 'momentum_1h': 0.025},
        'regime': {'volatility': 0.5, 'trend_strength': 0.2}
    }
    
    result = arbiter.predict(data_high_vol)
    # High volume should amplify BUY signal
    assert result['action'] == 'BUY'
    assert result['confidence'] >= 0.70


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
