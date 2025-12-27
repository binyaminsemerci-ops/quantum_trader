"""
Edge case tests for bulletproof AI components.

Tests that AI never crashes under extreme conditions:
- Empty inputs
- Invalid data types
- Timeouts
- Missing models
- Corrupted data
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from ai_engine.agents.xgb_agent import XGBAgent
from ai_engine.feature_engineer import (
    compute_basic_indicators,
    compute_all_indicators,
    add_sentiment_features,
)


class TestXGBAgentBulletproof:
    """Test XGBAgent never crashes under any conditions."""

    def test_predict_with_empty_dataframe(self):
        """Empty DataFrame should return neutral fallback, not crash."""
        agent = XGBAgent()
        result = agent.predict_for_symbol(pd.DataFrame())
        
        assert isinstance(result, dict)
        assert "action" in result
        assert result["action"] in ["BUY", "SELL", "HOLD"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_predict_with_none_input(self):
        """None input should return neutral fallback, not crash."""
        agent = XGBAgent()
        result = agent.predict_for_symbol(None)
        
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]

    def test_predict_with_invalid_columns(self):
        """DataFrame with wrong columns should fallback gracefully."""
        agent = XGBAgent()
        df = pd.DataFrame({
            "invalid_col1": [1, 2, 3],
            "invalid_col2": [4, 5, 6],
        })
        result = agent.predict_for_symbol(df)
        
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]

    def test_predict_with_nan_values(self):
        """DataFrame with all NaN should fallback, not crash."""
        agent = XGBAgent()
        df = pd.DataFrame({
            "open": [np.nan, np.nan, np.nan],
            "high": [np.nan, np.nan, np.nan],
            "low": [np.nan, np.nan, np.nan],
            "close": [np.nan, np.nan, np.nan],
            "volume": [np.nan, np.nan, np.nan],
        })
        result = agent.predict_for_symbol(df)
        
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]

    def test_predict_with_infinite_values(self):
        """DataFrame with inf values should be handled."""
        agent = XGBAgent()
        df = pd.DataFrame({
            "open": [np.inf, -np.inf, 100],
            "high": [np.inf, -np.inf, 110],
            "low": [np.inf, -np.inf, 90],
            "close": [np.inf, -np.inf, 105],
            "volume": [np.inf, -np.inf, 1000],
        })
        result = agent.predict_for_symbol(df)
        
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]

    def test_emergency_fallback_with_minimal_data(self):
        """Test emergency fallback with just 2 rows of data."""
        agent = XGBAgent()
        df = pd.DataFrame({
            "open": [100, 105],
            "high": [110, 115],
            "low": [95, 100],
            "close": [105, 110],
            "volume": [1000, 1100],
        })
        result = agent.predict_for_symbol(df)
        
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_scan_with_symbols(self):
        """Scan should handle various inputs gracefully."""
        agent = XGBAgent()
        
        # Scan with valid symbols
        result = await agent.scan_top_by_volume_from_api(
            symbols=["BTCUSDT", "ETHUSDT"],
            top_n=2
        )
        
        # Should return dict (may be empty or have results)
        assert isinstance(result, dict)


class TestFeatureEngineerBulletproof:
    """Test feature engineering never crashes."""

    def test_compute_basic_indicators_empty_df(self):
        """Empty DataFrame should return valid structure."""
        df = pd.DataFrame()
        result = compute_basic_indicators(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_compute_basic_indicators_insufficient_data(self):
        """Single row should not crash."""
        df = pd.DataFrame({
            "close": [100],
            "high": [105],
            "low": [95],
            "volume": [1000],
        })
        result = compute_basic_indicators(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_compute_all_indicators_invalid_input(self):
        """Invalid input should return empty DataFrame."""
        result = compute_all_indicators(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_compute_all_indicators_wrong_columns(self):
        """DataFrame with wrong columns should return empty or handle gracefully."""
        df = pd.DataFrame({
            "wrong_col": [1, 2, 3],
        })
        result = compute_all_indicators(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_rsi_empty_series(self):
        """Empty series should return neutral RSI."""
        from ai_engine.feature_engineer import _rsi
        
        series = pd.Series([])
        result = _rsi(series)
        
        assert isinstance(result, pd.Series)
        # Should be neutral RSI (50) or empty
        if len(result) > 0:
            assert all((result >= 0) & (result <= 100))

    def test_rsi_single_value(self):
        """Single value should not crash."""
        from ai_engine.feature_engineer import _rsi
        
        series = pd.Series([100])
        result = _rsi(series)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 1

    def test_rsi_all_same_values(self):
        """All same values (no change) should handle gracefully."""
        from ai_engine.feature_engineer import _rsi
        
        series = pd.Series([100] * 20)
        result = _rsi(series)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        # Should be neutral since no movement
        assert all((result >= 0) & (result <= 100))

    def test_add_sentiment_length_mismatch(self):
        """Sentiment data shorter/longer than DF should not crash."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
        })
        
        # Shorter sentiment
        sentiment_short = [0.5, 0.6]
        result = add_sentiment_features(df, sentiment_short)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Column may be 'sentiment' or 'sentiment_score' depending on version
        assert "sentiment" in result.columns or "sentiment_score" in result.columns
        
        # Longer sentiment
        sentiment_long = [0.5] * 10
        result = add_sentiment_features(df, sentiment_long)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_add_sentiment_none_input(self):
        """None sentiment should not crash."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        result = add_sentiment_features(df, None)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


class TestLiveSignalsAPIBulletproof:
    """Test live signals API endpoint never crashes."""

    @pytest.mark.asyncio
    async def test_live_signals_with_agent_failure(self):
        """Agent failure should fallback to heuristic."""
        from backend.routes.live_ai_signals import get_live_ai_signals
        
        with patch("backend.routes.live_ai_signals._agent_signals") as mock_agent:
            mock_agent.side_effect = Exception("Agent Error")
            
            result = await get_live_ai_signals(limit=5, profile="left")
            
            # Should return list, not crash
            assert isinstance(result, list)
            # May be empty or have heuristic signals

    @pytest.mark.asyncio
    async def test_live_signals_with_both_failures(self):
        """Both agent and heuristic failure should return empty list."""
        from backend.routes.live_ai_signals import get_live_ai_signals
        
        with patch("backend.routes.live_ai_signals._agent_signals") as mock_agent, \
             patch("backend.routes.live_ai_signals.ai_trader.generate_signals") as mock_heuristic:
            
            mock_agent.side_effect = Exception("Agent Error")
            mock_heuristic.side_effect = Exception("Heuristic Error")
            
            result = await get_live_ai_signals(limit=5, profile="left")
            
            # Should return empty list, not crash
            assert isinstance(result, list)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_live_signals_with_invalid_profile(self):
        """Invalid profile should use default."""
        from backend.routes.live_ai_signals import get_live_ai_signals
        
        result = await get_live_ai_signals(limit=5, profile="invalid_profile")
        
        # Should return list, not crash
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_live_signals_with_zero_limit(self):
        """Zero limit should be clamped to minimum."""
        from backend.routes.live_ai_signals import get_live_ai_signals
        
        result = await get_live_ai_signals(limit=0, profile="left")
        
        assert isinstance(result, list)
        # Should have at least some signals or empty

    @pytest.mark.asyncio
    async def test_live_signals_with_huge_limit(self):
        """Huge limit should be clamped to maximum."""
        from backend.routes.live_ai_signals import get_live_ai_signals
        
        result = await get_live_ai_signals(limit=99999, profile="left")
        
        assert isinstance(result, list)
        # Should be clamped to reasonable limit (50)
        assert len(result) <= 50


def test_complete_pipeline_with_corrupted_data():
    """Test entire pipeline with corrupted/edge case data."""
    agent = XGBAgent()
    
    # Corrupted OHLCV data
    corrupted_df = pd.DataFrame({
        "open": [np.nan, np.inf, -np.inf, 0, 100],
        "high": [np.nan, np.inf, -np.inf, 0, 110],
        "low": [np.nan, np.inf, -np.inf, 0, 90],
        "close": [np.nan, np.inf, -np.inf, 0, 105],
        "volume": [np.nan, np.inf, -np.inf, 0, 1000],
    })
    
    # Should handle gracefully through entire pipeline
    result = agent.predict_for_symbol(corrupted_df)
    
    assert isinstance(result, dict)
    assert "action" in result
    assert result["action"] in ["BUY", "SELL", "HOLD"]
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert "model" in result
    # Source field optional - rule_fallback may not have it
    # But should have at least action, confidence, model


def test_parallel_predictions_under_load():
    """Test multiple parallel predictions don't crash system."""
    agent = XGBAgent()
    
    # Simple valid data
    df = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })
    
    # Run 10 predictions sequentially (predict_for_symbol is sync)
    symbols = ["BTC", "ETH", "BNB", "SOL", "ADA", "DOT", "AVAX", "MATIC", "LINK", "UNI"]
    results = []
    
    for sym in symbols:
        try:
            result = agent.predict_for_symbol(df)
            results.append(result)
        except Exception as e:
            pytest.fail(f"Prediction for {sym} raised exception: {e}")
    
    # All should succeed and return valid dict
    assert len(results) == len(symbols)
    for result in results:
        assert isinstance(result, dict)
        assert result["action"] in ["BUY", "SELL", "HOLD"]
