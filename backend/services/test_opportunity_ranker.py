"""
Unit tests for OpportunityRanker module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

from opportunity_ranker import (
    OpportunityRanker,
    SymbolMetrics,
    TrendDirection,
)


# ============================================================================
# MOCK IMPLEMENTATIONS
# ============================================================================

class MockMarketData:
    """Mock market data client for testing."""
    
    def __init__(self, candles_override=None, spread=0.0005, liquidity=50_000_000):
        self.candles_override = candles_override
        self.spread = spread
        self.liquidity = liquidity
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if self.candles_override is not None:
            return self.candles_override
        
        # Generate simple trending data
        timestamps = pd.date_range(end=datetime.utcnow(), periods=limit, freq='1H')
        prices = 50000 + np.linspace(0, 2000, limit) + np.random.normal(0, 100, limit)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + 50,
            'low': prices - 50,
            'close': prices,
            'volume': np.random.uniform(100, 500, limit),
        })
    
    def get_spread(self, symbol: str) -> float:
        return self.spread
    
    def get_liquidity(self, symbol: str) -> float:
        return self.liquidity


class MockTradeLogs:
    """Mock trade log repository."""
    
    def __init__(self, winrate=0.6):
        self.winrate = winrate
    
    def get_symbol_winrate(self, symbol: str, last_n: int = 200) -> float:
        return self.winrate


class MockRegimeDetector:
    """Mock regime detector."""
    
    def __init__(self, regime="BULL"):
        self.regime = regime
    
    def get_global_regime(self) -> str:
        return self.regime


class MockOpportunityStore:
    """Mock opportunity store."""
    
    def __init__(self):
        self.data = {}
    
    def update(self, rankings: Dict[str, float]) -> None:
        self.data = rankings
    
    def get(self) -> Dict[str, float]:
        return self.data


# ============================================================================
# TEST CASES
# ============================================================================

class TestOpportunityRanker(unittest.TestCase):
    """Test OpportunityRanker core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.market_data = MockMarketData()
        self.trade_logs = MockTradeLogs()
        self.regime_detector = MockRegimeDetector()
        self.opportunity_store = MockOpportunityStore()
        
        self.ranker = OpportunityRanker(
            market_data=self.market_data,
            trade_logs=self.trade_logs,
            regime_detector=self.regime_detector,
            opportunity_store=self.opportunity_store,
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1h",
            candle_limit=200,
        )
    
    def test_initialization(self):
        """Test ranker initializes correctly."""
        self.assertEqual(len(self.ranker.symbols), 2)
        self.assertEqual(self.ranker.timeframe, "1h")
        self.assertEqual(self.ranker.candle_limit, 200)
    
    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        invalid_weights = {
            'trend_strength': 0.5,
            'volatility_quality': 0.3,
            # Sum = 0.8, not 1.0
        }
        
        with self.assertRaises(ValueError):
            OpportunityRanker(
                market_data=self.market_data,
                trade_logs=self.trade_logs,
                regime_detector=self.regime_detector,
                opportunity_store=self.opportunity_store,
                symbols=["BTCUSDT"],
                weights=invalid_weights,
            )
    
    def test_compute_symbol_scores(self):
        """Test computing scores for multiple symbols."""
        results = self.ranker.compute_symbol_scores()
        
        self.assertEqual(len(results), 2)
        self.assertIn("BTCUSDT", results)
        self.assertIn("ETHUSDT", results)
        
        for symbol, metrics in results.items():
            self.assertIsInstance(metrics, SymbolMetrics)
            self.assertGreaterEqual(metrics.final_score, 0.0)
            self.assertLessEqual(metrics.final_score, 1.0)
    
    def test_update_rankings(self):
        """Test full ranking update workflow."""
        rankings = self.ranker.update_rankings()
        
        # Check rankings are returned
        self.assertIsInstance(rankings, dict)
        self.assertGreater(len(rankings), 0)
        
        # Check store was updated
        stored = self.opportunity_store.get()
        self.assertEqual(rankings, stored)
        
        # Check sorting (descending)
        scores = list(rankings.values())
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_threshold_filtering(self):
        """Test minimum score threshold filtering."""
        ranker_high_threshold = OpportunityRanker(
            market_data=self.market_data,
            trade_logs=self.trade_logs,
            regime_detector=self.regime_detector,
            opportunity_store=MockOpportunityStore(),
            symbols=["BTCUSDT", "ETHUSDT"],
            min_score_threshold=0.99,  # Very high threshold
        )
        
        rankings = ranker_high_threshold.update_rankings()
        
        # Likely no symbols pass 0.99 threshold
        self.assertLessEqual(len(rankings), 2)
        
        # All remaining symbols must be >= threshold
        for score in rankings.values():
            self.assertGreaterEqual(score, 0.99)
    
    def test_get_top_n(self):
        """Test retrieving top N symbols."""
        self.ranker.update_rankings()
        
        top_1 = self.ranker.get_top_n(n=1)
        self.assertEqual(len(top_1), 1)
        
        top_5 = self.ranker.get_top_n(n=5)
        self.assertLessEqual(len(top_5), 2)  # Only 2 symbols available


class TestMetricCalculations(unittest.TestCase):
    """Test individual metric calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.market_data = MockMarketData()
        self.trade_logs = MockTradeLogs()
        self.regime_detector = MockRegimeDetector()
        self.opportunity_store = MockOpportunityStore()
        
        self.ranker = OpportunityRanker(
            market_data=self.market_data,
            trade_logs=self.trade_logs,
            regime_detector=self.regime_detector,
            opportunity_store=self.opportunity_store,
            symbols=["BTCUSDT"],
        )
    
    def _create_trending_candles(self, periods=200, trend='up') -> pd.DataFrame:
        """Helper to create trending candle data."""
        if trend == 'up':
            prices = 50000 + np.linspace(0, 5000, periods)
        elif trend == 'down':
            prices = 50000 - np.linspace(0, 5000, periods)
        else:
            prices = np.full(periods, 50000) + np.random.normal(0, 100, periods)
        
        return pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H'),
            'open': prices,
            'high': prices + 100,
            'low': prices - 100,
            'close': prices,
            'volume': np.full(periods, 1000),
        })
    
    def test_trend_strength_uptrend(self):
        """Test trend strength for strong uptrend."""
        df = self._create_trending_candles(trend='up')
        score = self.ranker._compute_trend_strength(df)
        
        self.assertGreaterEqual(score, 0.6)  # Should be high for strong trend
        self.assertLessEqual(score, 1.0)
    
    def test_trend_strength_downtrend(self):
        """Test trend strength for downtrend."""
        df = self._create_trending_candles(trend='down')
        score = self.ranker._compute_trend_strength(df)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_trend_strength_neutral(self):
        """Test trend strength for ranging market."""
        df = self._create_trending_candles(trend='neutral')
        score = self.ranker._compute_trend_strength(df)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.7)  # Should be lower for no trend
    
    def test_volatility_quality_optimal(self):
        """Test volatility quality for optimal volatility."""
        # Create data with ~2% ATR (optimal range)
        periods = 200
        prices = np.full(periods, 50000) + np.random.normal(0, 500, periods)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 500,
            'low': prices - 500,
            'close': prices,
            'volume': np.full(periods, 1000),
        })
        
        score = self.ranker._compute_volatility_quality(df)
        
        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 1.0)
    
    def test_liquidity_score(self):
        """Test liquidity scoring."""
        # High liquidity
        self.market_data.liquidity = 500_000_000
        score_high = self.ranker._compute_liquidity_score("BTCUSDT")
        
        # Low liquidity
        self.market_data.liquidity = 500_000
        score_low = self.ranker._compute_liquidity_score("BTCUSDT")
        
        self.assertGreater(score_high, score_low)
        self.assertGreaterEqual(score_low, 0.0)
        self.assertLessEqual(score_high, 1.0)
    
    def test_spread_score(self):
        """Test spread scoring."""
        # Excellent spread
        self.market_data.spread = 0.0002
        score_excellent = self.ranker._compute_spread_score("BTCUSDT")
        
        # Poor spread
        self.market_data.spread = 0.003
        score_poor = self.ranker._compute_spread_score("BTCUSDT")
        
        self.assertGreater(score_excellent, score_poor)
        self.assertGreaterEqual(score_excellent, 0.8)
    
    def test_symbol_winrate(self):
        """Test symbol winrate scoring."""
        # High winrate
        self.trade_logs.winrate = 0.75
        score_high = self.ranker._compute_symbol_winrate("BTCUSDT")
        
        # Low winrate
        self.trade_logs.winrate = 0.35
        score_low = self.ranker._compute_symbol_winrate("BTCUSDT")
        
        self.assertAlmostEqual(score_high, 0.75)
        self.assertAlmostEqual(score_low, 0.35)
    
    def test_regime_score_bull_alignment(self):
        """Test regime score for bull market with bullish symbol."""
        self.regime_detector.regime = "BULL"
        df = self._create_trending_candles(trend='up')
        
        score = self.ranker._compute_regime_score(df, "BULL")
        
        self.assertGreaterEqual(score, 0.8)  # Should be high for alignment
    
    def test_regime_score_bear_misalignment(self):
        """Test regime score for bear market with bullish symbol."""
        df = self._create_trending_candles(trend='up')
        
        score = self.ranker._compute_regime_score(df, "BEAR")
        
        self.assertLessEqual(score, 0.5)  # Should be low for misalignment
    
    def test_noise_score(self):
        """Test noise scoring."""
        # Clean price action (large bodies, small wicks)
        periods = 50
        prices = 50000 + np.linspace(0, 1000, periods)
        
        df_clean = pd.DataFrame({
            'open': prices,
            'high': prices + 20,  # Small wicks
            'low': prices - 20,
            'close': prices + 15,  # Large body
            'volume': np.full(periods, 1000),
        })
        
        # Noisy price action (small bodies, large wicks)
        df_noisy = pd.DataFrame({
            'open': prices,
            'high': prices + 500,  # Large wicks
            'low': prices - 500,
            'close': prices + 10,  # Small body
            'volume': np.full(periods, 1000),
        })
        
        score_clean = self.ranker._compute_noise_score(df_clean)
        score_noisy = self.ranker._compute_noise_score(df_noisy)
        
        self.assertGreater(score_clean, score_noisy)
    
    def test_trend_direction_detection(self):
        """Test trend direction classification."""
        df_bull = self._create_trending_candles(trend='up')
        df_bear = self._create_trending_candles(trend='down')
        df_neutral = self._create_trending_candles(trend='neutral')
        
        direction_bull = self.ranker._detect_trend_direction(df_bull)
        direction_bear = self.ranker._detect_trend_direction(df_bear)
        direction_neutral = self.ranker._detect_trend_direction(df_neutral)
        
        self.assertEqual(direction_bull, TrendDirection.BULLISH)
        self.assertEqual(direction_bear, TrendDirection.BEARISH)
        self.assertIn(direction_neutral, [TrendDirection.NEUTRAL, TrendDirection.BULLISH, TrendDirection.BEARISH])


class TestScoreAggregation(unittest.TestCase):
    """Test score aggregation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ranker = OpportunityRanker(
            market_data=MockMarketData(),
            trade_logs=MockTradeLogs(),
            regime_detector=MockRegimeDetector(),
            opportunity_store=MockOpportunityStore(),
            symbols=["BTCUSDT"],
        )
    
    def test_aggregate_scores_perfect(self):
        """Test aggregation with perfect scores."""
        metrics = {
            'trend_strength': 1.0,
            'volatility_quality': 1.0,
            'liquidity_score': 1.0,
            'spread_score': 1.0,
            'symbol_winrate_score': 1.0,
            'regime_score': 1.0,
            'noise_score': 1.0,
        }
        
        final_score = self.ranker._aggregate_scores(metrics)
        
        self.assertAlmostEqual(final_score, 1.0)
    
    def test_aggregate_scores_zero(self):
        """Test aggregation with zero scores."""
        metrics = {
            'trend_strength': 0.0,
            'volatility_quality': 0.0,
            'liquidity_score': 0.0,
            'spread_score': 0.0,
            'symbol_winrate_score': 0.0,
            'regime_score': 0.0,
            'noise_score': 0.0,
        }
        
        final_score = self.ranker._aggregate_scores(metrics)
        
        self.assertAlmostEqual(final_score, 0.0)
    
    def test_aggregate_scores_weighted(self):
        """Test that weights are applied correctly."""
        # Only trend_strength = 1.0, others = 0.0
        metrics = {
            'trend_strength': 1.0,
            'volatility_quality': 0.0,
            'liquidity_score': 0.0,
            'spread_score': 0.0,
            'symbol_winrate_score': 0.0,
            'regime_score': 0.0,
            'noise_score': 0.0,
        }
        
        final_score = self.ranker._aggregate_scores(metrics)
        
        # Should equal trend_strength weight (default 0.25)
        expected = self.ranker.DEFAULT_WEIGHTS['trend_strength']
        self.assertAlmostEqual(final_score, expected, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test handling of insufficient candle data."""
        short_df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.utcnow(), periods=10, freq='1H'),
            'open': [50000] * 10,
            'high': [50100] * 10,
            'low': [49900] * 10,
            'close': [50000] * 10,
            'volume': [1000] * 10,
        })
        
        market_data = MockMarketData(candles_override=short_df)
        ranker = OpportunityRanker(
            market_data=market_data,
            trade_logs=MockTradeLogs(),
            regime_detector=MockRegimeDetector(),
            opportunity_store=MockOpportunityStore(),
            symbols=["BTCUSDT"],
        )
        
        # Should handle gracefully (not raise)
        results = ranker.compute_symbol_scores()
        
        # Symbol should be skipped due to insufficient data
        self.assertEqual(len(results), 0)
    
    def test_empty_symbol_list(self):
        """Test with no symbols."""
        ranker = OpportunityRanker(
            market_data=MockMarketData(),
            trade_logs=MockTradeLogs(),
            regime_detector=MockRegimeDetector(),
            opportunity_store=MockOpportunityStore(),
            symbols=[],
        )
        
        rankings = ranker.update_rankings()
        
        self.assertEqual(len(rankings), 0)
    
    def test_custom_weights(self):
        """Test with custom metric weights."""
        custom_weights = {
            'trend_strength': 0.5,
            'volatility_quality': 0.2,
            'liquidity_score': 0.1,
            'spread_score': 0.1,
            'symbol_winrate_score': 0.05,
            'regime_score': 0.03,
            'noise_score': 0.02,
        }
        
        ranker = OpportunityRanker(
            market_data=MockMarketData(),
            trade_logs=MockTradeLogs(),
            regime_detector=MockRegimeDetector(),
            opportunity_store=MockOpportunityStore(),
            symbols=["BTCUSDT"],
            weights=custom_weights,
        )
        
        self.assertEqual(ranker.weights, custom_weights)


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
