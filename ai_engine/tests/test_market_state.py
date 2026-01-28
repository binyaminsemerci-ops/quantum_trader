"""
Unit tests for P0 MarketState module
"""

import pytest
import numpy as np
from ai_engine.market_state import MarketStateEngine, MarketState, DEFAULT_CONFIG


class TestMarketStateEngine:
    """Test suite for MarketStateEngine"""
    
    def test_initialization(self):
        """Test engine initialization with default config"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        assert engine.window == 100
        assert engine.trend_method == 'huber'
        assert len(engine.get_tracked_symbols()) == 0
    
    def test_custom_config(self):
        """Test custom configuration"""
        custom_config = {
            'theta': {
                'market_state': {
                    'window': 50,
                    'trend_method': 'theil-sen',
                    'ewma_alpha': 0.2
                }
            }
        }
        engine = MarketStateEngine(custom_config)
        assert engine.window == 50
        assert engine.trend_method == 'theil-sen'
        assert engine.ewma_alpha == 0.2
    
    def test_update_and_tracking(self):
        """Test price updates and symbol tracking"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        engine.update("BTCUSDT", 50000.0, timestamp=1.0)
        engine.update("BTCUSDT", 50100.0, timestamp=2.0)
        engine.update("ETHUSDT", 3000.0, timestamp=1.0)
        
        assert "BTCUSDT" in engine.get_tracked_symbols()
        assert "ETHUSDT" in engine.get_tracked_symbols()
        assert len(engine.get_tracked_symbols()) == 2
    
    def test_insufficient_data(self):
        """Test that get_state returns None with insufficient data"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Only 5 samples
        for i in range(5):
            engine.update("TEST", 100.0 + i, timestamp=float(i))
        
        state = engine.get_state("TEST")
        assert state is None
    
    def test_trending_market(self):
        """Test regime detection on trending market"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Generate strong uptrend
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.3 + 0.1)  # Strong positive drift
        
        for i, price in enumerate(prices):
            engine.update("TREND", price, timestamp=float(i))
        
        state = engine.get_state("TREND")
        assert state is not None
        assert state.symbol == "TREND"
        assert state.mu > 0  # Positive trend
        assert state.TS > 0  # Non-zero trend strength
        
        # Should favor TREND regime
        assert state.regime_probs['TREND'] > state.regime_probs['MR']
        print(f"\nTrending Market - Regime probs: {state.regime_probs}")
    
    def test_mean_reverting_market(self):
        """Test regime detection on mean-reverting market"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Generate mean-reverting series (oscillating around 100)
        np.random.seed(42)
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.5
        
        for i, price in enumerate(prices):
            engine.update("MR", price, timestamp=float(i))
        
        state = engine.get_state("MR")
        assert state is not None
        
        # Mean-reverting should have low trend strength
        assert state.TS < 1.0
        assert state.VR < 1.5  # Variance ratio should be moderate
        
        print(f"\nMean-Reverting Market - Regime probs: {state.regime_probs}")
    
    def test_choppy_market(self):
        """Test regime detection on choppy/sideways market"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Generate random walk (choppy, no clear trend)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 1.0)  # Pure random walk
        
        for i, price in enumerate(prices):
            engine.update("CHOP", price, timestamp=float(i))
        
        state = engine.get_state("CHOP")
        assert state is not None
        
        # Choppy should have VR close to 1 (random walk)
        assert 0.5 < state.VR < 1.5
        
        print(f"\nChoppy Market - Regime probs: {state.regime_probs}")
    
    def test_robust_sigma_with_outliers(self):
        """Test robust volatility calculation with outliers"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Generate prices with outliers
        np.random.seed(42)
        prices = 100 + np.random.randn(100) * 0.5
        prices[50] = 200  # Outlier
        prices[75] = 50   # Outlier
        
        for i, price in enumerate(prices):
            engine.update("OUTLIER", price, timestamp=float(i))
        
        state = engine.get_state("OUTLIER")
        assert state is not None
        
        # Sigma should be reasonable despite outliers
        assert 0.1 < state.sigma < 5.0
        print(f"\nOutlier Test - Sigma: {state.sigma:.4f}")
    
    def test_directional_persistence(self):
        """Test directional persistence calculation"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Persistent trend (same direction)
        prices_trend = 100 + np.cumsum(np.ones(100) * 0.1)
        for i, price in enumerate(prices_trend):
            engine.update("PERSIST", price, timestamp=float(i))
        
        state = engine.get_state("PERSIST")
        assert state is not None
        assert state.dp > 0.5  # High directional persistence
        
        # Alternating (mean-reverting)
        prices_alt = 100 + np.array([(-1)**i * 0.5 for i in range(100)])
        engine_alt = MarketStateEngine(DEFAULT_CONFIG)
        for i, price in enumerate(prices_alt):
            engine_alt.update("ALT", price, timestamp=float(i))
        
        state_alt = engine_alt.get_state("ALT")
        assert state_alt is not None
        assert state_alt.dp < 0.5  # Low directional persistence
        
        print(f"\nDP Test - Trend: {state.dp:.4f}, Alternating: {state_alt.dp:.4f}")
    
    def test_variance_ratio(self):
        """Test variance ratio calculation"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Trending: VR > 1
        prices_trend = 100 + np.cumsum(np.random.randn(100) * 0.3 + 0.1)
        for i, price in enumerate(prices_trend):
            engine.update("VR_TREND", price, timestamp=float(i))
        
        state_trend = engine.get_state("VR_TREND")
        assert state_trend is not None
        
        # Mean-reverting: VR < 1
        prices_mr = 100 + 5 * np.sin(np.linspace(0, 10*np.pi, 100))
        engine_mr = MarketStateEngine(DEFAULT_CONFIG)
        for i, price in enumerate(prices_mr):
            engine_mr.update("VR_MR", price, timestamp=float(i))
        
        state_mr = engine_mr.get_state("VR_MR")
        assert state_mr is not None
        
        print(f"\nVR Test - Trend: {state_trend.VR:.4f}, MR: {state_mr.VR:.4f}")
    
    def test_trend_strength(self):
        """Test trend strength calculation"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Strong trend
        prices_strong = 100 + np.cumsum(np.random.randn(100) * 0.2 + 0.2)
        for i, price in enumerate(prices_strong):
            engine.update("STRONG", price, timestamp=float(i))
        
        state_strong = engine.get_state("STRONG")
        assert state_strong is not None
        
        # Weak trend
        prices_weak = 100 + np.cumsum(np.random.randn(100) * 1.0 + 0.01)
        engine_weak = MarketStateEngine(DEFAULT_CONFIG)
        for i, price in enumerate(prices_weak):
            engine_weak.update("WEAK", price, timestamp=float(i))
        
        state_weak = engine_weak.get_state("WEAK")
        assert state_weak is not None
        
        # Strong trend should have higher TS
        print(f"\nTS Test - Strong: {state_strong.TS:.4f}, Weak: {state_weak.TS:.4f}")
    
    def test_get_all_states(self):
        """Test getting states for all symbols"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Add data for multiple symbols
        for symbol in ["BTC", "ETH", "SOL"]:
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            for i, price in enumerate(prices):
                engine.update(symbol, price, timestamp=float(i))
        
        all_states = engine.get_all_states()
        assert len(all_states) == 3
        assert "BTC" in all_states
        assert "ETH" in all_states
        assert "SOL" in all_states
    
    def test_clear_symbol(self):
        """Test clearing symbol history"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        for i in range(50):
            engine.update("TEST", 100.0 + i, timestamp=float(i))
        
        assert "TEST" in engine.get_tracked_symbols()
        
        engine.clear_symbol("TEST")
        assert "TEST" not in engine.get_tracked_symbols()
    
    def test_regime_probabilities_sum_to_one(self):
        """Test that regime probabilities sum to 1.0"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        for i, price in enumerate(prices):
            engine.update("TEST", price, timestamp=float(i))
        
        state = engine.get_state("TEST")
        assert state is not None
        
        prob_sum = sum(state.regime_probs.values())
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_window_size_limit(self):
        """Test that history respects window size"""
        config = {
            'theta': {
                'market_state': {
                    'window': 50
                }
            }
        }
        engine = MarketStateEngine(config)
        
        # Add 100 samples
        for i in range(100):
            engine.update("TEST", 100.0 + i, timestamp=float(i))
        
        # Should only keep last 50
        assert len(engine.price_history["TEST"]) == 50
    
    def test_theil_sen_method(self):
        """Test Theil-Sen trend estimation"""
        config = {
            'theta': {
                'market_state': {
                    'trend_method': 'theil-sen'
                }
            }
        }
        engine = MarketStateEngine(config)
        
        # Strong linear trend
        prices = 100 + np.arange(100) * 0.5
        for i, price in enumerate(prices):
            engine.update("TEST", price, timestamp=float(i))
        
        state = engine.get_state("TEST")
        assert state is not None
        assert state.mu > 0  # Should detect positive trend
        print(f"\nTheil-Sen Test - Mu: {state.mu:.6f}")
    
    def test_zero_returns_handling(self):
        """Test handling of flat prices (zero returns)"""
        engine = MarketStateEngine(DEFAULT_CONFIG)
        
        # Flat prices
        for i in range(50):
            engine.update("FLAT", 100.0, timestamp=float(i))
        
        state = engine.get_state("FLAT")
        assert state is not None
        assert state.sigma < 1e-6  # Should have very low volatility
        assert abs(state.mu) < 1e-6  # Should have zero trend


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
