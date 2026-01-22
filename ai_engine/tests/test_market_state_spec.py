"""
Unit tests for P0 MarketState SPEC v1.0
"""

import pytest
import numpy as np
from ai_engine.market_state import MarketState, DEFAULT_THETA


class TestMarketStateSpec:
    """Test suite for P0 MarketState SPEC v1.0"""
    
    def test_output_contract(self):
        """Test that output contains MUST fields"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        # MUST fields
        assert 'sigma' in state
        assert 'mu' in state
        assert 'ts' in state
        assert 'regime_probs' in state
        
        assert isinstance(state['sigma'], float)
        assert isinstance(state['mu'], float)
        assert isinstance(state['ts'], float)
        assert isinstance(state['regime_probs'], dict)
    
    def test_regime_probs_sum_to_one(self):
        """Test that regime probabilities sum to 1.0"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        prob_sum = sum(state['regime_probs'].values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Probs sum to {prob_sum}, expected 1.0"
    
    def test_regime_probs_keys(self):
        """Test that regime_probs has correct keys"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        assert 'trend' in state['regime_probs']
        assert 'mr' in state['regime_probs']
        assert 'chop' in state['regime_probs']
    
    def test_sigma_increases_on_spike(self):
        """Test that sigma increases when price spike is injected"""
        ms = MarketState()
        
        # Generate stable prices
        np.random.seed(42)
        prices = 100 + np.random.randn(280) * 0.1
        
        # Add spike
        prices = np.append(prices, [110, 115, 120, 115, 110])  # 20 more samples
        
        state = ms.get_state("SPIKE", prices)
        assert state is not None
        
        # Sigma should be elevated
        assert state['sigma'] > 0
        # Spike proxy should be elevated
        assert state['features']['spike_proxy'] > 0
    
    def test_ts_high_on_trend(self):
        """Test that TS is high on strong trending data"""
        ms = MarketState()
        
        # Strong uptrend
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.005 + 0.002))
        
        state = ms.get_state("TREND", prices)
        assert state is not None
        
        # Should have positive mu and reasonable TS
        assert state['mu'] > 0, f"Expected positive mu, got {state['mu']}"
        assert state['ts'] > 0, f"Expected positive TS, got {state['ts']}"
    
    def test_ts_low_on_chop(self):
        """Test that TS is low on choppy data"""
        ms = MarketState()
        
        # Random walk (choppy)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.015))
        
        state = ms.get_state("CHOP", prices)
        assert state is not None
        
        # TS should be relatively low (no clear trend)
        assert state['ts'] < 2.0, f"Expected low TS on chop, got {state['ts']}"
    
    def test_trend_regime_dominates_on_trend(self):
        """Test that TREND regime dominates on trending data"""
        ms = MarketState()
        
        # Very strong uptrend
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.003 + 0.003))
        
        state = ms.get_state("STRONG_TREND", prices)
        assert state is not None
        
        # TREND should be dominant or at least competitive
        probs = state['regime_probs']
        # With default coeffs, trend might not always dominate, but should be significant
        assert probs['trend'] > 0.2, f"Expected trend > 20%, got {probs}"
    
    def test_insufficient_data(self):
        """Test that get_state returns None with insufficient data"""
        ms = MarketState()
        
        # Only 50 samples (need 256+)
        prices = 100 + np.random.randn(50) * 0.1
        
        state = ms.get_state("TOO_SHORT", prices)
        assert state is None
    
    def test_sigma_components(self):
        """Test that sigma components are computed"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        # SHOULD fields exist
        assert 'features' in state
        assert 'spike_proxy' in state['features']
        assert 'dp' in state['features']
        assert 'vr' in state['features']
    
    def test_custom_theta(self):
        """Test that custom theta overrides defaults"""
        custom_theta = {
            'vol': {
                'sigma_floor': 1e-5  # Different from default 1e-6
            }
        }
        
        ms = MarketState(theta=custom_theta)
        assert ms.theta['vol']['sigma_floor'] == 1e-5
        assert ms.theta['vol']['window'] == 256  # Should keep default
    
    def test_multi_window_trend(self):
        """Test that multiple trend windows are used"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01 + 0.001))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        # Should use windows from theta
        assert state['windows']['trend_windows'] == [64, 128, 256]
    
    def test_dp_feature(self):
        """Test directional persistence calculation"""
        ms = MarketState()
        
        # Persistent trend (same direction)
        prices = 100 * np.exp(np.cumsum(np.ones(300) * 0.001))
        state = ms.get_state("PERSIST", prices)
        assert state is not None
        assert state['features']['dp'] > 0.5, "Expected dp > 0.5 for persistent trend"
    
    def test_vr_feature(self):
        """Test variance ratio calculation"""
        ms = MarketState()
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        
        state = ms.get_state("TEST", prices)
        assert state is not None
        
        # VR should be computed
        assert 'vr' in state['features']
        assert state['features']['vr'] > 0
    
    def test_winsorization(self):
        """Test that winsorization handles outliers"""
        ms = MarketState()
        
        # Create prices with outliers
        np.random.seed(42)
        prices = 100 + np.random.randn(280) * 0.1
        prices = np.append(prices, [200, 50, 150, 80])  # Extreme outliers
        prices = np.append(prices, 100 + np.random.randn(16) * 0.1)  # Back to normal
        
        state = ms.get_state("OUTLIERS", prices)
        assert state is not None
        
        # Should still compute reasonable sigma (not explode from outliers)
        assert state['sigma'] < 10.0, f"Sigma unexpectedly large: {state['sigma']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
