"""Tests for RegimeDetector module."""

import pytest
from backend.services.regime_detector import (
    RegimeDetector,
    RegimeConfig,
    RegimeIndicators,
    RegimeType,
    detect_regime,
)


class TestRegimeConfig:
    """Test regime configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeConfig()
        
        assert config.atr_ratio_low == 0.005
        assert config.atr_ratio_normal == 0.015
        assert config.atr_ratio_high == 0.03
        assert config.adx_trending == 25.0
        assert config.adx_strong_trend == 40.0
        assert config.range_width_threshold == 0.02
        assert config.ema_alignment_pct == 0.02
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegimeConfig(
            atr_ratio_low=0.01,
            atr_ratio_normal=0.02,
            atr_ratio_high=0.05,
            adx_trending=30.0
        )
        
        assert config.atr_ratio_low == 0.01
        assert config.atr_ratio_normal == 0.02
        assert config.atr_ratio_high == 0.05
        assert config.adx_trending == 30.0


class TestVolatilityClassification:
    """Test volatility regime classification."""
    
    def test_low_volatility(self):
        """Test LOW_VOL detection."""
        detector = RegimeDetector()
        
        # BTC at $50k with ATR of $200 = 0.4% ratio
        indicators = RegimeIndicators(
            price=50000.0,
            atr=200.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.LOW_VOL
        assert result.volatility_regime == RegimeType.LOW_VOL
        assert result.atr_ratio == 0.004
        assert result.details["atr_ratio_pct"] == 0.4
    
    def test_normal_volatility(self):
        """Test NORMAL_VOL detection."""
        detector = RegimeDetector()
        
        # BTC at $50k with ATR of $500 = 1.0% ratio
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.NORMAL_VOL
        assert result.volatility_regime == RegimeType.NORMAL_VOL
        assert result.atr_ratio == 0.01
        assert 0.5 <= result.details["atr_ratio_pct"] <= 1.5
    
    def test_high_volatility(self):
        """Test HIGH_VOL detection."""
        detector = RegimeDetector()
        
        # BTC at $50k with ATR of $1000 = 2.0% ratio
        indicators = RegimeIndicators(
            price=50000.0,
            atr=1000.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.HIGH_VOL
        assert result.volatility_regime == RegimeType.HIGH_VOL
        assert result.atr_ratio == 0.02
        assert 1.5 <= result.details["atr_ratio_pct"] <= 3.0
    
    def test_extreme_volatility(self):
        """Test EXTREME_VOL detection."""
        detector = RegimeDetector()
        
        # BTC at $50k with ATR of $2000 = 4.0% ratio (crisis mode)
        indicators = RegimeIndicators(
            price=50000.0,
            atr=2000.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.EXTREME_VOL
        assert result.volatility_regime == RegimeType.EXTREME_VOL
        assert result.atr_ratio == 0.04
        assert result.details["atr_ratio_pct"] > 3.0


class TestTrendClassification:
    """Test trend regime classification."""
    
    def test_trending_with_adx(self):
        """Test TRENDING detection using ADX."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            adx=30.0  # Above trending threshold of 25
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.trend_regime == "TRENDING"
        assert result.details["adx"] == 30.0
    
    def test_ranging_with_adx(self):
        """Test RANGING detection using ADX."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            adx=18.0  # Below trending threshold of 25
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.trend_regime == "RANGING"
        assert result.details["adx"] == 18.0
    
    def test_trending_with_range_width(self):
        """Test TRENDING detection using range width."""
        detector = RegimeDetector()
        
        # Wide range: $48k - $52k = 8% range
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            range_high=52000.0,
            range_low=48000.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.trend_regime == "TRENDING"
        assert result.details["range_width_pct"] == 8.0
    
    def test_ranging_with_range_width(self):
        """Test RANGING detection using range width."""
        detector = RegimeDetector()
        
        # Narrow range: $49.5k - $50.5k = 2% range
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            range_high=50500.0,
            range_low=49500.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.trend_regime == "RANGING"
        assert result.details["range_width_pct"] == 2.0
    
    def test_unknown_trend_without_indicators(self):
        """Test UNKNOWN trend when no indicators available."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0
            # No ADX, no range data
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.trend_regime == "UNKNOWN"


class TestEMAAlignment:
    """Test EMA alignment calculations."""
    
    def test_ema_above_price(self):
        """Test price above EMA200."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=52000.0,
            atr=500.0,
            ema_200=50000.0  # Price 4% above EMA
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.details["ema_200"] == 50000.0
        assert result.details["ema_distance_pct"] == pytest.approx(4.0, abs=0.1)
    
    def test_ema_below_price(self):
        """Test price below EMA200."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=48000.0,
            atr=500.0,
            ema_200=50000.0  # Price 4% below EMA
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.details["ema_distance_pct"] == pytest.approx(-4.0, abs=0.1)


class TestRegimeParameters:
    """Test regime parameter suggestions."""
    
    def test_low_vol_parameters(self):
        """Test LOW_VOL parameter adjustments."""
        detector = RegimeDetector()
        params = detector.get_regime_parameters(RegimeType.LOW_VOL)
        
        assert params["risk_multiplier"] == 1.2  # More aggressive
        assert params["sl_multiplier"] == 0.8     # Tighter stops
        assert params["min_confidence_adj"] == -0.05  # Lower confidence OK
    
    def test_normal_vol_parameters(self):
        """Test NORMAL_VOL parameter adjustments."""
        detector = RegimeDetector()
        params = detector.get_regime_parameters(RegimeType.NORMAL_VOL)
        
        assert params["risk_multiplier"] == 1.0
        assert params["sl_multiplier"] == 1.0
        assert params["tp_multiplier"] == 1.0
        assert params["min_confidence_adj"] == 0.0
    
    def test_high_vol_parameters(self):
        """Test HIGH_VOL parameter adjustments."""
        detector = RegimeDetector()
        params = detector.get_regime_parameters(RegimeType.HIGH_VOL)
        
        assert params["risk_multiplier"] == 0.8   # More conservative
        assert params["sl_multiplier"] == 1.2      # Wider stops
        assert params["min_confidence_adj"] == 0.05  # Higher confidence required
    
    def test_extreme_vol_parameters(self):
        """Test EXTREME_VOL parameter adjustments."""
        detector = RegimeDetector()
        params = detector.get_regime_parameters(RegimeType.EXTREME_VOL)
        
        assert params["risk_multiplier"] == 0.5   # Very conservative
        assert params["sl_multiplier"] == 1.5      # Much wider stops
        assert params["min_confidence_adj"] == 0.10  # Much higher confidence
        assert params["max_position_adj"] == -1    # Reduce positions


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_detect_regime_function(self):
        """Test detect_regime convenience function."""
        result = detect_regime(
            price=50000.0,
            atr=500.0,
            ema_200=48000.0,
            adx=30.0
        )
        
        assert result.regime == RegimeType.NORMAL_VOL
        assert result.trend_regime == "TRENDING"
        assert result.atr_ratio == 0.01
        assert result.details["adx"] == 30.0
    
    def test_detect_regime_minimal(self):
        """Test detect_regime with minimal inputs."""
        result = detect_regime(
            price=50000.0,
            atr=200.0
        )
        
        assert result.regime == RegimeType.LOW_VOL
        assert result.trend_regime == "UNKNOWN"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_price(self):
        """Test handling of zero price."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=0.0,
            atr=500.0
        )
        
        result = detector.detect_regime(indicators)
        
        # Should handle gracefully without division by zero
        assert result.atr_ratio == 0.0
    
    def test_zero_atr(self):
        """Test handling of zero ATR."""
        detector = RegimeDetector()
        
        indicators = RegimeIndicators(
            price=50000.0,
            atr=0.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.atr_ratio == 0.0
        assert result.regime == RegimeType.LOW_VOL
    
    def test_boundary_values(self):
        """Test exact boundary threshold values."""
        detector = RegimeDetector()
        
        # Test at exact LOW_VOL threshold (0.5%)
        indicators = RegimeIndicators(
            price=50000.0,
            atr=250.0  # Exactly 0.005 ratio
        )
        result = detector.detect_regime(indicators)
        assert result.regime == RegimeType.NORMAL_VOL  # Boundary goes to higher regime
        
        # Test at exact ADX trending threshold
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            adx=25.0  # Exactly at threshold
        )
        result = detector.detect_regime(indicators)
        assert result.trend_regime == "TRENDING"  # At threshold = trending
    
    def test_very_small_price(self):
        """Test with very small price (altcoin)."""
        detector = RegimeDetector()
        
        # Altcoin at $0.10 with ATR of $0.002 = 2% ratio
        indicators = RegimeIndicators(
            price=0.10,
            atr=0.002
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.HIGH_VOL
        assert result.atr_ratio == pytest.approx(0.02, abs=0.001)
    
    def test_very_large_price(self):
        """Test with very large price."""
        detector = RegimeDetector()
        
        # Large cap at $100k with ATR of $500 = 0.5% ratio
        indicators = RegimeIndicators(
            price=100000.0,
            atr=500.0
        )
        
        result = detector.detect_regime(indicators)
        
        assert result.regime == RegimeType.NORMAL_VOL
        assert result.atr_ratio == 0.005


class TestCustomConfig:
    """Test custom configuration scenarios."""
    
    def test_tighter_thresholds(self):
        """Test with tighter volatility thresholds."""
        config = RegimeConfig(
            atr_ratio_low=0.003,
            atr_ratio_normal=0.01,
            atr_ratio_high=0.02
        )
        detector = RegimeDetector(config)
        
        # Same conditions but with tighter thresholds
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0  # 1% ratio
        )
        
        result = detector.detect_regime(indicators)
        
        # With tighter thresholds, 1% is now at HIGH_VOL boundary
        assert result.regime == RegimeType.HIGH_VOL
    
    def test_higher_adx_threshold(self):
        """Test with higher ADX trending threshold."""
        config = RegimeConfig(adx_trending=35.0)
        detector = RegimeDetector(config)
        
        indicators = RegimeIndicators(
            price=50000.0,
            atr=500.0,
            adx=30.0
        )
        
        result = detector.detect_regime(indicators)
        
        # ADX 30 is now below trending threshold
        assert result.trend_regime == "RANGING"


class TestRealWorldScenarios:
    """Test realistic market scenarios."""
    
    def test_btc_bull_market(self):
        """Test BTC in strong bull market."""
        result = detect_regime(
            price=65000.0,
            atr=1200.0,        # ~1.8% (elevated but not extreme)
            ema_200=55000.0,   # Price well above EMA
            adx=45.0           # Strong trend
        )
        
        assert result.regime == RegimeType.HIGH_VOL
        assert result.trend_regime == "TRENDING"
        assert result.details["ema_distance_pct"] > 15.0
    
    def test_btc_consolidation(self):
        """Test BTC in consolidation phase."""
        result = detect_regime(
            price=50000.0,
            atr=400.0,         # ~0.8% (low-normal)
            ema_200=50200.0,   # Price near EMA
            adx=18.0,          # Weak trend
            range_high=51000.0,
            range_low=49000.0  # 4% range
        )
        
        assert result.regime == RegimeType.NORMAL_VOL
        assert result.trend_regime == "RANGING"
    
    def test_btc_flash_crash(self):
        """Test BTC during flash crash / extreme volatility."""
        result = detect_regime(
            price=45000.0,
            atr=3500.0,        # ~7.8% (extreme!)
            ema_200=55000.0,   # Price well below EMA
            adx=55.0           # Very strong trend (down)
        )
        
        assert result.regime == RegimeType.EXTREME_VOL
        assert result.trend_regime == "TRENDING"
        assert result.details["atr_ratio_pct"] > 7.0
    
    def test_altcoin_low_liquidity(self):
        """Test low liquidity altcoin."""
        result = detect_regime(
            price=1.50,
            atr=0.08,          # ~5.3% (very high)
            adx=22.0           # Ranging
        )
        
        assert result.regime == RegimeType.EXTREME_VOL
        assert result.trend_regime == "RANGING"
    
    def test_stablecoin_like_behavior(self):
        """Test stable, low volatility asset."""
        result = detect_regime(
            price=100.0,
            atr=0.20,          # 0.2% (very stable)
            adx=12.0,
            range_high=100.5,
            range_low=99.5
        )
        
        assert result.regime == RegimeType.LOW_VOL
        assert result.trend_regime == "RANGING"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
