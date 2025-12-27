"""Tests for Exit Policy Regime Configuration."""

import pytest
import os
from backend.services.execution.exit_policy_regime_config import (
    RegimeExitConfig,
    get_exit_params,
    get_all_regimes,
    get_risk_reward_ratio,
    validate_config,
    load_from_env,
    DEFAULT_REGIME_CONFIGS,
)


class TestRegimeExitConfig:
    """Test RegimeExitConfig dataclass."""
    
    def test_create_config(self):
        """Test creating a regime configuration."""
        config = RegimeExitConfig(
            k1_SL=1.5,
            k2_TP=3.0,
            breakeven_R=0.5,
            trailing_multiplier=2.0,
            max_duration_hours=24,
            description="Test config"
        )
        
        assert config.k1_SL == 1.5
        assert config.k2_TP == 3.0
        assert config.breakeven_R == 0.5
        assert config.trailing_multiplier == 2.0
        assert config.max_duration_hours == 24
        assert config.description == "Test config"


class TestDefaultConfigurations:
    """Test default regime configurations."""
    
    def test_all_regimes_have_configs(self):
        """Test that all expected regimes have configurations."""
        expected_regimes = [
            "LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "EXTREME_VOL",
            "TRENDING", "RANGING"
        ]
        
        for regime in expected_regimes:
            assert regime in DEFAULT_REGIME_CONFIGS
            assert isinstance(DEFAULT_REGIME_CONFIGS[regime], RegimeExitConfig)
    
    def test_low_vol_config(self):
        """Test LOW_VOL configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["LOW_VOL"]
        
        assert config.k1_SL == 1.2  # Tighter stop
        assert config.k2_TP == 3.5  # Larger target
        assert config.max_duration_hours == 48  # Longer hold
        assert config.k2_TP / config.k1_SL > 2.0  # Good R/R
    
    def test_normal_vol_config(self):
        """Test NORMAL_VOL configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["NORMAL_VOL"]
        
        assert config.k1_SL == 1.5
        assert config.k2_TP == 3.0
        assert config.breakeven_R == 0.5
        assert config.max_duration_hours == 36
    
    def test_high_vol_config(self):
        """Test HIGH_VOL configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["HIGH_VOL"]
        
        assert config.k1_SL == 2.0  # Wider stop
        assert config.k2_TP == 4.0  # Larger target
        assert config.max_duration_hours == 24  # Shorter hold
        assert config.k2_TP / config.k1_SL == 2.0
    
    def test_extreme_vol_config(self):
        """Test EXTREME_VOL configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["EXTREME_VOL"]
        
        assert config.k1_SL == 2.5  # Very wide stop
        assert config.k2_TP == 5.0  # Very large target
        assert config.max_duration_hours == 12  # Very short hold
    
    def test_trending_config(self):
        """Test TRENDING configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["TRENDING"]
        
        assert config.k1_SL > DEFAULT_REGIME_CONFIGS["NORMAL_VOL"].k1_SL  # Wider for trends
        assert config.k2_TP > DEFAULT_REGIME_CONFIGS["NORMAL_VOL"].k2_TP  # Larger targets
        assert config.max_duration_hours == 72  # Much longer hold
    
    def test_ranging_config(self):
        """Test RANGING configuration parameters."""
        config = DEFAULT_REGIME_CONFIGS["RANGING"]
        
        assert config.k1_SL == 1.0  # Tight stop
        assert config.k2_TP == 2.0  # Smaller target
        assert config.max_duration_hours == 18  # Short hold


class TestGetExitParams:
    """Test get_exit_params function."""
    
    def test_get_normal_vol_params(self):
        """Test getting NORMAL_VOL parameters."""
        params = get_exit_params("NORMAL_VOL")
        
        assert isinstance(params, RegimeExitConfig)
        assert params.k1_SL == 1.5
        assert params.k2_TP == 3.0
    
    def test_get_high_vol_params(self):
        """Test getting HIGH_VOL parameters."""
        params = get_exit_params("HIGH_VOL")
        
        assert params.k1_SL == 2.0
        assert params.k2_TP == 4.0
    
    def test_unknown_regime_raises_error(self):
        """Test that unknown regime raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_exit_params("UNKNOWN_REGIME")
        
        assert "Unknown regime" in str(exc_info.value)
        assert "UNKNOWN_REGIME" in str(exc_info.value)
    
    def test_custom_configs(self):
        """Test using custom configurations."""
        custom = {
            "TEST_REGIME": RegimeExitConfig(
                k1_SL=1.0,
                k2_TP=2.0,
                breakeven_R=0.3,
                trailing_multiplier=1.5,
                max_duration_hours=12,
                description="Custom test"
            )
        }
        
        params = get_exit_params("TEST_REGIME", custom_configs=custom)
        assert params.k1_SL == 1.0
        assert params.k2_TP == 2.0


class TestGetAllRegimes:
    """Test get_all_regimes function."""
    
    def test_returns_all_regime_names(self):
        """Test that all regime names are returned."""
        regimes = get_all_regimes()
        
        assert isinstance(regimes, list)
        assert len(regimes) == 6
        assert "LOW_VOL" in regimes
        assert "NORMAL_VOL" in regimes
        assert "HIGH_VOL" in regimes
        assert "EXTREME_VOL" in regimes
        assert "TRENDING" in regimes
        assert "RANGING" in regimes


class TestGetRiskRewardRatio:
    """Test get_risk_reward_ratio function."""
    
    def test_normal_vol_rr(self):
        """Test risk/reward ratio for NORMAL_VOL."""
        rr = get_risk_reward_ratio("NORMAL_VOL")
        assert rr == 2.0  # 3.0 / 1.5
    
    def test_high_vol_rr(self):
        """Test risk/reward ratio for HIGH_VOL."""
        rr = get_risk_reward_ratio("HIGH_VOL")
        assert rr == 2.0  # 4.0 / 2.0
    
    def test_trending_rr(self):
        """Test risk/reward ratio for TRENDING."""
        rr = get_risk_reward_ratio("TRENDING")
        assert rr == pytest.approx(2.5, abs=0.1)  # 4.5 / 1.8


class TestValidateConfig:
    """Test validate_config function."""
    
    def test_valid_config(self):
        """Test that valid config has no errors."""
        config = RegimeExitConfig(
            k1_SL=1.5,
            k2_TP=3.0,
            breakeven_R=0.5,
            trailing_multiplier=2.0,
            max_duration_hours=24,
            description="Valid"
        )
        
        errors = validate_config(config)
        assert errors == []
    
    def test_negative_k1_sl(self):
        """Test that negative k1_SL is invalid."""
        config = RegimeExitConfig(
            k1_SL=-1.0,
            k2_TP=3.0,
            breakeven_R=0.5,
            trailing_multiplier=2.0,
            max_duration_hours=24,
            description="Invalid"
        )
        
        errors = validate_config(config)
        assert any("k1_SL must be positive" in e for e in errors)
    
    def test_k2_tp_less_than_k1_sl(self):
        """Test that k2_TP <= k1_SL is invalid."""
        config = RegimeExitConfig(
            k1_SL=2.0,
            k2_TP=1.5,  # Less than k1_SL
            breakeven_R=0.5,
            trailing_multiplier=2.0,
            max_duration_hours=24,
            description="Invalid"
        )
        
        errors = validate_config(config)
        assert any("Risk/reward ratio" in e and "too low" in e for e in errors)
    
    def test_all_default_configs_are_valid(self):
        """Test that all default configurations are valid."""
        for regime, config in DEFAULT_REGIME_CONFIGS.items():
            errors = validate_config(config)
            assert errors == [], f"Regime {regime} has validation errors: {errors}"


class TestLoadFromEnv:
    """Test load_from_env function."""
    
    def test_load_defaults_without_env(self):
        """Test loading defaults when no env variables set."""
        # Clear any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("EXIT_REGIME_"):
                del os.environ[key]
        
        configs = load_from_env()
        
        # Should match defaults
        assert configs["NORMAL_VOL"].k1_SL == 1.5
        assert configs["NORMAL_VOL"].k2_TP == 3.0
    
    def test_override_k1_sl_from_env(self):
        """Test overriding k1_SL from environment."""
        os.environ["EXIT_REGIME_NORMAL_VOL_K1_SL"] = "2.5"
        
        try:
            configs = load_from_env()
            assert configs["NORMAL_VOL"].k1_SL == 2.5
            # Other params should remain default
            assert configs["NORMAL_VOL"].k2_TP == 3.0
        finally:
            del os.environ["EXIT_REGIME_NORMAL_VOL_K1_SL"]
    
    def test_override_multiple_params(self):
        """Test overriding multiple parameters."""
        os.environ["EXIT_REGIME_HIGH_VOL_K1_SL"] = "2.5"
        os.environ["EXIT_REGIME_HIGH_VOL_K2_TP"] = "5.0"
        os.environ["EXIT_REGIME_HIGH_VOL_MAX_DURATION_HOURS"] = "18"
        
        try:
            configs = load_from_env()
            assert configs["HIGH_VOL"].k1_SL == 2.5
            assert configs["HIGH_VOL"].k2_TP == 5.0
            assert configs["HIGH_VOL"].max_duration_hours == 18
            # Description should indicate override
            assert "env override" in configs["HIGH_VOL"].description
        finally:
            del os.environ["EXIT_REGIME_HIGH_VOL_K1_SL"]
            del os.environ["EXIT_REGIME_HIGH_VOL_K2_TP"]
            del os.environ["EXIT_REGIME_HIGH_VOL_MAX_DURATION_HOURS"]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_calculate_stops_and_targets_btc(self):
        """Test calculating stops and targets for BTC trade."""
        params = get_exit_params("NORMAL_VOL")
        
        entry_price = 50000.0
        atr = 1000.0
        
        stop_loss = entry_price - (params.k1_SL * atr)
        take_profit = entry_price + (params.k2_TP * atr)
        
        assert stop_loss == 48500.0  # 50000 - (1.5 * 1000)
        assert take_profit == 53000.0  # 50000 + (3.0 * 1000)
        
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        assert reward / risk == 2.0
    
    def test_high_vol_wider_stops(self):
        """Test that HIGH_VOL gives wider stops than NORMAL_VOL."""
        normal = get_exit_params("NORMAL_VOL")
        high = get_exit_params("HIGH_VOL")
        
        entry = 50000.0
        atr = 1000.0
        
        normal_sl = entry - (normal.k1_SL * atr)
        high_sl = entry - (high.k1_SL * atr)
        
        # High vol stop should be further away (lower price for LONG)
        assert high_sl < normal_sl
        assert (entry - high_sl) > (entry - normal_sl)
    
    def test_trending_longer_holds(self):
        """Test that TRENDING allows much longer holds."""
        normal = get_exit_params("NORMAL_VOL")
        trending = get_exit_params("TRENDING")
        
        # Trending should allow 2x longer holds
        assert trending.max_duration_hours > normal.max_duration_hours * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
