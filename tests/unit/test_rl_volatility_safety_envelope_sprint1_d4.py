"""
Tests for RL Volatility Safety Envelope - SPRINT 1 D4

Tests volatility-based safety limits for RL position sizing.
"""

import pytest
from unittest.mock import Mock, MagicMock
from backend.services.risk.rl_volatility_safety_envelope import (
    RLVolatilitySafetyEnvelope,
    VolatilityBucket,
    VolatilityLimits,
    EnvelopeResult,
    get_rl_volatility_envelope
)


class TestVolatilityBucketClassification:
    """Test volatility bucket classification based on ATR/price."""
    
    def test_low_volatility_bucket(self):
        """Test LOW volatility bucket (< 0.5% ATR)."""
        envelope = RLVolatilitySafetyEnvelope()
        
        # 0.3% ATR should be LOW
        bucket = envelope.get_volatility_bucket(atr_pct=0.003)
        assert bucket == VolatilityBucket.LOW
        
        # Edge case: 0.49% is still LOW
        bucket = envelope.get_volatility_bucket(atr_pct=0.0049)
        assert bucket == VolatilityBucket.LOW
    
    def test_normal_volatility_bucket(self):
        """Test NORMAL volatility bucket (0.5-1.5% ATR)."""
        envelope = RLVolatilitySafetyEnvelope()
        
        # 0.8% ATR should be NORMAL
        bucket = envelope.get_volatility_bucket(atr_pct=0.008)
        assert bucket == VolatilityBucket.NORMAL
        
        # 1.0% ATR should be NORMAL
        bucket = envelope.get_volatility_bucket(atr_pct=0.01)
        assert bucket == VolatilityBucket.NORMAL
        
        # Edge case: 1.4% is still NORMAL
        bucket = envelope.get_volatility_bucket(atr_pct=0.014)
        assert bucket == VolatilityBucket.NORMAL
    
    def test_high_volatility_bucket(self):
        """Test HIGH volatility bucket (1.5-3.0% ATR)."""
        envelope = RLVolatilitySafetyEnvelope()
        
        # 2.0% ATR should be HIGH
        bucket = envelope.get_volatility_bucket(atr_pct=0.02)
        assert bucket == VolatilityBucket.HIGH
        
        # 2.5% ATR should be HIGH
        bucket = envelope.get_volatility_bucket(atr_pct=0.025)
        assert bucket == VolatilityBucket.HIGH
        
        # Edge case: 2.9% is still HIGH
        bucket = envelope.get_volatility_bucket(atr_pct=0.029)
        assert bucket == VolatilityBucket.HIGH
    
    def test_extreme_volatility_bucket(self):
        """Test EXTREME volatility bucket (> 3.0% ATR)."""
        envelope = RLVolatilitySafetyEnvelope()
        
        # 3.5% ATR should be EXTREME
        bucket = envelope.get_volatility_bucket(atr_pct=0.035)
        assert bucket == VolatilityBucket.EXTREME
        
        # 5.0% ATR should be EXTREME
        bucket = envelope.get_volatility_bucket(atr_pct=0.05)
        assert bucket == VolatilityBucket.EXTREME


class TestPolicyStoreLimits:
    """Test PolicyStore integration for volatility limits."""
    
    @pytest.fixture
    def mock_policy_store(self):
        """Create mock PolicyStore."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            # LOW bucket
            "volatility.low.max_leverage": 25.0,
            "volatility.low.max_risk_pct": 0.10,
            # NORMAL bucket
            "volatility.normal.max_leverage": 20.0,
            "volatility.normal.max_risk_pct": 0.08,
            # HIGH bucket
            "volatility.high.max_leverage": 15.0,
            "volatility.high.max_risk_pct": 0.05,
            # EXTREME bucket
            "volatility.extreme.max_leverage": 10.0,
            "volatility.extreme.max_risk_pct": 0.03,
        }.get(key, default))
        return policy_store
    
    def test_low_volatility_limits(self, mock_policy_store):
        """Test limits for LOW volatility bucket."""
        envelope = RLVolatilitySafetyEnvelope(mock_policy_store)
        
        limits = envelope.get_limits_for_bucket(VolatilityBucket.LOW)
        
        assert limits.bucket == VolatilityBucket.LOW
        assert limits.max_leverage == 25.0
        assert limits.max_risk_pct == 0.10
    
    def test_normal_volatility_limits(self, mock_policy_store):
        """Test limits for NORMAL volatility bucket."""
        envelope = RLVolatilitySafetyEnvelope(mock_policy_store)
        
        limits = envelope.get_limits_for_bucket(VolatilityBucket.NORMAL)
        
        assert limits.bucket == VolatilityBucket.NORMAL
        assert limits.max_leverage == 20.0
        assert limits.max_risk_pct == 0.08
    
    def test_high_volatility_limits(self, mock_policy_store):
        """Test limits for HIGH volatility bucket."""
        envelope = RLVolatilitySafetyEnvelope(mock_policy_store)
        
        limits = envelope.get_limits_for_bucket(VolatilityBucket.HIGH)
        
        assert limits.bucket == VolatilityBucket.HIGH
        assert limits.max_leverage == 15.0
        assert limits.max_risk_pct == 0.05
    
    def test_extreme_volatility_limits(self, mock_policy_store):
        """Test limits for EXTREME volatility bucket."""
        envelope = RLVolatilitySafetyEnvelope(mock_policy_store)
        
        limits = envelope.get_limits_for_bucket(VolatilityBucket.EXTREME)
        
        assert limits.bucket == VolatilityBucket.EXTREME
        assert limits.max_leverage == 10.0
        assert limits.max_risk_pct == 0.03
    
    def test_custom_policy_limits(self):
        """Test custom policy limits override defaults."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "volatility.high.max_leverage": 12.0,  # Custom lower limit
            "volatility.high.max_risk_pct": 0.04,  # Custom lower limit
        }.get(key, default))
        
        envelope = RLVolatilitySafetyEnvelope(policy_store)
        limits = envelope.get_limits_for_bucket(VolatilityBucket.HIGH)
        
        assert limits.max_leverage == 12.0  # Custom value
        assert limits.max_risk_pct == 0.04  # Custom value
    
    def test_fallback_to_defaults_without_policy_store(self):
        """Test fallback to default limits when PolicyStore unavailable."""
        envelope = RLVolatilitySafetyEnvelope(policy_store=None)
        
        limits = envelope.get_limits_for_bucket(VolatilityBucket.HIGH)
        
        # Should use hardcoded defaults
        assert limits.bucket == VolatilityBucket.HIGH
        assert limits.max_leverage == 15.0
        assert limits.max_risk_pct == 0.05


class TestApplyLimits:
    """Test applying volatility-based limits to RL proposals."""
    
    @pytest.fixture
    def envelope(self):
        """Create envelope with mock PolicyStore."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "volatility.low.max_leverage": 25.0,
            "volatility.low.max_risk_pct": 0.10,
            "volatility.normal.max_leverage": 20.0,
            "volatility.normal.max_risk_pct": 0.08,
            "volatility.high.max_leverage": 15.0,
            "volatility.high.max_risk_pct": 0.05,
            "volatility.extreme.max_leverage": 10.0,
            "volatility.extreme.max_risk_pct": 0.03,
        }.get(key, default))
        return RLVolatilitySafetyEnvelope(policy_store)
    
    def test_no_capping_in_low_volatility(self, envelope):
        """Test no capping when volatility is LOW and RL proposal is within limits."""
        result = envelope.apply_limits(
            symbol="BTCUSDT",
            atr_pct=0.003,  # 0.3% ATR = LOW
            proposed_leverage=20.0,  # < 25.0 max
            proposed_risk_pct=0.08,  # < 0.10 max
            equity_usd=10000.0
        )
        
        assert result.volatility_bucket == VolatilityBucket.LOW
        assert result.capped_leverage == 20.0  # Unchanged
        assert result.capped_risk_pct == 0.08  # Unchanged
        assert result.was_capped is False
    
    def test_leverage_capping_in_high_volatility(self, envelope):
        """Test leverage capping when volatility is HIGH."""
        result = envelope.apply_limits(
            symbol="ETHUSDT",
            atr_pct=0.025,  # 2.5% ATR = HIGH
            proposed_leverage=20.0,  # > 15.0 max for HIGH
            proposed_risk_pct=0.04,  # < 0.05 max
            equity_usd=10000.0
        )
        
        assert result.volatility_bucket == VolatilityBucket.HIGH
        assert result.capped_leverage == 15.0  # Capped down
        assert result.capped_risk_pct == 0.04  # Unchanged
        assert result.was_capped is True
    
    def test_risk_capping_in_extreme_volatility(self, envelope):
        """Test risk % capping when volatility is EXTREME."""
        result = envelope.apply_limits(
            symbol="SOLUSDT",
            atr_pct=0.04,  # 4.0% ATR = EXTREME
            proposed_leverage=8.0,  # < 10.0 max
            proposed_risk_pct=0.05,  # > 0.03 max for EXTREME
            equity_usd=10000.0
        )
        
        assert result.volatility_bucket == VolatilityBucket.EXTREME
        assert result.capped_leverage == 8.0  # Unchanged
        assert result.capped_risk_pct == 0.03  # Capped down
        assert result.was_capped is True
    
    def test_both_leverage_and_risk_capping(self, envelope):
        """Test both leverage and risk % capping simultaneously."""
        result = envelope.apply_limits(
            symbol="BTCUSDT",
            atr_pct=0.035,  # 3.5% ATR = EXTREME
            proposed_leverage=25.0,  # > 10.0 max for EXTREME
            proposed_risk_pct=0.10,  # > 0.03 max for EXTREME
            equity_usd=10000.0
        )
        
        assert result.volatility_bucket == VolatilityBucket.EXTREME
        assert result.capped_leverage == 10.0  # Capped down
        assert result.capped_risk_pct == 0.03  # Capped down
        assert result.was_capped is True
        assert result.original_leverage == 25.0
        assert result.original_risk_pct == 0.10


class TestRLIntegration:
    """Test integration with RL Position Sizing Agent outputs."""
    
    @pytest.fixture
    def envelope(self):
        """Create envelope with test PolicyStore."""
        policy_store = Mock()
        policy_store.get = Mock(side_effect=lambda key, default=None: {
            "volatility.normal.max_leverage": 20.0,
            "volatility.normal.max_risk_pct": 0.08,
            "volatility.high.max_leverage": 15.0,
            "volatility.high.max_risk_pct": 0.05,
        }.get(key, default))
        return RLVolatilitySafetyEnvelope(policy_store)
    
    def test_rl_aggressive_proposal_in_normal_volatility(self, envelope):
        """Test RL aggressive proposal (25x, 10%) in NORMAL volatility."""
        # Simulate RL agent output
        rl_leverage = 25.0
        rl_risk_pct = 0.10
        
        # Apply envelope
        result = envelope.apply_limits(
            symbol="BTCUSDT",
            atr_pct=0.01,  # 1.0% ATR = NORMAL
            proposed_leverage=rl_leverage,
            proposed_risk_pct=rl_risk_pct,
            equity_usd=10000.0
        )
        
        # Should be capped to NORMAL limits
        assert result.capped_leverage == 20.0  # Capped from 25x
        assert result.capped_risk_pct == 0.08  # Capped from 10%
        assert result.was_capped is True
    
    def test_rl_conservative_proposal_passes_through(self, envelope):
        """Test RL conservative proposal passes through without capping."""
        # Simulate conservative RL output
        rl_leverage = 10.0
        rl_risk_pct = 0.03
        
        # Apply envelope in HIGH volatility
        result = envelope.apply_limits(
            symbol="ETHUSDT",
            atr_pct=0.02,  # 2.0% ATR = HIGH
            proposed_leverage=rl_leverage,
            proposed_risk_pct=rl_risk_pct,
            equity_usd=10000.0
        )
        
        # Should pass through (already below HIGH limits)
        assert result.capped_leverage == 10.0  # Unchanged
        assert result.capped_risk_pct == 0.03  # Unchanged
        assert result.was_capped is False
    
    def test_calculate_final_position_size(self, envelope):
        """Test final position size calculation with capped values."""
        # Apply envelope
        result = envelope.apply_limits(
            symbol="BTCUSDT",
            atr_pct=0.025,  # HIGH volatility
            proposed_leverage=20.0,
            proposed_risk_pct=0.08,
            equity_usd=10000.0
        )
        
        # Calculate final position size
        margin_usd, quantity = envelope.calculate_capped_position_size(
            equity_usd=10000.0,
            capped_risk_pct=result.capped_risk_pct,
            capped_leverage=result.capped_leverage,
            price=50000.0  # BTC price
        )
        
        # Verify calculations
        # HIGH vol: max_risk=5%, max_lev=15x
        assert result.capped_risk_pct == 0.05  # 5% of equity
        assert result.capped_leverage == 15.0  # 15x leverage
        
        expected_margin = 10000.0 * 0.05  # $500
        expected_notional = expected_margin * 15.0  # $7,500
        expected_quantity = expected_notional / 50000.0  # 0.15 BTC
        
        assert margin_usd == pytest.approx(expected_margin, rel=0.01)
        assert quantity == pytest.approx(expected_quantity, rel=0.01)


class TestSingletonPattern:
    """Test singleton pattern for envelope instance."""
    
    def test_singleton_returns_same_instance(self):
        """Test get_rl_volatility_envelope returns same instance."""
        envelope1 = get_rl_volatility_envelope()
        envelope2 = get_rl_volatility_envelope()
        
        assert envelope1 is envelope2
    
    def test_singleton_with_policy_store(self):
        """Test singleton initialization with PolicyStore."""
        policy_store = Mock()
        
        # Reset singleton for test
        import backend.services.risk.rl_volatility_safety_envelope as module
        module._envelope_instance = None
        
        envelope = get_rl_volatility_envelope(policy_store)
        
        assert envelope.policy_store is policy_store


class TestEnvelopeStatus:
    """Test envelope status and diagnostics."""
    
    def test_get_status_returns_configuration(self):
        """Test get_status returns envelope configuration."""
        envelope = RLVolatilitySafetyEnvelope()
        
        status = envelope.get_status()
        
        assert "thresholds" in status
        assert "limits_cached" in status
        assert "policy_store_available" in status
        assert status["thresholds"]["low"] == "0.50%"
        assert status["thresholds"]["normal"] == "1.50%"
        assert status["thresholds"]["high"] == "3.00%"
    
    def test_status_shows_cached_limits(self):
        """Test status shows cached limits after usage."""
        policy_store = Mock()
        policy_store.get = Mock(return_value=20.0)
        
        envelope = RLVolatilitySafetyEnvelope(policy_store)
        
        # Trigger cache by getting limits
        envelope.get_limits_for_bucket(VolatilityBucket.NORMAL)
        envelope.get_limits_for_bucket(VolatilityBucket.HIGH)
        
        status = envelope.get_status()
        
        assert status["limits_cached"] == 2
        assert "NORMAL" in status["cached_limits"]
        assert "HIGH" in status["cached_limits"]
