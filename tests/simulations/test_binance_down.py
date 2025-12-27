"""
Test: Binance API Down/Rate Limited Scenario
Sprint 3 Part 3 - Failure Simulation

Simulates Binance API failures and verifies:
1. SafeOrderExecutor handles -1003/-1015 errors
2. Global Rate Limiter prevents spam
3. Retry logic is policy-controlled
4. Logging contains [BINANCE][RATE_LIMIT] entries
5. Monitoring receives dependency-failure alerts
6. ESS doesn't trip unless actual drawdown

Run:
    pytest tests/simulations/test_binance_down.py -v -s
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone

from tests.simulations.harness import (
    FailureSimulationHarness,
    BinanceDownConfig,
    ScenarioStatus
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class TestBinanceDown:
    """Test suite for Binance API failure scenario"""
    
    @pytest.fixture
    async def harness(self):
        """Create test harness"""
        harness = FailureSimulationHarness()
        yield harness
    
    @pytest.mark.asyncio
    async def test_binance_down_default_config(self, harness):
        """
        Test Binance API failures with default configuration.
        
        Expected:
        - SafeOrderExecutor retries limited times (3x per attempt)
        - Rate limiter throttles requests
        - Logging shows [BINANCE][RATE_LIMIT] entries
        - Monitoring alert published
        - ESS doesn't trip (no actual loss)
        """
        result = await harness.run_binance_down_scenario()
        
        # Assertions
        assert result.status == ScenarioStatus.PASSED, f"Scenario failed: {result.errors}"
        assert result.checks_failed == 0, f"Checks failed: {result.checks_failed}"
        assert result.metrics["total_binance_errors"] > 0, "Should record Binance errors"
        assert result.metrics["avg_retries"] <= 3.0, "Retries should be limited"
        
        # Log results
        print(f"\n{'='*60}")
        print(f"BINANCE DOWN TEST RESULTS")
        print(f"{'='*60}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Checks: {result.checks_passed}/{result.checks_passed + result.checks_failed}")
        print(f"Binance errors: {result.metrics['total_binance_errors']}")
        print(f"Avg retries: {result.metrics['avg_retries']:.1f}")
        print(f"Rate limited: {result.metrics.get('rate_limited_requests', 0)}")
        print(f"\nObservations:")
        for obs in result.observations:
            print(f"  {obs}")
        print(f"{'='*60}\n")
    
    @pytest.mark.asyncio
    async def test_binance_down_rate_limit_1003(self, harness):
        """
        Test rate limit error -1003 specifically.
        
        Expected:
        - Error code -1003 logged
        - Exponential backoff applied
        - Request eventually gives up
        """
        config = BinanceDownConfig(
            error_codes=[-1003],
            failure_duration_seconds=30.0,
            trade_attempts=3,
            expected_retries_per_attempt=3
        )
        
        result = await harness.run_binance_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert harness.metrics["binance_errors"] > 0
        
        print(f"\nRate limit -1003: {harness.metrics['binance_errors']} errors logged")
    
    @pytest.mark.asyncio
    async def test_binance_down_ip_ban_1015(self, harness):
        """
        Test IP ban error -1015.
        
        Expected:
        - Error code -1015 logged
        - Longer backoff applied
        - No spam to Binance
        """
        config = BinanceDownConfig(
            error_codes=[-1015],
            failure_duration_seconds=60.0,
            trade_attempts=5,
            expected_retries_per_attempt=2  # Less retries for IP ban
        )
        
        result = await harness.run_binance_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["avg_retries"] <= 2.0, "Should limit retries for IP ban"
    
    @pytest.mark.asyncio
    async def test_binance_down_mixed_errors(self, harness):
        """
        Test mixed error codes (-1003, -1015).
        
        Expected:
        - Both error types handled
        - Different retry strategies applied
        - System remains stable
        """
        config = BinanceDownConfig(
            error_codes=[-1003, -1015, -1003, -1015],
            trade_attempts=8,
            expected_retries_per_attempt=3
        )
        
        result = await harness.run_binance_down_scenario(config)
        
        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["total_binance_errors"] >= 8, "Should log multiple error types"
    
    @pytest.mark.asyncio
    async def test_binance_down_rate_limiter_prevents_spam(self, harness):
        """
        Test that rate limiter prevents Binance spam.
        
        Expected:
        - Rapid requests throttled
        - Only limited requests allowed per second
        - Excess requests queued or rejected
        """
        result = await harness.run_binance_down_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        # Check rate limiting observations
        rate_limit_obs = [obs for obs in result.observations if "rate limit" in obs.lower()]
        assert len(rate_limit_obs) > 0, "Should have rate limiting observations"
        
        print(f"\nRate limiter observations:")
        for obs in rate_limit_obs:
            print(f"  {obs}")
    
    @pytest.mark.asyncio
    async def test_binance_down_monitoring_alert(self, harness):
        """
        Test monitoring receives external dependency failure alert.
        
        Expected:
        - Alert published with error details
        - Alert includes error codes and counts
        - Monitoring can track Binance health
        """
        result = await harness.run_binance_down_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        assert harness.metrics["health_alerts"] > 0, "Should publish monitoring alert"
        
        print(f"\nMonitoring alerts: {harness.metrics['health_alerts']}")
    
    @pytest.mark.asyncio
    async def test_binance_down_no_ess_trip(self, harness):
        """
        Test that ESS doesn't trip from API errors alone.
        
        Expected:
        - Binance errors logged
        - No actual trading loss
        - ESS remains ARMED
        - Orders can resume after recovery
        """
        result = await harness.run_binance_down_scenario()
        
        assert result.status == ScenarioStatus.PASSED
        
        # ESS should not trip (no drawdown from API errors)
        ess_trip_obs = [obs for obs in result.observations if "ESS" in obs and "not trip" in obs]
        assert len(ess_trip_obs) > 0, "Should confirm ESS did not trip"
        
        print(f"\nESS status: {ess_trip_obs[0] if ess_trip_obs else 'Not checked'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
